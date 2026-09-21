import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import PERIODES, PERIODE_VOLGORDE, get_filter_options, query_data

st.set_page_config(layout="wide", page_title="Risicoanalyse")

st.header("Risicoanalyse")

# Lichte filteropties ophalen zonder de volledige dataset te laden.
filter_options = get_filter_options()
beschikbare_jaren = sorted(filter_options["jaren"], reverse=True)

if not beschikbare_jaren:
    st.error("🚨 Kritieke fout: geen beschikbare jaren gevonden in Parquet.")
    st.stop()

# Sidebar met behoud van dezelfde labels, defaults en session-state-key.
st.sidebar.header("📅 Filter op jaren")
geselecteerde_jaren = st.sidebar.multiselect(
    "Selecteer gewenste jaren:",
    options=beschikbare_jaren,
    default=beschikbare_jaren,
)

# Gedeelde periodekeuze. De vaste key bewaart de selectie tussen pagina's die
# dezelfde centrale PERIODES-definitie en session-state-key gebruiken.
geselecteerde_periodes = st.sidebar.multiselect(
    "Selecteer gewenste seizoenen of halfjaren:",
    options=list(PERIODE_VOLGORDE),
    default=list(PERIODE_VOLGORDE),
    key="shared_periodes_filter",
    help=(
        "Winter: december t/m februari; voorjaar: maart t/m mei; "
        "zomer: juni t/m augustus; herfst: september t/m november; "
        "zomerhalfjaar: april t/m september; "
        "winterhalfjaar: oktober t/m maart. "
        "Bij meerdere keuzes worden de maanden gecombineerd."
    ),
)
st.sidebar.markdown("---")
st.sidebar.info("Navigeer via het menu hierboven naar de verschillende analyses.")

# Predicate- en projection-pushdown. Alleen voor deze risicoanalyse relevante
# kolommen worden uit Parquet gelezen. Een lege jaarselectie betekent, net als
# voorheen, dat er geen jaarbeperking wordt toegepast.
RISICO_COLUMNS = (
    "Datum",
    "Meetpunt",
    "Stof",
    "Stofgroep",
    "Waarde",
    "Eenheid",
    "Limietsymbool",
    "Signaleringswaarde",
)
df_filtered = query_data(
    jaren=tuple(geselecteerde_jaren),
    periodes=tuple(geselecteerde_periodes),
    kolommen=RISICO_COLUMNS,
)

if df_filtered.empty:
    st.error("🚨 Kritieke fout: de geselecteerde meetgegevens zijn leeg.")
    st.stop()

st.header("⚠️ Risicoanalyse op basis van signaleringswaarden)")
st.markdown("Deze analyse toont stoffen waarvan **normen** bekend zijn. Deze stoffen worden getoetst aan een generieke signaleringswaarde van **0.1 ug/l**.")

# 1. Basis data voorbereiden
df_risico_basis = df_filtered.dropna(subset=['Signaleringswaarde']).copy()
# Filter < rapportagegrens weg (Vectorized is sneller)
mask_detectie_risico = ~df_risico_basis['Limietsymbool'].astype(str).str.contains('<', na=False)
df_risico_basis = df_risico_basis[mask_detectie_risico]

col_risico_1, col_risico_2, col_risico_3 = st.columns(3)

# Filters
opts_mp = sorted(df_risico_basis['Meetpunt'].unique())
sel_mp = col_risico_1.multiselect("📍 Selecteer meetpunt(en)", opts_mp, default=opts_mp, key="tab4_meetpunt")

opts_grp = sorted(df_risico_basis['Stofgroep'].unique())
sel_grp = col_risico_2.multiselect("📂 Selecteer stofgroep", opts_grp, default=opts_grp, key="tab4_stofgroep")

# Filter tussentijds om stof-opties te beperken
df_risico = df_risico_basis[
    df_risico_basis['Meetpunt'].isin(sel_mp) & 
    df_risico_basis['Stofgroep'].isin(sel_grp)
].copy()

opts_stof = sorted(df_risico['Stof'].unique())
sel_stof = col_risico_3.multiselect("🔎 Selecteer stof(fen)", opts_stof, default=opts_stof, key="tab4_stof")

# Laatste filter
df_risico = df_risico[df_risico['Stof'].isin(sel_stof)]

st.markdown("---")

if not df_risico.empty:
    # Vectorized berekening van overschrijding
    df_risico['Boven_Signalering'] = df_risico['Waarde'] > df_risico['Signaleringswaarde']
    
    # Datum verwerking voor grafieken
    df_risico['Jaar'] = df_risico['Datum'].dt.year
    df_risico['Maand'] = df_risico['Datum'].dt.strftime('%Y-%m') # Voor chronologische sortering
    df_risico['MaandNr'] = df_risico['Datum'].dt.month # Voor aggregatie per maand
    maand_namen = {
        1: 'Jan', 2: 'Feb', 3: 'Mrt', 4: 'Apr', 5: 'Mei', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
    }
    df_risico['MaandNaam'] = df_risico['MaandNr'].map(maand_namen)


    # ---------------------------------------------------------
    # GRAFIEK 1: Risico Heatmap (Met tooltip van 'boosdoener')
    # ---------------------------------------------------------
    st.subheader("Risico-intensiteit per maand (heatmap)")
    st.markdown("Deze heatmap toont de **maximale overschrijding** per meetpunt en maand. Beweeg over een vakje om te zien **welke stof** de piek veroorzaakte.")

    # Bereken percentage t.o.v. de norm (100% = op de norm)
    df_risico['Percentage_van_drempelwaarde'] = (df_risico['Waarde'] / df_risico['Signaleringswaarde']) * 100
    
    # NIEUW: Filter de dataset om alleen overschrijdingen (> 100%) mee te nemen
    df_overschrijdingen = df_risico[df_risico['Percentage_van_drempelwaarde'] > 100].copy()

    # STAP 1: PRE-AGGREGATIE
    # We zoeken voor elke combinatie van Meetpunt & Maand de regel met de hoogste overschrijding.
    
    if not df_overschrijdingen.empty:
        # idxmax() zoekt de index van de maximale overschrijding
        idx_max = df_overschrijdingen.groupby(['Meetpunt', 'MaandNr'], observed=True)['Percentage_van_drempelwaarde'].idxmax().dropna()
    
        if not idx_max.empty:
            # Selecteer die specifieke rijen uit de originele dataset
            df_heatmap_data = df_overschrijdingen.loc[idx_max].copy()

            # Data voorbereiden voor px.imshow (vereist matrix-formaat)
            MaandNamen_Order = ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]
            MaandNr_Order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            # Pivot de data naar een matrix voor de Z-waarden (Percentage_van_drempelwaarde)
            df_pivot = df_heatmap_data.pivot_table(
                index='Meetpunt', 
                columns='MaandNr', 
                values='Percentage_van_drempelwaarde', 
                fill_value=0, # Vul ontbrekende (geen overschrijding) vakjes met 0%
                observed=True
            )
            # Herschik de kolommen op volgorde van de maandnummers
            df_pivot = df_pivot.reindex(columns=MaandNr_Order, fill_value=0)
            
            # Pivot de data naar een matrix voor de Stofnaam (de tooltip)
            df_pivot_stof = df_heatmap_data.pivot_table(
                index='Meetpunt', 
                columns='MaandNr', 
                values='Stof', 
                aggfunc=lambda x: x.iloc[0], # Selecteer de stofnaam die bij de max hoort
                observed=True
            )
            # Herschik en vul ontbrekende waarden met een duidelijke tekst
            df_pivot_stof = df_pivot_stof.reindex(columns=MaandNr_Order, fill_value="Geen overschrijding")
            
            # STAP 2: PLOTTEN met px.imshow
            fig_heat = px.imshow(
                df_pivot.values, # De matrix van Z-waarden
                x=MaandNamen_Order, # De X-as labels
                y=df_pivot.index,   # De Y-as labels (Meetpunten)
                color_continuous_scale='Reds',
                aspect="auto",
                labels=dict(
                    x="Maand", 
                    y="Meetpunt", 
                    color="Max % v. drempelwaarde"
                ),
                title='Risico-intensiteit (Rood = Grote overschrijding)',
                height=600
            )

            # Voeg customdata en hovertemplate achteraf toe via update_traces
            fig_heat.update_traces(
                customdata=df_pivot_stof.values,
                hovertemplate=(
                    "<b>Stof: %{customdata}</b><br>" +
                    "Meetpunt: %{y}<br>" +
                    "Maand: %{x}<br>" +
                    "Max % v. drempelwaarde: %{z:.0f}%<br>" +
                    "<extra></extra>"
                )
            )
            
            # Update de layout om de kleurschaal te beperken
            max_val = df_pivot.values.max()
            fig_heat.update_layout(
                coloraxis_colorbar_title="% v. drempelwaarde",
                xaxis_title=None,
                yaxis_title=None,
                # Zorg dat de kleur-as start bij de norm van 100%
                coloraxis={'cmin': 100, 'cmax': max_val} if max_val > 100 else {}
            )
            
            st.plotly_chart(fig_heat, width='stretch')
        else:
            st.info("Geen metingen beschikbaar in de geselecteerde filters met een geldige signaleringswaarde boven de drempelwaarde.")
    else:
        st.info("Geen metingen boven de signaleringswaarde gevonden om de heatmap te genereren.")

    st.markdown("---")
    
    # ---------------------------------------------------------
    # PRIORITEITENLIJST: TABEL VAN OVERTREDEN STOFFEN
    # ---------------------------------------------------------
    st.subheader("Prioriteringslijst")
    aggregatiemethode_risico = st.radio(
        "Aggregatiemethode voor risico-intensiteit:",
        options=["Gemiddelde", "Mediaan"],
        index=0,
        key="risico_aggregatiemethode",
        horizontal=True,
        help=(
            "Deze keuze bepaalt uitsluitend de centrale risico-intensiteit "
            "in de prioriteringslijst."
        ),
    )
    risico_agg_func = "mean" if aggregatiemethode_risico == "Gemiddelde" else "median"
    risico_agg_label = aggregatiemethode_risico.lower()
    st.markdown(
        "Deze tabel toont welke stoffen de meeste overschrijdingen veroorzaken "
        f"en hoe hoog de **{risico_agg_label} risico-intensiteit** daarbij is."
    )

    df_alleen_overschrijdingen = df_risico[df_risico['Boven_Signalering']].copy()

    if not df_alleen_overschrijdingen.empty:
        # Aggregeer de data per Stof
        df_prioriteit = df_alleen_overschrijdingen.groupby('Stof', observed=True).agg(
            Aantal_Overschrijdingen=('Stof', 'size'),
            Centrale_Perc_Overschrijding=('Percentage_van_drempelwaarde', risico_agg_func),
            Max_Perc_Overschrijding=('Percentage_van_drempelwaarde', 'max')
        ).reset_index()

        # Rond af, maar behoud numerieke waarden voor correcte sortering.
        df_prioriteit['Centrale_Perc_Overschrijding'] = df_prioriteit['Centrale_Perc_Overschrijding'].round(1)
        df_prioriteit['Max_Perc_Overschrijding'] = df_prioriteit['Max_Perc_Overschrijding'].round(1)

        # Vind de stofgroep en het meetpunt voor context (optioneel, maar nuttig)
        df_context = df_alleen_overschrijdingen.groupby('Stof', observed=True).agg(
            Stofgroep=('Stofgroep', lambda x: x.mode()[0]), # Meest voorkomende stofgroep
            Meetpunten=('Meetpunt', lambda x: ', '.join(sorted(x.unique()))) # Unieke meetpunten
        ).reset_index()

        # Merge contextuele data
        df_prioriteit = pd.merge(df_prioriteit, df_context, on='Stof')

        # Herschik en sorteer de kolommen
        df_prioriteit = df_prioriteit[[
            'Stof', 
            'Stofgroep',
            'Aantal_Overschrijdingen',
            'Centrale_Perc_Overschrijding',
            'Max_Perc_Overschrijding',
            'Meetpunten'
        ]].sort_values(by=['Aantal_Overschrijdingen', 'Max_Perc_Overschrijding'], ascending=[False, False])
        
        # Toon de top 20
        st.dataframe(
            df_prioriteit.head(20),
            width='stretch',
            column_config={
                "Stof": st.column_config.TextColumn("Stof", help="Naam van de stof"),
                "Stofgroep": st.column_config.TextColumn("Stofgroep"),
                "Aantal_Overschrijdingen": st.column_config.NumberColumn(
                    "Aantal Overschrijdingen",
                    help="Totaal aantal metingen boven de signaleringswaarde"
                ),
                "Centrale_Perc_Overschrijding": st.column_config.NumberColumn(
                    f"{aggregatiemethode_risico} % van drempelwaarde",
                    help=(
                        f"De {risico_agg_label} risico-intensiteit van alle "
                        "overschrijdende metingen voor deze stof."
                    ),
                    format="%.1f%%",
                ),
                "Max_Perc_Overschrijding": st.column_config.NumberColumn(
                    "Max % van drempelwaarde",
                    help="De hoogste geregistreerde risico-intensiteit voor deze stof.",
                    format="%.1f%%",
                ),
                "Meetpunten": st.column_config.TextColumn("Meetpunten"),
            }
        )

    else:
        st.info("Geen overschrijdingen gevonden om een prioriteitenlijst te maken.")

    st.markdown("---")

    # ---------------------------------------------------------
    # SPIDER CHARTS
    # ---------------------------------------------------------
    st.subheader("🕸️ Seizoenspatroon: signaleringswaardeoverschrijdingen per maand en jaar")
    st.info("Deze grafiek toont in welke maanden de meeste overschrijdingen plaatsvinden. Elke lijn vertegenwoordigt een jaar.")

    # Gebruik hier df_risico, maar filter eerst op Overschrijdingen voor de Spider Chart!
    df_trends_spider_basis = df_risico[df_risico['Boven_Signalering'] == True].copy()
    
    if not df_trends_spider_basis.empty:
        # 1. Maak de locatiefilter (gebruik sorted unique voor snelheid)
        alle_meetpunten = sorted(df_trends_spider_basis['Meetpunt'].unique())

        selected_meetpunten_spider = st.multiselect(
            "📍 Selecteer meetpunt(en) voor seizoensanalyse:",
            options=alle_meetpunten,
            # Let op: Default moet de Meetpunten bevatten die overschrijdingen hebben
            default=alle_meetpunten, 
            key="tab4_spider_meetpunt"
        )

        # Filter de data op basis van de selectie
        df_filtered_spider = df_trends_spider_basis[df_trends_spider_basis['Meetpunt'].isin(selected_meetpunten_spider)].copy()

        if not df_filtered_spider.empty:
            
            # 2. Data voorbereiden: Jaar toevoegen (MaandNr en MaandNaam zijn al berekend)
            df_filtered_spider['Jaar'] = df_filtered_spider['Datum'].dt.year

            # Meetpunt is category, dus .astype(str) is nodig voor concatenatie
            df_filtered_spider['Analyse_Groep'] = df_filtered_spider['Meetpunt'].astype(str) + ' (' + df_filtered_spider['Jaar'].astype(str) + ')'

            # 3. Aggregeren: Tel overschrijdingen per Analyse_Groep en Maand
            seasonal_counts = df_filtered_spider.groupby(['Analyse_Groep', 'MaandNr'], observed=True).size().reset_index(name='Aantal')
            
            # 4. Zorg dat ALLE maanden (1-12) bestaan voor elke Analyse_Groep (via MultiIndex)
            unieke_groepen = seasonal_counts['Analyse_Groep'].unique()
            full_index = pd.MultiIndex.from_product([unieke_groepen, range(1, 13)], names=['Analyse_Groep', 'MaandNr']).to_frame(index=False)
            
            # Merge de tellingen hierin, vul NaN op met 0
            df_radar = pd.merge(full_index, seasonal_counts, on=['Analyse_Groep', 'MaandNr'], how='left').fillna(0)
            
            # 5. Maandnummers omzetten naar Namen voor de grafiek
            df_radar['MaandNaam'] = df_radar['MaandNr'].map(maand_namen)
            
            # 6. Plotly Line Polar (Spider Chart)
            fig_radar = px.line_polar(
                df_radar, r='Aantal', theta='MaandNaam', color='Analyse_Groep',
                line_close=True, markers=True,
                title=f"Aantal signaleringsoverschrijdingen per maand",
                category_orders={"MaandNaam": ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]}
            )
            
            fig_radar.update_traces(fill='toself', opacity=0.3) 
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, df_radar['Aantal'].max() * 1.1])),
                legend_title_text='Meetpunt (Jaar)'
            )
            
            st.plotly_chart(fig_radar, width='stretch')
        else:
            st.warning("Selecteer één of meer meetpunten om de seizoensanalyse te zien.")
    else:
        st.info("Geen overschrijdingen gevonden om een seizoenspatroon te tonen.")
    
        
    # ---------------------------------------------------------
    # NIEUWE FUNCTIONALITEITEN (VERDIEPING)
    # ---------------------------------------------------------

    st.subheader("Verdieping singaleringswaardeoverschrijdingen")

    # Kolom voor status label
    df_risico['Status'] = np.where(df_risico['Boven_Signalering'], 'Overschrijding', 'Voldoet')

    col_a, col_b = st.columns([1, 2])

    with col_a:
        # GRAFIEK 2: Taartdiagram (Voldoet vs Niet Voldoet)
        st.markdown("**Verhouding voldoet vs. overschrijding**")
        fig_pie = px.pie(
            df_risico, 
            names='Status', 
            title='Totaal overzicht selectie',
            color='Status',
            color_discrete_map={'Voldoet': 'lightgreen', 'Overschrijding': 'crimson'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, width='stretch')

    with col_b:
        # GRAFIEK 3: Welke Stofgroepen veroorzaken de overschrijdingen?
        st.markdown("**Aantal overschrijdingen per stofgroep**")
        df_alleen_overschrijdingen = df_risico[df_risico['Boven_Signalering']].copy()
        
        if not df_alleen_overschrijdingen.empty:
            df_grp_count = df_alleen_overschrijdingen.groupby('Stofgroep', observed=True).size().reset_index(name='Aantal overschrijdingen')
            fig_grp = px.bar(
                df_grp_count.sort_values('Aantal overschrijdingen', ascending=True),
                x='Aantal overschrijdingen',
                y='Stofgroep',
                orientation='h',
                text_auto=True,
                color='Aantal overschrijdingen',
                color_continuous_scale='Reds'
            )
            fig_grp.update_layout(showlegend=False)
            st.plotly_chart(fig_grp, width='stretch')
        else:
            st.info("Geen overschrijdingen om weer te geven per stofgroep.")

    # ---------------------------------------------------------
    # GRAFIEK 4: Stapelgrafiek Tijdlijn (Maanden x Stofgroep)
    # ---------------------------------------------------------
    st.subheader("Trends in overschrijdingen")
    st.markdown("Onderstaande grafiek toont het **aantal overschrijdingen per maand**, opgebouwd uit de verschillende **stofgroepen**.")

    if not df_alleen_overschrijdingen.empty:
        # Groeperen op Maand en Stofgroep
        df_timeline = df_alleen_overschrijdingen.groupby(['Maand', 'Stofgroep'], observed=True).size().reset_index(name='Aantal')
        
        fig_stack = px.bar(
            df_timeline.sort_values('Maand'), # Sorteer chronologisch
            x='Maand',
            y='Aantal',
            color='Stofgroep',
            title='Aantal signaleringswaardeoverschrijdingen per maand (gestapeld per stofgroep)',
            labels={'Maand': 'Maand (Jaar-Mnd)', 'Aantal': 'Aantal overschrijdingen'},
            barmode='stack'
        )
        st.plotly_chart(fig_stack, width='stretch')
    else:
        st.info("Er zijn geen overschrijdingen in de geselecteerde periode.")
    
    # ---------------------------------------------------------
    # Trends in overschrijdingen per jaar
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Trends in overschrijdingen per jaar")
    aggregatiemethode_jaartrend = st.radio(
        "Aggregatiemethode voor de samenvattende jaarlijn:",
        options=["Gemiddelde", "Mediaan"],
        index=0,
        key="jaartrend_aggregatiemethode",
        horizontal=True,
        help=(
            "Deze keuze geldt alleen voor de opvallende samenvattende lijn "
            "in deze grafiek. De lijnen per meetpunt blijven ongewijzigd."
        ),
    )
    jaartrend_agg_func = (
        "mean"
        if aggregatiemethode_jaartrend == "Gemiddelde"
        else "median"
    )
    jaartrend_agg_label = aggregatiemethode_jaartrend.lower()
    st.info(
        "De dunne lijnen tonen het aantal overschrijdingen per individueel "
        "meetpunt. De dikke zwarte lijn toont het "
        f"**{jaartrend_agg_label} aantal overschrijdingen per meetpunt**."
    )

    # Filter alleen de regels die daadwerkelijk een overschrijding zijn.
    df_trends = df_risico[df_risico['Boven_Signalering']].copy()
    if not df_trends.empty:
        # Tel overschrijdingen per jaar en per meetpunt.
        counts_per_mp = (
            df_trends.groupby(['Jaar', 'Meetpunt'], observed=True)
            .size()
            .reset_index(name='Aantal_Overschrijdingen')
        )

        # Bereken de dedicated jaaraggregatie over de beschikbare meetpunten.
        centraal_per_year = (
            counts_per_mp.groupby('Jaar')['Aantal_Overschrijdingen']
            .agg(jaartrend_agg_func)
            .reset_index(name='Centraal_Aantal')
        )

        # Eén gecombineerde grafiek: individuele meetpunten als subtiele lijnen.
        fig_jaartrend = px.line(
            counts_per_mp,
            x='Jaar',
            y='Aantal_Overschrijdingen',
            color='Meetpunt',
            markers=True,
            title=(
                "Aantal signaleringswaardeoverschrijdingen per meetpunt "
                f"met {jaartrend_agg_label} referentielijn"
            ),
            labels={
                'Aantal_Overschrijdingen': 'Aantal overschrijdingen',
                'Jaar': 'Jaar',
            },
        )
        fig_jaartrend.update_traces(
            line=dict(width=1.5),
            marker=dict(size=6),
            opacity=0.55,
        )

        # De gekozen centrummaat springt visueel duidelijk uit.
        fig_jaartrend.add_trace(
            go.Scatter(
                x=centraal_per_year['Jaar'],
                y=centraal_per_year['Centraal_Aantal'],
                mode='lines+markers',
                name=f"{aggregatiemethode_jaartrend} alle meetpunten",
                line=dict(color='black', width=5),
                marker=dict(
                    color='white',
                    size=11,
                    line=dict(color='black', width=3),
                    symbol='diamond',
                ),
                hovertemplate=(
                    f"<b>{aggregatiemethode_jaartrend}</b><br>"
                    "Jaar: %{x}<br>"
                    "Aantal overschrijdingen: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        fig_jaartrend.update_xaxes(type='category', tickformat='d')
        fig_jaartrend.update_layout(
            legend_title_text='Meetpunt / samenvatting',
            hovermode='x unified',
        )
        st.plotly_chart(fig_jaartrend, width='stretch')
    else:
        st.info("Onvoldoende data om een trendgrafiek van overschrijdingen te maken.")
        
            
    # ---------------------------------------------
    # Actuele signaleringswaarde overschrijdingen
    # ---------------------------------------------
    st.markdown("---")
    st.subheader("Actuele signaleringswaarde overschrijdingen")
            
    df_overtredingen = df_risico[df_risico['Boven_Signalering']].sort_values('Datum', ascending=False)

    if not df_overtredingen.empty:
        st.dataframe(
            df_overtredingen[['Datum', 'Meetpunt', 'Stof', 'Waarde', 'Signaleringswaarde', 'Eenheid']],
            width='stretch'
        )
    else:
        st.success("Er zijn stoffen zonder norm aangetroffen, maar geen enkele meting kwam boven de drempelwaarde van 0.1 ug/l uit.")

else:
    st.warning("Geen data beschikbaar met de huidige filters.")
    st.warning("Geen aangetroffen metingen (boven rapportagegrens) gevonden voor stoffen zonder norm (ug/l).")