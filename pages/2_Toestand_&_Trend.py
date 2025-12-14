import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar, calculate_trends_optimized

# Pagina configuratie
st.set_page_config(layout="wide", page_title="Toestand & trendontwikkeling")

st.header("📈 Toestand en trendontwikkeling")

# Data laden en Sidebar initialiseren
df_main = load_data()
df_filtered = get_shared_sidebar(df_main)

st.info("Voor enkele stoffen is correctie met achtergrondwaardes van toepassing voor de KRW. In deze tool is dat niet meegenomen.")

# --- BESTAANDE CODE: FILTERS ---
st.write("### 🔍 Selectie filters")

stof_filter_type = st.radio(
    "Welk type stoffen wil je selecteren?",
    options=["Alle stoffen", "KRW-stoffen", "Niet-genormeerde stoffen"],
    index=0,
    horizontal=True
)

# Pre-calculatie unieke stoffen op basis van radio button
unique_stoffen_series = df_main['Stof'].unique()

if "KRW-stoffen" in stof_filter_type:
    mask_norm = df_main['JG_MKN'].notna() | df_main['MAC_MKN'].notna()
    beschikbare_stoffen = df_main.loc[mask_norm, 'Stof'].unique()
elif "Niet-genormeerde stoffen" in stof_filter_type:
    mask_norm = df_main['JG_MKN'].notna() | df_main['MAC_MKN'].notna()
    df_main_non_norm = df_main.loc[~mask_norm].copy()
    if 'Stof' in df_main_non_norm.columns:
        beschikbare_stoffen = df_main_non_norm['Stof'].unique()
    else:
        beschikbare_stoffen = []
else:
    beschikbare_stoffen = unique_stoffen_series

st.markdown("---")

# --- BESTAANDE CODE: VISUALISATIE RAW DATA ---
st.subheader("Specifieke selectie toestand & trend")
col1, col2, col3 = st.columns(3)

with col1:
    sorted_meetpunten = sorted(df_filtered['Meetpunt'].unique())
    default_meetpunten = st.session_state.get("tab1_meetpunten_select", sorted_meetpunten[:1] if sorted_meetpunten else [])
    selected_meetpunten = st.multiselect(
        "Selecteer meetpunt(en)",
        options=sorted_meetpunten,
        default=default_meetpunten,
        key="tab1_meetpunten_select"
    )

with col2:
    stofgroep_opties = sorted(df_filtered['Stofgroep'].unique())
    stofgroep_selected = st.multiselect(
        "Stofgroep",
        options=stofgroep_opties,
        default=stofgroep_opties[0] if stofgroep_opties else None,
        key="tab1_stofgroep_select"
    )

with col3:
    if stofgroep_selected:
        stoffen_in_groep = df_filtered[df_filtered['Stofgroep'].isin(stofgroep_selected)]['Stof'].unique()
        beschikbare_stoffen = np.intersect1d(beschikbare_stoffen, stoffen_in_groep)

    beschikbare_stoffen = sorted(beschikbare_stoffen)
    current_selection = st.session_state.get("tab1_stoffen_select", [])
    valid_default = [s for s in current_selection if s in beschikbare_stoffen]

    if not valid_default and len(beschikbare_stoffen) > 0:
        valid_default = [beschikbare_stoffen[0]]

    selected_stoffen = st.multiselect(
        "Selecteer Stof(fen)",
        options=beschikbare_stoffen,
        default=valid_default,
        key="tab1_stoffen_select"
    )

if selected_meetpunten and selected_stoffen:
    df_trend = df_filtered[
        (df_filtered['Meetpunt'].isin(selected_meetpunten)) &
        (df_filtered['Stof'].isin(selected_stoffen))
    ].copy()

    if not df_trend.empty:
        df_trend['Meting Type'] = np.where(
            df_trend['Limietsymbool'].astype(str).str.contains('<'),
            '< Onder rapportagegrens',
            'Gemeten waarde'
        )

        unique_stoffen_plot = sorted(df_trend['Stof'].unique())
        stof_info_df = df_trend.groupby('Stof', observed=True)[['Eenheid', 'JG_MKN', 'MAC_MKN']].first()

        fig = px.scatter(
            df_trend,
            x='Datum',
            y='Waarde',
            color='Meetpunt',
            symbol='Meting Type',
            symbol_map={'< Onder rapportagegrens': 'x', 'Gemeten waarde': 'circle'},
            facet_row='Stof',
            title='Geselecteerde individuele metingen over tijd',
            hover_data={'Datum': True, 'Waarde': ':.4f', 'Eenheid': True, 'Meetpunt': True},
            category_orders={"Stof": unique_stoffen_plot},
            height=350 * len(unique_stoffen_plot)
        )
        
        # Layout updates voor de individuele plot...
        fig.update_xaxes(matches='x')
        fig.update_yaxes(matches=None)
        
        for i, stof_naam in enumerate(unique_stoffen_plot):
            row_index = len(unique_stoffen_plot) - i
            if stof_naam in stof_info_df.index:
                info = stof_info_df.loc[stof_naam]
                eenheid = info['Eenheid']
                jg_norm = info['JG_MKN']
                mac_norm = info['MAC_MKN']
                
                # Bepaal y-as range
                max_waarde_data = df_trend[df_trend['Stof'] == stof_naam]['Waarde'].max()
                current_max = max_waarde_data if pd.notna(max_waarde_data) else 0
                vergelijkings_waarden = [current_max]
                if pd.notna(jg_norm): vergelijkings_waarden.append(jg_norm)
                if pd.notna(mac_norm): vergelijkings_waarden.append(mac_norm)
                target_top = max(vergelijkings_waarden) * 1.15 if vergelijkings_waarden else 10

                fig.update_yaxes(
                    title_text=f"Waarde ({eenheid})", range=[0, target_top], 
                    row=row_index, col=1, showticklabels=True
                )
                
                if pd.notna(jg_norm):
                    fig.add_hline(
                        y=jg_norm, line_dash="dash", line_color="darkorange", line_width=2, 
                        row=row_index, col=1, 
                        annotation_text=f"JG: {jg_norm:.4f}", annotation_position="top right"
                    )
                if pd.notna(mac_norm):
                    fig.add_hline(
                        y=mac_norm, line_dash="dot", line_color="red", line_width=2, 
                        row=row_index, col=1, 
                        annotation_text=f"MAC: {mac_norm:.2f}", annotation_position="top left"
                    )

        fig.update_layout(margin=dict(l=80, r=80))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Geen data gevonden voor de geselecteerde combinatie.")


# ==============================================================================
# NIEUWE SECTIE: OPWAARTSE TREND ANALYSE
# ==============================================================================
st.markdown("---")
st.header("⚠️ Opwaartse trends (o.b.v. jaargemiddelden)")

with st.expander("❓ Toelichting: wat betekent de trendscore (helling)?"):
    st.markdown("""
        De **trendscore (helling)** is de uitkomst van een lineaire regressie over de berekende jaargemiddelden. 
        Het geeft de **gemiddelde jaarlijkse toename** van de concentratie van de stof weer.

        ### Prioritering en urgentie
        Een positieve helling duidt op een stijgende trend. De urgentie wordt bepaald door de **tijd tot normoverschrijding**:
        
        $$
        \\text{Tijd tot Normoverschrijding (jaar)} \\approx \\frac{\\text{JG-MKN} - \\text{Laatste Jaargemiddelde}}{\\text{Trendscore (Helling)}}
        $$

        | Tijd tot normoverschrijding | Prioritering | Mogelijke acties |
        | :--- | :--- | :--- |
        | **0 jaar (Overschreden!)** | **Kritisch Urgent** | Norm is al overschreden. |
        | **< 5 jaar** | **Zeer Urgent** | Onmiddellijke actie en brononderzoek noodzakelijk. |
        | **5 tot 10 jaar** | **Urgent** | Hoog op de agenda; start maatregelontwikkeling. |
        | **10 tot 20 jaar** | **Aandachtspunt** | Periodieke monitoring; meenemen in beleidsplannen. |
        | **> 20 jaar** | **Reguliere Monitoring** | Geen directe actie nodig. |
        
        *Nvt = Geen norm beschikbaar of trend is niet relevant.*
    """)

# Filters voor Trendanalyse
st.write("### 🔍 Selectie filters voor trendanalyse")
c_trend_filter, c_lt_filter = st.columns([1, 1])

with c_trend_filter:
    trend_filter_type = st.radio(
        "Welk type stoffen wil je analyseren?",
        options=["Alle stoffen", "KRW-stoffen", "Niet-genormeerde stoffen"],
        index=0,
        horizontal=True,
        key="trend_stof_filter" 
    )

with c_lt_filter:
    lt_waarde_optie = st.radio(
        "Hoe om te gaan met '< Onder rapportagegrens'-waarden?",
        options=["Gebruik gemeten waarde", "Sluit uit van berekening"], 
        index=0,
        horizontal=True,
        key="lt_waarde_optie"
    )

# 1. Data Filteren
if "KRW-stoffen" in trend_filter_type:
    mask_norm = df_filtered['JG_MKN'].notna() | df_filtered['MAC_MKN'].notna()
    df_trend_filtered = df_filtered[mask_norm].copy()
elif "Niet-genormeerde stoffen" in trend_filter_type:
    mask_norm = df_filtered['JG_MKN'].notna() | df_filtered['MAC_MKN'].notna()
    df_trend_filtered = df_filtered[~mask_norm].copy()
else:
    df_trend_filtered = df_filtered.copy()

# 2. Berekening (GEOPTIMALISEERD DEEL)
if not df_trend_filtered.empty:
    
    # Maak een aparte lookup dataframe voor de normen uit de hoofddataset
    # Dit gebeurt buiten de loop voor O(1) opzoeken van normen.
    df_norm_lookup = df_main[['Stof', 'JG_MKN']].drop_duplicates()

    # AANROEP NIEUWE FUNCTIE (GECACEDE)
    df_trends = calculate_trends_optimized(
        df_in=df_trend_filtered, 
        lt_optie=lt_waarde_optie, 
        norm_lookup_df=df_norm_lookup
    )
    
    if not df_trends.empty:
        
        # NIEUWE CODE: FILTEREN VAN REEDS OVERSCHREDEN NORMEN (Tijd_tot_JG_normoverschrijding == 0.0)
        df_trends = df_trends[df_trends['Tijd_tot_JG_normoverschrijding'] != 0.0].copy()
        
        if df_trends.empty:
            st.success("Geen stijgende trends gevonden die de norm nog niet overschreden hebben.")
        else:
            # Sorteer op urgentie (Tijd tot norm oplopend: inf is laagst)
            df_trends = df_trends.sort_values('Tijd_tot_JG_normoverschrijding', ascending=True).reset_index(drop=True)
            
            # Maak een nette label voor de selectbox (aangepast omdat 0.0 nu gefilterd is)
            def maak_label(row):
                tijd = row['Tijd_tot_JG_normoverschrijding']
                if np.isinf(tijd):
                    status = "Info (Geen norm of trend niet relevant)"
                else:
                    status = f"Overschrijding in {tijd:.1f} jaar"
                return f"{row['Stof']} @ {row['Meetpunt']} | {status}"

            df_trends['Display_Label'] = df_trends.apply(maak_label, axis=1)

            c_tabel, c_grafiek = st.columns([1, 2])
            
            with c_tabel:
                st.write(f"**Gevonden trends:** {len(df_trends)}")
                
                # Configuratie voor nette weergave in de tabel
                column_config = {
                    "Tijd_tot_JG_normoverschrijding": st.column_config.NumberColumn(
                        "Tijd tot JG-norm (jr)",
                        help="Jaren tot overschrijding. inf = Geen norm.",
                        format="%.1f"
                    ),
                    "Trendscore": st.column_config.NumberColumn(
                        "Helling (Trend)",
                        format="%.5f"
                    ),
                    "Startwaarde": st.column_config.NumberColumn("gemiddelde conc. eerste jaar", format="%.4f"),
                    "Eindwaarde": st.column_config.NumberColumn("gemiddelde conc. laatste jaar", format="%.4f"),
                    "Meetpunt": st.column_config.TextColumn("Locatie"),
                    "Stof": st.column_config.TextColumn("Stofnaam"),
                    "Aantal_jaren": st.column_config.NumberColumn("Aantal jaren tot voorspelde normoverschrijding", format="%d")
                }

                # Weergave tabel
                st.dataframe(
                    df_trends[['Meetpunt', 'Stof', 'Trendscore', 'Tijd_tot_JG_normoverschrijding', 'Aantal_jaren', 'Startwaarde', 'Eindwaarde']],
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                    column_config=column_config
                )

            with c_grafiek:
                st.write("### Trendgrafiek ")
                st.info("Selecteer hieronder een trend om de grafiek te bekijken. De lijst is gesorteerd op urgentie (zoals de tabel).")
                
                # DE ALTERNATIEVE SELECTIE METHODE (ROBUUST)
                gekozen_label = st.selectbox(
                    "Selecteer trend:",
                    options=df_trends['Display_Label']
                )
                
                # Haal de data op die hoort bij het label
                gekozen_rij = df_trends[df_trends['Display_Label'] == gekozen_label].iloc[0]
                
                mp_keuze = gekozen_rij['Meetpunt']
                stof_keuze = gekozen_rij['Stof']
                slope = gekozen_rij['Trendscore']
                tijd_tot_norm = gekozen_rij['Tijd_tot_JG_normoverschrijding']
                
                # Data ophalen voor plot: Filter de df_trend_filtered (de dataset van de sidebar)
                # En bereken de jaargemiddelden opnieuw voor de plot.
                df_plot_source = df_trend_filtered[
                    (df_trend_filtered['Meetpunt'] == mp_keuze) & 
                    (df_trend_filtered['Stof'] == stof_keuze)
                ].copy()
                
                # Pas de < logica toe voor de plot consistent met de berekening
                if lt_waarde_optie == "Sluit uit van berekening":
                    mask_lt_plot = df_plot_source['Limietsymbool'].astype(str).str.contains('<', na=False)
                    df_plot_source.loc[mask_lt_plot, 'Waarde'] = np.nan
                
                df_plot_source['Jaar'] = df_plot_source['Datum'].dt.year
                # Gebruik de 'Waarde' kolom (die eventueel op NaN is gezet door de lt_waarde_optie)
                plot_data = df_plot_source.groupby('Jaar')['Waarde'].mean().reset_index().dropna(subset=['Waarde'])

                # Trendlijn berekenen
                z = np.polyfit(plot_data['Jaar'], plot_data['Waarde'], 1)
                p = np.poly1d(z)
                plot_data['Trendlijn'] = p(plot_data['Jaar'])
                
                # Plotten
                fig_trend = go.Figure()
                
                # Punten
                fig_trend.add_trace(go.Scatter(
                    x=plot_data['Jaar'], y=plot_data['Waarde'],
                    mode='markers+lines', name='Jaargemiddelde',
                    marker=dict(size=10, color='blue'), hoverinfo='all'
                ))
                
                # Lijn
                fig_trend.add_trace(go.Scatter(
                    x=plot_data['Jaar'], y=plot_data['Trendlijn'],
                    mode='lines', name=f'Trendlijn (helling={slope:.4f})',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='**Trendwaarde:** %{y:.4f}<br>**Jaar:** %{x}<extra></extra>'
                ))
                
                # Norm
                # Norm ophalen via de snelle lookup tabel
                jg_norm = df_norm_lookup[df_norm_lookup['Stof'] == stof_keuze]['JG_MKN'].iloc[0] if not df_norm_lookup[df_norm_lookup['Stof'] == stof_keuze].empty else np.nan
                
                # Titel bepalen
                titel_suffix = ""
                # De 0.0 case is nu uitgesloten door de filter, dus alleen inf of > 0.0
                if np.isinf(tijd_tot_norm):
                    titel_suffix = " (Nvt)"
                else:
                    titel_suffix = f" (Nog {tijd_tot_norm:.1f} jaar tot norm)"
                
                if pd.notna(jg_norm):
                     fig_trend.add_hline(y=jg_norm, line_dash="solid", line_color="orange", annotation_text=f"JG-norm: {jg_norm}")

                fig_trend.update_layout(
                    title=f"{stof_keuze} @ {mp_keuze} {titel_suffix}",
                    xaxis_title="Jaar",
                    yaxis_title="Concentratie (gem)",
                    hovermode="x unified",
                    xaxis=dict(
                        tickformat='d',  # Format als geheel getal
                        dtick=1,         # Zet de stapgrootte op 1 eenheid (jaar)
                        showgrid=True    # Maakt de discrete stappen duidelijker
                    )
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            
    else:
        st.success("Geen stijgende trends gevonden in de huidige selectie/jaren.")
else:
    st.warning("Geen data beschikbaar voor trendanalyse.")