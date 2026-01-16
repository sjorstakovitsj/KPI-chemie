# 2_Toestand_&_Trend.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar, calculate_trends_optimized, calculate_declining_exceedances_optimized

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
# SECTIE: TREND ANALYSE
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
        index=1,
        horizontal=True,
        key="trend_stof_filter" 
    )

with c_lt_filter:
    lt_waarde_optie = st.radio(
        "Hoe om te gaan met '< Onder rapportagegrens'-waarden?",
        options=["Gebruik gemeten waarde", "Sluit uit van berekening"], 
        index=1,
        horizontal=True,
        key="lt_waarde_optie"
    )

# 1. Data Filteren op Type Stof
if "KRW-stoffen" in trend_filter_type:
    mask_norm = df_filtered['JG_MKN'].notna() | df_filtered['MAC_MKN'].notna()
    df_trend_filtered = df_filtered[mask_norm].copy()
elif "Niet-genormeerde stoffen" in trend_filter_type:
    mask_norm = df_filtered['JG_MKN'].notna() | df_filtered['MAC_MKN'].notna()
    df_trend_filtered = df_filtered[~mask_norm].copy()
else:
    df_trend_filtered = df_filtered.copy()

# Globale variabelen die we straks hergebruiken
df_norm_lookup = df_main[['Stof', 'JG_MKN']].drop_duplicates()

# ==============================================================================
# BEREKENINGEN & STATISTIEKEN (Voorbereiding voor beide tabellen)
# ==============================================================================
stats_df = pd.DataFrame()

if not df_trend_filtered.empty:
    
    df_check = df_trend_filtered.copy()
    
    # Pas LT-optie toe op df_check voor statistiekberekening
    if lt_waarde_optie == "Sluit uit van berekening":
        is_lt = df_check['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_check.loc[is_lt, 'Waarde'] = np.nan
        df_check = df_check.dropna(subset=['Waarde'])
    
    # Check voor minimaal 7 metingen (Geldt voor BEIDE analyses)
    counts = df_check.groupby(['Meetpunt', 'Stof']).size().reset_index(name='n_obs')
    valid_groups = counts[counts['n_obs'] >= 7][['Meetpunt', 'Stof']]
    df_trend_filtered = pd.merge(df_trend_filtered, valid_groups, on=['Meetpunt', 'Stof'], how='inner')
    
    # -----------------------------------------------------------
    # BEREKEN GLOBAL MEAN & RSD (voor weergave in tabel)
    # -----------------------------------------------------------
    # We gebruiken df_check (waar LT opties al verwerkt zijn) gefilterd op de valid groups
    df_stats_source = pd.merge(df_check, valid_groups, on=['Meetpunt', 'Stof'], how='inner')
    
    # Groepeer en bereken mean en std
    stats_agg = df_stats_source.groupby(['Meetpunt', 'Stof'])['Waarde'].agg(['mean', 'std']).reset_index()
    stats_agg['RSD'] = (stats_agg['std'] / stats_agg['mean']) * 100
    stats_agg = stats_agg.rename(columns={'mean': 'Gemiddelde_Waarde'})
    
    stats_df = stats_agg[['Meetpunt', 'Stof', 'Gemiddelde_Waarde', 'RSD']]


# ==============================================================================
# 1. OPWAARTSE TRENDS
# ==============================================================================

if not df_trend_filtered.empty:
    
    # AANROEP FUNCTIE (Rising Trends)
    df_trends = calculate_trends_optimized(
        df_in=df_trend_filtered, 
        lt_optie=lt_waarde_optie, 
        norm_lookup_df=df_norm_lookup
    )
    
    if not df_trends.empty:
        
        # Filter 1: Verberg trends die > 20 jaar duren
        df_trends = df_trends[df_trends['Tijd_tot_JG_normoverschrijding'] <= 20].copy()

        # Filter 2: Verberg stoffen die de norm REEDS overschrijden (Tijd > 0)
        df_trends = df_trends[df_trends['Tijd_tot_JG_normoverschrijding'] > 0].copy()
        
        if df_trends.empty:
            st.success("Geen kritieke opwaartse trends gevonden (tijd tot norm tussen 0 en 20 jaar).")
        else:
            # Merge statistieken (Gemiddelde & RSD)
            df_trends = pd.merge(df_trends, stats_df, on=['Meetpunt', 'Stof'], how='left')

            # Sorteer op urgentie (Tijd tot norm klein -> groot)
            df_trends = df_trends.sort_values('Tijd_tot_JG_normoverschrijding', ascending=True).reset_index(drop=True)
            
            # Voeg Rank kolom toe
            df_trends.insert(0, 'Nr.', range(1, 1 + len(df_trends)))
            
            def maak_label(row):
                tijd = row['Tijd_tot_JG_normoverschrijding']
                if np.isinf(tijd):
                    status = "Info (Geen norm of trend niet relevant)"
                else:
                    status = f"Overschrijding in {tijd:.1f} jaar"
                # AANGEPAST: Nr. toegevoegd in label
                return f"{int(row['Nr.'])}. {row['Stof']} @ {row['Meetpunt']} | {status}"

            df_trends['Display_Label'] = df_trends.apply(maak_label, axis=1)

            c_tabel, c_grafiek = st.columns([1, 2])
            
            with c_tabel:
                st.write(f"**Gevonden trends:** {len(df_trends)}")
                st.caption("Stoffen met stijgende trend, nog onder de norm, normoverschrijding < 20jr.")
                
                column_config = {
                    "Nr.": st.column_config.NumberColumn("Nr.", format="%d", width="small"),
                    "Tijd_tot_JG_normoverschrijding": st.column_config.NumberColumn(
                        "Tijd tot JG-norm (jr)",
                        help="Jaren tot overschrijding. inf = Geen norm.",
                        format="%.1f"
                    ),
                    "Trendscore": st.column_config.NumberColumn(
                        "Helling (Trend)",
                        format="%.5f"
                    ),
                    "Gemiddelde_Waarde": st.column_config.NumberColumn("Gem. Conc.", format="%.4f"),
                    "RSD": st.column_config.NumberColumn("RSD (%)", format="%.1f%%"),
                    "Meetpunt": st.column_config.TextColumn("Locatie"),
                    "Stof": st.column_config.TextColumn("Stofnaam"),
                    "n_metingen_boven_rg": st.column_config.NumberColumn("Metingen > RG", format="%d")
                }

                st.dataframe(
                    df_trends[['Nr.', 'Meetpunt', 'Stof', 'n_metingen_boven_rg', 'Trendscore', 'Tijd_tot_JG_normoverschrijding', 'Gemiddelde_Waarde', 'RSD']],
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                    column_config=column_config
                )

            with c_grafiek:
                st.write("### Trendgrafiek (Opwaarts)")
                st.info("Selecteer hieronder een trend om de grafiek te bekijken.")
                
                gekozen_label = st.selectbox(
                    "Selecteer trend:",
                    options=df_trends['Display_Label']
                )
                
                gekozen_rij = df_trends[df_trends['Display_Label'] == gekozen_label].iloc[0]
                mp_keuze = gekozen_rij['Meetpunt']
                stof_keuze = gekozen_rij['Stof']
                slope = gekozen_rij['Trendscore']
                tijd_tot_norm = gekozen_rij['Tijd_tot_JG_normoverschrijding']
                
                # RAW DATA Ophalen (voor de scatter plot)
                df_plot_source = df_trend_filtered[
                    (df_trend_filtered['Meetpunt'] == mp_keuze) & 
                    (df_trend_filtered['Stof'] == stof_keuze)
                ].copy()
                
                if lt_waarde_optie == "Sluit uit van berekening":
                    mask_lt_plot = df_plot_source['Limietsymbool'].astype(str).str.contains('<', na=False)
                    df_plot_source.loc[mask_lt_plot, 'Waarde'] = np.nan
                    df_plot_source = df_plot_source.dropna(subset=['Waarde'])
                
                # JAARGEMIDDELDE (voor de trendlijn berekening, om consistent te zijn met de helling in de tabel)
                df_plot_source['Jaar'] = df_plot_source['Datum'].dt.year
                plot_data_agg = df_plot_source.groupby('Jaar')['Waarde'].mean().reset_index()

                z = np.polyfit(plot_data_agg['Jaar'], plot_data_agg['Waarde'], 1)
                p = np.poly1d(z)
                plot_data_agg['Trendlijn'] = p(plot_data_agg['Jaar'])
                
                # Maak een fictieve datum (1 juli) aan de jaardata vast om de trendlijn op de datum-as te kunnen plotten
                plot_data_agg['Datum_Plot'] = pd.to_datetime(plot_data_agg['Jaar'].astype(str) + '-07-01')

                fig_trend = go.Figure()
                
                # 1. Scatter: Individuele punten (Datum vs Waarde)
                fig_trend.add_trace(go.Scatter(
                    x=df_plot_source['Datum'], 
                    y=df_plot_source['Waarde'],
                    mode='markers', 
                    name='Individuele meting',
                    marker=dict(size=8, color='blue', opacity=0.6), 
                    hoverinfo='x+y'
                ))
                
                # 2. Line: Trendlijn (berekend op jaren, geplot op datum-as)
                fig_trend.add_trace(go.Scatter(
                    x=plot_data_agg['Datum_Plot'], 
                    y=plot_data_agg['Trendlijn'],
                    mode='lines', 
                    name=f'Trendlijn (helling={slope:.4f})',
                    line=dict(color='red', dash='dash'),
                    hovertemplate='**Trendwaarde:** %{y:.4f}<br>**Jaar:** %{x}<extra></extra>'
                ))
                
                jg_norm = df_norm_lookup[df_norm_lookup['Stof'] == stof_keuze]['JG_MKN'].iloc[0] if not df_norm_lookup[df_norm_lookup['Stof'] == stof_keuze].empty else np.nan
                
                titel_suffix = ""
                if np.isinf(tijd_tot_norm):
                    titel_suffix = " (Nvt)"
                else:
                    titel_suffix = f" (Nog {tijd_tot_norm:.1f} jaar tot norm)"
                
                if pd.notna(jg_norm):
                     fig_trend.add_hline(y=jg_norm, line_dash="solid", line_color="orange", annotation_text=f"JG-norm: {jg_norm}")

                fig_trend.update_layout(
                    title=f"{stof_keuze} @ {mp_keuze} {titel_suffix}",
                    xaxis_title="Datum",
                    yaxis_title="Concentratie",
                    hovermode="closest"
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
            
    else:
        st.success("Geen stijgende trends gevonden in de huidige selectie/jaren.")
else:
    st.warning("Geen data beschikbaar voor trendanalyse.")


# ==============================================================================
# SECTIE: OVERSCHRIJDING MAAR DALEND/STAGNAIR
# ==============================================================================
st.markdown("---")
st.header("📉 Normoverschrijdingen met dalende of stagnante trend")
st.info("Deze sectie toont stoffen die momenteel de JG-MKN norm overschrijden, maar waarbij de trend niet stijgt (helling ≤ 0). Dit zijn potentiële 'goede nieuws' gevallen of situaties die stabiliseren.")

if not df_trend_filtered.empty:
    
    # Gebruik de reeds gefilterde dataset (df_trend_filtered bevat al de >7 metingen check)
    df_declining = calculate_declining_exceedances_optimized(
        df_in=df_trend_filtered,
        lt_optie=lt_waarde_optie,
        norm_lookup_df=df_norm_lookup
    )

    if df_declining.empty:
        st.write("Geen stoffen gevonden die de norm overschrijden met een dalende of stagnante trend in deze selectie.")
    else:
        # Filter: Verberg trends die > 20 jaar duren om onder de norm te komen
        df_declining = df_declining[df_declining['Tijd_tot_onder_norm'] <= 20].copy()

        if df_declining.empty:
            st.success("Er zijn wel dalende overschrijdingen, maar geen enkele komt binnen 20 jaar onder de norm.")
        else:
            # Merge statistieken (Gemiddelde & RSD)
            df_declining = pd.merge(df_declining, stats_df, on=['Meetpunt', 'Stof'], how='left')
            
            # Sorteer op 'Tijd_tot_onder_norm' (Laag -> Hoog, dus snelst opgelost bovenaan)
            df_declining = df_declining.sort_values('Tijd_tot_onder_norm', ascending=True).reset_index(drop=True)
            
            # Voeg Rank kolom toe
            df_declining.insert(0, 'Nr.', range(1, 1 + len(df_declining)))

            c_table_dec, c_graph_dec = st.columns([1, 2])

            with c_table_dec:
                st.write(f"**Gevonden records (binnen 20 jaar opgelost):** {len(df_declining)}")
                
                column_config_dec = {
                    "Nr.": st.column_config.NumberColumn("Nr.", format="%d", width="small"),
                    "JG_MKN": st.column_config.NumberColumn("JG Norm", format="%.4f"),
                    "Trendscore": st.column_config.NumberColumn("Helling (Trend)", format="%.5f"),
                    "Gemiddelde_Waarde": st.column_config.NumberColumn("Gem. Conc.", format="%.4f"),
                    "RSD": st.column_config.NumberColumn("RSD (%)", format="%.1f%%"),
                    "Meetpunt": st.column_config.TextColumn("Locatie"),
                    "Stof": st.column_config.TextColumn("Stofnaam"),
                    "Tijd_tot_onder_norm": st.column_config.NumberColumn("Jaren tot onder norm", format="%.1f"),
                    "n_metingen_boven_rg": st.column_config.NumberColumn("Metingen > RG", format="%d")
                }

                st.dataframe(
                    df_declining[['Nr.', 'Meetpunt', 'Stof', 'n_metingen_boven_rg', 'Trendscore', 'JG_MKN', 'Gemiddelde_Waarde', 'RSD', 'Tijd_tot_onder_norm']],
                    use_container_width=True,
                    height=500,
                    hide_index=True,
                    column_config=column_config_dec
                )

            with c_graph_dec:
                st.write("### Trendgrafiek (Stagnant/Dalend)")
                
                # Label maken voor selector MET Nr.
                df_declining['Display_Label'] = df_declining.apply(
                    lambda row: f"{int(row['Nr.'])}. {row['Stof']} @ {row['Meetpunt']} (Helling: {row['Trendscore']:.4f})", axis=1
                )
                
                keuze_dalend = st.selectbox(
                    "Selecteer record:",
                    options=df_declining['Display_Label'],
                    key="select_dalend"
                )
                
                rij_dalend = df_declining[df_declining['Display_Label'] == keuze_dalend].iloc[0]
                
                mp_d = rij_dalend['Meetpunt']
                stof_d = rij_dalend['Stof']
                slope_d = rij_dalend['Trendscore']
                norm_d = rij_dalend['JG_MKN']

                # RAW DATA ophalen voor plot
                df_plot_d = df_trend_filtered[
                    (df_trend_filtered['Meetpunt'] == mp_d) & 
                    (df_trend_filtered['Stof'] == stof_d)
                ].copy()

                if lt_waarde_optie == "Sluit uit van berekening":
                    mask_lt_d = df_plot_d['Limietsymbool'].astype(str).str.contains('<', na=False)
                    df_plot_d.loc[mask_lt_d, 'Waarde'] = np.nan
                    df_plot_d = df_plot_d.dropna(subset=['Waarde'])
                
                # Trendlijn berekening (op aggregatie om consistent te blijven met helling)
                df_plot_d['Jaar'] = df_plot_d['Datum'].dt.year
                plot_data_d_agg = df_plot_d.groupby('Jaar')['Waarde'].mean().reset_index()

                z_d = np.polyfit(plot_data_d_agg['Jaar'], plot_data_d_agg['Waarde'], 1)
                p_d = np.poly1d(z_d)
                plot_data_d_agg['Trendlijn'] = p_d(plot_data_d_agg['Jaar'])
                
                # Fictieve datum voor plotting van de lijn
                plot_data_d_agg['Datum_Plot'] = pd.to_datetime(plot_data_d_agg['Jaar'].astype(str) + '-07-01')

                fig_d = go.Figure()
                
                # 1. Scatter: Individuele punten
                fig_d.add_trace(go.Scatter(
                    x=df_plot_d['Datum'], 
                    y=df_plot_d['Waarde'],
                    mode='markers', 
                    name='Individuele meting',
                    marker=dict(size=8, color='green', opacity=0.6), 
                    hoverinfo='x+y'
                ))
                
                # 2. Line: Trendlijn
                fig_d.add_trace(go.Scatter(
                    x=plot_data_d_agg['Datum_Plot'], 
                    y=plot_data_d_agg['Trendlijn'],
                    mode='lines', 
                    name=f'Trendlijn (helling={slope_d:.4f})',
                    line=dict(color='gray', dash='dot'),
                    hovertemplate='**Trendwaarde:** %{y:.4f}<br>**Jaar:** %{x}<extra></extra>'
                ))
                
                fig_d.add_hline(y=norm_d, line_dash="solid", line_color="orange", annotation_text=f"JG-norm: {norm_d}")

                fig_d.update_layout(
                    title=f"{stof_d} @ {mp_d}",
                    xaxis_title="Datum",
                    yaxis_title="Concentratie",
                    hovermode="closest"
                )
                
                st.plotly_chart(fig_d, use_container_width=True)