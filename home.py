# Home.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Importeer de nieuwe functies
from utils import (
    create_gauge,
    get_filter_options,
    query_data,
    calculate_metrics,
    calculate_compliance_details,
    prepare_heatmap_data,
    prepare_sunburst_data,
)

st.set_page_config(layout="wide", page_title="Waterkwaliteit KPI Dashboard", page_icon="💧")
st.title("💧 Dashboard chemische waterkwaliteit MN")

# Beheer toekomstige updates door uitsluitend nieuwe dictionaries bovenaan
# deze lijst toe te voegen. De nieuwste update staat altijd als eerste.
APP_UPDATES = [
    {
        "datum": "21 september 2026",
        "titel": "Seizoensfilters en uitgebreidere trendanalyse",
        "wijzigingen": [
            "De analysepagina's ondersteunen nu winter, voorjaar, zomer, herfst, zomerhalfjaar en winterhalfjaar.",
            "De losse zomerhalfjaarfilter in de ruimtelijke analyse is vervangen door de centrale periodefilter.",
            "Bij individuele meetreeksen is Mann-Kendall toegevoegd; bij een beperkte seizoensselectie wordt automatisch Seasonal Mann-Kendall gebruikt.",
            "Metingen onder de rapportagegrens worden uitgesloten van de Mann-Kendall-toets en de aantallen onder en boven de rapportagegrens worden getoond.",
            "Niet-relevante aggregatiekeuzes zijn van diverse pagina's verwijderd en detailmeters op Home zijn standaard ingeklapt.",
            "De risicoanalyse heeft aparte keuzes voor gemiddelde of mediaan bij de prioriteringslijst en de jaartrend.",
            "De jaartrend in de risicoanalyse combineert meetpuntlijnen met een duidelijk gemarkeerde gemiddelde- of mediaanlijn.",
        ],
    },
    {
        "datum": "11 september 2026",
        "titel": "Betere norm- en risicobeoordeling",
        "wijzigingen": [
            "Stofspecifieke signaleringswaarden worden nu centraal beheerd via een koppeltabel.",
            "Als geen stofspecifieke signaleringswaarde beschikbaar is, blijft voor geschikte stoffen de generieke waarde van 0,1 µg/l gelden.",
            "Metingen van geselecteerde metalen worden gecorrigeerd voor de natuurlijke achtergrondconcentratie.",
            "De oorspronkelijke meetwaarde blijft beschikbaar, zodat de achtergrondcorrectie controleerbaar is.",
            "De KRW-check, jaargemiddelden, MAC-controle, gauges en overschrijdingstabellen gebruiken de gecorrigeerde concentraties.",
            "In Home en KRW normcheck is zichtbaar hoeveel meetregels zijn gecorrigeerd en welke waarden zijn gebruikt.",
            "De verwerking en validatie van koppeltabellen en categorische gegevens is robuuster en toekomstbestendiger gemaakt.",
        ],
    },
]


@st.dialog("Wat is er nieuw?", width="large")
def toon_updatevenster():
    """Toont één update per venster met navigatie door de updatehistorie."""
    maximaal_index = max(len(APP_UPDATES) - 1, 0)
    huidig_index = min(
        max(int(st.session_state.get("update_index", 0)), 0),
        maximaal_index,
    )
    st.session_state.update_index = huidig_index
    update = APP_UPDATES[huidig_index]

    st.subheader(update["titel"])
    st.caption(
        f"Update van {update['datum']}  |  "
        f"{huidig_index + 1} van {len(APP_UPDATES)}"
    )
    for wijziging in update["wijzigingen"]:
        st.markdown(f"- {wijziging}")

    st.divider()
    vorige_col, sluit_col, volgende_col = st.columns([1, 2, 1])

    with vorige_col:
        if st.button(
            "← Nieuwere update",
            disabled=huidig_index == 0,
            width="stretch",
        ):
            st.session_state.update_index = huidig_index - 1
            st.rerun()

    with sluit_col:
        if st.button("Sluiten", type="primary", width="stretch"):
            st.session_state.updatevenster_open = False
            st.rerun()

    with volgende_col:
        if st.button(
            "Oudere update →",
            disabled=huidig_index >= maximaal_index,
            width="stretch",
        ):
            st.session_state.update_index = huidig_index + 1
            st.rerun()


if "updatevenster_open" not in st.session_state:
    st.session_state.updatevenster_open = True
if "update_index" not in st.session_state:
    st.session_state.update_index = 0

if st.session_state.updatevenster_open:
    toon_updatevenster()

if st.button("Bekijk de laatste updates", key="open_updatevenster"):
    st.session_state.update_index = 0
    st.session_state.updatevenster_open = True
    st.rerun()

# 1. Lichte filteropties ophalen via een afzonderlijk gecachete DuckDB-query.
# Hierdoor hoeft de volledige analytische dataset niet te worden geladen om de
# beschikbare jaren en het totale aantal meetpunten te bepalen.
filter_options = get_filter_options()
available_filter_years = sorted(filter_options["jaren"], reverse=True)

if not available_filter_years:
    st.error(
        "🚨 Kritieke fout: er zijn geen beschikbare jaren in het "
        "Parquetbestand gevonden."
    )
    st.stop()

# 2. Sidebar en query-pushdown.
st.sidebar.header("📅 Filter op jaren")
selected_years = st.sidebar.multiselect(
    "Selecteer gewenste jaren:",
    options=available_filter_years,
    default=available_filter_years,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Navigeer via het menu hierboven naar de verschillende analyses."
)

# Lege jaarselectie behoudt het eerdere gedrag: geen jaarbeperking.
# Alleen de kolommen die Home daadwerkelijk gebruikt worden uit Parquet gelezen.
HOME_COLUMNS = (
    "Datum",
    "Meetpunt",
    "Stof",
    "Waarde",
    "Waarde_Origineel",
    "Eenheid",
    "Limietsymbool",
    "Latitude",
    "Longitude",
    "JG_MKN",
    "MAC_MKN",
    "Achtergrondconcentratie",
    "Achtergrondcorrectie_Toegepast",
)

df_filtered = query_data(
    jaren=tuple(selected_years),
    kolommen=HOME_COLUMNS,
)

if df_filtered.empty:
    st.error("🚨 Kritieke fout: de geselecteerde meetgegevens zijn leeg.")
    st.stop()

# 3. Dynamische Scorekaart: Recent vs Selectie
available_years = sorted(df_filtered['Datum'].dt.year.unique(), reverse=True)

if len(available_years) >= 1:
    current_year = available_years[0]
    df_curr = df_filtered[df_filtered['Datum'].dt.year == current_year]
    
    st.subheader(f"Vergelijking huidig jaar ten opzichte van voorgaande jaren: **{current_year}**")

    # Opties opbouwen
    comparison_options = {}
    for year in available_years[1:]:
        comparison_options[f"voorgaand jaar: {year}"] = {'years': [year], 'is_period': False}

    if len(available_years) > 1:
        prev_years = available_years[1:]
        is_multi_year = len(prev_years) > 1
        comparison_options[f"Periode: {prev_years[-1]} t/m {prev_years[0]} (gemiddeld)"] = {'years': prev_years, 'is_period': is_multi_year}
        
    if not comparison_options:
        st.info("Niet genoeg historische data (minimaal 2 jaar) om te vergelijken.")
    else:
        # UI Selectie
        default_index = 0
        keys = list(comparison_options.keys())
        if keys[0].startswith("Periode:"): default_index = 0
        selected_option = st.selectbox(f"Selecteer vergelijking voor **{current_year}**:", options=keys, index=default_index)
        
        # Data ophalen
        comp_info = comparison_options[selected_option]
        df_comp = df_filtered[df_filtered['Datum'].dt.year.isin(comp_info['years'])].copy()
    
        # Berekeningen (via Utils)
        count_curr, viol_curr, pct_curr = calculate_metrics(df_curr, is_period_average=False)
        comp_count, comp_viol, comp_pct = calculate_metrics(df_comp, is_period_average=comp_info['is_period'])
        
        # Metrics Tonen
        col1, col2, col3 = st.columns(3)
        lbl_avg = "gemiddeld" if comp_info['is_period'] else ""
        
        col1.metric("Aantal metingen " + lbl_avg, f"{count_curr:.0f}", f"{count_curr - comp_count:.0f}")
        col2.metric("Aantal overschrijdingen " + lbl_avg, f"{viol_curr:.0f}", f"{viol_curr - comp_viol:.0f}", delta_color="inverse")
        col3.metric("Percentage normoverschrijdingen " + lbl_avg, f"{pct_curr:.1f}%", f"{pct_curr - comp_pct:.1f}%", delta_color="inverse")
    
    st.divider()

# 4. KRW-check & Heatmap
st.header("🔴 KRW-check: overschrijdende stoffen")
# De achtergrondcorrectie is tijdens de Parquet-build centraal toegepast.
# Toon hier transparant hoeveel meetregels in de huidige selectie zijn gecorrigeerd.
if 'Achtergrondcorrectie_Toegepast' in df_filtered.columns:
    aantal_gecorrigeerd = int(
        df_filtered['Achtergrondcorrectie_Toegepast'].fillna(False).sum()
    )
    aantal_totaal = len(df_filtered)
    st.info(
        "Achtergrondcorrectie is centraal toegepast op "
        f"{aantal_gecorrigeerd:,} van de {aantal_totaal:,} meetregels in de huidige selectie. "
        "De oorspronkelijke concentratie blijft beschikbaar als 'Waarde_Origineel'."
    )
else:
    st.warning(
        "De kolom 'Achtergrondcorrectie_Toegepast' ontbreekt. "
        "Controleer of de actuele utils.py en de achtergrondcorrectietabel worden gebruikt."
    )

# Compliance data berekenen (via Utils)
df_failures = calculate_compliance_details(df_filtered)

if df_failures.empty:
    st.success("Geen normoverschrijdingen gevonden! 🎉")
else:
    st.subheader("🧩 Heatmap overschrijdingsfactoren")
    
    # Heatmap data voorbereiden (ZWARE LOGICA NU IN UTILS)
    factor_matrix, text_matrix, viol_stof, viol_mp = prepare_heatmap_data(df_filtered)
    
    if factor_matrix is None:
         st.info("Geen data beschikbaar na filtering.")
    else:
        # Plotting Logic (Blijft in Home want dit is View)
        Z_MAX = factor_matrix.max().max() if factor_matrix.max().max() > 1.0 else 2.0
        split_frac = 1.0 / Z_MAX
        custom_colorscale = [[0.0, 'rgb(220, 220, 220)'], [split_frac, 'rgb(220, 220, 220)'], [split_frac, 'rgb(255, 235, 130)'], [1.0, 'rgb(189, 0, 38)']]

        fig_heatmap = go.Figure(data=[
            go.Heatmap(
                z=factor_matrix.values, x=factor_matrix.columns, y=factor_matrix.index,
                text=text_matrix.values, texttemplate="%{text}", textfont={"size": 10},
                colorscale=custom_colorscale, zmin=0.0, zmax=Z_MAX,
                hovertemplate="<b>%{x}</b><br>%{y}<br>Max Factor: %{z:.1f}x<br><br>Status:<br>%{text}<extra></extra>",
                xgap=1, ygap=1
            )
        ])
        fig_heatmap.update_layout(
            height=max(400, len(viol_stof) * 30 + 150),
            xaxis=dict(title="Meetpunt", tickangle=-45), yaxis=dict(title="Stof", autorange="reversed"),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig_heatmap, width="stretch")
        st.caption("Grijze cellen = geen overschrijding.")
    
    st.divider()

    # Detailweergave & Sunburst
    meetpunten_met_fouten = sorted(df_failures['Meetpunt'].unique())
    selected_mp = st.selectbox("Selecteer een meetpunt voor detailweergave:", meetpunten_met_fouten)
    
    df_mp_fail = df_failures[df_failures['Meetpunt'] == selected_mp].copy()
    
    # Data voorbereiden voor Sunburst (via Utils)
    df_viz = prepare_sunburst_data(df_mp_fail)
    
    col_graph, col_list = st.columns([1, 1])
    with col_graph:
        fig_sun = go.Figure(go.Sunburst(
            ids=["Totaal"] + df_viz['Stof'].tolist(),
            labels=["Chemische<br>Toestand"] + df_viz['Stof'].tolist(),
            parents=[""] + ["Totaal"] * len(df_viz),
            marker=dict(colors=["#DDDDDD"] + df_viz['Color'].tolist()),
            hovertext=[""] + [f"Type: {r['Categorie']}<br>Factor: {r['Factor']:.1f}x" for i, r in df_viz.iterrows()],
            hoverinfo="label+text"
        ))
        fig_sun.update_layout(margin=dict(t=30, l=0, r=0, b=10), title=f"Overschrijdingen: {selected_mp}", height=500)
        st.plotly_chart(fig_sun, width="stretch")

    with col_list:
        st.markdown(f"**Details ({selected_mp})**")
        st_display = df_mp_fail[['Jaar', 'Stof', 'Normtype', 'Factor']].sort_values(by=['Factor', 'Stof', 'Jaar'], ascending=[False, True, False])
        st.dataframe(
            st_display, width="stretch", hide_index=True,
            column_config={
                "Factor": st.column_config.ProgressColumn("Factor (x norm)", format="%.1f x", min_value=0, max_value=5),
                "Jaar": st.column_config.NumberColumn("Jaar", format="%d")
            }
        )

st.divider()

# 5. Overzicht toestand & kaart
st.header("📊 Overzicht toestand & kaart meetpunten")

# Bereken totalen voor gauges
df_mac_ov = df_filtered.dropna(subset=['MAC_MKN'])
if 'Limietsymbool' in df_mac_ov.columns: df_mac_ov = df_mac_ov[df_mac_ov['Limietsymbool'] != '<']

df_jg_ov = df_filtered.dropna(subset=['JG_MKN'])
if 'Limietsymbool' in df_jg_ov.columns: df_jg_ov = df_jg_ov[df_jg_ov['Limietsymbool'] != '<']

# Metrics hergebruiken we niet direct uit utils omdat dit "overall" filters zijn
pct_jg_total = (df_jg_ov['Waarde'] <= df_jg_ov['JG_MKN']).mean() * 100 if not df_jg_ov.empty else 0
pct_mac_total = (df_mac_ov['Waarde'] <= df_mac_ov['MAC_MKN']).mean() * 100 if not df_mac_ov.empty else 0

col_kpi, col_gauges, col_map = st.columns([1, 2, 2])

with col_kpi:
    st.metric("Unieke meetpunten", len(filter_options["meetpunten"]))
    # Metingen zonder norm: totaal - (metingen met JG of MAC)
    df_with_norm = df_filtered.dropna(subset=['JG_MKN', 'MAC_MKN'], how='all')
    st.metric("Metingen zonder JG/MAC norm", len(df_filtered) - len(df_with_norm))
    st.metric("Metingen met JG-norm", len(df_jg_ov))
    st.metric("Metingen met MAC-norm", len(df_mac_ov))

with col_gauges:
    s1, s2 = st.columns(2)
    s1.plotly_chart(create_gauge(pct_jg_total, "Totaal: voldoet JG (%)"), width="stretch")
    s2.plotly_chart(create_gauge(pct_mac_total, "Totaal: voldoet MAC (%)"), width="stretch")

with col_map:
    df_map = df_filtered[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()
    if not df_map.empty:
        fig_map = px.scatter_map(
            df_map, lat='Latitude', lon='Longitude', hover_name='Meetpunt',
            size_max=15, zoom=8, map_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=12, color='red'))
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, width="stretch")

st.divider()

# 6. Detail Gauges per meetpunt
unieke_meetpunten = sorted(df_filtered['Meetpunt'].unique())

# Pre-calc stats (vectorized)
jg_stats = df_jg_ov.groupby('Meetpunt', observed=True).apply(lambda x: (x['Waarde'] <= x['JG_MKN']).mean() * 100)
mac_stats = df_mac_ov.groupby('Meetpunt', observed=True).apply(lambda x: (x['Waarde'] <= x['MAC_MKN']).mean() * 100)

with st.expander(f"Toon detailmeters voor alle {len(unieke_meetpunten)} meetpunten", expanded=False):
    for mp in unieke_meetpunten:
        p_jg = jg_stats.get(mp, None)
        p_mac = mac_stats.get(mp, None)
        if pd.isna(p_jg) and pd.isna(p_mac): continue

        st.markdown(f"**{mp}**")
        c1, c2 = st.columns(2)
        if pd.notna(p_jg): c1.plotly_chart(create_gauge(p_jg, f"JG: {mp}", 95), width="stretch", key=f"g_jg_{mp}")
        if pd.notna(p_mac): c2.plotly_chart(create_gauge(p_mac, f"MAC: {mp}", 95), width="stretch", key=f"g_mac_{mp}")

st.markdown("---")
st.subheader("⚠️ Meest recente overschrijdingen")

# Vergelijk alleen met beschikbare normen. Ontbrekende normen (pd.NA) gelden niet
# als overschrijding en worden vóór boolean-evaluatie expliciet False gemaakt.
jg_over = (df_filtered['Waarde'] > df_filtered['JG_MKN']).fillna(False).astype(bool)
mac_over = (df_filtered['Waarde'] > df_filtered['MAC_MKN']).fillna(False).astype(bool)
mask_any_over = jg_over | mac_over

df_violations = df_filtered.loc[mask_any_over].copy()

if not df_violations.empty:
    # Hergebruik de opgeschoonde maskers, zodat np.select uitsluitend echte booleans ontvangt.
    jg_over_violations = jg_over.loc[df_violations.index]
    mac_over_violations = mac_over.loc[df_violations.index]

    df_violations['Type'] = np.select(
        [
            (jg_over_violations & mac_over_violations).to_numpy(dtype=bool),
            jg_over_violations.to_numpy(dtype=bool),
            mac_over_violations.to_numpy(dtype=bool),
        ],
        ['JG+MAC', 'JG', 'MAC'],
        default='Onbekend',
    )
    # Toon gecorrigeerde en oorspronkelijke concentraties wanneer beschikbaar.
    violation_cols = ['Datum', 'Meetpunt', 'Stof']
    if 'Waarde_Origineel' in df_violations.columns:
        violation_cols.append('Waarde_Origineel')
    if 'Achtergrondconcentratie' in df_violations.columns:
        violation_cols.append('Achtergrondconcentratie')
    violation_cols.append('Waarde')
    if 'Achtergrondcorrectie_Toegepast' in df_violations.columns:
        violation_cols.append('Achtergrondcorrectie_Toegepast')
    violation_cols.extend(['Eenheid', 'JG_MKN', 'MAC_MKN', 'Type'])

    st.dataframe(
        df_violations[violation_cols]
        .sort_values('Datum', ascending=False).head(15),
        width="stretch",
        column_config={
            'Waarde_Origineel': st.column_config.NumberColumn(
                'Oorspronkelijke concentratie', format='%.4g'
            ),
            'Achtergrondconcentratie': st.column_config.NumberColumn(
                'Achtergrondconcentratie', format='%.4g'
            ),
            'Waarde': st.column_config.NumberColumn(
                'Gecorrigeerde concentratie', format='%.4g'
            ),
            'Achtergrondcorrectie_Toegepast': st.column_config.CheckboxColumn(
                'Achtergrondcorrectie toegepast'
            ),
        },
    )
else:
    st.success("Geen overschrijdingen gevonden.")