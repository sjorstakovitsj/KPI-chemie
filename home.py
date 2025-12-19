# Home.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
# Importeer de nieuwe functies
from utils import (
    load_data, create_gauge, get_shared_sidebar, 
    calculate_metrics, calculate_compliance_details, 
    prepare_heatmap_data, prepare_sunburst_data
)

st.set_page_config(layout="wide", page_title="Waterkwaliteit KPI Dashboard", page_icon="💧")
st.title("💧 Dashboard chemische waterkwaliteit MN")

# 1. Data Laden
df_main = load_data()

if df_main.empty:
    st.error("🚨 Kritieke fout: de geladen data is leeg.")
    st.stop()

# 2. Sidebar & Filteren
df_filtered = get_shared_sidebar(df_main)

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
st.info("Achtergrondwaarde correctie is niet meegenomen in dit dashboard.")

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
        st.plotly_chart(fig_heatmap, use_container_width=True)
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
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_list:
        st.markdown(f"**Details ({selected_mp})**")
        st_display = df_mp_fail[['Jaar', 'Stof', 'Normtype', 'Factor']].sort_values(by=['Factor', 'Stof', 'Jaar'], ascending=[False, True, False])
        st.dataframe(
            st_display, use_container_width=True, hide_index=True,
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
    st.metric("Unieke meetpunten", df_main['Meetpunt'].nunique())
    # Metingen zonder norm: totaal - (metingen met JG of MAC)
    df_with_norm = df_filtered.dropna(subset=['JG_MKN', 'MAC_MKN'], how='all')
    st.metric("Metingen zonder JG/MAC norm", len(df_filtered) - len(df_with_norm))
    st.metric("Metingen met JG-norm", len(df_jg_ov))
    st.metric("Metingen met MAC-norm", len(df_mac_ov))

with col_gauges:
    s1, s2 = st.columns(2)
    s1.plotly_chart(create_gauge(pct_jg_total, "Totaal: voldoet JG (%)"), use_container_width=True)
    s2.plotly_chart(create_gauge(pct_mac_total, "Totaal: voldoet MAC (%)"), use_container_width=True)

with col_map:
    df_map = df_filtered[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()
    if not df_map.empty:
        fig_map = px.scatter_mapbox(
            df_map, lat='Latitude', lon='Longitude', hover_name='Meetpunt',
            size_max=15, zoom=8, mapbox_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=12, color='red'))
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# 6. Detail Gauges per meetpunt
unieke_meetpunten = sorted(df_filtered['Meetpunt'].unique())

# Pre-calc stats (vectorized)
jg_stats = df_jg_ov.groupby('Meetpunt', observed=True).apply(lambda x: (x['Waarde'] <= x['JG_MKN']).mean() * 100)
mac_stats = df_mac_ov.groupby('Meetpunt', observed=True).apply(lambda x: (x['Waarde'] <= x['MAC_MKN']).mean() * 100)

with st.expander(f"Toon detailmeters voor alle {len(unieke_meetpunten)} meetpunten", expanded=True):
    for mp in unieke_meetpunten:
        p_jg = jg_stats.get(mp, None)
        p_mac = mac_stats.get(mp, None)
        if pd.isna(p_jg) and pd.isna(p_mac): continue

        st.markdown(f"**{mp}**")
        c1, c2 = st.columns(2)
        if pd.notna(p_jg): c1.plotly_chart(create_gauge(p_jg, f"JG: {mp}", 95), use_container_width=True, key=f"g_jg_{mp}")
        if pd.notna(p_mac): c2.plotly_chart(create_gauge(p_mac, f"MAC: {mp}", 95), use_container_width=True, key=f"g_mac_{mp}")

st.markdown("---")
st.subheader("⚠️ Meest recente overschrijdingen")

mask_any_over = (df_filtered['Waarde'] > df_filtered['JG_MKN']) | (df_filtered['Waarde'] > df_filtered['MAC_MKN'])
df_violations = df_filtered[mask_any_over].copy()

if not df_violations.empty:
    df_violations['Type'] = np.where(
        (df_violations['Waarde'] > df_violations['JG_MKN']) & (df_violations['Waarde'] > df_violations['MAC_MKN']), "JG+MAC",
        np.where(df_violations['Waarde'] > df_violations['JG_MKN'], "JG", "MAC")
    )
    st.dataframe(
        df_violations[['Datum', 'Meetpunt', 'Stof', 'Waarde', 'Eenheid', 'JG_MKN', 'MAC_MKN', 'Type']]
        .sort_values('Datum', ascending=False).head(15),
        use_container_width=True
    )
else:
    st.success("Geen overschrijdingen gevonden.")