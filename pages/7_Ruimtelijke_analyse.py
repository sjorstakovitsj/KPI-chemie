import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar
from datetime import datetime

st.set_page_config(layout="wide", page_title="Ruimtelijke analyse")

st.header("Ruimtelijke analyse")

# Data en Sidebar
df_main = load_data()
df_filtered = get_shared_sidebar(df_main)

st.header("🔍 Ruimtelijke analyse")

# We werken hier verder met een kopie van df_filtered (al gefilterd op jaren sidebar)
# Optimalisatie: Kopieer alleen relevante kolommen
# AANGEPAST: 'Limietsymbool' toegevoegd aan cols_needed voor de filter functionaliteit
cols_needed = ['Datum', 'Meetpunt', 'Stof', 'Stofgroep', 'Waarde', 'Eenheid', 'Limietsymbool']
df_space = df_filtered[cols_needed].copy()

# Datum bereik bepalen
if not df_space.empty:
    min_d, max_d = df_space['Datum'].min().date(), df_space['Datum'].max().date()
else:
    min_d, max_d = datetime(2022,1,1).date(), datetime(2025,12,31).date()

# Filters UI
with st.container():
    c_zomer, c_start, c_end = st.columns([1.5, 2, 2])

    c_zomer.write("") # Spacing voor uitlijning
    
    # Checkbox 1: Zomerhalfjaar
    zomerhalfjaar = c_zomer.checkbox("Alleen zomerhalfjaar (apr-sep)", value=False)
    
    # AANGEPAST: Checkbox 2: Alleen detecties (>RG)
    alleen_detecties = c_zomer.checkbox("Alleen aangetroffen waarden (>RG)", value=True)

    start_date = c_start.date_input("Startdatum", value=min_d, min_value=min_d, max_value=max_d)
    end_date = c_end.date_input("Einddatum", value=max_d, min_value=min_d, max_value=max_d)

    c_loc, c_grp, c_stof = st.columns([2, 2, 2])

    # 1. Meetpunt Selectie
    loc_opts = sorted(df_space['Meetpunt'].unique())
    sel_loc = c_loc.multiselect("Meetpunt", loc_opts, default=loc_opts)

    # 2. Stofgroep Selectie
    grp_opts = sorted(df_space['Stofgroep'].unique())
    sel_grp = c_grp.multiselect("Stofgroep", grp_opts, default=[grp_opts[0]] if grp_opts else None)

    # --- DYNAMISCHE STOF OPTIES LOGICA ---
    # We berekenen hier welke stoffen beschikbaar zijn op basis van ALLE voorgaande filters
    # (Tijd, Locatie, Stofgroep EN Detectielimiet)
    
    # Start met basis datum filter
    mask_opt = (df_space['Datum'].dt.date >= start_date) & (df_space['Datum'].dt.date <= end_date)
    
    # Filter op zomerhalfjaar indien aangevinkt
    if zomerhalfjaar:
        mask_opt &= (df_space['Datum'].dt.month >= 4) & (df_space['Datum'].dt.month <= 9)
        
    # Filter op detectielimiet (>RG) voor de opties
    if alleen_detecties:
        mask_opt &= ~df_space['Limietsymbool'].astype(str).str.contains('<', na=False)

    # Filter op geselecteerde locaties
    if sel_loc:
        mask_opt &= df_space['Meetpunt'].isin(sel_loc)
        
    # Filter op geselecteerde stofgroep
    if sel_grp:
        mask_opt &= df_space['Stofgroep'].isin(sel_grp)
    
    # Haal de unieke stoffen op die voldoen aan dit masker
    stof_opts = sorted(df_space.loc[mask_opt, 'Stof'].unique())

    # 3. Stof Selectie
    sel_stof = c_stof.multiselect("Stof", stof_opts, default=stof_opts[:1] if stof_opts else [])

# --- DATA FILTERING LOGICA (VECTORIZED) ---
# Nu passen we de filters definitief toe voor de grafieken.
# (We doen dit opnieuw om de logica schoon te houden en dff_final correct op te bouwen)

# 1. Datum Filter
mask_date = (df_space['Datum'].dt.date >= start_date) & (df_space['Datum'].dt.date <= end_date)

# 2. Zomer Filter
if zomerhalfjaar:
    mask_zomer = (df_space['Datum'].dt.month >= 4) & (df_space['Datum'].dt.month <= 9)
    mask_date = mask_date & mask_zomer

# 3. Categorische filters
mask_loc = df_space['Meetpunt'].isin(sel_loc) if sel_loc else pd.Series(False, index=df_space.index)
mask_grp = df_space['Stofgroep'].isin(sel_grp) if sel_grp else pd.Series(True, index=df_space.index)
mask_stof = df_space['Stof'].isin(sel_stof) if sel_stof else pd.Series(False, index=df_space.index)

# 4. Detectie Limiet Filter (>RG)
if alleen_detecties:
    mask_limit = ~df_space['Limietsymbool'].astype(str).str.contains('<', na=False)
else:
    mask_limit = pd.Series(True, index=df_space.index)

# Pas alle filters in één keer toe
dff_final = df_space[mask_date & mask_loc & mask_grp & mask_stof & mask_limit].copy()

if dff_final.empty:
    st.warning("Geen data beschikbaar voor de geselecteerde criteria.")
else:
    # --- PLOTS ---

    # A. Tijdlijn & Kaart
    col_kaart, col_tijd = st.columns(2)

    # Aggregaties voor plots (observed=True voor speed)
    time_agg = dff_final.groupby(['Datum', 'Meetpunt'], observed=True)['Waarde'].mean().reset_index()
    tijd_fig = px.line(time_agg, x="Datum", y="Waarde", color="Meetpunt", markers=True, title="Verloop in de tijd")
    
    # Zorg dat de x-as alleen hele jaartallen toont
    tijd_fig.update_xaxes(dtick="M12", tickformat="%Y", ticklabelmode="period")
    
    col_tijd.plotly_chart(tijd_fig, use_container_width=True)

    # Kaart data
    loc_agg = dff_final.groupby("Meetpunt", observed=True)["Waarde"].mean().reset_index()

    # Efficiënte coördinaten lookup (zonder grote merge op de hele dataset)
    coords_ref = df_main[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()

    # Merge alleen de geaggregeerde tabel
    loc_map = pd.merge(loc_agg, coords_ref, on='Meetpunt', how='inner')

    if not loc_map.empty:
        kaart_fig = px.scatter_mapbox(
            loc_map, lat="Latitude", lon="Longitude", color="Waarde",
            size=[15]*len(loc_map), hover_name="Meetpunt",
            color_continuous_scale="YlOrRd", zoom=7.5, mapbox_style="open-street-map",
            title="Gemiddelde waarde per locatie"
        )
        col_kaart.plotly_chart(kaart_fig, use_container_width=True)
    else:
        col_kaart.info("Geen coördinaten beschikbaar voor deze meetpunten.")

    # B. Boxplot & Strip
    col_box, col_strip = st.columns(2)

    box_fig = px.box(dff_final, x="Stof", y="Waarde", color="Meetpunt", title="Verdeling per stof")
    col_box.plotly_chart(box_fig, use_container_width=True)

    strip_fig = px.strip(dff_final, x="Waarde", y="Stof", color="Meetpunt", title="Individuele meetwaarden")
    col_strip.plotly_chart(strip_fig, use_container_width=True)

    # C. Heatmap & Bar
    col_heat, col_sub = st.columns(2)

    # Pivot voor heatmap
    heat_data = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().unstack()
    heat_fig = px.imshow(heat_data, aspect="auto", color_continuous_scale="YlGnBu", title="Heatmap van gemiddelde waarden", text_auto=".3f")
    col_heat.plotly_chart(heat_fig, use_container_width=True)

    sub_agg = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().reset_index()
    sub_fig = px.bar(sub_agg, x="Stof", y="Waarde", color="Meetpunt", barmode="group", title="Gemiddelde per meetpunt", text=sub_agg['Waarde'].apply(lambda x: f'{x:.3f}'))
    col_sub.plotly_chart(sub_fig, use_container_width=True)

    # D. Fold Change Analyse
    st.markdown("---")
    st.subheader("Log2 fold change analyse")

    # Opties beperken tot meetpunten die in de huidige gefilterde set zitten
    beschikbare_mp_fc = sorted(dff_final['Meetpunt'].unique())

    if len(beschikbare_mp_fc) > 1:
        ref_mp = st.selectbox("Selecteer referentie meetpunt", beschikbare_mp_fc)

        # Bereken gemiddelden
        means = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().reset_index()

        # Split in Ref en Rest
        ref_data = means[means['Meetpunt'] == ref_mp][['Stof', 'Waarde']].rename(columns={'Waarde': 'Ref_Waarde'})

        # Merge terug
        fc_data = pd.merge(means, ref_data, on='Stof', how='inner')

        # Vectorized berekening
        fc_data = fc_data[fc_data['Ref_Waarde'] > 0]
        fc_data['Log2FC'] = np.log2(fc_data['Waarde'] / fc_data['Ref_Waarde'])

        # Filter referentie zelf eruit voor plot
        fc_plot = fc_data[fc_data['Meetpunt'] != ref_mp]

        if not fc_plot.empty:
            fig_fc = px.scatter(
                fc_plot, x="Log2FC", y="Stof", color="Meetpunt",
                title=f"Fold Change t.o.v. {ref_mp}",
                hover_data={'Waarde':':.2f', 'Ref_Waarde':':.2f'}
            )
            fig_fc.add_vline(x=0, line_dash="dash", line_color="black")
            fig_fc.add_vline(x=1, line_dash="dot", line_color="gray")
            fig_fc.add_vline(x=-1, line_dash="dot", line_color="gray")
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Geen overlappende stoffen gevonden om te vergelijken.")
    else:
        st.info("Selecteer minimaal 2 meetpunten voor Fold Change analyse.")