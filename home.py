# Home.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_data, create_gauge, get_shared_sidebar

# Configuratie moet als aller-allereerste
st.set_page_config(layout="wide", page_title="Waterkwaliteit KPI Dashboard", page_icon="💧")

st.title("💧 Dashboard chemische waterkwaliteit MN")

# Data laden (gecached)
df_main = load_data()

if df_main.empty:
    st.error("🚨 Kritieke fout: de geladen en opgeschoonde data is leeg.")
    st.stop()

# Sidebar aanroepen en gefilterde data ophalen
df_filtered = get_shared_sidebar(df_main)


# Hulpfunctie om de KPI's voor een dataframe te berekenen
def calculate_metrics(df_in: pd.DataFrame, is_period_average: bool = False):
    """
    Berekent metingen, overschrijdingen en percentage.
    Indien is_period_average=True, berekent het gemiddelde per jaar in de periode.
    """
    if df_in.empty:
        return 0, 0, 0.0
    
    # Voeg jaar toe voor groepering
    df_in['Jaar'] = df_in['Datum'].dt.year
    
    # Bepaal het aantal unieke jaren
    unique_years = df_in['Jaar'].nunique()

    # Masker voor overschrijdingen (JG of MAC)
    mask_jg_over = (df_in['Waarde'] > df_in['JG_MKN'])
    mask_mac_over = (df_in['Waarde'] > df_in['MAC_MKN'])
    
    # Totaal over de (gehele of enkel) dataset
    total_count = len(df_in)
    total_viol_count = (mask_jg_over | mask_mac_over).sum()
    
    if is_period_average and unique_years > 0:
        # Als het een periode is, bereken de gemiddelden per jaar
        avg_count = total_count / unique_years
        avg_viol_count = total_viol_count / unique_years
        
        # Het percentage wordt over de totale periode berekend voor stabiliteit
        pct_viol = (total_viol_count / total_count * 100)
        
        return avg_count, avg_viol_count, pct_viol
    else:
        # Huidig jaar (of een enkel vergelijkingsjaar)
        pct_viol = (total_viol_count / total_count * 100) if total_count > 0 else 0.0
        return total_count, total_viol_count, pct_viol


# --- DYNAMISCHE SCOREKAART: Recent vs Selectie ---
available_years = sorted(df_filtered['Datum'].dt.year.unique(), reverse=True)

if len(available_years) >= 1:
    current_year = available_years[0]
    df_curr = df_filtered[df_filtered['Datum'].dt.year == current_year]
    
    st.subheader(f"Vergelijking huidig jaar ten opzichte van voorgaande jaren: **{current_year}**")

    # 1. Bepaal vergelijkingsopties
    comparison_options = {}
    
    # Opties voor individuele voorgaande jaren
    for year in available_years[1:]:
        comparison_options[f"voorgaand jaar: {year}"] = {'years': [year], 'is_period': False}

    # Optie voor alle voorgaande jaren als periode (alleen als er > 1 jaar beschikbaar is)
    if len(available_years) > 1:
        prev_years = available_years[1:]
        start_year = prev_years[-1]
        end_year = prev_years[0]
        # Als er meer dan één voorgaand jaar is, markeer het als periode
        is_multi_year = len(prev_years) > 1
        comparison_options[f"Periode: {start_year} t/m {end_year} (gemiddeld)"] = {'years': prev_years, 'is_period': is_multi_year}
        
    if not comparison_options:
        st.info("Niet genoeg historische data (minimaal 2 jaar) om te vergelijken.")
    else:
        # 2. Selectbox
        # Zet de periode-optie standaard bovenaan als deze bestaat
        default_index = 0
        keys = list(comparison_options.keys())
        if keys[0].startswith("Periode:"):
             default_index = 0 # Standaard de periode tonen

        selected_option = st.selectbox(
            f"Selecteer het jaar of de periode om **{current_year}** mee te vergelijken:",
            options=keys,
            index=default_index
        )
        
        # 3. Data voor vergelijking ophalen
        comp_info = comparison_options[selected_option]
        comparison_years = comp_info['years']
        is_period_comparison = comp_info['is_period']
        
        df_comp = df_filtered[df_filtered['Datum'].dt.year.isin(comparison_years)].copy()
    
        # 4. Berekeningen uitvoeren
        # Huidige jaar (geen gemiddelde)
        count_curr, viol_curr, pct_curr = calculate_metrics(df_curr, is_period_average=False)
        
        # Vergelijking (wel of geen gemiddelde)
        comp_count, comp_viol, comp_pct = calculate_metrics(df_comp, is_period_average=is_period_comparison)
            
        # 5. Delta's berekenen
        delta_count = count_curr - comp_count
        delta_viol = viol_curr - comp_viol
        delta_pct = pct_curr - comp_pct
    
        # 6. Metrics tonen
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        # Pas de labels aan om duidelijk te maken wat er vergeleken wordt
        comp_label = "gemiddeld" if is_period_comparison else ""
        
        col_kpi1.metric(
            label=f"Aantal metingen{comp_label}", 
            value=f"{count_curr:.0f} (vs: {comp_count:.0f})", 
            delta=f"{delta_count:.0f}"
        )
        
        col_kpi2.metric(
            label=f"Aantal overschrijdingen {comp_label}", 
            value=f"{viol_curr:.0f} (vs: {comp_viol:.0f})", 
            delta=f"{delta_viol:.0f}",
            delta_color="inverse" # Rood als het stijgt (slecht), groen als het daalt
        )
        
        col_kpi3.metric(
            label=f"Percentage normoverschrijdingen {comp_label}",
            value=f"{pct_curr:.1f}% (vs: {comp_pct:.1f}%)",
            delta=f"{delta_pct:.1f}%",
            delta_color="inverse"
        )
    
    st.divider()

# -----------------------------------------------------

st.header("📊 Overzicht toestand & kaart meetpunten")

# Berekeningen over totale gefilterde set
col_kpi_info, col_kpi_gauges, col_map = st.columns([1, 2, 2])

# MAC Overall
df_mac_overall = df_filtered.dropna(subset=['MAC_MKN'])
pct_mac_total = 0
if not df_mac_overall.empty:
    voldoet = (df_mac_overall['Waarde'] <= df_mac_overall['MAC_MKN']).sum()
    totaal = len(df_mac_overall)
    pct_mac_total = round((voldoet / totaal * 100), 1)

# JG Overall
df_jg_overall = df_filtered.dropna(subset=['JG_MKN'])
pct_jg_total = 0
if not df_jg_overall.empty:
    voldoet = (df_jg_overall['Waarde'] <= df_jg_overall['JG_MKN']).sum()
    totaal = len(df_jg_overall)
    pct_jg_total = round((voldoet / totaal * 100), 1)

# --- NIEUWE METRIC: Metingen zonder norm ---
# Een meting heeft geen JG of MAC-norm als BEIDE kolommen NaN zijn.
# Eerst bepalen we de metingen waar WEL een norm beschikbaar is (niet NaN voor JG OF MAC)
df_with_norm = df_filtered.dropna(subset=['JG_MKN', 'MAC_MKN'], how='all')

# Het verschil is het aantal metingen zonder norm
metingen_zonder_norm = len(df_filtered) - len(df_with_norm)
# -------------------------------------------


with col_kpi_info:
    st.metric(label="Unieke meetpunten", value=df_main['Meetpunt'].nunique())
    # Nieuwe metric
    st.metric(label="Metingen zonder JG/MAC norm", value=metingen_zonder_norm)
    st.metric(label="Metingen met JG-norm", value=len(df_jg_overall))
    st.metric(label="Metingen met MAC-norm", value=len(df_mac_overall))

with col_kpi_gauges:
    sub1, sub2 = st.columns(2)
    sub1.plotly_chart(create_gauge(pct_jg_total, "Totaal: voldoet JG (%)"), use_container_width=True)
    sub2.plotly_chart(create_gauge(pct_mac_total, "Totaal: voldoet MAC (%)"), use_container_width=True)

with col_map:
    df_map = df_filtered[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()
    if not df_map.empty:
        center_lat = df_map['Latitude'].mean()
        center_lon = df_map['Longitude'].mean()
        
        fig_map = px.scatter_mapbox(
            df_map, lat='Latitude', lon='Longitude', hover_name='Meetpunt',
            size_max=15, zoom=8, mapbox_style="open-street-map"
        )
        fig_map.update_traces(marker=dict(size=12, color='red'))
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, mapbox_center={"lat": center_lat, "lon": center_lon})
        st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# KPI per specifiek meetpunt
unieke_meetpunten = sorted(df_filtered['Meetpunt'].unique())

# JG Stats per punt
jg_stats = df_jg_overall.groupby('Meetpunt', observed=True).apply(
    lambda x: (x['Waarde'] <= x['JG_MKN']).mean() * 100
)
# MAC Stats per punt
mac_stats = df_mac_overall.groupby('Meetpunt', observed=True).apply(
    lambda x: (x['Waarde'] <= x['MAC_MKN']).mean() * 100
)

with st.expander(f"Toon detailmeters voor alle {len(unieke_meetpunten)} meetpunten", expanded=True):
    for meetpunt in unieke_meetpunten:
        pct_jg_mp = jg_stats.get(meetpunt, None)
        pct_mac_mp = mac_stats.get(meetpunt, None)
        
        if pd.isna(pct_jg_mp) and pd.isna(pct_mac_mp):
                continue

        st.markdown(f"**{meetpunt}**")
        c1, c2 = st.columns(2)
        
        if pd.notna(pct_jg_mp):
            c1.plotly_chart(create_gauge(pct_jg_mp, f"JG: {meetpunt}", 95), use_container_width=True, key=f"g_jg_{meetpunt}")
        else:
            c1.info("Geen JG data")
            
        if pd.notna(pct_mac_mp):
            c2.plotly_chart(create_gauge(pct_mac_mp, f"MAC: {meetpunt}", 95), use_container_width=True, key=f"g_mac_{meetpunt}")
        else:
            c2.info("Geen MAC data")
        
        st.markdown("---")

st.subheader("⚠️ Meest recente overschrijdingen")

mask_jg_over = (df_filtered['Waarde'] > df_filtered['JG_MKN'])
mask_mac_over = (df_filtered['Waarde'] > df_filtered['MAC_MKN'])

df_violations = df_filtered[mask_jg_over | mask_mac_over].copy()

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