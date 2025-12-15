# Home.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
# NIEUWE SECTIE: Chemische Toestand & Bollenschema (Geavanceerd)
# -----------------------------------------------------
st.header("🔴 KRW-check: overschrijdende stoffen")
st.info("Voor enkele stoffen is correctie met achtergrondwaardes van toepassing voor de KRW. In deze tool is dat niet meegenomen.")

def calculate_compliance_details(df_in):
    """
    Berekent overschrijdingen inclusief overschrijdingsfactor.
    """
    required_cols = ['Meetpunt', 'Datum', 'Stof', 'Waarde', 'JG_MKN', 'MAC_MKN']
    if not all(col in df_in.columns for col in required_cols):
        return pd.DataFrame()

    df_calc = df_in.copy()
    df_calc['Jaar'] = df_calc['Datum'].dt.year

    # --- STAP 1: JG Toetsing ---
    if 'Limietsymbool' in df_calc.columns:
        # Volgens verzoek: waarden zonder '<' symbool
        df_jg_valid = df_calc[df_calc['Limietsymbool'] != '<'].copy()
    else:
        df_jg_valid = df_calc.copy()

    jg_means = df_jg_valid.groupby(['Meetpunt', 'Jaar', 'Stof', 'JG_MKN'])['Waarde'].mean().reset_index()
    jg_means.rename(columns={'Waarde': 'JaarGemiddelde'}, inplace=True)
    
    # Filter fouten en bereken factor
    jg_failures = jg_means[jg_means['JaarGemiddelde'] > jg_means['JG_MKN']].copy()
    jg_failures['Normtype'] = 'JG-MKN'
    jg_failures['Factor'] = jg_failures['JaarGemiddelde'] / jg_failures['JG_MKN']

    # --- STAP 2: MAC Toetsing ---
    mac_maxs = df_calc.groupby(['Meetpunt', 'Jaar', 'Stof', 'MAC_MKN'])['Waarde'].max().reset_index()
    mac_maxs.rename(columns={'Waarde': 'JaarMax'}, inplace=True)
    
    # Filter fouten en bereken factor
    mac_failures = mac_maxs[mac_maxs['JaarMax'] > mac_maxs['MAC_MKN']].copy()
    mac_failures['Normtype'] = 'MAC-MKN'
    mac_failures['Factor'] = mac_failures['JaarMax'] / mac_failures['MAC_MKN']

    # --- STAP 3: Samenvoegen ---
    combined = pd.concat([
        jg_failures[['Meetpunt', 'Jaar', 'Stof', 'Normtype', 'Factor']], 
        mac_failures[['Meetpunt', 'Jaar', 'Stof', 'Normtype', 'Factor']]
    ])
    
    return combined

# Data ophalen
df_failures = calculate_compliance_details(df_filtered)

if df_failures.empty:
    st.success("Geen normoverschrijdingen gevonden! 🎉")
else:
    meetpunten_met_fouten = sorted(df_failures['Meetpunt'].unique())
    selected_mp = st.selectbox("Selecteer een meetpunt voor detailweergave:", meetpunten_met_fouten)
    
    # Filter op meetpunt
    df_mp_fail = df_failures[df_failures['Meetpunt'] == selected_mp].copy()
    
    # --- LOGICA VOOR KLEUREN EN CATEGORIEËN ---
    stof_summary = []
    
    for stof, group in df_mp_fail.groupby('Stof'):
        types = group['Normtype'].unique()
        max_factor = group['Factor'].max()
        
        # Categorie bepalen
        if 'JG-MKN' in types and 'MAC-MKN' in types:
            cat = "JG + MAC"
            base_color_scale = 'Reds' # Rood voor dubbel falen
        elif 'JG-MKN' in types:
            cat = "JG (Gemiddelde)"
            base_color_scale = 'Oranges' # Oranje voor chronisch
        else:
            cat = "MAC (Piek)"
            base_color_scale = 'Purples' # Paars voor acuut
            
        # Kleurintensiteit bepalen: 1.0 = licht, 5.0+ = donkerst
        # Normale factor tussen 0.0 en 1.0
        norm_val = min((max_factor - 1) / 4, 1.0) 
        # Start de schaal niet op wit (0.0), maar bij 0.3 zodat het altijd zichtbaar is
        color_val = 0.3 + (norm_val * 0.7) 
        
        # Gebruik de kleurenfuncties van Plotly om de tint op te halen
        hex_color = px.colors.sample_colorscale(base_color_scale, [color_val])[0]
        
        stof_summary.append({
            'Stof': stof,
            'Categorie': cat,
            'Factor': max_factor,
            'Color': hex_color
        })
        
    df_viz = pd.DataFrame(stof_summary)
    
    # --- VISUALISATIE ---
    col_graph, col_list = st.columns([1, 1])
    
    with col_graph:
        # Sunburst constructie met go
        ids = ["Totaal"] + df_viz['Stof'].tolist()
        labels = ["Chemische<br>Toestand"] + df_viz['Stof'].tolist()
        parents = [""] + ["Totaal"] * len(df_viz)
        
        # Kleurenlijst: Root is grijs, rest komt uit dataframe
        colors = ["#DDDDDD"] + df_viz['Color'].tolist()
        
        # Hover info maken
        hover_text = [""] + [f"Type: {row['Categorie']}<br>Max factor: {row['Factor']:.1f}x" for i, row in df_viz.iterrows()]

        fig_sun = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            marker=dict(colors=colors),
            hovertext=hover_text,
            hoverinfo="label+text",
            insidetextorientation='radial'
        ))
        
        fig_sun.update_layout(
            margin=dict(t=30, l=0, r=0, b=10),
            title=f"Overschrijdingen: {selected_mp}",
            height=500
        )
        
        st.plotly_chart(fig_sun, use_container_width=True)
        
        # Legenda voor kleuren
        st.caption("🎨 **Legenda:** 🟧 **Oranje** = JG-norm | 🟪 **Paars** = MAC-norm | 🟥 **Rood** = Beide. (Donkerder = hogere overschrijdingsfactor)")

with col_list:
        st.markdown(f"**Details ({selected_mp})**")
        
        # Sorteer de data
        st_display = df_mp_fail[['Jaar', 'Stof', 'Normtype', 'Factor']].sort_values(by=['Factor', 'Stof', 'Jaar'], ascending=[False, True, False])
        
        # We gebruiken st.dataframe met column_config voor een native visualisatie
        st.dataframe(
            st_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Factor": st.column_config.ProgressColumn(
                    "Factor (x norm)",     # Titel van de kolom
                    format="%.1f x",       # Formaat (bijv 2.5 x)
                    min_value=0,           # Minimum van de balk
                    max_value=5,           # Maximum (alles daarboven is volle balk)
                ),
                "Jaar": st.column_config.NumberColumn(
                    "Jaar",
                    format="%d"            # Zorg dat jaar zonder komma wordt getoond
                )
            }
        )
        st.caption("De tabel toont de stoffen die gemiddeld (JG) per jaar of elke individuele (MAC) de norm overschrijden. Bij JG zijn metingen onder rapportagegrens buiten beschouwing gelaten.")

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
    
