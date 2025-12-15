import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar

st.set_page_config(layout="wide", page_title="KRW normcheck")

st.header("KRW normcheck")

# Data en Sidebar
df_main = load_data()
df_filtered = get_shared_sidebar(df_main)

st.header("✅ KRW normcheck per meetpunt")

col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

with col_filter_1:
    all_meetpunten_norm = sorted(df_main['Meetpunt'].unique())
    default_meetpunten_norm = all_meetpunten_norm[:3] if all_meetpunten_norm else []
    selected_meetpunten_norm = st.multiselect(
        "📍 Selecteer meetpunt(en)",
        options=all_meetpunten_norm,
        default=default_meetpunten_norm,
        key="tab2_meetpunten_select"
    )

with col_filter_2:
    # 1. Bepaal eenmalig welke stoffen een norm hebben (JG_MKN of MAC_MKN is niet NaN)
    stoffen_met_norm_df = df_filtered[df_filtered[['JG_MKN', 'MAC_MKN']].notna().any(axis=1)].copy()

    # 2. Gebruik ALLEEN de stofgroepen die deze genormeerde stoffen bevatten
    stofgroep_opties = sorted(stoffen_met_norm_df['Stofgroep'].unique())
    stofgroep_selected_tab2 = st.multiselect(
        "Selecteer stofgroep",
        options=stofgroep_opties,
        default=stofgroep_opties[0] if stofgroep_opties else None,
        key="tab2_stofgroep_select"
    )

with col_filter_3:
    if stofgroep_selected_tab2:
        # Filter de genormeerde stoffen-dataframe op de geselecteerde stofgroepen
        filtered_stoffen_df = stoffen_met_norm_df[stoffen_met_norm_df['Stofgroep'].isin(stofgroep_selected_tab2)]
        
        # 3. Nu bevat all_stoffen_norm enkel stoffen die:
        #    a) een norm hebben (MAC of JG)
        #    b) behoren tot de geselecteerde stofgroep(en)
        all_stoffen_norm = sorted(filtered_stoffen_df['Stof'].unique())
    else:
        # Als er geen stofgroep geselecteerd is, toon alle genormeerde stoffen
        all_stoffen_norm = sorted(stoffen_met_norm_df['Stof'].unique())

    # De logica voor slimme defaults is nu overbodig omdat all_stoffen_norm al is gefilterd
    # We kunnen de 'default_stoffen_norm' gewoon als de eerste 3 van de gefilterde lijst nemen.
    default_stoffen_norm = all_stoffen_norm[:3] if len(all_stoffen_norm) > 0 else []

    selected_stoffen_norm = st.multiselect(
        "🔎 Selecteer stof(fen)",
        options=all_stoffen_norm,
        default=default_stoffen_norm,
        key="tab2_stoffen_select"
    )

# Checkbox voor filteren op detecties
alleen_detecties = st.checkbox("Alleen aangetroffen waarden meenemen (>RG) in normcheck", value=True)

if not selected_stoffen_norm or not selected_meetpunten_norm:
    st.warning("Selecteer minstens één stof en één meetpunt.")
else:
    # --- JG-MKN Toetsing (Jaargemiddelden) ---
    st.subheader("Jaargemiddelde toetsing (JG-MKN)")
    st.markdown("Onderstaande grafieken tonen het **jaargemiddelde** per stof en meetpunt, afgezet tegen de norm (rode stippellijn).")

    # 1. Filter basis selectie
    df_jg = df_filtered[
        (df_filtered['Stof'].isin(selected_stoffen_norm)) &
        (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
        (df_filtered['JG_MKN'].notna())
    ].copy()

    # 2. Filter op < waarden indien aangevinkt
    if alleen_detecties and not df_jg.empty:
        df_jg = df_jg[~df_jg['Limietsymbool'].astype(str).str.contains('<', na=False)]

    if not df_jg.empty:
        df_jg['Jaar'] = df_jg['Datum'].dt.year

        # 3. Bereken jaargemiddelde per jaar/meetpunt/stof
        # observed=True is belangrijk voor performance
        df_gemiddelde = df_jg.groupby(['Jaar', 'Meetpunt', 'Stof'], observed=True)['Waarde'].mean().reset_index()

        # Voeg norm toe
        norm_lookup = df_jg[['Stof', 'JG_MKN', 'Eenheid']].drop_duplicates()
        df_gemiddelde = pd.merge(df_gemiddelde, norm_lookup, on='Stof', how='left')

        # Maak Jaar string voor categorische x-as
        df_gemiddelde['JaarStr'] = df_gemiddelde['Jaar'].astype(str)

        # 4. Loop per stof om een duidelijke grafiek te maken
        # (Samenvoegen in één grafiek werkt slecht door verschillende schalen/eenheden)
        for stof in selected_stoffen_norm:
            df_stof = df_gemiddelde[df_gemiddelde['Stof'] == stof]
            
            if df_stof.empty:
                continue

            # Haal norm en eenheid op voor titel/lijn
            norm_waarde = df_stof['JG_MKN'].iloc[0]
            eenheid = df_stof['Eenheid'].iloc[0]

            # Bepaal kleuren: Rood als gemiddelde > norm, anders standaard kleur per meetpunt?
            # Standaard Plotly group bar is beter per meetpunt gekleurd voor vergelijking locaties.
            # We voegen de norm toe als lijn.
            
            fig = px.bar(
                df_stof,
                x="JaarStr",
                y="Waarde",
                color="Meetpunt",
                barmode="group",
                title=f"Toetsing {stof} (Norm: {norm_waarde} {eenheid})",
                labels={"Waarde": f"Concentratie ({eenheid})", "JaarStr": "Jaar"},
                hover_data={"Waarde": ":.2f", "JG_MKN": ":.2f"}
            )

            # Voeg de Norm lijn toe
            fig.add_hline(
                y=norm_waarde, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"JG-MKN ({norm_waarde})",
                annotation_position="top right"
            )
            
            # NIEUW: Forceer discrete weergave van jaartallen
            fig.update_xaxes(type='category', tickangle=0) 

            st.plotly_chart(fig, use_container_width=True)
            
        # Optioneel: Toon de data in een tabel
        with st.expander("Toon berekende jaargemiddelden in tabel"):
            display_cols = ['Jaar', 'Meetpunt', 'Stof', 'Waarde', 'JG_MKN']
            st.dataframe(df_gemiddelde[display_cols].sort_values(['Stof', 'Jaar', 'Meetpunt']))

    else:
        st.info("Geen data beschikbaar met JG-MKN norm voor de huidige selectie (of alles weggefilterd door >RG filter).")

    st.markdown("---")

    # --- MAC-MKN Toetsing ---
    st.subheader("Maximale aanvaardbare concentratie (MAC-MKN)")
    st.markdown("Percentage metingen dat de pieknorm overschrijdt.")

    df_mac = df_filtered[
        (df_filtered['Stof'].isin(selected_stoffen_norm)) &
        (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
        (df_filtered['MAC_MKN'].notna())
    ].copy()

    # Filter op < waarden indien aangevinkt
    if alleen_detecties and not df_mac.empty:
        df_mac = df_mac[~df_mac['Limietsymbool'].astype(str).str.contains('<', na=False)]

    if not df_mac.empty:
        df_mac['Overschrijding'] = df_mac['Waarde'] > df_mac['MAC_MKN']

        # Hier behouden we de "Percentage" weergave, want MAC gaat over pieken, niet jaargemiddelden
        mac_overschrijding_pct = df_mac.groupby(['Meetpunt', 'Stof'], observed=True)['Overschrijding'].mean().reset_index()
        mac_overschrijding_pct['Overschrijding (%)'] = (mac_overschrijding_pct['Overschrijding'] * 100).round(2)

        mac_overschrijding_pct['PlotLabel'] = mac_overschrijding_pct['Meetpunt'].astype(str) + ' (' + mac_overschrijding_pct['Stof'].astype(str) + ')'

        fig_mac = px.bar(
            mac_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
            x='PlotLabel', y='Overschrijding (%)',
            title='Percentage individuele metingen boven MAC-MKN',
            color='Meetpunt',
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_mac, use_container_width=True)
    else:
        if alleen_detecties:
            st.info("Geen waarden boven de rapportagegrens gevonden voor de MAC-MKN selectie.")
        else:
            st.info("Geen MAC-MKN data voor selectie.")