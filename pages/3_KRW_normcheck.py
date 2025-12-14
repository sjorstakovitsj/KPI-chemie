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
    stofgroep_opties = sorted(df_filtered['Stofgroep'].unique())
    stofgroep_selected_tab2 = st.multiselect(
        "Selecteer stofgroep",
        options=stofgroep_opties,
        default=stofgroep_opties[0] if stofgroep_opties else None,
        key="tab2_stofgroep_select"
    )

with col_filter_3:
    if stofgroep_selected_tab2:
        filtered_stoffen_df = df_filtered[df_filtered['Stofgroep'].isin(stofgroep_selected_tab2)]
        all_stoffen_norm = sorted(filtered_stoffen_df['Stof'].unique())
    else:
        all_stoffen_norm = sorted(df_filtered['Stof'].unique())

    # Slimme defaults: stoffen met norm
    stoffen_met_norm = df_filtered.dropna(subset=['JG_MKN', 'MAC_MKN'])['Stof'].unique()
    mogelijke_defaults = np.intersect1d(stoffen_met_norm, all_stoffen_norm)

    default_stoffen_norm = sorted(mogelijke_defaults)[:3] if len(mogelijke_defaults) > 0 else []

    selected_stoffen_norm = st.multiselect(
        "🔎 Selecteer stof(fen)",
        options=all_stoffen_norm,
        default=default_stoffen_norm,
        key="tab2_stoffen_select"
    )

if not selected_stoffen_norm or not selected_meetpunten_norm:
    st.warning("Selecteer minstens één stof en één meetpunt.")
else:
    # JG-MKN Toetsing
    st.subheader("Jaargemiddelde toetsing (JG-MKN)")

    # Filter eerst, daarna berekenen
    df_jg = df_filtered[
        (df_filtered['Stof'].isin(selected_stoffen_norm)) &
        (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
        (df_filtered['JG_MKN'].notna())
    ].copy()

    if not df_jg.empty:
        df_jg['Jaar'] = df_jg['Datum'].dt.year

        # Bereken jaargemiddelde
        # observed=True is belangrijk bij categorical data voor performance
        df_gemiddelde = df_jg.groupby(['Jaar', 'Meetpunt', 'Stof'], observed=True)['Waarde'].mean().reset_index()

        # Voeg norm toe (is constant per stof)
        norm_lookup = df_jg[['Stof', 'JG_MKN']].drop_duplicates()
        df_gemiddelde = pd.merge(df_gemiddelde, norm_lookup, on='Stof', how='left')

        df_gemiddelde['Overschrijding'] = df_gemiddelde['Waarde'] > df_gemiddelde['JG_MKN']

        jg_overschrijding_pct = df_gemiddelde.groupby(['Meetpunt', 'Stof'], observed=True)['Overschrijding'].mean().reset_index()
        jg_overschrijding_pct['Overschrijding (%)'] = (jg_overschrijding_pct['Overschrijding'] * 100).round(2)

        # Convert categories to str for plotting labels
        jg_overschrijding_pct['PlotLabel'] = jg_overschrijding_pct['Meetpunt'].astype(str) + ' (' + jg_overschrijding_pct['Stof'].astype(str) + ')'

        fig_jg = px.bar(
            jg_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
            x='PlotLabel',
            y='Overschrijding (%)',
            title='Percentage jaren boven JG-MKN norm',
            color='Meetpunt',
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_jg, use_container_width=True)
    else:
        st.info("Geen JG-MKN data voor selectie.")

    st.markdown("---")

    # MAC-MKN Toetsing
    st.subheader("Maximale aanvaardbare concentratie (MAC-MKN)")

    df_mac = df_filtered[
        (df_filtered['Stof'].isin(selected_stoffen_norm)) &
        (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
        (df_filtered['MAC_MKN'].notna())
    ].copy()

    if not df_mac.empty:
        df_mac['Overschrijding'] = df_mac['Waarde'] > df_mac['MAC_MKN']

        mac_overschrijding_pct = df_mac.groupby(['Meetpunt', 'Stof'], observed=True)['Overschrijding'].mean().reset_index()
        mac_overschrijding_pct['Overschrijding (%)'] = (mac_overschrijding_pct['Overschrijding'] * 100).round(2)

        mac_overschrijding_pct['PlotLabel'] = mac_overschrijding_pct['Meetpunt'].astype(str) + ' (' + mac_overschrijding_pct['Stof'].astype(str) + ')'

        fig_mac = px.bar(
            mac_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
            x='PlotLabel', y='Overschrijding (%)',
            title='Percentage metingen boven MAC-MKN',
            color='Meetpunt',
            color_continuous_scale=px.colors.sequential.Reds
        )
        st.plotly_chart(fig_mac, use_container_width=True)
    else:
        st.info("Geen MAC-MKN data voor selectie.")