import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar, load_pfas_ref

st.set_page_config(layout="wide", page_title="PFAS effectbeoordeling")

st.header("PFAS effectbeoordeling")

# Data en Sidebar
df_main = load_data()
df_filtered = get_shared_sidebar(df_main)

st.header("Effectbeoordeling PFAS")
df_pfas_ref = load_pfas_ref()

if not df_pfas_ref.empty:
    # 1. Voorbereiding & Merging (Vectorized)
    # We maken een tijdelijke clean kolom in df_filtered voor de merge
    df_filtered['Stof_Clean'] = df_filtered['Stof'].str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True).str.strip()
    df_pfas_ref['Stofnaam'] = df_pfas_ref['Stofnaam'].str.lower().str.strip()

    # Inner join om alleen PFAS stoffen over te houden
    df_pfas_calc = pd.merge(df_filtered, df_pfas_ref, left_on='Stof_Clean', right_on='Stofnaam', how='inner')

    if not df_pfas_calc.empty:
        # 2. Globale Berekeningen (één keer voor alle grafieken)
        # Detectie limiet logica: als '<' in symbool, dan waarde = 0 voor berekening
        detected = ~df_pfas_calc['Limietsymbool'].astype(str).str.contains('<', na=False)

        df_pfas_calc['RPF_calc'] = np.where(detected, df_pfas_calc['RPF'], 0)
        df_pfas_calc['RBF_calc'] = np.where(detected, df_pfas_calc['RBF'], 0)

        # Waarde omrekenen van ug/l naar ng/l (* 1000)
        df_pfas_calc['Waarde_ng'] = df_pfas_calc['Waarde'] * 1000

        # PEQ en Bioaccumulatie berekenen
        df_pfas_calc['PEQ_Waarde'] = df_pfas_calc['Waarde_ng'] * df_pfas_calc['RPF_calc']
        df_pfas_calc['Bioacc_Waarde'] = df_pfas_calc['PEQ_Waarde'] * df_pfas_calc['RBF_calc']

        # --- DEEL A: Originele Grafieken (Per Meetpunt) ---
        mp_opts = sorted(df_pfas_calc['Meetpunt'].unique())
        sel_mp_pfas = st.selectbox("Selecteer meetpunt", mp_opts)

        # Filter voor de staafgrafieken
        df_plot = df_pfas_calc[df_pfas_calc['Meetpunt'] == sel_mp_pfas].copy()

        col_bar1, col_bar2 = st.columns(2)

        with col_bar1:
            fig_rpf = px.bar(df_plot.sort_values('Datum'), x='Datum', y='PEQ_Waarde', color='Stof', title='Relatieve toxiciteit (RPF-PEQ)')
            fig_rpf.add_hline(y=4.4, line_dash="dash", line_color="red", annotation_text="Norm: 4.4")
            st.plotly_chart(fig_rpf, use_container_width=True)

        with col_bar2:
            fig_rbf = px.bar(df_plot.sort_values('Datum'), x='Datum', y='Bioacc_Waarde', color='Stof', title='Bioaccumulatie')
            fig_rbf.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Drempel: 0.3")
            st.plotly_chart(fig_rbf, use_container_width=True)

        # --- DEEL B: NIEUWE SPIDER CHART (Seizoenspatroon) ---
        st.markdown("---")
        st.subheader("🕸️ Seizoenspatroon: PFAS relatieve toxiciteit (RPF-PEQ)")
        st.info("Deze grafiek toont de gemiddelde totale PEQ-waarde per maand. Hierbij wordt enkel gekeken naar relatieve toxiciteit (RPF), niet naar bioaccumulatie.")

        # 1. Selectie voor Spider Chart
        # Gebruik sel_mp_pfas (van hierboven) als slimme default
        default_spider = [sel_mp_pfas] if sel_mp_pfas in mp_opts else mp_opts[:1]

        selected_meetpunten_spider = st.multiselect(
            "📍 Selecteer meetpunt(en) voor RPF-PEQ seizoensanalyse:",
            options=mp_opts,
            default=default_spider,
            key="pfas_spider_multiselect"
        )

        if selected_meetpunten_spider:
            # 2. Filteren (gebruik de reeds berekende df_pfas_calc)
            df_spider_pfas = df_pfas_calc[df_pfas_calc['Meetpunt'].isin(selected_meetpunten_spider)].copy()

            if not df_spider_pfas.empty:
                # 3. Aggregeren
                # Stap A: Sommeer PEQ per Datum + Meetpunt (Totaal PEQ van alle stoffen op één dag)
                # observed=True is belangrijk voor performance bij categorische data (Meetpunt)
                df_daily_sum = df_spider_pfas.groupby(['Datum', 'Meetpunt'], observed=True)['PEQ_Waarde'].sum().reset_index(name='Dag_Totaal_PEQ')

                # Stap B: Tijdvariabelen toevoegen (Vectorized)
                df_daily_sum['Jaar'] = df_daily_sum['Datum'].dt.year
                df_daily_sum['MaandNr'] = df_daily_sum['Datum'].dt.month

                # Maak label voor de grafiek
                df_daily_sum['Analyse_Groep'] = df_daily_sum['Meetpunt'].astype(str) + ' (' + df_daily_sum['Jaar'].astype(str) + ')'

                # Stap C: Gemiddelde per Maand berekenen
                df_monthly_avg = df_daily_sum.groupby(['Analyse_Groep', 'MaandNr'])['Dag_Totaal_PEQ'].mean().reset_index(name='Gemiddelde_PEQ')

                # 4. Gaten vullen (Zorg dat maand 1 t/m 12 bestaat voor elke lijn)
                unieke_groepen_pfas = df_monthly_avg['Analyse_Groep'].unique()

                if len(unieke_groepen_pfas) > 0:
                    full_index_pfas = pd.MultiIndex.from_product(
                        [unieke_groepen_pfas, range(1, 13)], 
                        names=['Analyse_Groep', 'MaandNr']
                    ).to_frame(index=False)

                    df_radar_pfas = pd.merge(full_index_pfas, df_monthly_avg, on=['Analyse_Groep', 'MaandNr'], how='left').fillna(0)

                    # 5. Maandnamen mappen
                    maand_namen = {
                        1: 'Jan', 2: 'Feb', 3: 'Mrt', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
                    }
                    df_radar_pfas['MaandNaam'] = df_radar_pfas['MaandNr'].map(maand_namen)

                    # 6. Plotten
                    fig_spider_pfas = px.line_polar(
                        df_radar_pfas, 
                        r='Gemiddelde_PEQ', 
                        theta='MaandNaam', 
                        color='Analyse_Groep', 
                        line_close=True,
                        markers=True,
                        title="Gemiddelde RPF-PEQ toxiciteit (ng/l) per maand",
                        category_orders={"MaandNaam": ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]}
                    )

                    fig_spider_pfas.update_traces(fill='toself', opacity=0.3)
                    fig_spider_pfas.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, df_radar_pfas['Gemiddelde_PEQ'].max() * 1.1])
                        ),
                        legend_title_text='Meetpunt (Jaar)'
                    )

                    st.plotly_chart(fig_spider_pfas, use_container_width=True)
                else:
                    st.info("Geen data over om te plotten na aggregatie.")
            else:
                st.info("Geen PFAS data gevonden voor de geselecteerde meetpunten.")
        else:
            st.info("Selecteer ten minste één meetpunt om de spider chart te genereren.")

        st.divider()

    else:
        st.warning("Geen PFAS matches gevonden in de huidige dataset (check spelling stoffen).")