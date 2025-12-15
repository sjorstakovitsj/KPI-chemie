import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from utils import load_data, get_shared_sidebar

st.set_page_config(layout="wide", page_title="Herkomst stoffen")

st.header("Herkomst stoffen")

# Data en Sidebar
df_main = load_data()
df_filtered = get_shared_sidebar(df_main)

st.header("🏭 Herkomst en verdeling van stoffen")

# ---------------------------------------------------------
# DATA VOORBEREIDING
# ---------------------------------------------------------
st.info("Hieronder worden de gegevens getoond op basis van de **jaren geselecteerd in de zijbalk**.")

# We gebruiken df_filtered (al gefilterd op jaar in sidebar)
df_herkomst = df_filtered.dropna(subset=['Stof']).copy()

# Vectorized check voor detecties (sneller dan string contains in loop)
# We nemen aan dat load_data 'Limietsymbool' al heeft opgeschoond (geen NaN)
mask_detectie = ~df_herkomst['Limietsymbool'].astype(str).str.contains('<', na=False)
df_detecties = df_herkomst[mask_detectie].copy()

# ---------------------------------------------------------
# A. TOTAAL OVERZICHT
# ---------------------------------------------------------
st.subheader("Totaalbeeld verdeling (Geselecteerde periode)")
st.markdown("### 🟢 uitgevoerde metingen vs. 🟠 aangetroffen")

col_total_all, col_total_det = st.columns(2)

with col_total_all:
    st.markdown("**Totaal: alle metingen**")
    # Value_counts is zeer snel op categorical data
    dist_total = df_herkomst['Stofgroep'].value_counts().reset_index()
    dist_total.columns = ['Stofgroep', 'Aantal metingen']

    fig_pie_total = px.pie(
        dist_total, values='Aantal metingen', names='Stofgroep',
        title='Aantal metingen (incl. <RG)', hole=0.4
    )
    st.plotly_chart(fig_pie_total, use_container_width=True)

    st.markdown("*Top 25 meest gemeten (totaal)*")
    # Limit voor display performance
    top_stoffen = df_herkomst['Stof'].value_counts().head(25).reset_index()
    top_stoffen.columns = ['Stof', 'Aantal']
    st.dataframe(top_stoffen, use_container_width=True)

with col_total_det:
    st.markdown("**Filter: aangetroffen stoffen (boven rapportagegrens)**")
    if not df_detecties.empty:
        dist_det = df_detecties['Stofgroep'].value_counts().reset_index()
        dist_det.columns = ['Stofgroep', 'Aantal x aangetroffen']

        fig_pie_det = px.pie(
            dist_det, values='Aantal x aangetroffen', names='Stofgroep',
            title='Aantal keer aangetroffen (>RG)', hole=0.4
        )
        st.plotly_chart(fig_pie_det, use_container_width=True)

        st.markdown("*Top 25 meest aangetroffen (> RG)*")
        top_stoffen_det = df_detecties['Stof'].value_counts().head(25).reset_index()
        top_stoffen_det.columns = ['Stof', 'Aantal']
        st.dataframe(top_stoffen_det, use_container_width=True)
    else:
        st.warning("Geen waarden boven rapportagegrens gevonden in totale dataset.")

st.divider()

# ---------------------------------------------------------
# B. TRENDANALYSE STOFGROEPEN (ALLE JAREN)
# ---------------------------------------------------------
st.subheader("📈 Trendverdeling Stofgroepen (Alle jaren)")
st.markdown("De grafiek toont data uit de **volledige dataset**, ongeacht de jaar-filter in de zijbalk.")

col_trend_sel1, col_trend_sel2 = st.columns([1, 3])

with col_trend_sel1:
    # AANGEPAST: Standaard aangevinkt (value=True)
    alleen_detecties_trend = st.checkbox("Toon alleen aangetroffen stoffen (>RG)", value=True, key="tab6_trend_detectie")

with col_trend_sel2:
    # Gebruik categories voor snelle lookup
    all_meetpunten_trend = sorted(df_main['Meetpunt'].unique())
    selected_meetpunten_trend = st.multiselect(
        "📍 Selecteer meetpunt(en)",
        options=all_meetpunten_trend,
        default=all_meetpunten_trend,
        key="tab6_trend_meetpunt"
    )

if selected_meetpunten_trend:
    # Filter op df_main (alles)
    mask_mp = df_main['Meetpunt'].isin(selected_meetpunten_trend)

    if alleen_detecties_trend:
        # Combineer masks direct voor snelheid
        mask_det_main = ~df_main['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_trend = df_main[mask_mp & mask_det_main].copy()
    else:
        df_trend = df_main[mask_mp].copy()

    if not df_trend.empty:
        df_trend['Jaar'] = df_trend['Datum'].dt.year

        # Groupby op categories is heel snel
        df_grouped = df_trend.groupby(['Jaar', 'Stofgroep'], observed=True).size().reset_index(name='Aantal')

        # Bereken percentages vectorized (niet nodig voor pie, maar was in oude code)
        # totaal_per_jaar = df_grouped.groupby('Jaar')['Aantal'].transform('sum')
        # df_grouped['Percentage'] = (df_grouped['Aantal'] / totaal_per_jaar) * 100

        # Drop rows waar percentage 0 of NaN is (kan gebeuren door observed=True bij lege cats)
        df_grouped = df_grouped[df_grouped['Aantal'] > 0]

        # --- NIEUW: Vervaang de staafgrafiek met taartdiagrammen per jaar ---
        if not df_grouped.empty:
            st.markdown("##### Procentuele Verdeling per Jaar (Aangetroffen stoffen)")
            
            jaren = sorted(df_grouped['Jaar'].unique())
            num_jaren = len(jaren)
            
            # Gebruik maximaal 4 kolommen voor de layout
            num_cols = min(num_jaren, 4) 
            cols = st.columns(num_cols)
            
            # Plot de taartdiagrammen
            for i, jaar in enumerate(jaren):
                df_jaar = df_grouped[df_grouped['Jaar'] == jaar].copy()
                
                fig_pie = px.pie(
                    df_jaar, 
                    values='Aantal', 
                    names='Stofgroep',
                    title=f'Verdeling {jaar}', 
                    hole=0.3
                )
                
                # Toon percentages in de slices
                # Toon de legenda alleen bij het eerste diagram (i == 0)
                fig_pie.update_traces(
                    textposition='inside', 
                    textinfo='percent', 
                    showlegend=True if i == 0 else False
                )
                
                # Zorg ervoor dat Plotly Express de kleuren consistent houdt
                
                # Toon het diagram in de juiste kolom
                with cols[i % num_cols]:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Vang het geval op dat er meer dan 4 jaar zijn en de legenda niet duidelijk is
            if num_jaren > 4:
                 st.info("De kleuren zijn consistent over alle diagrammen; de legenda wordt alleen bij de eerste weergegeven.")

        else:
            st.info("Geen data na groepering.")
        # -------------------------------------------------------------------
    else:
        st.info("Geen data gevonden voor deze selectie.")
else:
    st.warning("Selecteer ten minste één meetpunt.")

st.divider()

# ---------------------------------------------------------
# C. PER MEETPUNT
# ---------------------------------------------------------
st.subheader("Overzicht per specifiek meetpunt")

meetpunten_list = sorted(df_herkomst['Meetpunt'].unique())
if meetpunten_list:
    selected_mp_herkomst = st.selectbox("Selecteer meetpunt:", meetpunten_list)

    # Filteren
    df_mp_all = df_herkomst[df_herkomst['Meetpunt'] == selected_mp_herkomst]
    df_mp_det = df_detecties[df_detecties['Meetpunt'] == selected_mp_herkomst]

    if not df_mp_all.empty:
        col_mp_1, col_mp_2 = st.columns(2)

        with col_mp_1:
            st.markdown(f"**{selected_mp_herkomst}: Alle metingen**")
            counts_all = df_mp_all['Stofgroep'].value_counts().reset_index()
            counts_all.columns = ['Stofgroep', 'Aantal']
            st.plotly_chart(px.pie(counts_all, values='Aantal', names='Stofgroep'), use_container_width=True)

        with col_mp_2:
            st.markdown(f"**{selected_mp_herkomst}: > Rapportagegrens**")
            if not df_mp_det.empty:
                counts_det = df_mp_det['Stofgroep'].value_counts().reset_index()
                counts_det.columns = ['Stofgroep', 'Aantal']
                st.plotly_chart(px.pie(counts_det, values='Aantal', names='Stofgroep'), use_container_width=True)
            else:
                st.info("Geen detecties op dit meetpunt.")

        st.divider()
        st.markdown(f"### 📋 Detailoverzicht per stofgroep: {selected_mp_herkomst}")

        # 1. Filteroptie: Alles of alleen detecties
        # AANGEPAST: Standaard aangevinkt (value=True)
        show_only_detected = st.checkbox(
            "Toon in tabel alleen waarden boven rapportagegrens (>RG)",
            value=True,
            key="tab6_table_checkbox"
        )

        # Bepaal welke dataset we gebruiken
        df_table_source = df_mp_det if show_only_detected else df_mp_all

        if not df_table_source.empty:
            # 2. Beschikbare stofgroepen bepalen (dynamisch op basis van filter)
            # We gebruiken dropna() en unique() om alleen relevante groepen te tonen
            beschikbare_groepen = sorted(list(df_table_source['Stofgroep'].dropna().unique()))

            if beschikbare_groepen:
                # 3. Dropdown menu
                selected_stofgroep_table = st.selectbox(
                    "Selecteer een stofgroep om de details te bekijken:",
                    options=beschikbare_groepen,
                    key="tab6_table_selectbox"
                )

                # 4. Data filteren op gekozen groep
                df_table_filtered = df_table_source[df_table_source['Stofgroep'] == selected_stofgroep_table].copy()

                # 5. Tabel samenstellen (Aggregatie per stof voor overzichtelijkheid)
                table_agg = df_table_filtered.groupby('Stof', observed=True).agg(
                    Aantal_Metingen=('Waarde', 'count'),
                    Gemiddelde=('Waarde', 'mean'),
                    Maximum=('Waarde', 'max'),
                    Laatste_Datum=('Datum', 'max'),
                    Eenheid=('Eenheid', 'first')
                ).reset_index().sort_values('Maximum', ascending=False)

                # Opmaak verfraaien
                table_agg['Laatste_Datum'] = table_agg['Laatste_Datum'].dt.strftime('%Y-%m-%d')
                table_agg['Gemiddelde'] = table_agg['Gemiddelde'].map('{:,.4f}'.format)
                table_agg['Maximum'] = table_agg['Maximum'].map('{:,.4f}'.format)

                # 6. Weergeven
                st.dataframe(table_agg, use_container_width=True)
            else:
                st.info("Geen stofgroepen gevonden in de huidige selectie.")
        else:
            st.warning("Geen metingen gevonden voor de huidige filterinstellingen.")

        # ---------------------------------------------------------
        # D. STOFFEN ZONDER GROEP (ONBEKEND)
        # ---------------------------------------------------------
        st.divider()
        st.subheader("⚠️ Stoffen zonder toegewezen stofgroep")
            
        # We kijken naar df_herkomst (dat is df_filtered binnen tab 6 context)
        # In de load_data functie krijgen niet-gematchte stoffen de label 'Onbekend'
        df_onbekend = df_herkomst[df_herkomst['Stofgroep'] == 'Onbekend'].copy()
            
        if not df_onbekend.empty:
            st.warning(f"Er zijn **{df_onbekend['Stof'].nunique()}** unieke stoffen aangetroffen die niet in de configuratielijst staan.")
            st.info("Onderstaande tabel toont stoffen die de categorie 'Onbekend' hebben gekregen.")
                
            # We tonen de unieke stoffen, hoe vaak ze voorkomen en de meest recente datum
            onbekend_summary = df_onbekend.groupby('Stof', observed=True).agg(
                Aantal_Metingen=('Waarde', 'count'),
                Laatste_Datum=('Datum', 'max'),
                Voorbeeld_Waarde=('Waarde', 'first'),
                Eenheid=('Eenheid', 'first')
                ).sort_values('Aantal_Metingen', ascending=False).reset_index()
                
            # Formatteer de datum voor netheid
            onbekend_summary['Laatste_Datum'] = onbekend_summary['Laatste_Datum'].dt.strftime('%Y-%m-%d')
                
            st.dataframe(onbekend_summary, use_container_width=True)
        else:
            st.success("✅ Alle aangetroffen stoffen zijn succesvol ingedeeld in een stofgroep.")
                
            if not df_table_source.empty:
                # observed=True zorgt dat we alleen relevante groepen zien
                beschikbare_groepen = df_table_source['Stofgroep'].unique()
                # Filter lege categorieen eruit
                beschikbare_groepen = [g for g in beschikbare_groepen if g in df_table_source['Stofgroep'].values]
                    
                if beschikbare_groepen:
                    selected_groep = st.selectbox("Kies stofgroep:", sorted(beschikbare_groepen))

                    df_detail = df_table_source[df_table_source['Stofgroep'] == selected_groep]

                    # Aggregatie
                    detail_agg = df_detail.groupby('Stof', observed=True).agg(
                        Aantal=('Waarde', 'count'),
                        Gemiddelde=('Waarde', 'mean'),
                        Max=('Waarde', 'max'),
                        Eenheid=('Eenheid', 'first')
                        ).sort_values('Aantal', ascending=False).reset_index()
                        
                    detail_agg['Gemiddelde'] = detail_agg['Gemiddelde'].map('{:,.4f}'.format)
                    detail_agg['Max'] = detail_agg['Max'].map('{:,.4f}'.format)

                    st.dataframe(detail_agg, use_container_width=True)
                else:
                    st.info("Geen stofgroepen beschikbaar in deze dataset.")
            else:
                st.warning("Geen data beschikbaar voor tabel.")