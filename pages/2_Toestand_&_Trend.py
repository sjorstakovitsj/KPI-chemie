"""Thin Streamlit page for chemical state and trend development."""
from __future__ import annotations

import streamlit as st

from chemistry_app.config.periods import PERIODE_VOLGORDE
from chemistry_app.services.state_trend_dashboard_service import (
    build_state_trend_dashboard,
    get_state_trend_options,
    get_state_trend_years,
)
from chemistry_app.ui.state_trend_charts import (
    create_measurements_figure,
    create_regression_figure,
)
from chemistry_app.ui.state_trend_renderers import (
    render_mann_kendall_results,
    render_trend_table,
)

st.set_page_config(layout="wide", page_title="Toestand & trendontwikkeling")
st.header("📈 Toestand en trendontwikkeling")

available_years = get_state_trend_years()
if not available_years:
    st.error("🚨 Kritieke fout: geen beschikbare jaren gevonden in Parquet.")
    st.stop()

st.sidebar.header("📅 Filter op jaren")
years = st.sidebar.multiselect("Selecteer gewenste jaren:", options=available_years, default=available_years)
periods = st.sidebar.multiselect(
    "Selecteer gewenste seizoenen of halfjaren:",
    options=list(PERIODE_VOLGORDE),
    default=list(PERIODE_VOLGORDE),
    key="shared_periodes_filter",
)
st.sidebar.markdown("---")
st.sidebar.info("Navigeer via het menu hierboven naar de verschillende analyses.")
if not years:
    st.warning("Selecteer minimaal één jaar.")
    st.stop()

# De specifieke meetreeks- en Mann-Kendallselectie staat los van het
# stoffilter voor de pagina-brede trendanalyses.
base_options = get_state_trend_options(
    years=tuple(years),
    periods=tuple(periods),
    substance_filter="Alle stoffen",
)

st.markdown("---")
st.subheader("Specifieke selectie toestand & trend")
col1, col2, col3 = st.columns(3)
with col1:
    current_locations = st.session_state.get("tab1_meetpunten_select", [])
    location_default = [x for x in current_locations if x in base_options.locations]
    if not location_default and base_options.locations:
        location_default = [base_options.locations[0]]
    locations = st.multiselect(
        "Selecteer meetpunt(en)", base_options.locations, default=location_default, key="tab1_meetpunten_select"
    )
with col2:
    current_groups = st.session_state.get("tab1_stofgroep_select", [])
    group_default = [x for x in current_groups if x in base_options.substance_groups]
    if not group_default and base_options.substance_groups:
        group_default = [base_options.substance_groups[0]]
    groups = st.multiselect(
        "Stofgroep", base_options.substance_groups, default=group_default, key="tab1_stofgroep_select"
    )
refined_options = get_state_trend_options(
    years=tuple(years),
    periods=tuple(periods),
    substance_filter="Alle stoffen",
    groups=tuple(groups),
)
with col3:
    current_substances = st.session_state.get("tab1_stoffen_select", [])
    substance_default = [x for x in current_substances if x in refined_options.substances]
    if not substance_default and refined_options.substances:
        substance_default = [refined_options.substances[0]]
    substances = st.multiselect(
        "Selecteer Stof(fen)", refined_options.substances, default=substance_default, key="tab1_stoffen_select"
    )

st.write("### 🔍 Selectie filters voor trendanalyse")
filter_col, limit_col = st.columns(2)
with filter_col:
    trend_filter = st.radio(
        "Welk type stoffen wil je analyseren?",
        options=("Alle stoffen", "KRW-stoffen", "Niet-genormeerde stoffen"),
        index=1,
        horizontal=True,
        key="trend_stof_filter",
    )
with limit_col:
    limit_option = st.radio(
        "Hoe om te gaan met '< Onder rapportagegrens'-waarden?",
        options=("Gebruik gemeten waarde", "Sluit uit van berekening"),
        index=1,
        horizontal=True,
        key="lt_waarde_optie",
    )

if not locations or not substances:
    st.info("Selecteer minimaal één meetpunt en één stof.")
    st.stop()

dashboard = build_state_trend_dashboard(
    years=tuple(years),
    periods=tuple(periods),
    substance_filter="Alle stoffen",
    locations=tuple(locations),
    groups=tuple(groups),
    substances=tuple(substances),
    limit_value_option=limit_option,
    trend_substance_filter=trend_filter,
)

st.plotly_chart(create_measurements_figure(dashboard.selected_measurements), width="stretch")
method = "Seasonal Mann-Kendall" if dashboard.mann_kendall.seasonal else "Mann-Kendall"
render_mann_kendall_results(dashboard.mann_kendall.rows, title=f"Mann-Kendall-resultaten: {method}")

st.header("⚠️ Opwaartse trends")
rising_columns = ("Nr.", "Meetpunt", "Stof", "n_metingen_boven_rg", "Trendscore", "Tijd_tot_JG_normoverschrijding", "Gemiddelde_Waarde", "RSD")
render_trend_table(dashboard.rising_trends, title="Urgente opwaartse trends", columns=rising_columns)
if not dashboard.rising_trends.empty:
    rising_label = st.selectbox("Selecteer opwaartse trend:", dashboard.rising_trends["Display_Label"], key="select_opwaarts")
    rising_row = dashboard.rising_trends.loc[dashboard.rising_trends["Display_Label"].eq(rising_label)].iloc[0]
    rising_data = dashboard.trend_measurements.loc[
        dashboard.trend_measurements["Meetpunt"].eq(rising_row["Meetpunt"])
        & dashboard.trend_measurements["Stof"].eq(rising_row["Stof"])
    ]
    st.plotly_chart(
        create_regression_figure(rising_data, title=rising_label, standard=rising_row.get("JG_MKN")), width="stretch"
    )

st.header("📉 Normoverschrijdingen met dalende of stagnante trend")
declining_columns = ("Nr.", "Meetpunt", "Stof", "n_metingen_boven_rg", "Trendscore", "JG_MKN", "Gemiddelde_Waarde", "RSD", "Tijd_tot_onder_norm")
render_trend_table(dashboard.declining_trends, title="Dalende of stagnante trends", columns=declining_columns)
if not dashboard.declining_trends.empty:
    declining_label = st.selectbox("Selecteer record:", dashboard.declining_trends["Display_Label"], key="select_dalend")
    declining_row = dashboard.declining_trends.loc[dashboard.declining_trends["Display_Label"].eq(declining_label)].iloc[0]
    declining_data = dashboard.trend_measurements.loc[
        dashboard.trend_measurements["Meetpunt"].eq(declining_row["Meetpunt"])
        & dashboard.trend_measurements["Stof"].eq(declining_row["Stof"])
    ]
    st.plotly_chart(
        create_regression_figure(declining_data, title=declining_label, standard=declining_row.get("JG_MKN")), width="stretch"
    )
