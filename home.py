"""Thin Streamlit UI layer for the chemistry Home dashboard."""
from __future__ import annotations

import streamlit as st

from chemistry_app.services.home_service import (
    build_comparison_options,
    build_home_dashboard,
    get_home_years,
)
from chemistry_app.ui.charts import (
    create_compliance_sunburst,
    create_exceedance_heatmap,
)
from chemistry_app.ui.home_metrics import (
    render_background_summary,
    render_home_metrics,
)


st.set_page_config(
    layout="wide",
    page_title="Waterkwaliteit KPI Dashboard",
    page_icon="💧",
)
st.title("💧 Dashboard chemische waterkwaliteit MN")

with st.expander("Laatste wijzigingen", expanded=False):
    st.markdown(
        """
- Centrale normtoetsing voor JG-MKN en MAC-MKN.
- Hiaatwaarden worden centraal uitgesloten.
- Achtergrondcorrectie en oorspronkelijke concentratie zijn traceerbaar.
- Home gebruikt services voor data, KPI's en visualisatievoorbereiding.
"""
    )

available_years = get_home_years()
if not available_years:
    st.error("Geen meetjaren beschikbaar in de meet-Parquet.")
    st.stop()

st.sidebar.header("📅 Filter op jaren")
selected_years = st.sidebar.multiselect(
    "Selecteer gewenste jaren:",
    options=available_years,
    default=available_years,
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Navigeer via het menu hierboven naar de verschillende analyses."
)

if not selected_years:
    st.warning("Selecteer minimaal één jaar om het dashboard op te bouwen.")
    st.stop()

current_year = max(selected_years)
comparison_options = build_comparison_options(
    tuple(available_years),
    current_year,
)

st.subheader(
    "Vergelijking huidig jaar ten opzichte van voorgaande jaren: "
    f"**{current_year}**"
)

if not comparison_options:
    st.info(
        "Niet genoeg historische data om de huidige selectie te vergelijken."
    )
    st.stop()

selected_label = st.selectbox(
    "Vergelijkingsperiode",
    [option.label for option in comparison_options],
)
comparison = next(
    option
    for option in comparison_options
    if option.label == selected_label
)

dashboard = build_home_dashboard(
    selected_years=tuple(selected_years),
    current_year=current_year,
    comparison=comparison,
)

render_home_metrics(
    dashboard.current_metrics,
    dashboard.comparison_metrics,
    comparison_label=comparison.label,
    average_label=comparison.average_per_year,
)

st.divider()
st.header("🔴 KRW-check: overschrijdende stoffen")
render_background_summary(dashboard.background)

if dashboard.heatmap.factors.empty:
    st.success("Geen normoverschrijdingen gevonden! 🎉")
else:
    st.subheader("🧩 Heatmap overschrijdingsfactoren")
    st.plotly_chart(
        create_exceedance_heatmap(dashboard.heatmap),
        width="stretch",
    )
    st.caption("Grijze cellen = geen overschrijding.")

if not dashboard.sunburst.rows.empty:
    st.subheader("☀️ Verdeling overschrijdingen")
    st.plotly_chart(
        create_compliance_sunburst(
            dashboard.sunburst,
            title="Overschrijdingen per stof",
        ),
        width="stretch",
    )

st.divider()
st.subheader("⚠️ Recente normoverschrijdingen")
if dashboard.recent_exceedances.empty:
    st.success("Geen recente normoverschrijdingen in de geselecteerde jaren.")
else:
    st.dataframe(
        dashboard.recent_exceedances,
        width="stretch",
        hide_index=True,
    )
