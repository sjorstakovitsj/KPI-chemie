"""Thin Streamlit page for the KRW normcheck."""
from __future__ import annotations

import streamlit as st

from chemistry_app.services.krw_normcheck_dashboard_service import (
    build_krw_normcheck_dashboard,
)
from chemistry_app.services.krw_normcheck_service import (
    get_krw_normcheck_options,
)
from chemistry_app.ui.krw_normcheck_renderers import (
    render_jg_results,
    render_mac_results,
)


def _initialize_multiselect(
    key: str,
    valid_options: tuple[str, ...],
    *,
    fallback_count: int,
) -> None:
    """Keep session-state selections valid after cascading filters change."""
    current = st.session_state.get(key, [])
    valid = [value for value in current if value in valid_options]
    if not valid and valid_options:
        valid = list(valid_options[:fallback_count])
    st.session_state[key] = valid


st.set_page_config(
    layout="wide",
    page_title="KRW normcheck",
)
st.header("KRW normcheck")
st.subheader("✅ KRW normcheck per meetpunt")

all_options = get_krw_normcheck_options()
years = st.sidebar.multiselect(
    "Selecteer gewenste jaren:",
    options=all_options.years,
    default=all_options.years,
)
if not years:
    st.warning("Selecteer minimaal één jaar.")
    st.stop()

year_options = get_krw_normcheck_options(
    years=tuple(years),
)

_initialize_multiselect(
    "tab2_meetpunten_select",
    year_options.locations,
    fallback_count=3,
)
locations = st.multiselect(
    "📍 Selecteer meetpunt(en)",
    options=year_options.locations,
    key="tab2_meetpunten_select",
)

location_options = get_krw_normcheck_options(
    years=tuple(years),
    locations=tuple(locations),
)
_initialize_multiselect(
    "tab2_stofgroep_select",
    location_options.substance_groups,
    fallback_count=1,
)
groups = st.multiselect(
    "Selecteer stofgroep",
    options=location_options.substance_groups,
    key="tab2_stofgroep_select",
)

substance_options = get_krw_normcheck_options(
    years=tuple(years),
    locations=tuple(locations),
    substance_groups=tuple(groups),
)
_initialize_multiselect(
    "tab2_stoffen_select",
    substance_options.substances,
    fallback_count=3,
)
substances = st.multiselect(
    "🔎 Selecteer stof(fen)",
    options=substance_options.substances,
    key="tab2_stoffen_select",
)

detections_only = st.checkbox(
    "Alleen aangetroffen waarden meenemen (>RG) in normcheck",
    value=True,
)

if not locations or not substances:
    st.warning("Selecteer minstens één stof en één meetpunt.")
    st.stop()

dashboard = build_krw_normcheck_dashboard(
    years=tuple(years),
    locations=tuple(locations),
    substance_groups=tuple(groups),
    substances=tuple(substances),
    detections_only=detections_only,
)

st.info(
    "Achtergrondcorrectie is toegepast op "
    f"{dashboard.background_corrected_rows:,} meetregels "
    "binnen de huidige selectie."
)

st.subheader("Jaargemiddelde toetsing (JG-MKN)")
st.caption(
    "Jaargemiddelden per stof en meetpunt, afgezet tegen de JG-MKN."
)
render_jg_results(dashboard.jg.annual_means)

st.markdown("---")
st.subheader("Maximale aanvaardbare concentratie (MAC-MKN)")
st.caption(
    "Aandeel individuele metingen dat de pieknorm overschrijdt."
)
render_mac_results(
    dashboard.mac.exceedance_percentages,
    dashboard.mac.measurement_details,
)
