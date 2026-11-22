"""Streamlit adapters for shared sidebar filters and UI error translation."""
from __future__ import annotations
from dataclasses import dataclass
import streamlit as st
from chemistry_app.config.periods import PERIODE_VOLGORDE
from chemistry_app.infrastructure.exceptions import ChemistryAppError
from chemistry_app.infrastructure.measurements_repository import MeasurementsRepository
from chemistry_app.infrastructure.models import MeasurementFilters
from chemistry_app.services.filter_service import FilterOptions,get_filter_options
@dataclass(frozen=True, slots=True)
class SidebarSelection:
    filters: MeasurementFilters
    spatial_aggregation: str
@st.cache_data(show_spinner=False,max_entries=16)
def _cached_options(repository: MeasurementsRepository,artifact_version: str) -> FilterOptions:
    del artifact_version
    return get_filter_options(repository)
def render_shared_sidebar(repository: MeasurementsRepository,*,show_aggregation: bool=True) -> SidebarSelection | None:
    try: options=_cached_options(repository,repository.version)
    except ChemistryAppError as exc:
        st.error(f"Filteropties kunnen niet worden geladen: {exc}"); return None
    st.sidebar.header("ðŸ“… Periodefilters")
    years=st.sidebar.multiselect("Selecteer gewenste jaren:",options=sorted(options.years,reverse=True),default=sorted(options.years,reverse=True),key="shared_jaren_filter")
    periods=st.sidebar.multiselect("Selecteer gewenste seizoenen of halfjaren:",options=list(PERIODE_VOLGORDE),default=list(PERIODE_VOLGORDE),key="shared_periodes_filter")
    compartments=st.sidebar.multiselect("Compartiment:",options=list(options.compartments),default=list(options.compartments),key="shared_compartiment_filter")
    conditions=st.sidebar.multiselect("Hoedanigheid:",options=list(options.conditions),default=list(options.conditions),key="shared_hoedanigheid_filter")
    qualities=st.sidebar.multiselect("Kwaliteitsoordeel:",options=list(options.quality_codes),default=list(options.quality_codes),key="shared_kwaliteit_filter")
    aggregation="Gemiddelde"
    if show_aggregation:
        st.sidebar.markdown("---"); aggregation=st.sidebar.radio("Bereken ruimtelijke waarden als:",options=["Gemiddelde","Mediaan"],key="ruimtelijke_aggregatiemethode")
    return SidebarSelection(MeasurementFilters(years=tuple(years),periods=tuple(periods),compartments=tuple(compartments),conditions=tuple(conditions),quality_codes=tuple(qualities)).normalized(),aggregation)

