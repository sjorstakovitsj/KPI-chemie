"""Tijdelijke compatibiliteitslaag.

Nieuwe code importeert rechtstreeks uit ``chemistry_app``. Deze module wordt
verwijderd nadat alle Streamlit-pages zijn gemigreerd.
"""
from __future__ import annotations
import warnings
from chemistry_app.compatibility.legacy_utils import (
    STOFGROEPEN_MAPPING,
    UITGESLOTEN_ELEMENTEN,
    apply_achtergrondcorrectie,
    calculate_compliance,
    calculate_compliance_details,
    calculate_declining_exceedances_optimized,
    calculate_metrics,
    calculate_trends,
    calculate_trends_optimized,
    create_gauge,
    get_filter_options,
    get_shared_sidebar,
    get_shared_sidebar_filters,
    load_achtergrondcorrecties,
    load_data,
    load_pfas_ref,
    load_signaleringswaarden,
    match_stofgroep_optimized,
    prepare_heatmap_data,
    prepare_sunburst_data,
    query_data,
)
warnings.warn(
    "utils.py is deprecated; importeer rechtstreeks uit chemistry_app.",
    DeprecationWarning,
    stacklevel=2,
)
__all__ = [
    "STOFGROEPEN_MAPPING",
    "UITGESLOTEN_ELEMENTEN",
    "apply_achtergrondcorrectie",
    "calculate_compliance",
    "calculate_compliance_details",
    "calculate_declining_exceedances_optimized",
    "calculate_metrics",
    "calculate_trends",
    "calculate_trends_optimized",
    "create_gauge",
    "get_filter_options",
    "get_shared_sidebar",
    "get_shared_sidebar_filters",
    "load_achtergrondcorrecties",
    "load_data",
    "load_pfas_ref",
    "load_signaleringswaarden",
    "match_stofgroep_optimized",
    "prepare_heatmap_data",
    "prepare_sunburst_data",
    "query_data",
]
