"""Streamlit rendering for the four Home metrics."""
from __future__ import annotations

import streamlit as st

from chemistry_app.services.home_service import (
    BackgroundCorrectionSummary,
    HomeMetrics,
)


def render_home_metrics(
    current: HomeMetrics,
    comparison: HomeMetrics,
    *,
    comparison_label: str,
    average_label: bool,
) -> None:
    """Render the metric cards and their explanation without calculating data."""
    suffix = " gemiddeld" if average_label else ""

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        f"Aantal metingen{suffix}",
        f"{current.measurement_count:.0f}",
        f"{current.measurement_count - comparison.measurement_count:.0f}",
    )
    col2.metric(
        f"Overschrijdende normbeoordelingen{suffix}",
        f"{current.exceeding_assessment_count:.0f}",
        (
            f"{current.exceeding_assessment_count - comparison.exceeding_assessment_count:.0f}"
        ),
        delta_color="inverse",
    )
    col3.metric(
        f"Toetsbare normbeoordelingen{suffix}",
        f"{current.assessable_assessment_count:.0f}",
        (
            f"{current.assessable_assessment_count - comparison.assessable_assessment_count:.0f}"
        ),
    )
    col4.metric(
        f"Percentage overschrijdende normbeoordelingen{suffix}",
        f"{current.exceedance_percentage:.1f}%",
        (
            f"{current.exceedance_percentage - comparison.exceedance_percentage:.1f}%"
        ),
        delta_color="inverse",
    )

    _render_metric_explanation(
        current,
        comparison_label=comparison_label,
        average_label=average_label,
    )


def _render_metric_explanation(
    metrics: HomeMetrics,
    *,
    comparison_label: str,
    average_label: bool,
) -> None:
    """Explain aggregation levels and the comparison delta."""
    comparison_note = (
        "Bij de meerjarige vergelijkingsperiode worden aantallen als gemiddeld "
        "aantal per beschikbaar jaar weergegeven."
        if average_label
        else "De vergelijkingswaarden hebben betrekking op één kalenderjaar."
    )

    with st.expander(
        "ℹ️ Toelichting op de vier metrics",
        expanded=False,
    ):
        st.markdown(
            f"""
- **Aantal metingen**: individuele numerieke meetwaarden, exclusief ontbrekende waarden en bekende hiaatcodes.
- **Overschrijdende normbeoordelingen**: jaarlijkse JG-MKN- en MAC-MKN-beoordelingen met status `overschrijdt`. Dit is niet het aantal losse meetwaarden boven een norm.
- **Toetsbare normbeoordelingen**: beoordelingen met status `voldoet` of `overschrijdt`. Beoordelingen zonder bruikbare meetwaarde of geldige norm vallen erbuiten.
- **Percentage overschrijdende normbeoordelingen**: `{metrics.exceeding_assessment_count:.0f} ÷ {metrics.assessable_assessment_count:.0f} × 100 = {metrics.exceedance_percentage:.1f}%`.

De kleine waarde onder iedere metric is het verschil met **{comparison_label}**. {comparison_note}
"""
        )

def render_background_summary(
    summary: BackgroundCorrectionSummary,
) -> None:
    """Render an explicit background-correction status message."""
    if summary.corrected_rows > 0:
        original_note = (
            " De oorspronkelijke concentratie blijft beschikbaar als "
            "'Waarde_Origineel'."
            if summary.original_values_available
            else ""
        )
        st.info(
            "Achtergrondcorrectie is bij het samenstellen van de "
            "dashboarddata toegepast op "
            f"{summary.corrected_rows:,} van de "
            f"{summary.total_rows:,} meetregels in de huidige selectie."
            f"{original_note}"
        )
        return

    st.warning(
        "In de huidige selectie zijn geen meetregels als "
        "achtergrondgecorrigeerd gemarkeerd."
    )

