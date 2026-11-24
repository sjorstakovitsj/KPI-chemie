"""Central orchestration for the KRW normcheck page."""

from __future__ import annotations

import pandas as pd

from chemistry_app.services.krw_jg_service import (
    prepare_jg_assessments,
)
from chemistry_app.services.krw_mac_service import (
    prepare_mac_assessments,
)
from chemistry_app.services.krw_normcheck_service import (
    JgAssessmentData,
    KrwNormcheckDashboard,
    MacAssessmentData,
    get_krw_normcheck_options,
    load_krw_measurements,
    select_krw_measurements,
)


def build_krw_normcheck_dashboard(
    *,
    years: tuple[int, ...],
    locations: tuple[str, ...],
    substance_groups: tuple[str, ...],
    substances: tuple[str, ...],
    detections_only: bool,
) -> KrwNormcheckDashboard:
    """Build all non-visual data for the KRW normcheck page."""
    measurements = load_krw_measurements(years)

    options = get_krw_normcheck_options(
        years=years,
        locations=locations,
        substance_groups=substance_groups,
    )

    selected = select_krw_measurements(
        measurements,
        locations=locations,
        substance_groups=substance_groups,
        substances=substances,
    )

    if selected.empty:
        return KrwNormcheckDashboard(
            measurements=selected,
            options=options,
            background_corrected_rows=0,
            jg=JgAssessmentData(
                annual_means=pd.DataFrame(),
                substances=(),
            ),
            mac=MacAssessmentData(
                exceedance_percentages=pd.DataFrame(),
                measurement_details=pd.DataFrame(),
            ),
        )

    jg_results = prepare_jg_assessments(
        selected,
        locations=locations,
        substances=substances,
        detections_only=detections_only,
    )

    mac_summary, mac_details = prepare_mac_assessments(
        selected,
        locations=locations,
        substances=substances,
        detections_only=detections_only,
    )

    corrected_rows = 0

    if (
        "Achtergrondcorrectie_Toegepast"
        in selected.columns
    ):
        corrected_rows = int(
            selected[
                "Achtergrondcorrectie_Toegepast"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    if "Stof" in jg_results.columns:
        jg_substances = tuple(
            sorted(
                jg_results["Stof"]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda values: values.ne("")]
                .unique()
                .tolist(),
                key=str.casefold,
            )
        )
    else:
        jg_substances = ()

    return KrwNormcheckDashboard(
        measurements=selected,
        options=options,
        background_corrected_rows=corrected_rows,
        jg=JgAssessmentData(
            annual_means=jg_results,
            substances=jg_substances,
        ),
        mac=MacAssessmentData(
            exceedance_percentages=mac_summary,
            measurement_details=mac_details,
        ),
    )
