"""Application service for the chemistry home dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chemistry_app.domain.censored_values import (
    normalize_missing_measurement_values,
)
from chemistry_app.domain.compliance import calculate_compliance
from chemistry_app.domain.schemas import ComplianceStatus
from chemistry_app.domain.krw_assessment_policy import (
    prepare_krw_assessment_measurements,
)
from chemistry_app.services.page_data_service import get_filter_options, query_data
from chemistry_app.services.visualization_data_service import (
    HeatmapData,
    SunburstData,
    prepare_heatmap_from_compliance,
    prepare_sunburst_from_compliance,
)

HOME_COLUMNS = (
    "Datum",
    "Meetpunt",
    "Locatiecode",
    "Stof",
    "Parametercode",
    "Stofgroep",
    "Hoedanigheid",
    "Waarde",
    "Waarde_Origineel",
    "Eenheid",
    "Limietsymbool",
    "Latitude",
    "Longitude",
    "JG_MKN",
    "MAC_MKN",
    "Achtergrondconcentratie",
    "Achtergrondcorrectie_Toegepast",
)


@dataclass(frozen=True)
class HomeMetrics:
    measurement_count: float
    exceeding_assessment_count: float
    assessable_assessment_count: float
    exceedance_percentage: float


@dataclass(frozen=True)
class BackgroundCorrectionSummary:
    total_rows: int
    corrected_rows: int
    original_values_available: bool


@dataclass(frozen=True)
class ComparisonDefinition:
    label: str
    years: tuple[int, ...]
    average_per_year: bool


@dataclass(frozen=True)
class HomeDashboardData:
    measurements: pd.DataFrame
    current_measurements: pd.DataFrame
    comparison_measurements: pd.DataFrame
    current_metrics: HomeMetrics
    comparison_metrics: HomeMetrics
    background: BackgroundCorrectionSummary
    heatmap: HeatmapData
    sunburst: SunburstData
    recent_exceedances: pd.DataFrame




def get_home_years() -> tuple[int, ...]:
    """Return available Home years in descending display order."""
    options = get_filter_options()

    # page_data_service gebruikt momenteel de Nederlandse legacy-sleutel
    # 'jaren'. 'years' blijft ondersteund voor een toekomstig typed contract.
    years = options.get(
        "jaren",
        options.get("years", ()),
    )

    return tuple(
        sorted(
            {int(year) for year in years},
            reverse=True,
        )
    )

def build_comparison_options(
    available_years: tuple[int, ...],
    current_year: int,
) -> tuple[ComparisonDefinition, ...]:
    """Define valid Home comparison choices independently of Streamlit."""
    normalized_years = tuple(
        sorted(
            {int(year) for year in available_years},
            reverse=True,
        )
    )
    previous_years = tuple(
        year
        for year in normalized_years
        if year < current_year
    )

    options: list[ComparisonDefinition] = []

    previous_calendar_year = current_year - 1
    if previous_calendar_year in previous_years:
        options.append(
            ComparisonDefinition(
                label=f"Vergelijk met {previous_calendar_year}",
                years=(previous_calendar_year,),
                average_per_year=False,
            )
        )

    if previous_years:
        period_start = min(previous_years)
        period_end = max(previous_years)
        options.append(
            ComparisonDefinition(
                label=(
                    f"Gemiddelde voorgaande jaren "
                    f"({period_start} t/m {period_end})"
                ),
                years=previous_years,
                average_per_year=True,
            )
        )

    return tuple(options)

def calculate_home_metrics(
    measurements: pd.DataFrame,
    *,
    average_per_year: bool = False,
) -> HomeMetrics:
    """Calculate the four Home KPIs from one compliance assessment."""
    results = calculate_compliance(measurements)
    assessable = results.loc[
        results["Status"].isin(
            [
                ComplianceStatus.COMPLIANT.value,
                ComplianceStatus.EXCEEDS.value,
            ]
        )
    ]

    valid_values = normalize_missing_measurement_values(
        measurements["Waarde"]
    )
    measurement_count = float(valid_values.notna().sum())
    assessment_count = float(len(assessable))
    exceedance_count = float(
        assessable["Status"]
        .eq(ComplianceStatus.EXCEEDS.value)
        .sum()
    )
    percentage = (
        exceedance_count / assessment_count * 100
        if assessment_count
        else 0.0
    )

    if average_per_year:
        dates = pd.to_datetime(
            measurements.loc[valid_values.notna(), "Datum"],
            errors="coerce",
        )
        year_count = max(int(dates.dt.year.nunique()), 1)
        measurement_count /= year_count
        assessment_count /= year_count
        exceedance_count /= year_count

    return HomeMetrics(
        measurement_count=measurement_count,
        exceeding_assessment_count=exceedance_count,
        assessable_assessment_count=assessment_count,
        exceedance_percentage=percentage,
    )


def summarize_background_correction(
    measurements: pd.DataFrame,
) -> BackgroundCorrectionSummary:
    """Summarize the background-correction state for presentation."""
    corrected_rows = 0
    if "Achtergrondcorrectie_Toegepast" in measurements.columns:
        corrected_rows = int(
            measurements["Achtergrondcorrectie_Toegepast"]
            .fillna(False)
            .astype(bool)
            .sum()
        )

    return BackgroundCorrectionSummary(
        total_rows=len(measurements),
        corrected_rows=corrected_rows,
        original_values_available=(
            "Waarde_Origineel" in measurements.columns
        ),
    )


def select_period(
    measurements: pd.DataFrame,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """Return an isolated frame for the requested calendar years."""
    dates = pd.to_datetime(measurements["Datum"], errors="coerce")
    return measurements.loc[dates.dt.year.isin(years)].copy()



def prepare_recent_exceedances(
    results: pd.DataFrame,
    *,
    limit: int | None = 100,
) -> pd.DataFrame:
    """Prepare a stable, presentation-ready exceedance table.

    Both the central compliance shape (``Overschrijdingsfactor``) and the
    legacy detail shape (``Factor``) are supported during the migration.
    """
    legacy_columns = ("Meetpunt", "Jaar", "Stof", "Normtype", "Factor")
    compliance_columns = (
        "Meetpunt",
        "Jaar",
        "Stof",
        "Normtype",
        "Toetswaarde",
        "Normwaarde",
        "Overschrijdingsfactor",
    )

    if results.empty:
        columns = (
            legacy_columns
            if "Factor" in results.columns
            else compliance_columns
        )
        return pd.DataFrame(columns=list(columns))

    recent = results.copy()
    if "Status" in recent.columns:
        recent = recent.loc[
            recent["Status"].eq(ComplianceStatus.EXCEEDS.value)
        ].copy()

    factor_column = (
        "Overschrijdingsfactor"
        if "Overschrijdingsfactor" in recent.columns
        else "Factor"
        if "Factor" in recent.columns
        else None
    )
    if factor_column is None:
        raise KeyError(
            "Overschrijdingsresultaten missen 'Overschrijdingsfactor' "
            "of legacykolom 'Factor'."
        )

    recent[factor_column] = pd.to_numeric(
        recent[factor_column], errors="coerce"
    ).round(2)
    selected_columns = (
        legacy_columns
        if factor_column == "Factor"
        else compliance_columns
    )
    selected_columns = tuple(
        column for column in selected_columns if column in recent.columns
    )

    prepared = (
        recent.loc[:, list(selected_columns)]
        .sort_values(
            ["Jaar", factor_column, "Stof"],
            ascending=[False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if limit is not None:
        prepared = prepared.head(max(int(limit), 0)).reset_index(drop=True)
    return prepared


def display_locations(measurements: pd.DataFrame) -> tuple[str, ...]:
    """Return unique display locations in deterministic case-insensitive order."""
    if "Meetpunt" not in measurements.columns:
        return ()
    return tuple(
        sorted(
            measurements["Meetpunt"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .unique()
            .tolist(),
            key=str.casefold,
        )
    )

def build_home_dashboard(
    *,
    selected_years: tuple[int, ...],
    current_year: int,
    comparison: ComparisonDefinition,
) -> HomeDashboardData:
    """Build all non-visual data required by Home in one service call."""
    measurements = query_data(
        jaren=selected_years,
        kolommen=HOME_COLUMNS,
    )
    measurements = prepare_krw_assessment_measurements(measurements)
    current = select_period(measurements, (current_year,))
    comparison_frame = select_period(measurements, comparison.years)

    current_metrics = calculate_home_metrics(current)
    comparison_metrics = calculate_home_metrics(
        comparison_frame,
        average_per_year=comparison.average_per_year,
    )
    compliance_results = calculate_compliance(measurements)

    return HomeDashboardData(
        measurements=measurements,
        current_measurements=current,
        comparison_measurements=comparison_frame,
        current_metrics=current_metrics,
        comparison_metrics=comparison_metrics,
        background=summarize_background_correction(measurements),
        heatmap=prepare_heatmap_from_compliance(compliance_results),
        sunburst=prepare_sunburst_from_compliance(compliance_results),
        recent_exceedances=prepare_recent_exceedances(
            compliance_results
        ),
    )
