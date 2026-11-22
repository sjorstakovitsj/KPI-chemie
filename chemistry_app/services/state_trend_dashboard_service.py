"""Central orchestration for the Toestand & Trend page."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chemistry_app.services.state_trend_analysis_service import (
    MannKendallResults,
    calculate_declining_trends,
    calculate_mann_kendall_results,
    calculate_rising_trends,
    prepare_trend_measurements,
    select_measurements,
    valid_trend_groups,
)
from chemistry_app.services.state_trend_filters import (
    SelectionOptions,
    build_selection_options,
    filter_substances,
)
from chemistry_app.services.state_trend_page_service import (
    StateTrendData,
    get_available_years,
    load_state_trend_data,
    use_seasonal_mann_kendall,
)


@dataclass(frozen=True)
class StateTrendDashboard:
    data: StateTrendData
    options: SelectionOptions
    selected_measurements: pd.DataFrame
    trend_measurements: pd.DataFrame
    mann_kendall: MannKendallResults
    rising_trends: pd.DataFrame
    declining_trends: pd.DataFrame


def get_state_trend_years() -> tuple[int, ...]:
    """Expose available years through the page-specific facade."""
    return get_available_years()


def get_state_trend_options(
    *,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    substance_filter: str,
    groups: tuple[str, ...] = (),
) -> SelectionOptions:
    """Load presentation-ready filter options for the current period."""
    data = load_state_trend_data(years=years, periods=periods)
    return build_selection_options(
        data.measurements,
        data.substance_lookup,
        substance_filter,
        groups,
    )


def _empty_mann_kendall(*, seasonal: bool) -> MannKendallResults:
    return MannKendallResults(rows=pd.DataFrame(), seasonal=seasonal)


def _filter_trend_substances(
    frame: pd.DataFrame,
    lookup: pd.DataFrame,
    filter_type: str,
) -> pd.DataFrame:
    allowed = set(filter_substances(lookup, filter_type))
    return frame.loc[frame["Stof"].isin(allowed)].copy()


def _filter_time_horizon(
    frame: pd.DataFrame,
    *,
    column: str,
    positive_only: bool,
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return frame.copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = values.le(20)
    if positive_only:
        mask &= values.gt(0)
    return frame.loc[mask.fillna(False)].copy()



def _filter_rising_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only statistically coherent upward-trend rows."""
    if frame.empty:
        return frame.copy()

    required = {"Trendscore", "n_metingen_boven_rg"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()

    slope = pd.to_numeric(frame["Trendscore"], errors="coerce")
    count = pd.to_numeric(frame["n_metingen_boven_rg"], errors="coerce")
    mask = slope.gt(0) & count.ge(7)
    return frame.loc[mask.fillna(False)].copy()


def _filter_declining_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only statistically coherent declining or stagnant rows."""
    if frame.empty:
        return frame.copy()

    required = {"Trendscore", "n_metingen_boven_rg"}
    if not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()

    slope = pd.to_numeric(frame["Trendscore"], errors="coerce")
    count = pd.to_numeric(frame["n_metingen_boven_rg"], errors="coerce")
    mask = slope.le(0) & count.ge(7)
    return frame.loc[mask.fillna(False)].copy()


def _validate_trend_semantics(
    rising: pd.DataFrame,
    declining: pd.DataFrame,
) -> None:
    """Fail fast if inconsistent trend rows survive the semantic filters."""
    if not rising.empty:
        slope = pd.to_numeric(rising["Trendscore"], errors="coerce")
        count = pd.to_numeric(
            rising["n_metingen_boven_rg"], errors="coerce"
        )
        if slope.le(0).any() or count.lt(7).any():
            raise ValueError(
                "Opwaartse trendresultaten bevatten een niet-positieve "
                "Trendscore of minder dan zeven bruikbare metingen."
            )

    if not declining.empty:
        slope = pd.to_numeric(declining["Trendscore"], errors="coerce")
        count = pd.to_numeric(
            declining["n_metingen_boven_rg"], errors="coerce"
        )
        if slope.gt(0).any() or count.lt(7).any():
            raise ValueError(
                "Dalende trendresultaten bevatten een positieve Trendscore "
                "of minder dan zeven bruikbare metingen."
            )

def _prepare_display_rows(
    frame: pd.DataFrame,
    *,
    time_column: str,
) -> pd.DataFrame:
    """Voeg sortering, rangnummers en presentatielabels toe."""
    if frame.empty:
        return frame.copy()

    result = frame.copy().reset_index(drop=True)

    sort_columns = [
        column
        for column in (
            time_column,
            "Trendscore",
        )
        if column in result.columns
    ]

    if sort_columns:
        ascending = [
            column == time_column
            for column in sort_columns
        ]

        result = (
            result.sort_values(
                sort_columns,
                ascending=ascending,
                kind="stable",
            )
            .reset_index(drop=True)
        )

    result.insert(
        0,
        "Nr.",
        range(1, len(result) + 1),
    )

    if "Trendscore" in result.columns:
        slopes = pd.to_numeric(
            result["Trendscore"],
            errors="coerce",
        )
    else:
        slopes = pd.Series(
            pd.NA,
            index=result.index,
            dtype="Float64",
        )

    if "Stof" in result.columns:
        substances = (
            result["Stof"]
            .astype("string")
            .fillna("")
        )
    else:
        substances = pd.Series(
            "",
            index=result.index,
            dtype="string",
        )

    if "Meetpunt" in result.columns:
        locations = (
            result["Meetpunt"]
            .astype("string")
            .fillna("")
        )
    else:
        locations = pd.Series(
            "",
            index=result.index,
            dtype="string",
        )

    result["Display_Label"] = [
        (
            f"{number}. {substance} @ {location} "
            f"(Helling: {slope:.4f})"
            if pd.notna(slope)
            else f"{number}. {substance} @ {location}"
        )
        for number, substance, location, slope in zip(
            result["Nr."],
            substances,
            locations,
            slopes,
        )
    ]

    return result


def build_state_trend_dashboard(
    *,
    years: tuple[int, ...],
    periods: tuple[str, ...],
    substance_filter: str,
    locations: tuple[str, ...],
    groups: tuple[str, ...],
    substances: tuple[str, ...],
    limit_value_option: str,
    trend_substance_filter: str | None = None,
) -> StateTrendDashboard:
    """Build all non-visual data required by Toestand & Trend."""
    data = load_state_trend_data(years=years, periods=periods)
    options = build_selection_options(
        data.measurements,
        data.substance_lookup,
        substance_filter,
        groups,
    )
    selected = select_measurements(data.measurements, locations, substances)
    seasonal = use_seasonal_mann_kendall(periods)

    if selected.empty:
        return StateTrendDashboard(
            data=data,
            options=options,
            selected_measurements=selected,
            trend_measurements=pd.DataFrame(),
            mann_kendall=_empty_mann_kendall(seasonal=seasonal),
            rising_trends=pd.DataFrame(),
            declining_trends=pd.DataFrame(),
        )

    mann_kendall = calculate_mann_kendall_results(selected, seasonal=seasonal)
    # Trendanalyse is pagina-breed en staat los van de specifieke selectie
    # voor meetreeks en Mann-Kendall. Dit behoudt het gedrag van de oude pagina.
    trend_frame = _filter_trend_substances(
        data.measurements,
        data.substance_lookup,
        trend_substance_filter or substance_filter,
    )
    prepared_trend_frame = prepare_trend_measurements(
    trend_frame,
    limit_value_option=limit_value_option,
)

    valid = valid_trend_groups(
    prepared_trend_frame,
    minimum_measurements=7,
)

    if valid.empty:
        rising = pd.DataFrame()
        declining = pd.DataFrame()
    else:
        rising = calculate_rising_trends(
            valid,
            limit_value_option,
            data.substance_lookup,
        )
        declining = calculate_declining_trends(
            valid,
            limit_value_option,
            data.substance_lookup,
        )

    rising = _filter_rising_semantics(rising)
    rising = _filter_time_horizon(
        rising,
        column="Tijd_tot_JG_normoverschrijding",
        positive_only=True,
    )

    declining = _filter_declining_semantics(declining)
    declining = _filter_time_horizon(
        declining,
        column="Tijd_tot_onder_norm",
        positive_only=False,
    )

    _validate_trend_semantics(rising, declining)

    rising = _prepare_display_rows(
        rising,
        time_column="Tijd_tot_JG_normoverschrijding",
    )
    declining = _prepare_display_rows(
        declining,
        time_column="Tijd_tot_onder_norm",
    )

    return StateTrendDashboard(
        data=data,
        options=options,
        selected_measurements=selected,
        trend_measurements=valid,
        mann_kendall=mann_kendall,
        rising_trends=rising,
        declining_trends=declining,
    )
