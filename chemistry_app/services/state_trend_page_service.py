"""Page-data service for Toestand & Trend.

The shared page-data service preserves both the presentation field ``Meetpunt``
and the canonical source identifier ``Locatiecode``. This page deliberately uses
``Locatiecode`` as ``Meetpunt`` so every filter, table, analysis and chart shows
only canonical source location names.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chemistry_app.config.periods import PERIODES
from chemistry_app.services.page_data_service import (
    get_filter_options,
    query_data,
)

PAGE_COLUMNS = (
    "Datum",
    "Meetpunt",
    "Locatiecode",
    "Stof",
    "Stofgroep",
    "Waarde",
    "Eenheid",
    "Limietsymbool",
    "JG_MKN",
    "MAC_MKN",
)


@dataclass(frozen=True)
class StateTrendData:
    measurements: pd.DataFrame
    substance_lookup: pd.DataFrame
    available_years: tuple[int, ...]


def _normalize_location_code(series: pd.Series) -> pd.Series:
    """Normalize canonical source codes without deriving codes from names."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )


def _use_canonical_locations(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Use canonical ``Locatiecode`` for filtering and presentation.

    This function never converts a presentation name into a code. It requires
    the source-provided canonical code and copies that value to ``Meetpunt`` for
    this page's internal and visible location identity.
    """
    if "Locatiecode" not in measurements.columns:
        raise KeyError(
            "Toestand & Trend vereist de canonieke bronkolom "
            "'Locatiecode'."
        )

    result = measurements.copy()
    canonical_locations = _normalize_location_code(
        result["Locatiecode"]
    )

    missing_mask = canonical_locations.isna()
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        examples = (
            result.loc[missing_mask, "Meetpunt"]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .head(5)
            .tolist()
            if "Meetpunt" in result.columns
            else []
        )
        example_text = (
            f" Voorbeelden van presentatienamen: {examples}."
            if examples
            else ""
        )
        raise ValueError(
            "Toestand & Trend bevat "
            f"{missing_count:,} meetregels zonder canonieke Locatiecode."
            + example_text
        )

    result["Locatiecode"] = canonical_locations
    result["Meetpunt"] = canonical_locations
    return result


def get_available_years() -> tuple[int, ...]:
    """Return unique available measurement years in descending order."""
    return tuple(
        sorted(
            {
                int(year)
                for year in get_filter_options().get("jaren", ())
            },
            reverse=True,
        )
    )


def selected_months(
    periods: tuple[str, ...],
) -> frozenset[int]:
    """Return all calendar months represented by selected periods."""
    return frozenset(
        month
        for period in periods
        for month in PERIODES[period]
    )


def use_seasonal_mann_kendall(
    periods: tuple[str, ...],
) -> bool:
    """Use Seasonal Mann-Kendall when the selection is not the full year."""
    return bool(periods) and selected_months(periods) != frozenset(
        range(1, 13)
    )


def load_state_trend_data(
    *,
    years: tuple[int, ...],
    periods: tuple[str, ...],
) -> StateTrendData:
    """Load canonical location data and the substance/norm lookup."""
    query_arguments: dict[str, object] = {
        "kolommen": PAGE_COLUMNS,
    }
    if years:
        query_arguments["jaren"] = years
    if periods:
        query_arguments["periodes"] = periods

    measurements = query_data(**query_arguments)
    measurements = _use_canonical_locations(measurements)

    lookup = query_data(
        kolommen=(
            "Stof",
            "Stofgroep",
            "JG_MKN",
            "MAC_MKN",
        )
    ).drop_duplicates()

    return StateTrendData(
        measurements=measurements,
        substance_lookup=lookup,
        available_years=get_available_years(),
    )
