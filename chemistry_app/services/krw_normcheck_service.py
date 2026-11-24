"""Application-service contracts and filter options for KRW normcheck."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chemistry_app.services.page_data_service import (
    get_filter_options,
    query_data,
)

KRW_COLUMNS = (
    "Datum",
    "Meetpunt",
    "Locatiecode",
    "Stof",
    "Stofgroep",
    "Waarde",
    "Waarde_Origineel",
    "Eenheid",
    "Limietsymbool",
    "JG_MKN",
    "MAC_MKN",
    "Achtergrondconcentratie",
    "Achtergrondcorrectie_Toegepast",
)


@dataclass(frozen=True)
class KrwNormcheckOptions:
    years: tuple[int, ...]
    locations: tuple[str, ...]
    substance_groups: tuple[str, ...]
    substances: tuple[str, ...]


@dataclass(frozen=True)
class JgAssessmentData:
    annual_means: pd.DataFrame
    substances: tuple[str, ...]


@dataclass(frozen=True)
class MacAssessmentData:
    exceedance_percentages: pd.DataFrame
    measurement_details: pd.DataFrame


@dataclass(frozen=True)
class KrwNormcheckDashboard:
    measurements: pd.DataFrame
    options: KrwNormcheckOptions
    background_corrected_rows: int
    jg: JgAssessmentData
    mac: MacAssessmentData


def _normalize_location_code(series: pd.Series) -> pd.Series:
    """Normalize canonical location codes for filtering and display."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )


def _normalize_filter_values(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return ()
    normalized = _normalize_location_code(
        pd.Series(values, dtype="string")
    )
    return tuple(normalized.dropna().astype(str).tolist())


def _sorted_text(series: pd.Series) -> tuple[str, ...]:
    return tuple(
        sorted(
            series.dropna()
            .astype("string")
            .str.strip()
            .loc[lambda values: values.ne("")]
            .astype(str)
            .unique()
            .tolist(),
            key=str.casefold,
        )
    )


def load_krw_measurements(
    years: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Load KRW data using canonical ``Locatiecode`` as ``Meetpunt``."""
    query_arguments: dict[str, object] = {"kolommen": KRW_COLUMNS}
    if years:
        query_arguments["jaren"] = years

    result = query_data(**query_arguments).copy()
    if "Locatiecode" not in result.columns:
        raise KeyError(
            "De KRW-normcheck vereist de canonieke bronkolom "
            "'Locatiecode'."
        )

    canonical_locations = _normalize_location_code(
        result["Locatiecode"]
    )
    missing_count = int(canonical_locations.isna().sum())
    if missing_count:
        raise ValueError(
            "De KRW-normcheck bevat "
            f"{missing_count:,} meetregels zonder canonieke Locatiecode."
        )

    result["Locatiecode"] = canonical_locations
    result["Meetpunt"] = canonical_locations

    for column in ("Stof", "Stofgroep"):
        if column in result.columns:
            result[column] = (
                result[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )
    return result


def _normed_substances(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty:
        return ()
    required = {"Stof", "JG_MKN", "MAC_MKN"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(
            "Ontbrekende kolommen voor normclassificatie: "
            + ", ".join(missing)
        )

    normalized = frame.loc[
        frame["Stof"].notna(),
        ["Stof", "JG_MKN", "MAC_MKN"],
    ].copy()
    normalized["Stof"] = normalized["Stof"].astype("string").str.strip()
    normalized = normalized.loc[normalized["Stof"].ne("")]
    normalized["has_standard"] = (
        pd.to_numeric(normalized["JG_MKN"], errors="coerce").notna()
        | pd.to_numeric(normalized["MAC_MKN"], errors="coerce").notna()
    )
    status = normalized.groupby(
        "Stof", observed=True, sort=False
    )["has_standard"].any()
    return tuple(
        sorted(
            status.loc[status].index.astype(str).tolist(),
            key=str.casefold,
        )
    )


def get_krw_normcheck_options(
    *,
    years: tuple[int, ...] = (),
    locations: tuple[str, ...] = (),
    substance_groups: tuple[str, ...] = (),
) -> KrwNormcheckOptions:
    frame = load_krw_measurements(years)
    normalized_locations = _normalize_filter_values(locations)

    group_frame = frame
    if normalized_locations:
        group_frame = group_frame.loc[
            group_frame["Meetpunt"].isin(normalized_locations)
        ].copy()

    substance_frame = group_frame
    if substance_groups:
        substance_frame = substance_frame.loc[
            substance_frame["Stofgroep"].isin(substance_groups)
        ].copy()

    available_years = tuple(
        sorted(
            {
                int(year)
                for year in get_filter_options().get("jaren", ())
            },
            reverse=True,
        )
    )
    return KrwNormcheckOptions(
        years=available_years,
        locations=_sorted_text(frame["Meetpunt"]),
        substance_groups=_sorted_text(group_frame["Stofgroep"]),
        substances=_normed_substances(substance_frame),
    )


def select_krw_measurements(
    frame: pd.DataFrame,
    *,
    locations: tuple[str, ...],
    substance_groups: tuple[str, ...],
    substances: tuple[str, ...],
) -> pd.DataFrame:
    result = frame.copy()
    normalized_locations = _normalize_filter_values(locations)
    if normalized_locations:
        result = result.loc[
            result["Meetpunt"].isin(normalized_locations)
        ].copy()
    if substance_groups:
        result = result.loc[
            result["Stofgroep"].isin(substance_groups)
        ].copy()
    if substances:
        result = result.loc[result["Stof"].isin(substances)].copy()
    return result
