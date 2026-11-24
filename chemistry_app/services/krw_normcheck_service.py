"""Canonical-only data boundary and filter options for KRW normcheck."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from chemistry_app.domain.krw_assessment_policy import (
    prepare_krw_assessment_measurements,
)
from chemistry_app.services.page_data_service import get_filter_options, query_data

KRW_COLUMNS = (
    "Datum",
    "Locatiecode",
    "Stof",
    "Parametercode",
    "Stofgroep",
    "Hoedanigheid",
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


def _normalize_code(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )


def _normalize_location_values(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(
        _normalize_code(pd.Series(values, dtype="string"))
        .dropna()
        .astype(str)
        .tolist()
    )


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


def load_krw_measurements(years: tuple[int, ...] = ()) -> pd.DataFrame:
    """Load only canonical locations; presentation names never enter KRW data."""
    arguments: dict[str, object] = {"kolommen": KRW_COLUMNS}
    if years:
        arguments["jaren"] = years
    result = query_data(**arguments).copy()

    if "Locatiecode" not in result.columns:
        raise KeyError("KRW normcheck vereist 'Locatiecode'.")
    canonical = _normalize_code(result["Locatiecode"])
    if canonical.isna().any():
        raise ValueError(
            f"KRW normcheck bevat {int(canonical.isna().sum()):,} "
            "regels zonder canonieke Locatiecode."
        )

    result["Locatiecode"] = canonical
    # Compatibility alias for existing JG/MAC services. Values are always the
    # canonical code, never a presentation name from page_data_service.
    result["Meetpunt"] = canonical

    for column in ("Stof", "Parametercode", "Stofgroep", "Hoedanigheid"):
        result[column] = (
            result[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
    return prepare_krw_assessment_measurements(result)


def _normed_substances(frame: pd.DataFrame) -> tuple[str, ...]:
    if frame.empty:
        return ()
    normalized = frame.loc[
        frame["Stof"].notna(), ["Stof", "JG_MKN", "MAC_MKN"]
    ].copy()
    normalized["has_standard"] = (
        pd.to_numeric(normalized["JG_MKN"], errors="coerce").notna()
        | pd.to_numeric(normalized["MAC_MKN"], errors="coerce").notna()
    )
    status = normalized.groupby("Stof", observed=True)["has_standard"].any()
    return tuple(sorted(status.loc[status].index.astype(str), key=str.casefold))


def get_krw_normcheck_options(
    *,
    years: tuple[int, ...] = (),
    locations: tuple[str, ...] = (),
    substance_groups: tuple[str, ...] = (),
) -> KrwNormcheckOptions:
    frame = load_krw_measurements(years)
    normalized_locations = _normalize_location_values(locations)
    group_frame = frame
    if normalized_locations:
        group_frame = group_frame.loc[
            group_frame["Locatiecode"].isin(normalized_locations)
        ].copy()
    substance_frame = group_frame
    if substance_groups:
        substance_frame = substance_frame.loc[
            substance_frame["Stofgroep"].isin(substance_groups)
        ].copy()

    years_available = tuple(
        sorted(
            {int(year) for year in get_filter_options().get("jaren", ())},
            reverse=True,
        )
    )
    return KrwNormcheckOptions(
        years=years_available,
        locations=_sorted_text(frame["Locatiecode"]),
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
    normalized_locations = _normalize_location_values(locations)
    if normalized_locations:
        result = result.loc[
            result["Locatiecode"].isin(normalized_locations)
        ].copy()
    if substance_groups:
        result = result.loc[
            result["Stofgroep"].isin(substance_groups)
        ].copy()
    if substances:
        result = result.loc[result["Stof"].isin(substances)].copy()
    # Maintain compatibility only after canonical filtering.
    result["Meetpunt"] = result["Locatiecode"]
    return result
