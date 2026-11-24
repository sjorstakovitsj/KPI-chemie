"""Shared KRW assessment policy for all dashboards and norm checks."""
from __future__ import annotations

import pandas as pd

METALS_GROUP = "metalen & elementen"
DISSOLVED_CONDITION_CODE = "nf"


def normalize_canonical_location(series: pd.Series) -> pd.Series:
    """Normalize source-provided canonical location codes."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
        .replace("", pd.NA)
    )


def prepare_krw_assessment_measurements(
    measurements: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the shared location and fraction contract before assessment.

    Contract:
    - ``Locatiecode`` is the only location identity.
    - ``Meetpunt`` becomes an exact compatibility alias of ``Locatiecode``.
    - For ``Metalen & elementen`` only condition code ``nf`` is retained.
    - Other substance groups remain unchanged.
    """
    required = {"Locatiecode", "Stofgroep", "Hoedanigheid"}
    missing = sorted(required - set(measurements.columns))
    if missing:
        raise KeyError(
            "KRW-beoordelingscontract mist vereiste kolommen: "
            + ", ".join(missing)
        )

    result = measurements.copy()
    canonical = normalize_canonical_location(result["Locatiecode"])
    if canonical.isna().any():
        raise ValueError(
            "KRW-beoordeling bevat "
            f"{int(canonical.isna().sum()):,} regels zonder "
            "canonieke Locatiecode."
        )

    groups = (
        result["Stofgroep"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    conditions = (
        result["Hoedanigheid"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    is_metal = groups.eq(METALS_GROUP)
    is_dissolved = conditions.eq(DISSOLVED_CONDITION_CODE)
    result = result.loc[(~is_metal | is_dissolved).fillna(False)].copy()

    result["Locatiecode"] = canonical.loc[result.index]
    result["Meetpunt"] = result["Locatiecode"]
    result["Hoedanigheid"] = conditions.loc[result.index]
    return result
