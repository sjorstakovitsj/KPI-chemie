"""Single policy for missing and censored chemistry measurements."""
from __future__ import annotations

from enum import StrEnum

import pandas as pd

# Central source of truth. Values with an absolute magnitude at or above the
# threshold are source-system fill values in this dataset, never concentrations.
MISSING_VALUE_SENTINELS = frozenset(
    {
        -999,
        -9_999,
        -99_999,
        -999_999,
        -9_999_999,
        -9_999_999_999,
        9_999_999_999,
        999_999_999_999,
        1_000_000_000_000,
    }
)
MISSING_VALUE_ABSOLUTE_THRESHOLD = 1_000_000_000


class CensoredValueStrategy(StrEnum):
    EXCLUDE = "exclude"
    USE_REPORTED_VALUE = "reported_value"
    HALF_REPORTED_LIMIT = "half_limit"
    ZERO = "zero"


LEGACY_STRATEGY_LABELS = {
    "Sluit uit van berekening": CensoredValueStrategy.EXCLUDE,
    "Gebruik gerapporteerde waarde": CensoredValueStrategy.USE_REPORTED_VALUE,
    "Halve rapportagegrens": CensoredValueStrategy.HALF_REPORTED_LIMIT,
    "Nul": CensoredValueStrategy.ZERO,
}


def parse_censored_strategy(
    value: CensoredValueStrategy | str,
) -> CensoredValueStrategy:
    if isinstance(value, CensoredValueStrategy):
        return value
    if value in LEGACY_STRATEGY_LABELS:
        return LEGACY_STRATEGY_LABELS[value]
    try:
        return CensoredValueStrategy(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CensoredValueStrategy)
        raise ValueError(
            f"Onbekende detectiegrensstrategie: {value}. Geldig: {allowed}."
        ) from exc


def missing_measurement_value_mask(values: pd.Series) -> pd.Series:
    """Identify known sentinels and high-magnitude source fill values."""
    numeric = pd.to_numeric(values, errors="coerce").astype("Float64")
    return (
        numeric.isin(MISSING_VALUE_SENTINELS)
        | numeric.abs().ge(MISSING_VALUE_ABSOLUTE_THRESHOLD)
    ).fillna(False).astype(bool)


def normalize_missing_measurement_values(values: pd.Series) -> pd.Series:
    """Return nullable floats with all missing-value codes replaced by NA."""
    numeric = pd.to_numeric(values, errors="coerce").astype("Float64")
    return numeric.mask(missing_measurement_value_mask(numeric))


def censored_mask(limit_symbols: pd.Series) -> pd.Series:
    return (
        limit_symbols.astype("string")
        .str.strip()
        .str.startswith("<", na=False)
        .astype(bool)
    )


def apply_censored_value_strategy(
    frame: pd.DataFrame,
    strategy: CensoredValueStrategy | str = CensoredValueStrategy.EXCLUDE,
    *,
    value_column: str = "Waarde",
    limit_symbol_column: str = "Limietsymbool",
    output_column: str = "Waarde_Analyse",
    censored_column: str = "Is_Gecensureerd",
) -> pd.DataFrame:
    """Return a copy with one consistent analysis value and censor flag."""
    if value_column not in frame:
        raise KeyError(f"Ontbrekende meetwaardekolom: {value_column}")

    result = frame.copy()
    values = normalize_missing_measurement_values(result[value_column])
    mask = (
        censored_mask(result[limit_symbol_column])
        if limit_symbol_column in result
        else pd.Series(False, index=result.index, dtype=bool)
    )
    selected = parse_censored_strategy(strategy)
    analysis = values.copy()

    if selected is CensoredValueStrategy.EXCLUDE:
        analysis.loc[mask] = pd.NA
    elif selected is CensoredValueStrategy.HALF_REPORTED_LIMIT:
        analysis.loc[mask] = values.loc[mask] / 2
    elif selected is CensoredValueStrategy.ZERO:
        analysis.loc[mask] = 0.0

    result[output_column] = analysis
    result[censored_column] = mask
    result.attrs["censored_value_strategy"] = selected.value
    return result
