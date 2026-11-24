"""Central JG-MKN and MAC-MKN compliance assessment."""
from __future__ import annotations

import pandas as pd

from chemistry_app.domain.censored_values import (
    CensoredValueStrategy,
    apply_censored_value_strategy,
    normalize_missing_measurement_values,
)
from chemistry_app.domain.schemas import (
    COMPLIANCE_RESULT_COLUMNS,
    ComplianceStatus,
    StandardType,
)



def calculate_compliance(
    measurements: pd.DataFrame,
    *,
    strategy: CensoredValueStrategy | str = CensoredValueStrategy.EXCLUDE,
    location_column: str = "Meetpunt",
    location_code_column: str = "Locatiecode",
    date_column: str = "Datum",
    substance_column: str = "Stof",
    value_column: str = "Waarde",
    limit_symbol_column: str = "Limietsymbool",
    jg_column: str = "JG_MKN",
    mac_column: str = "MAC_MKN",
) -> pd.DataFrame:
    """Assess JG-MKN and MAC-MKN from one source of truth.

    JG-MKN uses the annual arithmetic mean. MAC-MKN uses the annual maximum
    individual analysis value. The location name is used when available;
    otherwise the location code is used. Rows without any usable location or
    substance are excluded. Missing/non-positive standards receive
    ``geen_norm``.
    """
    required = {
        location_column,
        date_column,
        substance_column,
        value_column,
        jg_column,
        mac_column,
    }
    missing = sorted(required - set(measurements.columns))
    if missing:
        raise KeyError(
            "Ontbrekende kolommen voor normtoetsing: " + ", ".join(missing)
        )

    if measurements.empty:
        return pd.DataFrame(columns=COMPLIANCE_RESULT_COLUMNS)

    calc = measurements.copy()

    text_columns = [location_column, substance_column]
    if location_code_column in calc.columns:
        text_columns.append(location_code_column)

    for column in text_columns:
        calc[column] = calc[column].replace(r"^\s*$", pd.NA, regex=True)

    # Canonical source code is the only location identity for compliance.
    if location_code_column in calc.columns:
        calc["__location"] = (
            calc[location_code_column]
            .astype("string")
            .str.strip()
            .str.casefold()
            .replace("", pd.NA)
        )
    else:
        calc["__location"] = calc[location_column]


    missing_context = calc[["__location", substance_column]].isna().any(axis=1)
    calc = calc.loc[~missing_context].copy()
    if calc.empty:
        return pd.DataFrame(columns=COMPLIANCE_RESULT_COLUMNS)

    calc[date_column] = pd.to_datetime(calc[date_column], errors="coerce")
    invalid_dates = calc[date_column].isna()
    if invalid_dates.any():
        raise ValueError(
            "Normtoetsing bevat "
            f"{int(invalid_dates.sum())} ongeldige of ontbrekende datums."
        )

    # Normalize known missing-value codes silently before assessment.
    calc[value_column] = normalize_missing_measurement_values(
        calc[value_column]
    )

    calc = apply_censored_value_strategy(
        calc,
        strategy,
        value_column=value_column,
        limit_symbol_column=limit_symbol_column,
        output_column="__analysis",
    )
    calc["__year"] = calc[date_column].dt.year.astype(int)

    keys = ["__location", "__year", substance_column]
    rows: list[dict[str, object]] = []
    assessments = (
        (StandardType.JG_MKN, jg_column, "mean"),
        (StandardType.MAC_MKN, mac_column, "max"),
    )

    for standard_type, standard_column, aggregation in assessments:
        standard_counts = (
            calc.groupby(keys, observed=True, dropna=False)[standard_column]
            .nunique(dropna=True)
        )
        conflicts = standard_counts.loc[standard_counts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                f"Meerdere {standard_type.value}-waarden gevonden binnen "
                "dezelfde locatie-, jaar- en stofcontext."
            )

        grouped_results = (
            calc.groupby(keys, observed=True, dropna=False)
            .agg(
                __assessment=("__analysis", aggregation),
                __standard=(standard_column, "first"),
                __count=("__analysis", "count"),
            )
            .reset_index()
        )

        for _, record in grouped_results.iterrows():
            location = record["__location"]
            year = record["__year"]
            substance = record[substance_column]
            value = record["__assessment"]
            standard = record["__standard"]
            count = int(record["__count"])

            value_float = None if pd.isna(value) else float(value)
            standard_float = None if pd.isna(standard) else float(standard)

            if value_float is None:
                status = ComplianceStatus.NO_VALUE
                factor = None
            elif standard_float is None or standard_float <= 0:
                status = ComplianceStatus.NO_STANDARD
                factor = None
            else:
                factor = value_float / standard_float
                status = (
                    ComplianceStatus.EXCEEDS
                    if value_float > standard_float
                    else ComplianceStatus.COMPLIANT
                )

            rows.append(
                {
                    "Meetpunt": location,
                    "Jaar": int(year),
                    "Stof": substance,
                    "Normtype": standard_type.value,
                    "Toetswaarde": value_float,
                    "Normwaarde": standard_float,
                    "Status": status.value,
                    "Overschrijdingsfactor": factor,
                    "Aantal_metingen": count,
                }
            )

    return pd.DataFrame(rows, columns=COMPLIANCE_RESULT_COLUMNS)


def compliance_failures(results: pd.DataFrame) -> pd.DataFrame:
    """Return only assessment rows that exceed the applicable standard."""
    if results.empty:
        return results.copy()
    return results.loc[
        results["Status"].eq(ComplianceStatus.EXCEEDS.value)
    ].copy()


def calculate_compliance_details(
    measurements: pd.DataFrame,
    *,
    strategy: CensoredValueStrategy | str = CensoredValueStrategy.EXCLUDE,
) -> pd.DataFrame:
    """Return the legacy-shaped failure table from central assessment results."""
    failures = compliance_failures(
        calculate_compliance(measurements, strategy=strategy)
    )
    if failures.empty:
        return pd.DataFrame(
            columns=["Meetpunt", "Jaar", "Stof", "Normtype", "Factor"]
        )
    return failures.rename(
        columns={"Overschrijdingsfactor": "Factor"}
    )[["Meetpunt", "Jaar", "Stof", "Normtype", "Factor"]]


def calculate_metrics(
    measurements: pd.DataFrame,
    is_period_average: bool = False,
    *,
    strategy: CensoredValueStrategy | str = CensoredValueStrategy.EXCLUDE,
    value_column: str = "Waarde",
    date_column: str = "Datum",
) -> tuple[float, float, float]:
    """Return valid measurement count, exceedance count and assessment percentage.

    The first metric remains an individual measurement count. Exceedances and
    their percentage remain based on assessable annual norm assessments, which
    preserves the existing compliance interpretation.
    """
    results = calculate_compliance(
        measurements,
        strategy=strategy,
        date_column=date_column,
        value_column=value_column,
    )
    assessable = results.loc[
        results["Status"].isin(
            [
                ComplianceStatus.COMPLIANT.value,
                ComplianceStatus.EXCEEDS.value,
            ]
        )
    ]

    if value_column not in measurements.columns:
        raise KeyError(
            f"Ontbrekende kolom voor telling van metingen: {value_column}"
        )

    numeric_values = normalize_missing_measurement_values(
        measurements[value_column]
    )
    valid_measurement_mask = numeric_values.notna()
    measurement_count = int(valid_measurement_mask.sum())

    violations = int(
        assessable["Status"].eq(ComplianceStatus.EXCEEDS.value).sum()
    )
    assessment_count = len(assessable)
    percentage = (
        violations / assessment_count * 100
        if assessment_count
        else 0.0
    )

    if is_period_average:
        if date_column not in measurements.columns:
            raise KeyError(
                f"Ontbrekende datumkolom voor periodegemiddelde: {date_column}"
            )
        valid_dates = pd.to_datetime(
            measurements.loc[valid_measurement_mask, date_column],
            errors="coerce",
        )
        number_of_years = max(int(valid_dates.dt.year.nunique()), 1)
        return (
            measurement_count / number_of_years,
            violations / number_of_years,
            percentage,
        )

    return float(measurement_count), float(violations), percentage
