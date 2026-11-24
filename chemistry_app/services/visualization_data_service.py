"""Prepare chart data from shared compliance results."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from chemistry_app.domain.compliance import (
    compliance_failures,
)
from chemistry_app.services.compliance_service import (
    compliance_details,
)


@dataclass(frozen=True, slots=True)
class HeatmapData:
    factors: pd.DataFrame
    labels: pd.DataFrame
    substances: tuple[str, ...]
    locations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SunburstData:
    rows: pd.DataFrame


def _failures_from_compliance(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Convert central compliance failures to the chart-data contract."""
    failures = compliance_failures(results)
    columns = (
        "Meetpunt",
        "Jaar",
        "Stof",
        "Normtype",
        "Factor",
    )
    if failures.empty:
        return pd.DataFrame(columns=columns)

    return (
        failures.rename(
            columns={
                "Overschrijdingsfactor": "Factor",
            }
        )
        .loc[:, list(columns)]
        .copy()
    )


def _prepare_heatmap_from_failures(
    failures: pd.DataFrame,
) -> HeatmapData:
    if failures.empty:
        return HeatmapData(
            factors=pd.DataFrame(),
            labels=pd.DataFrame(),
            substances=(),
            locations=(),
        )

    grouped = failures.groupby(
        ["Stof", "Meetpunt"],
        observed=True,
    )
    factors = grouped["Factor"].max().unstack(
        fill_value=0.0
    )

    def label(group: pd.DataFrame) -> str:
        parts: list[str] = []
        for normtype, normgroup in group.groupby(
            "Normtype",
            observed=True,
        ):
            years = ", ".join(
                map(
                    str,
                    sorted(
                        normgroup["Jaar"]
                        .astype(int)
                        .unique()
                    ),
                )
            )
            parts.append(f"{normtype} ({years})")
        return "<br>".join(parts)

    labels = grouped.apply(
        label,
        include_groups=False,
    ).unstack(fill_value=" ")

    substances = tuple(
        sorted(factors.index.astype(str))
    )
    locations = tuple(
        sorted(factors.columns.astype(str))
    )

    return HeatmapData(
        factors=factors.reindex(
            index=substances,
            columns=locations,
            fill_value=0.0,
        ),
        labels=labels.reindex(
            index=substances,
            columns=locations,
            fill_value=" ",
        ),
        substances=substances,
        locations=locations,
    )


def _prepare_sunburst_from_failures(
    failures: pd.DataFrame,
) -> SunburstData:
    output_columns = (
        "Stof",
        "Categorie",
        "Factor",
    )
    if failures.empty:
        return SunburstData(
            rows=pd.DataFrame(
                columns=output_columns
            )
        )

    rows: list[dict[str, object]] = []
    for substance, group in failures.groupby(
        "Stof",
        observed=True,
        dropna=True,
    ):
        normtypes = set(
            group["Normtype"].dropna().astype(str)
        )
        if {"JG-MKN", "MAC-MKN"}.issubset(
            normtypes
        ):
            category = "JG + MAC"
        elif "JG-MKN" in normtypes:
            category = "JG (Gemiddelde)"
        elif "MAC-MKN" in normtypes:
            category = "MAC (Piek)"
        else:
            continue

        factors = pd.to_numeric(
            group["Factor"],
            errors="coerce",
        ).dropna()
        if factors.empty:
            continue

        rows.append(
            {
                "Stof": substance,
                "Categorie": category,
                "Factor": float(factors.max()),
            }
        )

    prepared = pd.DataFrame(
        rows,
        columns=output_columns,
    )
    if not prepared.empty:
        prepared = prepared.sort_values(
            ["Factor", "Stof"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

    return SunburstData(rows=prepared)


def prepare_heatmap_from_compliance(
    results: pd.DataFrame,
) -> HeatmapData:
    """Prepare Heatmap data from one central compliance result."""
    return _prepare_heatmap_from_failures(
        _failures_from_compliance(results)
    )


def prepare_sunburst_from_compliance(
    results: pd.DataFrame,
) -> SunburstData:
    """Prepare Sunburst data from one central compliance result."""
    return _prepare_sunburst_from_failures(
        _failures_from_compliance(results)
    )


# Compatibility wrappers for pages not yet migrated to a precomputed result.
def prepare_heatmap_data(
    measurements: pd.DataFrame,
) -> HeatmapData:
    return _prepare_heatmap_from_failures(
        compliance_details(measurements)
    )


def prepare_sunburst_data(
    measurements: pd.DataFrame,
) -> SunburstData:
    return _prepare_sunburst_from_failures(
        compliance_details(measurements)
    )
