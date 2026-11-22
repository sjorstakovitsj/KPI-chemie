"""Prepare chart data from central domain/service results only."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.services.compliance_service import compliance_details
@dataclass(frozen=True, slots=True)
class HeatmapData:
    factors: pd.DataFrame
    labels: pd.DataFrame
    substances: tuple[str,...]
    locations: tuple[str,...]
@dataclass(frozen=True, slots=True)
class SunburstData:
    rows: pd.DataFrame

def prepare_heatmap_data(measurements: pd.DataFrame) -> HeatmapData:
    failures=compliance_details(measurements)
    if failures.empty: return HeatmapData(pd.DataFrame(),pd.DataFrame(),(),())
    grouped=failures.groupby(["Stof","Meetpunt"],observed=True)
    factors=grouped["Factor"].max().unstack(fill_value=0.0)
    def label(group: pd.DataFrame) -> str:
        parts=[]
        for normtype,normgroup in group.groupby("Normtype",observed=True): parts.append(f"{normtype} ({', '.join(map(str,sorted(normgroup['Jaar'].astype(int).unique())))})")
        return "<br>".join(parts)
    labels=grouped.apply(label,include_groups=False).unstack(fill_value=" ")
    substances=tuple(sorted(factors.index.astype(str))); locations=tuple(sorted(factors.columns.astype(str)))
    return HeatmapData(factors.reindex(index=substances,columns=locations,fill_value=0.0),labels.reindex(index=substances,columns=locations,fill_value=" "),substances,locations)

def prepare_sunburst_data(
    measurements: pd.DataFrame,
) -> SunburstData:
    """Prepare one Sunburst row per exceeding substance.

    The category shows whether the substance exceeds JG-MKN, MAC-MKN,
    or both. The displayed factor is the highest exceedance factor for
    that substance across the selected measurements and years.
    """
    failures = compliance_details(measurements)

    output_columns = (
        "Stof",
        "Categorie",
        "Factor",
    )

    if failures.empty:
        return SunburstData(
            rows=pd.DataFrame(
                columns=output_columns,
            )
        )

    required_columns = {
        "Stof",
        "Normtype",
        "Factor",
    }
    missing_columns = sorted(
        required_columns - set(failures.columns)
    )

    if missing_columns:
        raise KeyError(
            "Ontbrekende kolommen voor Sunburst: "
            + ", ".join(missing_columns)
        )

    rows: list[dict[str, object]] = []

    for substance, group in failures.groupby(
        "Stof",
        observed=True,
        dropna=True,
    ):
        normtypes = set(
            group["Normtype"]
            .dropna()
            .astype(str)
        )

        if {
            "JG-MKN",
            "MAC-MKN",
        }.issubset(normtypes):
            category = "JG + MAC"
        elif "JG-MKN" in normtypes:
            category = "JG (Gemiddelde)"
        elif "MAC-MKN" in normtypes:
            category = "MAC (Piek)"
        else:
            # Onbekende normtypen horen niet in deze visualisatie.
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
        prepared = (
            prepared.sort_values(
                [
                    "Factor",
                    "Stof",
                ],
                ascending=[
                    False,
                    True,
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return SunburstData(rows=prepared)
