from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
@dataclass(frozen=True)
class SelectionOptions:
 locations: tuple[str,...]
 substance_groups: tuple[str,...]
 substances: tuple[str,...]
def sorted_text(series): return tuple(sorted(series.dropna().astype(str).str.strip().loc[lambda x:x.ne("")].unique(),key=str.casefold))
def filter_substances(
    lookup: pd.DataFrame,
    filter_type: str,
) -> tuple[str, ...]:
    """Classificeer stoffen op basis van alle beschikbare normrecords.

    Een stof is een KRW-stof zodra minimaal één record een geldige
    JG-MKN of MAC-MKN bevat. Alleen wanneer alle records voor de stof
    beide normen missen, is de stof niet-genormeerd.
    """
    required_columns = {
        "Stof",
        "JG_MKN",
        "MAC_MKN",
    }
    missing_columns = sorted(
        required_columns - set(lookup.columns)
    )

    if missing_columns:
        raise KeyError(
            "Ontbrekende kolommen voor stoffiltering: "
            + ", ".join(missing_columns)
        )

    normalized = lookup.loc[
        lookup["Stof"].notna(),
        [
            "Stof",
            "JG_MKN",
            "MAC_MKN",
        ],
    ].copy()

    normalized["Stof"] = (
        normalized["Stof"]
        .astype("string")
        .str.strip()
    )

    normalized = normalized.loc[
        normalized["Stof"].ne("")
    ]

    normalized["JG_MKN"] = pd.to_numeric(
        normalized["JG_MKN"],
        errors="coerce",
    )
    normalized["MAC_MKN"] = pd.to_numeric(
        normalized["MAC_MKN"],
        errors="coerce",
    )

    normalized["Heeft_norm"] = (
        normalized["JG_MKN"].notna()
        | normalized["MAC_MKN"].notna()
    )

    substance_status = (
        normalized.groupby(
            "Stof",
            observed=True,
            sort=False,
        )["Heeft_norm"]
        .any()
    )

    if filter_type == "KRW-stoffen":
        substances = substance_status.loc[
            substance_status
        ].index

    elif filter_type == "Niet-genormeerde stoffen":
        substances = substance_status.loc[
            ~substance_status
        ].index

    elif filter_type == "Alle stoffen":
        substances = substance_status.index

    else:
        raise ValueError(
            "Onbekend stoffilter: "
            f"{filter_type!r}. "
            "Geldig zijn: 'Alle stoffen', "
            "'KRW-stoffen' en "
            "'Niet-genormeerde stoffen'."
        )

    return tuple(
        sorted(
            substances.astype(str).tolist(),
            key=str.casefold,
        )
    )
def build_selection_options(measurements,lookup,kind,groups=()):
 substances=set(filter_substances(lookup,kind))
 if groups: substances &= set(measurements.loc[measurements["Stofgroep"].isin(groups),"Stof"].dropna().astype(str))
 return SelectionOptions(sorted_text(measurements["Meetpunt"]),sorted_text(measurements["Stofgroep"]),tuple(sorted(substances,key=str.casefold)))
