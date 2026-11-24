"""Preparation of annual JG-MKN assessments."""

from __future__ import annotations

import pandas as pd


JG_RESULT_COLUMNS = (
    "Jaar",
    "Meetpunt",
    "Stof",
    "Waarde",
    "measurement_count",
    "JG_MKN",
    "Eenheid",
    "Overschrijdingsfactor",
    "Overschrijdt",
)


def prepare_jg_assessments(
    frame: pd.DataFrame,
    *,
    locations: tuple[str, ...],
    substances: tuple[str, ...],
    detections_only: bool,
) -> pd.DataFrame:
    """Bereken een jaargemiddelde per meetpunt en stof."""

    required = {
        "Datum",
        "Meetpunt",
        "Stof",
        "Waarde",
        "Limietsymbool",
        "JG_MKN",
        "Eenheid",
    }
    missing = sorted(required - set(frame.columns))

    if missing:
        raise KeyError(
            "Ontbrekende kolommen voor JG-MKN: "
            + ", ".join(missing)
        )

    data = frame.loc[
        frame["Meetpunt"].isin(locations)
        & frame["Stof"].isin(substances)
        & frame["JG_MKN"].notna()
    ].copy()

    if detections_only and not data.empty:
        below_reporting_limit = (
            data["Limietsymbool"]
            .astype("string")
            .str.contains(
                "<",
                regex=False,
                na=False,
            )
        )

        data = data.loc[
            ~below_reporting_limit
        ].copy()

    if data.empty:
        return pd.DataFrame(
            columns=JG_RESULT_COLUMNS
        )

    data["Datum"] = pd.to_datetime(
        data["Datum"],
        errors="coerce",
    )
    data["Waarde"] = pd.to_numeric(
        data["Waarde"],
        errors="coerce",
    )
    data["JG_MKN"] = pd.to_numeric(
        data["JG_MKN"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "Datum",
            "Waarde",
            "JG_MKN",
        ]
    )

    data["Jaar"] = data["Datum"].dt.year

    annual = (
        data.groupby(
            [
                "Jaar",
                "Meetpunt",
                "Stof",
            ],
            observed=True,
        )
        .agg(
            Waarde=("Waarde", "mean"),
            measurement_count=("Waarde", "size"),
        )
        .reset_index()
    )

    norms = (
        data.groupby(
            [
                "Stof",
                "Eenheid",
            ],
            observed=True,
        )
        .agg(
            JG_MKN=("JG_MKN", "max"),
        )
        .reset_index()
    )

    # Eén normcontext per stof is vereist voor deze pagina.
    conflicting_norms = (
        norms.groupby(
            "Stof",
            observed=True,
        )["JG_MKN"]
        .nunique()
    )

    conflicts = conflicting_norms.loc[
        conflicting_norms.gt(1)
    ]

    if not conflicts.empty:
        raise ValueError(
            "Meerdere JG-MKN-waarden gevonden voor: "
            + ", ".join(
                conflicts.index.astype(str)
            )
        )

    norms = (
        norms.sort_values(
            [
                "Stof",
                "Eenheid",
            ],
            kind="stable",
        )
        .drop_duplicates(
            subset=["Stof"],
            keep="first",
        )
    )

    result = annual.merge(
        norms,
        on="Stof",
        how="left",
        validate="many_to_one",
    )

    result["Overschrijdingsfactor"] = (
        result["Waarde"]
        / result["JG_MKN"]
    )

    result["Overschrijdt"] = (
        result["Waarde"]
        > result["JG_MKN"]
    )

    return (
        result.loc[
            :,
            list(JG_RESULT_COLUMNS),
        ]
        .sort_values(
            [
                "Stof",
                "Jaar",
                "Meetpunt",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )