"""Preparation of individual MAC-MKN assessments."""

from __future__ import annotations

import pandas as pd


MAC_SUMMARY_COLUMNS = (
    "Meetpunt",
    "Stof",
    "measurement_count",
    "exceedance_count",
    "Overschrijding (%)",
    "MAC_MKN",
    "Eenheid",
    "PlotLabel",
)


def prepare_mac_assessments(
    frame: pd.DataFrame,
    *,
    locations: tuple[str, ...],
    substances: tuple[str, ...],
    detections_only: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bereken individuele MAC-MKN-overschrijdingen.

    Returns:
        Een samenvatting per meetpunt en stof, plus de gebruikte
        individuele meetrecords.
    """
    required = {
        "Datum",
        "Meetpunt",
        "Stof",
        "Waarde",
        "Limietsymbool",
        "MAC_MKN",
        "Eenheid",
    }
    missing = sorted(
        required - set(frame.columns)
    )

    if missing:
        raise KeyError(
            "Ontbrekende kolommen voor MAC-MKN: "
            + ", ".join(missing)
        )

    data = frame.loc[
        frame["Meetpunt"].isin(locations)
        & frame["Stof"].isin(substances)
        & frame["MAC_MKN"].notna()
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
        return (
            pd.DataFrame(
                columns=MAC_SUMMARY_COLUMNS
            ),
            data,
        )

    data["Datum"] = pd.to_datetime(
        data["Datum"],
        errors="coerce",
    )
    data["Waarde"] = pd.to_numeric(
        data["Waarde"],
        errors="coerce",
    )
    data["MAC_MKN"] = pd.to_numeric(
        data["MAC_MKN"],
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "Datum",
            "Waarde",
            "MAC_MKN",
        ]
    )

    data["Overschrijding"] = (
        data["Waarde"]
        > data["MAC_MKN"]
    )

    grouped = (
        data.groupby(
            [
                "Meetpunt",
                "Stof",
            ],
            observed=True,
        )
        .agg(
            measurement_count=(
                "Overschrijding",
                "size",
            ),
            exceedance_count=(
                "Overschrijding",
                "sum",
            ),
            MAC_MKN=(
                "MAC_MKN",
                "max",
            ),
            Eenheid=(
                "Eenheid",
                "first",
            ),
        )
        .reset_index()
    )

    grouped["Overschrijding (%)"] = (
        grouped["exceedance_count"]
        / grouped["measurement_count"]
        * 100
    ).round(2)

    grouped["PlotLabel"] = (
        grouped["Meetpunt"].astype(str)
        + " ("
        + grouped["Stof"].astype(str)
        + ")"
    )

    summary = (
        grouped.loc[
            :,
            list(MAC_SUMMARY_COLUMNS),
        ]
        .sort_values(
            [
                "Overschrijding (%)",
                "Stof",
                "Meetpunt",
            ],
            ascending=[
                False,
                True,
                True,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    details = (
        data.sort_values(
            [
                "Stof",
                "Meetpunt",
                "Datum",
            ],
            ascending=[
                True,
                True,
                False,
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    return summary, details