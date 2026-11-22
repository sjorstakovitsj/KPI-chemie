"""Analyse-orchestratie voor de pagina Toestand & Trend."""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.compatibility.legacy_utils import (
    calculate_declining_exceedances_optimized,
    calculate_trends_optimized,
)
from chemistry_app.domain.mann_kendall import (
    bereken_mann_kendall,
)


@dataclass(frozen=True)
class MannKendallResults:
    rows: pd.DataFrame
    seasonal: bool


def select_measurements(
    frame: pd.DataFrame,
    locations: tuple[str, ...],
    substances: tuple[str, ...],
) -> pd.DataFrame:
    """Selecteer meetrecords voor de gekozen locaties en stoffen."""
    return frame.loc[
        frame["Meetpunt"].isin(locations)
        & frame["Stof"].isin(substances)
    ].copy()


def calculate_mann_kendall_results(
    frame: pd.DataFrame,
    *,
    seasonal: bool,
) -> MannKendallResults:
    """Bereken Mann-Kendall per combinatie van meetpunt en stof."""
    rows: list[dict[str, object]] = []

    for (
        location,
        substance,
    ), group in frame.groupby(
        [
            "Meetpunt",
            "Stof",
        ],
        observed=True,
    ):
        rows.append(
            {
                "Meetpunt": location,
                "Stof": substance,
                **bereken_mann_kendall(
                    group,
                    seasonal=seasonal,
                ),
            }
        )

    return MannKendallResults(
        rows=pd.DataFrame(rows),
        seasonal=seasonal,
    )


def valid_trend_groups(
    frame: pd.DataFrame,
    minimum_measurements: int = 7,
) -> pd.DataFrame:
    """Behoud alleen meetpunt-stofgroepen met voldoende metingen."""
    counts = (
        frame.groupby(
            [
                "Meetpunt",
                "Stof",
            ],
            observed=True,
        )
        .size()
        .rename("n_obs")
        .reset_index()
    )

    valid = counts.loc[
        counts["n_obs"].ge(minimum_measurements),
        [
            "Meetpunt",
            "Stof",
        ],
    ]

    return frame.merge(
        valid,
        on=[
            "Meetpunt",
            "Stof",
        ],
        how="inner",
    )

def prepare_trend_measurements(
    frame: pd.DataFrame,
    *,
    limit_value_option: str,
) -> pd.DataFrame:
    """Maak één consistente dataset voor trendberekening en grafiek."""
    result = frame.copy()

    result["Waarde"] = pd.to_numeric(
        result["Waarde"],
        errors="coerce",
    )

    result["Datum"] = pd.to_datetime(
        result["Datum"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "Meetpunt",
            "Stof",
            "Datum",
            "Waarde",
        ]
    )

    if limit_value_option == "Sluit uit van berekening":
        below_reporting_limit = (
            result["Limietsymbool"]
            .astype("string")
            .str.contains(
                "<",
                regex=False,
                na=False,
            )
        )

        result = result.loc[
            ~below_reporting_limit
        ].copy()

    return result


def calculate_rising_trends(
    frame: pd.DataFrame,
    lt_option: str,
    norm_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Delegeer de bestaande opwaartse trendanalyse aan de compatibiliteitslaag."""
    return calculate_trends_optimized(
        frame=frame,
        lt_optie=lt_option,
        norm_lookup_df=norm_lookup,
    )


def calculate_declining_trends(
    frame: pd.DataFrame,
    lt_option: str,
    norm_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Delegeer de bestaande dalende trendanalyse aan de compatibiliteitslaag."""
    return calculate_declining_exceedances_optimized(
        frame=frame,
        lt_optie=lt_option,
        norm_lookup_df=norm_lookup,
    )