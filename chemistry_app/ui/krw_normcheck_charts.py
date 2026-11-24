"""Plotly figures for the KRW normcheck page."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_jg_figure(
    data: pd.DataFrame,
    *,
    substance: str,
) -> go.Figure:
    """Create a JG-MKN annual-mean chart for one substance."""
    required_columns = {
        "Jaar",
        "Meetpunt",
        "Stof",
        "Waarde",
        "JG_MKN",
        "Eenheid",
    }
    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Ontbrekende kolommen voor JG-figuur: "
            + ", ".join(missing_columns)
        )

    rows = data.loc[
        data["Stof"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq(substance.strip().casefold())
    ].copy()

    if rows.empty:
        return go.Figure()

    rows["Jaar"] = pd.to_numeric(
        rows["Jaar"],
        errors="coerce",
    )
    rows["Waarde"] = pd.to_numeric(
        rows["Waarde"],
        errors="coerce",
    )
    rows["JG_MKN"] = pd.to_numeric(
        rows["JG_MKN"],
        errors="coerce",
    )

    rows = rows.dropna(
        subset=[
            "Jaar",
            "Waarde",
            "JG_MKN",
        ]
    )

    if rows.empty:
        return go.Figure()

    rows["JaarStr"] = (
        rows["Jaar"]
        .astype("int64")
        .astype(str)
    )

    norm_values = rows["JG_MKN"].dropna().unique()
    if len(norm_values) != 1:
        raise ValueError(
            f"Voor {substance!r} zijn {len(norm_values)} "
            "verschillende JG-MKN-waarden gevonden."
        )

    units = (
        rows["Eenheid"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .unique()
    )
    if len(units) != 1:
        raise ValueError(
            f"Voor {substance!r} zijn {len(units)} "
            "verschillende eenheden gevonden."
        )

    norm = float(norm_values[0])
    unit = str(units[0])

    hover_data: dict[str, object] = {
        "JaarStr": True,
        "Meetpunt": True,
        "Waarde": ":.6g",
        "JG_MKN": ":.6g",
    }

    if "Overschrijdingsfactor" in rows.columns:
        hover_data["Overschrijdingsfactor"] = ":.3f"

    if "measurement_count" in rows.columns:
        hover_data["measurement_count"] = True

    figure = px.bar(
        rows,
        x="JaarStr",
        y="Waarde",
        color="Meetpunt",
        barmode="group",
        title=(
            f"Toetsing {substance} "
            f"(JG-MKN: {norm:g} {unit})"
        ),
        labels={
            "JaarStr": "Jaar",
            "Waarde": (
                f"Jaargemiddelde concentratie ({unit})"
            ),
            "Meetpunt": "Meetpunt",
        },
        hover_data=hover_data,
    )

    figure.add_hline(
        y=norm,
        line_dash="dash",
        line_color="red",
        annotation_text=(
            f"JG-MKN ({norm:g} {unit})"
        ),
        annotation_position="top right",
    )

    figure.update_xaxes(
        type="category",
        title="Jaar",
    )
    figure.update_yaxes(
        title=(
            f"Jaargemiddelde concentratie ({unit})"
        ),
    )
    figure.update_layout(
        legend_title_text="Meetpunt",
        hovermode="closest",
    )

    return figure


def create_mac_figure(
    data: pd.DataFrame,
) -> go.Figure:
    """Create a MAC-MKN chart from individual-measurement results."""
    required_columns = {
        "Meetpunt",
        "Stof",
        "measurement_count",
        "exceedance_count",
        "Overschrijding (%)",
        "PlotLabel",
    }
    missing_columns = sorted(
        required_columns - set(data.columns)
    )

    if missing_columns:
        raise KeyError(
            "Ontbrekende kolommen voor MAC-figuur: "
            + ", ".join(missing_columns)
        )

    if data.empty:
        return go.Figure()

    rows = data.copy()

    rows["measurement_count"] = pd.to_numeric(
        rows["measurement_count"],
        errors="coerce",
    )
    rows["exceedance_count"] = pd.to_numeric(
        rows["exceedance_count"],
        errors="coerce",
    )
    rows["Overschrijding (%)"] = pd.to_numeric(
        rows["Overschrijding (%)"],
        errors="coerce",
    )

    rows = rows.dropna(
        subset=[
            "measurement_count",
            "exceedance_count",
            "Overschrijding (%)",
        ]
    )

    if rows.empty:
        return go.Figure()

    rows = rows.sort_values(
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

    figure = px.bar(
        rows,
        x="PlotLabel",
        y="Overschrijding (%)",
        color="Meetpunt",
        title=(
            "Percentage individuele metingen "
            "boven de MAC-MKN"
        ),
        labels={
            "PlotLabel": "Meetpunt en stof",
            "Overschrijding (%)": (
                "Individuele overschrijdingen (%)"
            ),
            "Meetpunt": "Meetpunt",
        },
        hover_data={
            "Stof": True,
            "Meetpunt": True,
            "measurement_count": True,
            "exceedance_count": True,
            "Overschrijding (%)": ":.2f",
        },
    )

    figure.update_yaxes(
        range=[0, 100],
        ticksuffix="%",
    )
    figure.update_xaxes(
        tickangle=-45,
    )
    figure.update_layout(
        legend_title_text="Meetpunt",
        hovermode="closest",
    )

    return figure