"""Plotly figures for the Toestand & Trend page."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_measurements_figure(frame: pd.DataFrame) -> go.Figure:
    """Create a faceted measurement figure with reporting-limit symbols."""
    if frame.empty:
        return go.Figure()
    data = frame.copy()
    data["Meting Type"] = np.where(
        data["Limietsymbool"].astype(str).str.contains("<", na=False),
        "< Onder rapportagegrens",
        "Gemeten waarde",
    )
    substances = tuple(sorted(data["Stof"].dropna().astype(str).unique(), key=str.casefold))
    figure = px.scatter(
        data,
        x="Datum",
        y="Waarde",
        color="Meetpunt",
        symbol="Meting Type",
        symbol_map={"< Onder rapportagegrens": "x", "Gemeten waarde": "circle"},
        facet_row="Stof",
        category_orders={"Stof": substances},
        hover_data={"Datum": True, "Waarde": ":.4f", "Eenheid": True, "Meetpunt": True},
        title="Geselecteerde individuele metingen over tijd",
        height=max(350, 350 * len(substances)),
    )
    figure.update_xaxes(matches="x")
    figure.update_yaxes(matches=None)
    return figure


def create_regression_figure(
    frame: pd.DataFrame,
    *,
    title: str,
    standard: float | None = None,
) -> go.Figure:
    """Create individual points and a trend line based on annual averages."""
    if frame.empty:
        return go.Figure()
    data = frame.copy()
    data["Datum"] = pd.to_datetime(data["Datum"], errors="coerce")
    data["Waarde"] = pd.to_numeric(data["Waarde"], errors="coerce")
    data = data.dropna(subset=["Datum", "Waarde"])
    if data.empty:
        return go.Figure()
    data["Jaar"] = data["Datum"].dt.year
    annual = data.groupby("Jaar", observed=True)["Waarde"].mean().reset_index()
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["Datum"], y=data["Waarde"], mode="markers", name="Individuele meting"))
    if len(annual) >= 2:
        coefficients = np.polyfit(annual["Jaar"], annual["Waarde"], 1)
        annual["Trend"] = np.poly1d(coefficients)(annual["Jaar"])
        annual["Datum"] = pd.to_datetime(annual["Jaar"].astype(str) + "-07-01")
        figure.add_trace(go.Scatter(x=annual["Datum"], y=annual["Trend"], mode="lines", name=f"Trendlijn ({coefficients[0]:.4f})"))
    if standard is not None and pd.notna(standard):
        figure.add_hline(y=float(standard), line_color="orange", annotation_text=f"JG-norm: {standard}")
    figure.update_layout(title=title, xaxis_title="Datum", yaxis_title="Concentratie", hovermode="closest")
    return figure
