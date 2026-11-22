"""Plotly-only presentation functions. No chemistry calculations."""
from __future__ import annotations
import math
import pandas as pd
import plotly.graph_objects as go
from chemistry_app.services.visualization_data_service import HeatmapData,SunburstData

def create_gauge(percentage: float,title_text: str="Metingen onder Norm",drempel: int=95) -> go.Figure:
    value=0.0 if percentage is None or pd.isna(percentage) else min(max(float(percentage),0.0),100.0); threshold=min(max(int(drempel),0),100); color="green" if value>=threshold else "red"
    fig=go.Figure(go.Indicator(mode="gauge+number",value=value,domain={"x":[0,1],"y":[0,1]},title={"text":title_text,"font":{"size":14}},gauge={"axis":{"range":[0,100]},"bar":{"color":color},"steps":[{"range":[0,threshold],"color":"lightgray"},{"range":[threshold,100],"color":"green"}],"threshold":{"line":{"color":"red","width":4},"thickness":0.75,"value":threshold}})); fig.update_layout(height=250,margin=dict(l=30,r=30,t=50,b=10)); return fig

def create_compliance_heatmap(data: HeatmapData,title: str="Normoverschrijdingen") -> go.Figure:
    if data.factors.empty: return go.Figure().update_layout(title=title)
    return go.Figure(go.Heatmap(z=data.factors.to_numpy(),x=list(data.factors.columns),y=list(data.factors.index),text=data.labels.to_numpy(),hovertemplate="Stof: %{y}<br>Meetpunt: %{x}<br>Factor: %{z:.2f}<br>%{text}<extra></extra>",colorscale="Reds",zmin=0)).update_layout(title=title,xaxis_title="Meetpunt",yaxis_title="Stof")

_HOME_SUNBURST_COLORS = {
    "JG + MAC": "#8B0000",
    "JG (Gemiddelde)": "#FF8C00",
    "MAC (Piek)": "#DC143C",
}


def create_exceedance_heatmap(data: HeatmapData) -> go.Figure:
    """Create the Home exceedance-factor heatmap from prepared service data."""
    if data.factors.empty:
        return go.Figure()

    maximum = float(data.factors.max().max())
    z_max = maximum if maximum > 1.0 else 2.0
    split_fraction = 1.0 / z_max
    colorscale = [
        [0.0, "rgb(220, 220, 220)"],
        [split_fraction, "rgb(220, 220, 220)"],
        [split_fraction, "rgb(255, 235, 130)"],
        [1.0, "rgb(189, 0, 38)"],
    ]

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=data.factors.values,
                x=list(data.locations),
                y=list(data.substances),
                text=data.labels.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale=colorscale,
                zmin=0.0,
                zmax=z_max,
                hovertemplate=(
                    "<b>%{x}</b><br>%{y}<br>"
                    "Max Factor: %{z:.1f}x<br><br>"
                    "Status:<br>%{text}<extra></extra>"
                ),
                xgap=1,
                ygap=1,
            )
        ]
    )
    figure.update_layout(
        height=max(400, len(data.substances) * 30 + 150),
        xaxis={"title": "Meetpunt", "tickangle": -45},
        yaxis={"title": "Stof", "autorange": "reversed"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return figure


def create_compliance_sunburst(
    data: SunburstData,
    *,
    title: str,
) -> go.Figure:
    """Create the Home compliance sunburst from prepared service rows."""
    rows = data.rows
    if rows.empty:
        return go.Figure()

    colors = (
        rows["Categorie"]
        .map(_HOME_SUNBURST_COLORS)
        .fillna("#808080")
        .tolist()
    )
    hovertext = [
        f"Type: {row['Categorie']}<br>Factor: {row['Factor']:.1f}x"
        for _, row in rows.iterrows()
    ]

    figure = go.Figure(
        go.Sunburst(
            ids=["Totaal"] + rows["Stof"].tolist(),
            labels=["Chemische<br>Toestand"] + rows["Stof"].tolist(),
            parents=[""] + ["Totaal"] * len(rows),
            marker={"colors": ["#DDDDDD"] + colors},
            hovertext=[""] + hovertext,
            hoverinfo="label+text",
        )
    )
    figure.update_layout(
        margin={"t": 30, "l": 0, "r": 0, "b": 10},
        title=title,
        height=500,
    )
    return figure

