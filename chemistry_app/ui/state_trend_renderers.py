"""Streamlit renderers for Toestand & Trend results."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_mann_kendall_results(results: pd.DataFrame, *, title: str) -> None:
    with st.expander(title, expanded=False):
        if results.empty:
            st.info("Geen Mann-Kendall-resultaten beschikbaar.")
            return
        preferred = ("Meetpunt", "Stof", "Methode", "Aantal onder RG", "Aantal boven RG", "Tau", "p-waarde", "Trend")
        st.dataframe(results.loc[:, [c for c in preferred if c in results.columns]], width="stretch", hide_index=True)


def render_trend_table(frame: pd.DataFrame, *, title: str, columns: tuple[str, ...]) -> None:
    st.write(f"**{title}:** {len(frame)}")
    if frame.empty:
        st.info("Geen resultaten.")
        return
    st.dataframe(frame.loc[:, [c for c in columns if c in frame.columns]], width="stretch", hide_index=True, height=500)
