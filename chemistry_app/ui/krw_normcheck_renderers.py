"""Streamlit renderers for the KRW normcheck page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from chemistry_app.ui.krw_normcheck_charts import (
    create_jg_figure,
    create_mac_figure,
)


def render_jg_results(
    data: pd.DataFrame,
) -> None:
    """Render JG-MKN figures and the annual-mean table."""
    if data.empty:
        st.info(
            "Geen JG-MKN-resultaten voor de huidige selectie. "
            "Controleer de gekozen jaren, meetpunten, stoffen "
            "en de instelling voor de rapportagegrens."
        )
        return

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
        st.error(
            "De JG-MKN-resultaten missen vereiste kolommen: "
            + ", ".join(missing_columns)
        )
        return

    substances = tuple(
        sorted(
            data["Stof"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .unique()
            .tolist(),
            key=str.casefold,
        )
    )

    for substance in substances:
        figure = create_jg_figure(
            data,
            substance=substance,
        )

        if len(figure.data) == 0:
            st.warning(
                f"Voor {substance} konden geen geldige "
                "JG-MKN-resultaten worden gevisualiseerd."
            )
            continue

        st.plotly_chart(
            figure,
            width="stretch",
        )

    preferred_columns = (
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

    available_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    table = (
        data.loc[:, available_columns]
        .sort_values(
            [
                column
                for column in (
                    "Stof",
                    "Jaar",
                    "Meetpunt",
                )
                if column in available_columns
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    column_config: dict[str, object] = {}

    if "Waarde" in table.columns:
        column_config["Waarde"] = (
            st.column_config.NumberColumn(
                "Gecorrigeerd jaargemiddelde",
                format="%.6g",
            )
        )

    if "measurement_count" in table.columns:
        column_config["measurement_count"] = (
            st.column_config.NumberColumn(
                "Aantal gebruikte metingen",
                format="%d",
            )
        )

    if "JG_MKN" in table.columns:
        column_config["JG_MKN"] = (
            st.column_config.NumberColumn(
                "JG-MKN",
                format="%.6g",
            )
        )

    if "Overschrijdingsfactor" in table.columns:
        column_config["Overschrijdingsfactor"] = (
            st.column_config.NumberColumn(
                "Factor ten opzichte van norm",
                format="%.3f",
            )
        )

    if "Overschrijdt" in table.columns:
        column_config["Overschrijdt"] = (
            st.column_config.CheckboxColumn(
                "Overschrijdt JG-MKN"
            )
        )

    with st.expander(
        "Toon berekende jaargemiddelden in tabel",
        expanded=False,
    ):
        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )


def render_mac_results(
    summary: pd.DataFrame,
    details: pd.DataFrame,
) -> None:
    """Render MAC-MKN percentages and individual measurements."""
    if summary.empty:
        st.info(
            "Geen MAC-MKN-resultaten voor de huidige selectie. "
            "Controleer de gekozen jaren, meetpunten, stoffen "
            "en de instelling voor de rapportagegrens."
        )
        return

    required_columns = {
        "Meetpunt",
        "Stof",
        "measurement_count",
        "exceedance_count",
        "Overschrijding (%)",
        "PlotLabel",
    }
    missing_columns = sorted(
        required_columns - set(summary.columns)
    )

    if missing_columns:
        st.error(
            "De MAC-MKN-resultaten missen vereiste kolommen: "
            + ", ".join(missing_columns)
        )
        return

    figure = create_mac_figure(summary)

    if len(figure.data) == 0:
        st.warning(
            "De MAC-MKN-samenvatting bevat geen "
            "visualiseerbare resultaten."
        )
    else:
        st.plotly_chart(
            figure,
            width="stretch",
        )

    summary_columns = [
        column
        for column in (
            "Meetpunt",
            "Stof",
            "measurement_count",
            "exceedance_count",
            "Overschrijding (%)",
            "MAC_MKN",
            "Eenheid",
        )
        if column in summary.columns
    ]

    with st.expander(
        "Toon MAC-MKN-samenvatting",
        expanded=False,
    ):
        st.dataframe(
            summary.loc[:, summary_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "measurement_count": (
                    st.column_config.NumberColumn(
                        "Aantal gebruikte metingen",
                        format="%d",
                    )
                ),
                "exceedance_count": (
                    st.column_config.NumberColumn(
                        "Aantal overschrijdingen",
                        format="%d",
                    )
                ),
                "Overschrijding (%)": (
                    st.column_config.NumberColumn(
                        "Overschrijding",
                        format="%.2f%%",
                    )
                ),
                "MAC_MKN": (
                    st.column_config.NumberColumn(
                        "MAC-MKN",
                        format="%.6g",
                    )
                ),
            },
        )

    if details.empty:
        return

    preferred_detail_columns = (
        "Datum",
        "Meetpunt",
        "Stof",
        "Waarde_Origineel",
        "Achtergrondconcentratie",
        "Waarde",
        "Eenheid",
        "Limietsymbool",
        "MAC_MKN",
        "Overschrijding",
        "Achtergrondcorrectie_Toegepast",
    )

    detail_columns = [
        column
        for column in preferred_detail_columns
        if column in details.columns
    ]

    detail_table = (
        details.loc[:, detail_columns]
        .sort_values(
            [
                column
                for column in (
                    "Stof",
                    "Meetpunt",
                    "Datum",
                )
                if column in detail_columns
            ],
            ascending=[
                True,
                True,
                False,
            ][: len(
                [
                    column
                    for column in (
                        "Stof",
                        "Meetpunt",
                        "Datum",
                    )
                    if column in detail_columns
                ]
            )],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    detail_config: dict[str, object] = {}

    if "Waarde_Origineel" in detail_table.columns:
        detail_config["Waarde_Origineel"] = (
            st.column_config.NumberColumn(
                "Oorspronkelijke concentratie",
                format="%.6g",
            )
        )

    if "Achtergrondconcentratie" in detail_table.columns:
        detail_config["Achtergrondconcentratie"] = (
            st.column_config.NumberColumn(
                "Achtergrondconcentratie",
                format="%.6g",
            )
        )

    if "Waarde" in detail_table.columns:
        detail_config["Waarde"] = (
            st.column_config.NumberColumn(
                "Gecorrigeerde concentratie",
                format="%.6g",
            )
        )

    if "MAC_MKN" in detail_table.columns:
        detail_config["MAC_MKN"] = (
            st.column_config.NumberColumn(
                "MAC-MKN",
                format="%.6g",
            )
        )

    if "Overschrijding" in detail_table.columns:
        detail_config["Overschrijding"] = (
            st.column_config.CheckboxColumn(
                "Overschrijdt MAC-MKN"
            )
        )

    if (
        "Achtergrondcorrectie_Toegepast"
        in detail_table.columns
    ):
        detail_config[
            "Achtergrondcorrectie_Toegepast"
        ] = st.column_config.CheckboxColumn(
            "Achtergrondcorrectie toegepast"
        )

    with st.expander(
        "Toon individuele MAC-metingen "
        "en achtergrondcorrectie",
        expanded=False,
    ):
        st.dataframe(
            detail_table,
            width="stretch",
            hide_index=True,
            column_config=detail_config,
        )