"""Centrale periode-definities en validatiehelpers voor de chemieapp."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, TypeAlias

MonthTuple: TypeAlias = tuple[int, ...]

PERIODES: Final[dict[str, MonthTuple]] = {
    "Winter": (12, 1, 2),
    "Voorjaar": (3, 4, 5),
    "Zomer": (6, 7, 8),
    "Herfst": (9, 10, 11),
    "Zomerhalfjaar": (4, 5, 6, 7, 8, 9),
    "Winterhalfjaar": (10, 11, 12, 1, 2, 3),
}

# De expliciete volgorde voorkomt dat UI-volgorde ongemerkt verandert wanneer
# PERIODES later wordt aangepast of anders wordt opgebouwd.
PERIODE_VOLGORDE: Final[tuple[str, ...]] = (
    "Winter",
    "Voorjaar",
    "Zomer",
    "Herfst",
    "Zomerhalfjaar",
    "Winterhalfjaar",
)


def months_for_periods(periods: Iterable[str]) -> MonthTuple:
    """Geef de unieke kalendermaanden voor de geselecteerde perioden terug.

    Dubbele perioden en overlappende maanden worden eenmaal opgenomen. De
    maanden worden numeriek gesorteerd, wat geschikt is voor DuckDB/Pandas-
    filtering met ``month IN (...)``.

    Args:
        periods: Namen zoals opgenomen in :data:`PERIODES`.

    Returns:
        Een oplopend gesorteerde tuple met unieke maandnummers.

    Raises:
        ValueError: Als een of meer perioden onbekend zijn.
    """
    selected = tuple(dict.fromkeys(periods))
    unknown = sorted(set(selected).difference(PERIODES))

    if unknown:
        available = ", ".join(PERIODE_VOLGORDE)
        invalid = ", ".join(unknown)
        raise ValueError(
            f"Onbekende perioden: {invalid}. Geldige perioden: {available}."
        )

    return tuple(
        sorted(
            {
                month
                for period in selected
                for month in PERIODES[period]
            }
        )
    )


__all__ = ["MonthTuple", "PERIODES", "PERIODE_VOLGORDE", "months_for_periods"]
