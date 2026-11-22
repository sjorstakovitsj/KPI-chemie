"""Repository query models."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date

@dataclass(frozen=True, slots=True)
class MeasurementFilters:
    years: tuple[int,...]=()
    periods: tuple[str,...]=()
    substances: tuple[str,...]=()
    locations: tuple[str,...]=()
    substance_groups: tuple[str,...]=()
    units: tuple[str,...]=()
    compartments: tuple[str,...]=()
    conditions: tuple[str,...]=()
    quality_codes: tuple[str,...]=()
    statuses: tuple[str,...]=()
    date_from: date | None=None
    date_to: date | None=None

    def normalized(self) -> "MeasurementFilters":
        def unique(values): return tuple(sorted(set(values),key=str))
        return MeasurementFilters(
            years=tuple(sorted(set(int(v) for v in self.years))), periods=unique(self.periods),
            substances=unique(self.substances), locations=unique(self.locations),
            substance_groups=unique(self.substance_groups), units=unique(self.units),
            compartments=unique(self.compartments), conditions=unique(self.conditions),
            quality_codes=unique(self.quality_codes), statuses=unique(self.statuses),
            date_from=self.date_from,date_to=self.date_to,
        )
