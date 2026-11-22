"""Service models for available filters and selections."""
from __future__ import annotations
from dataclasses import dataclass
from chemistry_app.infrastructure.measurements_repository import MeasurementsRepository
from chemistry_app.infrastructure.models import MeasurementFilters
@dataclass(frozen=True, slots=True)
class FilterOptions:
    years: tuple[int,...]=(); locations: tuple[str,...]=(); substances: tuple[str,...]=(); substance_groups: tuple[str,...]=(); units: tuple[str,...]=(); compartments: tuple[str,...]=(); conditions: tuple[str,...]=(); quality_codes: tuple[str,...]=(); statuses: tuple[str,...]=()

def get_filter_options(repository: MeasurementsRepository) -> FilterOptions:
    raw=repository.filter_options()
    return FilterOptions(tuple(int(v) for v in raw.get("sample_year",())),tuple(raw.get("location_code",())),tuple(raw.get("parameter_name",())),tuple(raw.get("substance_group",())),tuple(raw.get("unit_code",())),tuple(raw.get("compartment_code",())),tuple(raw.get("condition_code",())),tuple(raw.get("quality_code",())),tuple(raw.get("status",())))

def create_measurement_filters(*,years=(),periods=(),substances=(),locations=(),substance_groups=(),units=(),compartments=(),conditions=(),quality_codes=(),statuses=()) -> MeasurementFilters:
    return MeasurementFilters(tuple(years),tuple(periods),tuple(substances),tuple(locations),tuple(substance_groups),tuple(units),tuple(compartments),tuple(conditions),tuple(quality_codes),tuple(statuses)).normalized()

