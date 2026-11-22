"""Temporary legacy-column adapter for pages still expecting Dutch names."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from chemistry_app.infrastructure.measurements_repository import MeasurementsRepository
from chemistry_app.infrastructure.models import MeasurementFilters
LEGACY_MAP={"sample_date":"Datum","location_name":"Meetpunt","parameter_name":"Stof","numeric_value_reported":"Waarde","limit_symbol":"Limietsymbool","unit_code":"Eenheid","substance_group":"Stofgroep","condition_code":"Hoedanigheid","compartment_code":"Compartiment","quality_code":"Kwaliteitsoordeel","status":"Status"}
def query_legacy(repository: MeasurementsRepository,filters: MeasurementFilters=MeasurementFilters(),*,columns: tuple[str,...]) -> pd.DataFrame:
    reverse={v:k for k,v in LEGACY_MAP.items()}; canonical=tuple(reverse.get(c,c) for c in columns); return repository.query(filters,columns=canonical).rename(columns=LEGACY_MAP)

