"""Read-only measurement queries without UI dependencies."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from chemistry_app.config.periods import months_for_periods
from chemistry_app.infrastructure.duckdb_client import execute_frame, explain_query, parquet_columns, quote_identifier, require_artifact
from chemistry_app.infrastructure.artifact_version import artifact_version
from chemistry_app.infrastructure.models import MeasurementFilters

DEFAULT_COLUMNS=("measurement_id","location_code","location_name","sample_date","parameter_code","parameter_name","condition_code","compartment_code","unit_code","numeric_value_reported","numeric_value_analysis","limit_symbol","is_censored","status","quality_code","substance_group")
FILTER_COLUMNS={"sample_year","sample_month","parameter_name","location_code","substance_group","unit_code","compartment_code","condition_code","quality_code","status","sample_date"}

class MeasurementsRepository:
    def __init__(self,parquet_path: Path,manifest_path: Path) -> None:
        self.parquet_path=require_artifact(parquet_path); self.manifest_path=Path(manifest_path).resolve()
        self.version=artifact_version(self.manifest_path,"chemistry_measurements")
        self.columns=parquet_columns(self.parquet_path); self.column_set=set(self.columns)
    def _projection(self,columns: Sequence[str] | None) -> tuple[str,...]:
        selected=tuple(columns) if columns is not None else tuple(c for c in DEFAULT_COLUMNS if c in self.column_set)
        if not selected: raise ValueError("Kolomprojectie mag niet leeg zijn.")
        unknown=sorted(set(selected)-self.column_set)
        if unknown: raise ValueError("Onbekende meetkolommen: "+", ".join(unknown))
        return selected
    def _build_query(self,filters: MeasurementFilters,columns: Sequence[str] | None) -> tuple[str,list[object]]:
        selected=self._projection(columns); sql="SELECT "+", ".join(quote_identifier(c,allowed=self.column_set) for c in selected)+" FROM read_parquet(?)"; params=[self.parquet_path.as_posix()]; conditions=[]
        f=filters.normalized()
        def add_in(column: str,values: tuple[object,...]) -> None:
            if not values: return
            if column not in self.column_set: raise ValueError(f"Filterkolom ontbreekt in Parquet: {column}")
            conditions.append(f"{quote_identifier(column,allowed=self.column_set)} IN ({', '.join('?' for _ in values)})"); params.extend(values)
        add_in("sample_year",f.years)
        if f.periods: add_in("sample_month",months_for_periods(f.periods))
        add_in("parameter_name",f.substances); add_in("location_code",f.locations); add_in("substance_group",f.substance_groups); add_in("unit_code",f.units); add_in("compartment_code",f.compartments); add_in("condition_code",f.conditions); add_in("quality_code",f.quality_codes); add_in("status",f.statuses)
        if f.date_from is not None: conditions.append(f"{quote_identifier('sample_date',allowed=self.column_set)} >= CAST(? AS DATE)"); params.append(f.date_from.isoformat())
        if f.date_to is not None: conditions.append(f"{quote_identifier('sample_date',allowed=self.column_set)} <= CAST(? AS DATE)"); params.append(f.date_to.isoformat())
        if conditions: sql+=" WHERE "+" AND ".join(conditions)
        return sql,params
    def query(self,filters: MeasurementFilters=MeasurementFilters(),*,columns: Sequence[str] | None=None) -> pd.DataFrame:
        sql,params=self._build_query(filters,columns); result=execute_frame(sql,params)
        if "sample_date" in result: result["sample_date"]=pd.to_datetime(result["sample_date"],errors="coerce")
        return result
    def explain(self,filters: MeasurementFilters=MeasurementFilters(),*,columns: Sequence[str] | None=None) -> str:
        sql,params=self._build_query(filters,columns); return explain_query(sql,params)
    def filter_options(self) -> dict[str,tuple]:
        return _filter_options_cached(self.parquet_path.as_posix(),self.version)

@lru_cache(maxsize=16)
def _filter_options_cached(path: str,version: str) -> dict[str,tuple]:
    del version
    columns=("sample_year","location_code","parameter_name","substance_group","unit_code","compartment_code","condition_code","quality_code","status")
    expressions=[]
    for column in columns: expressions.append(f"list_sort(list(DISTINCT {quote_identifier(column)})) AS {quote_identifier(column)}")
    row=execute_frame("SELECT "+", ".join(expressions)+" FROM read_parquet(?)",[path]).iloc[0]
    return {
    column: tuple(
        value
        for value in row[column]
        if value is not None
    )
    if row[column] is not None
    else tuple()
    for column in columns
}

