"""Critical data checks used by the build pipelines."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from chemistry_app.infrastructure.exceptions import MeasurementSchemaError, ParquetSchemaError

@dataclass(slots=True)
class BuildValidationReport:
    dataset: str
    input_rows: int = 0
    output_rows: int = 0
    duplicate_rows: int = 0
    rejected_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    def assert_balanced(self) -> None:
        if self.input_rows != self.output_rows + self.rejected_rows:
            raise MeasurementSchemaError(f"Rijbalans ongeldig voor {self.dataset}: input={self.input_rows}, output={self.output_rows}, rejected={self.rejected_rows}.")
    def as_dict(self) -> dict[str, object]:
        return {"dataset":self.dataset,"input_rows":self.input_rows,"output_rows":self.output_rows,"duplicate_rows":self.duplicate_rows,"rejected_rows":self.rejected_rows,"warnings":self.warnings}

MEASUREMENT_REQUIRED = frozenset({"locatie_code","parameter_code","parameter_omschrijving","datum","meetwaarde_lokaalid","numeriekewaarde","eenheid_code"})

def validate_source_columns(columns: list[str] | pd.Index) -> None:
    missing=sorted(MEASUREMENT_REQUIRED-set(columns))
    if missing: raise MeasurementSchemaError("Kritieke bronkolommen ontbreken: "+", ".join(missing))

def validate_measurement_chunk(frame: pd.DataFrame, *, offset: int=0) -> None:
    if frame.empty: return
    critical={"measurement_id":"meetwaarde-ID","location_code":"locatiecode","parameter_code":"parametercode","sample_date":"datum"}
    for column,label in critical.items():
        if column not in frame: raise MeasurementSchemaError(f"Interne buildkolom ontbreekt: {column}")
        bad=frame[column].isna() if column=="sample_date" else frame[column].astype("string").str.strip().isin(["",pd.NA]).fillna(True)
        if bad.any():
            examples=", ".join(str(offset+i+2) for i in frame.index[bad][:10])
            raise MeasurementSchemaError(f"Ontbrekende/ongeldige {label}; eerste CSV-regels: {examples}.")
    for c,low,high in (("latitude",-90,90),("longitude",-180,180)):
        if c in frame:
            bad=frame[c].notna() & ~frame[c].between(low,high)
            if bad.any(): raise MeasurementSchemaError(f"{c} buiten toegestaan bereik.")

def validate_unique_ids(ids: pd.Series) -> None:
    dup=ids[ids.duplicated(keep=False)]
    if not dup.empty: raise MeasurementSchemaError(f"measurement_id is niet uniek; eerste dubbele ID's: {', '.join(dup.astype(str).head(10).unique())}")

def validate_parquet(path: Path, *, expected_rows: int, required_columns: set[str]) -> None:
    try: frame=pd.read_parquet(path,columns=list(required_columns))
    except Exception as exc: raise ParquetSchemaError(f"Parquetvalidatie mislukt voor {path}: {exc}") from exc
    missing=required_columns-set(frame.columns)
    if missing: raise ParquetSchemaError("Parquet mist kolommen: "+", ".join(sorted(missing)))
    if len(frame)!=expected_rows: raise ParquetSchemaError(f"Parquet row count {len(frame)} wijkt af van verwacht {expected_rows}.")
