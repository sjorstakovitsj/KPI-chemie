"""Build the canonical chemistry measurements Parquet from source CSV."""
from __future__ import annotations
import os
from hashlib import sha256
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from chemistry_app.domain.censored_values import (
    CensoredValueStrategy,
    apply_censored_value_strategy,
    missing_measurement_value_mask,
    normalize_missing_measurement_values,
)
from chemistry_app.domain.substance_groups import classify_substance_group
from chemistry_app.infrastructure.exceptions import DataArtifactNotFoundError, DataArtifactReadError, MeasurementSchemaError
from chemistry_app.pipelines.validation import BuildValidationReport, validate_measurement_chunk, validate_parquet, validate_source_columns, validate_unique_ids
COLUMN_MAP={"meetwaarde_lokaalid":"source_measurement_id","locatie_code":"location_code","locatienaam":"location_name","locatie_lat_etrs89":"latitude","locatie_lon_etrs89":"longitude","datum":"sample_date","tijd":"sample_time","resultaattijd":"result_timestamp","parameter_code":"parameter_code","parameter_omschrijving":"parameter_name","grootheid_code":"quantity_code","hoedanigheid_code":"condition_code","compartiment_code":"compartment_code","eenheid_code":"unit_code","numeriekewaarde":"numeric_value_reported","alfanumeriekewaarde":"alphanumeric_value","limietsymbool":"limit_symbol","status":"status","kwaliteitsoordeel_code":"quality_code"}
REQUIRED_OUTPUT={"measurement_id","location_code","sample_date","parameter_code","numeric_value_reported","numeric_value_analysis","is_censored","substance_group"}
def _transform(frame: pd.DataFrame,offset: int) -> pd.DataFrame:
    frame=frame.rename(columns=COLUMN_MAP); out=pd.DataFrame(index=frame.index)
    for target in COLUMN_MAP.values(): out[target]=frame[target] if target in frame else pd.NA
    out["source_measurement_id"] = out["source_measurement_id"].astype("string").str.strip()
    for column in ("source_measurement_id","location_code","location_name","parameter_code","parameter_name","quantity_code","condition_code","compartment_code","unit_code","alphanumeric_value","limit_symbol","status","quality_code","sample_time"):
        out[column]=out[column].astype("string").str.strip()
    out["latitude"]=pd.to_numeric(out["latitude"],errors="coerce"); out["longitude"]=pd.to_numeric(out["longitude"],errors="coerce")
    out["sample_date"]=pd.to_datetime(out["sample_date"],format="%Y-%m-%d",errors="coerce")
    out["result_timestamp"]=pd.to_datetime(out["result_timestamp"],format="mixed",errors="coerce",dayfirst=False)
    raw_numeric_values=pd.to_numeric(out["numeric_value_reported"].astype("string").str.replace(",",".",regex=False),errors="coerce")
    out["is_missing_value_sentinel"]=missing_measurement_value_mask(raw_numeric_values)
    out["numeric_value_reported"]=normalize_missing_measurement_values(raw_numeric_values)
    legacy=out.rename(columns={"numeric_value_reported":"Waarde","limit_symbol":"Limietsymbool"})
    legacy=apply_censored_value_strategy(legacy,CensoredValueStrategy.EXCLUDE,output_column="numeric_value_analysis",censored_column="is_censored")
    out["numeric_value_analysis"]=legacy["numeric_value_analysis"].astype("float64"); out["is_censored"]=legacy["is_censored"].astype(bool)
    out["sample_year"]=out["sample_date"].dt.year.astype("Int16"); out["sample_month"]=out["sample_date"].dt.month.astype("Int8")
    out["substance_group"]=out["parameter_name"].fillna("").map(classify_substance_group).astype("string")
    out["source_row_number"]=pd.Series(range(offset+2,offset+2+len(out)),index=out.index,dtype="int64")
    # meetwaarde_lokaalid is in de bron een monster-/opdracht-ID en is daarom
    # niet uniek per parameter. Maak een technische meetregel-ID en behoud het
    # oorspronkelijke ID afzonderlijk voor herleidbaarheid.
    identity_columns = [
        "source_measurement_id", "location_code", "parameter_code",
        "sample_date", "sample_time", "result_timestamp",
        "numeric_value_reported", "limit_symbol", "unit_code",
    ]
    def make_measurement_id(row: pd.Series) -> str:
        canonical = "|".join("" if pd.isna(row[column]) else str(row[column]) for column in identity_columns)
        digest = sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"m_{int(row['source_row_number']):09d}_{digest}"
    out["measurement_id"] = out.apply(make_measurement_id, axis=1).astype("string")
    validate_measurement_chunk(out,offset=offset); return out

def build_measurements(source: Path,output: Path,*,chunksize: int=100_000) -> BuildValidationReport:
    source=Path(source); output=Path(output)
    if not source.is_file(): raise DataArtifactNotFoundError(f"Meetbestand niet gevonden: {source}",path=source)
    output.parent.mkdir(parents=True,exist_ok=True); temporary=output.with_suffix(output.suffix+".tmp"); writer=None; identifiers=[]; report=BuildValidationReport("chemistry_measurements")
    try:
        header=pd.read_csv(source,sep=";",nrows=0,encoding="utf-8-sig"); validate_source_columns(header.columns)
        for chunk in pd.read_csv(source,sep=";",dtype="string",encoding="utf-8-sig",chunksize=chunksize,low_memory=False):
            transformed=_transform(chunk,report.input_rows); report.input_rows+=len(chunk); report.output_rows+=len(transformed); identifiers.append(transformed["measurement_id"]); missing_source_ids=int(transformed["source_measurement_id"].isna().sum());
            if missing_source_ids: report.warnings.append(f"{missing_source_ids} bronregels zonder meetwaarde_lokaalid; technische measurement_id is wel aangemaakt.")
            sentinel_count=int(transformed["is_missing_value_sentinel"].sum())
            if sentinel_count: report.warnings.append(f"{sentinel_count} bronwaarden met bekende hiaatcode zijn als ontbrekende meetwaarde opgeslagen.")
            table=pa.Table.from_pandas(transformed,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(temporary,table.schema,compression="zstd")
            writer.write_table(table)
        if writer is None: raise MeasurementSchemaError("Het CSV-bronbestand bevat geen meetrecords.")
        writer.close(); writer=None
        validate_unique_ids(pd.concat(identifiers,ignore_index=True)); report.assert_balanced(); validate_parquet(temporary,expected_rows=report.output_rows,required_columns=REQUIRED_OUTPUT)
        os.replace(temporary,output); return report
    except (MeasurementSchemaError,DataArtifactNotFoundError): raise
    except Exception as exc: raise DataArtifactReadError(f"Bouw van metingen-Parquet mislukt: {exc}",path=source) from exc
    finally:
        if writer is not None: writer.close()
        temporary.unlink(missing_ok=True)
