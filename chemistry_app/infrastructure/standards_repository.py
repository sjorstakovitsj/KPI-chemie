"""Read-only standards repository."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from chemistry_app.infrastructure.duckdb_client import execute_frame,parquet_columns,quote_identifier,require_artifact
from chemistry_app.infrastructure.artifact_version import artifact_version
class StandardsRepository:
    DEFAULT_COLUMNS=("substance","standard_type","standard_value")
    def __init__(self,parquet_path: Path,manifest_path: Path) -> None:
        self.path=require_artifact(parquet_path); self.version=artifact_version(manifest_path,"chemistry_standards"); self.columns=set(parquet_columns(self.path))
    def query(self,*,substances: tuple[str,...]=(),standard_types: tuple[str,...]=(),columns: Sequence[str] | None=None) -> pd.DataFrame:
        selected=tuple(columns or self.DEFAULT_COLUMNS); unknown=set(selected)-self.columns
        if unknown: raise ValueError("Onbekende normkolommen: "+", ".join(sorted(unknown)))
        sql="SELECT "+", ".join(quote_identifier(c,allowed=self.columns) for c in selected)+" FROM read_parquet(?)"; params=[self.path.as_posix()]; where=[]
        for column,values in (("substance",substances),("standard_type",standard_types)):
            if values: where.append(f"{quote_identifier(column,allowed=self.columns)} IN ({', '.join('?' for _ in values)})"); params.extend(values)
        if where: sql+=" WHERE "+" AND ".join(where)
        return execute_frame(sql,params)

