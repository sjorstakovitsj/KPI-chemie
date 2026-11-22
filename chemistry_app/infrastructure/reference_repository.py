"""Read-only chemistry reference repository."""
from __future__ import annotations
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from chemistry_app.infrastructure.duckdb_client import execute_frame,parquet_columns,quote_identifier,require_artifact
from chemistry_app.infrastructure.artifact_version import artifact_version
class ReferenceRepository:
    def __init__(self,parquet_path: Path,manifest_path: Path) -> None:
        self.path=require_artifact(parquet_path); self.version=artifact_version(manifest_path,"chemistry_reference"); self.columns=set(parquet_columns(self.path))
    def query(self,*,reference_types: tuple[str,...]=(),columns: Sequence[str]) -> pd.DataFrame:
        selected=tuple(columns)
        if not selected: raise ValueError("Geef expliciet referentiekolommen op.")
        unknown=set(selected)-self.columns
        if unknown: raise ValueError("Onbekende referentiekolommen: "+", ".join(sorted(unknown)))
        sql="SELECT "+", ".join(quote_identifier(c,allowed=self.columns) for c in selected)+" FROM read_parquet(?)"; params=[self.path.as_posix()]
        if reference_types:
            if "reference_type" not in self.columns: raise ValueError("reference_type ontbreekt in artifact.")
            sql+=f" WHERE {quote_identifier('reference_type',allowed=self.columns)} IN ({', '.join('?' for _ in reference_types)})"; params.extend(reference_types)
        return execute_frame(sql,params)

