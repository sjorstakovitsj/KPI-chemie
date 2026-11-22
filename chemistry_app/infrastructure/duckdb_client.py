"""Small read-only DuckDB infrastructure layer."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator, Sequence
import re
import duckdb
import pandas as pd
from chemistry_app.infrastructure.exceptions import DataArtifactNotFoundError, DataArtifactReadError

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

def quote_identifier(identifier: str, *, allowed: set[str] | None = None) -> str:
    """Validate and quote a DuckDB identifier.

    Values are never interpolated; only validated identifiers are interpolated.
    """
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"Ongeldige SQL-identifier: {identifier!r}")
    if allowed is not None and identifier not in allowed:
        raise ValueError(f"Niet-toegestane kolom: {identifier}")
    return f'"{identifier}"'

@contextmanager
def read_only_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    connection=duckdb.connect(database=":memory:")
    try:
        connection.execute("SET enable_progress_bar = false")
        yield connection
    finally:
        connection.close()

def require_artifact(path: Path) -> Path:
    resolved=Path(path).resolve()
    if not resolved.is_file():
        raise DataArtifactNotFoundError(f"Parquetartifact niet gevonden: {resolved}",path=resolved)
    return resolved

def parquet_columns(path: Path) -> tuple[str,...]:
    artifact=require_artifact(path)
    try:
        with read_only_connection() as con:
            rows=con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[artifact.as_posix()]).fetchall()
    except duckdb.Error as exc:
        raise DataArtifactReadError(f"Parquetschema kan niet worden gelezen: {exc}",path=artifact) from exc
    return tuple(row[0] for row in rows)

def execute_frame(sql: str, parameters: Sequence[object]) -> pd.DataFrame:
    try:
        with read_only_connection() as con:
            return con.execute(sql,list(parameters)).fetchdf()
    except duckdb.Error as exc:
        raise DataArtifactReadError(f"DuckDB-query mislukt: {exc}") from exc

def explain_query(sql: str, parameters: Sequence[object]) -> str:
    try:
        with read_only_connection() as con:
            rows=con.execute("EXPLAIN "+sql,list(parameters)).fetchall()
    except duckdb.Error as exc:
        raise DataArtifactReadError(f"DuckDB EXPLAIN mislukt: {exc}") from exc
    return "\n".join(str(value) for row in rows for value in row)
