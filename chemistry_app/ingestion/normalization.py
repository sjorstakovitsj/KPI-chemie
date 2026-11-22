"""Reusable normalization helpers for chemistry CSV ingestion."""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

_SPACE_RE = re.compile(r"\s+")

def normalize_text(value: object) -> str | pd.NA:
    """Normalize Unicode, whitespace and case while preserving missing values."""
    if value is None or pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = _SPACE_RE.sub(" ", text)
    return text if text else pd.NA

def normalize_unit(value: object) -> str | pd.NA:
    """Normalize common micro symbols and spacing in a unit."""
    text = normalize_text(value)
    if pd.isna(text): return pd.NA
    return str(text).replace("µ", "u").replace("μ", "u").replace(" ", "")

def normalize_columns(columns: pd.Index) -> pd.Index:
    """Normalize CSV column names to stable snake-case-like labels."""
    normalized=[]
    for column in columns:
        value=str(normalize_text(column)).replace("-", "_").replace(" ", "_")
        normalized.append(value)
    return pd.Index(normalized)

def normalize_decimal(series: pd.Series) -> pd.Series:
    """Parse decimal comma/dot strings as nullable Float64 values."""
    values=series.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce").astype("Float64")

def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_text).astype("string")

def normalize_unit_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_unit).astype("string")
