"""PFAS service for relative potency/equivalency calculations."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.infrastructure.exceptions import ReferenceTableValidationError
@dataclass(frozen=True, slots=True)
class PfasSummary:
    measurement_count: int
    substance_count: int
    total_peq: float

def calculate_pfas_equivalents(measurements: pd.DataFrame,references: pd.DataFrame,*,substance_column: str="Stof",value_column: str="Waarde",reference_substance_column: str="Stof",rpf_column: str="RPF",output_column: str="PEQ") -> pd.DataFrame:
    required_m={substance_column,value_column}; required_r={reference_substance_column,rpf_column}
    if missing:=sorted(required_m-set(measurements)): raise KeyError("Ontbrekende PFAS-meetkolommen: "+", ".join(missing))
    if missing:=sorted(required_r-set(references)): raise ReferenceTableValidationError("Ontbrekende PFAS-referentiekolommen: "+", ".join(missing))
    ref=references[[reference_substance_column,rpf_column]].copy(); ref["__key"]=ref[reference_substance_column].astype("string").str.strip().str.casefold(); ref["__rpf"]=pd.to_numeric(ref[rpf_column],errors="coerce")
    if ref["__key"].isna().any() or ref["__rpf"].isna().any() or ref["__rpf"].lt(0).any(): raise ReferenceTableValidationError("PFAS-referentie bevat lege of ongeldige RPF-waarden.")
    if ref["__key"].duplicated().any(): raise ReferenceTableValidationError("PFAS-referentie bevat dubbele stofnamen.")
    result=measurements.copy(); result["__key"]=result[substance_column].astype("string").str.strip().str.casefold(); mapping=ref.set_index("__key")["__rpf"]
    result["RPF"]=result["__key"].map(mapping).astype("Float64"); result[output_column]=pd.to_numeric(result[value_column],errors="coerce").astype("Float64")*result["RPF"]
    return result.drop(columns="__key")
def summarize_pfas(result: pd.DataFrame,*,substance_column: str="Stof",peq_column: str="PEQ") -> PfasSummary:
    if result.empty: return PfasSummary(0,0,0.0)
    return PfasSummary(len(result),int(result[substance_column].nunique(dropna=True)),float(pd.to_numeric(result[peq_column],errors="coerce").sum()))
