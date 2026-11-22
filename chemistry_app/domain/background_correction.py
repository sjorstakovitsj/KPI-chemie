"""Pure background-correction logic for chemistry measurements."""
from __future__ import annotations
import re
import pandas as pd
from chemistry_app.ingestion.normalization import normalize_text_series, normalize_unit_series

_SUFFIX_RE=re.compile(r"\s*\((?:totaal|opgelost)\)\s*$",flags=re.IGNORECASE)

def apply_background_correction(
    measurements: pd.DataFrame, references: pd.DataFrame, *,
    substance_column: str="Stof", unit_column: str="Eenheid", value_column: str="Waarde",
    reference_substance_column: str="stofnaam", reference_unit_column: str="eenheid",
    reference_value_column: str="achtergrondconcentratie",
) -> pd.DataFrame:
    """Subtract matching background values, preserving all reported values."""
    required_m={substance_column,unit_column,value_column}; required_r={reference_substance_column,reference_unit_column,reference_value_column}
    missing_m=sorted(required_m-set(measurements)); missing_r=sorted(required_r-set(references))
    if missing_m: raise KeyError("Ontbrekende meetkolommen: "+", ".join(missing_m))
    if missing_r: raise KeyError("Ontbrekende referentiekolommen: "+", ".join(missing_r))
    result=measurements.copy(); result["Waarde_Origineel"]=pd.to_numeric(result[value_column],errors="coerce").astype("Float64")
    result["Achtergrondconcentratie"]=pd.Series(pd.NA,index=result.index,dtype="Float64")
    result["Achtergrondcorrectie_Toegepast"]=False; result["Achtergrondcorrectie_Status"]="geen_match"
    if references.empty: return result
    ref=references.copy(); ref["__stof"]=normalize_text_series(ref[reference_substance_column]); ref["__eenheid"]=normalize_unit_series(ref[reference_unit_column]); ref["__achtergrond"]=pd.to_numeric(ref[reference_value_column],errors="coerce").astype("Float64")
    if ref.duplicated(["__stof","__eenheid"]).any(): raise ValueError("Achtergrondreferentie bevat dubbele stof-eenheidcombinaties.")
    if ref["__achtergrond"].lt(0).any(): raise ValueError("Achtergrondconcentratie mag niet negatief zijn.")
    substances=result[substance_column].astype("string").str.replace(_SUFFIX_RE,"",regex=True)
    keys=pd.MultiIndex.from_arrays([normalize_text_series(substances),normalize_unit_series(result[unit_column])])
    mapping=ref.set_index(["__stof","__eenheid"])["__achtergrond"]
    matched=pd.Series(mapping.reindex(keys).to_numpy(),index=result.index,dtype="Float64")
    valid=matched.notna() & result["Waarde_Origineel"].notna()
    result.loc[matched.notna(),"Achtergrondconcentratie"]=matched.loc[matched.notna()]
    result.loc[valid,value_column]=(result.loc[valid,"Waarde_Origineel"]-matched.loc[valid]).clip(lower=0)
    result.loc[valid,"Achtergrondcorrectie_Toegepast"]=True; result.loc[valid,"Achtergrondcorrectie_Status"]="toegepast"
    result.loc[matched.notna() & ~valid,"Achtergrondcorrectie_Status"]="meetwaarde_ontbreekt"
    return result

def apply_achtergrondcorrectie(measurements: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    """Dutch compatibility alias."""
    return apply_background_correction(measurements,references)
