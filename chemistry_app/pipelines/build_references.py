"""Build standards and reference Parquets without Streamlit."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from chemistry_app.infrastructure.exceptions import DataArtifactNotFoundError, ReferenceTableValidationError
from chemistry_app.ingestion.reference_csv import load_achtergrondcorrecties,load_normen,load_pfas_ref,load_signaleringswaarden
from chemistry_app.pipelines.validation import BuildValidationReport,validate_parquet

def _atomic_parquet(frame: pd.DataFrame,path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    try: frame.to_parquet(temporary,index=False,engine="pyarrow",compression="zstd"); validate_parquet(temporary,expected_rows=len(frame),required_columns=set(frame.columns)); os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)
def build_references(*,norms_file: Path,signal_file: Path,pfas_file: Path,background_file: Path,standards_output: Path,reference_output: Path) -> tuple[BuildValidationReport,BuildValidationReport]:
    for path in (norms_file,signal_file,pfas_file,background_file):
        if not Path(path).is_file(): raise DataArtifactNotFoundError(f"Verplicht referentiebestand niet gevonden: {path}",path=Path(path))
    norms=load_normen(Path(norms_file)); signals=load_signaleringswaarden(Path(signal_file)); pfas=load_pfas_ref(Path(pfas_file)); background=load_achtergrondcorrecties(Path(background_file))
    standards_df = norms.copy()
    signal_rows = pd.DataFrame({
        "substance": signals["stofnaam"],
        "parameter_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "standard_type": "SIGNALERINGSWAARDE",
        "standard_value": signals["signaleringswaarde"].astype("Float64"),
        "unit_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "compartment_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "condition_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "quantity_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "value_processing_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "category": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "norm_subgroup_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "compartment_subgroup_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "krw_status": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "source_row_number": pd.Series(pd.NA, index=signals.index, dtype="Int64"),
        "standard_id": pd.Series(
            [f"signal_{index + 1:07d}" for index in range(len(signals))],
            index=signals.index,
            dtype="string",
        ),
        "valid_from": pd.NaT,
        "valid_to": pd.NaT,
        "standard_status_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "background_inclusion_code": pd.Series(pd.NA, index=signals.index, dtype="string"),
        "standard_description": "Signaleringswaarde",
        "source_standard_name": "Signaleringswaarde",
    })
    standards_df = pd.concat([standards_df, signal_rows], ignore_index=True, sort=False)
    reference_frames=[]
    pfas_copy=pfas.copy(); pfas_copy["reference_type"]="PFAS"; reference_frames.append(pfas_copy)
    background_copy=background.copy(); background_copy["reference_type"]="ACHTERGROND"; reference_frames.append(background_copy)
    reference_df=pd.concat(reference_frames,ignore_index=True,sort=False)
    _atomic_parquet(standards_df,Path(standards_output)); _atomic_parquet(reference_df,Path(reference_output))
    sr=BuildValidationReport("chemistry_standards",len(norms)+len(signals),len(standards_df)); rr=BuildValidationReport("chemistry_reference",len(pfas)+len(background),len(reference_df)); sr.assert_balanced(); rr.assert_balanced(); return sr,rr
