"""CLI orchestration for all chemistry data artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from chemistry_app.config.settings import (ACHTERGRONDCORRECTIE_FILE_PATH,DATA_FILE_PATH,NORMEN_FILE_PATH,PFAS_FILE_PATH,REFERENCE_DIR,SIGNALERINGSWAARDEN_FILE_PATH,PARQUET_DIR,ensure_data_directories)
from chemistry_app.pipelines.build_measurements import build_measurements
from chemistry_app.pipelines.build_references import build_references
from chemistry_app.pipelines.manifest import artifact_metadata,new_manifest,write_manifest_atomic

def build_all(*,source: Path=DATA_FILE_PATH,output_dir: Path=PARQUET_DIR,only: str="all",validate_only: bool=False) -> dict[str,object]:
    ensure_data_directories(); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    measurements=output_dir/"chemistry_measurements.parquet"; standards=output_dir/"chemistry_standards.parquet"; references=output_dir/"chemistry_reference.parquet"; manifest_path=output_dir/"build_manifest.json"
    manifest=new_manifest(Path(source)); reports=[]
    if only in {"all","measurements"}:
        report=build_measurements(Path(source),measurements); reports.append(report); manifest.artifacts.append(artifact_metadata("chemistry_measurements",measurements,report.output_rows))
    if only in {"all","references"}:
        sr,rr=build_references(norms_file=NORMEN_FILE_PATH,signal_file=SIGNALERINGSWAARDEN_FILE_PATH,pfas_file=PFAS_FILE_PATH,background_file=ACHTERGRONDCORRECTIE_FILE_PATH,standards_output=standards,reference_output=references); reports.extend([sr,rr]); manifest.artifacts.extend([artifact_metadata("chemistry_standards",standards,sr.output_rows),artifact_metadata("chemistry_reference",references,rr.output_rows)])
    manifest.validation={r.dataset:r.as_dict() for r in reports}
    if not validate_only: write_manifest_atomic(manifest,manifest_path)
    return manifest.as_dict()
def main() -> int:
    parser=argparse.ArgumentParser(description="Bouw chemische Parquet-artifacts zonder Streamlit.")
    parser.add_argument("--source",type=Path,default=DATA_FILE_PATH); parser.add_argument("--output-dir",type=Path,default=PARQUET_DIR); parser.add_argument("--only",choices=("all","measurements","references"),default="all"); parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args(); print(json.dumps(build_all(source=args.source,output_dir=args.output_dir,only=args.only,validate_only=args.validate_only),ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
