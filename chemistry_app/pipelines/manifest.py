"""Build manifest creation and atomic publication."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json, os
from pathlib import Path
from typing import Any
PIPELINE_VERSION="1.0.0"; SCHEMA_VERSION="1.0.0"
def file_sha256(path: Path, chunk_size: int=1024*1024) -> str:
    digest=sha256()
    with Path(path).open("rb") as stream:
        while chunk:=stream.read(chunk_size): digest.update(chunk)
    return digest.hexdigest()
@dataclass(slots=True)
class ArtifactManifest:
    name: str; path: str; rows: int; bytes: int; sha256: str
@dataclass(slots=True)
class BuildManifest:
    build_id: str; built_at_utc: str; pipeline_version: str=PIPELINE_VERSION; schema_version: str=SCHEMA_VERSION
    source: dict[str,Any]=field(default_factory=dict); artifacts: list[ArtifactManifest]=field(default_factory=list); validation: dict[str,Any]=field(default_factory=dict)
    def as_dict(self) -> dict[str,Any]:
        value=asdict(self); value["artifacts"]=[asdict(a) for a in self.artifacts]; return value

def source_metadata(path: Path) -> dict[str,Any]:
    stat=Path(path).stat(); return {"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":file_sha256(path)}
def artifact_metadata(name: str,path: Path,rows: int) -> ArtifactManifest:
    return ArtifactManifest(name,str(path),rows,path.stat().st_size,file_sha256(path))
def new_manifest(source: Path) -> BuildManifest:
    now=datetime.now(timezone.utc); return BuildManifest(now.strftime("%Y%m%dT%H%M%SZ"),now.isoformat(),source=source_metadata(source))
def write_manifest_atomic(manifest: BuildManifest,path: Path) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(manifest.as_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(temporary,path)
