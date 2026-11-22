"""Manifest-based artifact versioning for cache invalidation."""
from __future__ import annotations
from functools import lru_cache
import json
from pathlib import Path
from chemistry_app.infrastructure.exceptions import DataArtifactNotFoundError, DataArtifactReadError

@lru_cache(maxsize=32)
def _read_manifest_cached(path: str, mtime_ns: int, size: int) -> dict[str,object]:
    try: return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError,ValueError,TypeError) as exc: raise DataArtifactReadError(f"Buildmanifest kan niet worden gelezen: {exc}",path=Path(path)) from exc

def read_manifest(path: Path) -> dict[str,object]:
    path=Path(path).resolve()
    if not path.is_file(): raise DataArtifactNotFoundError(f"Buildmanifest niet gevonden: {path}",path=path)
    stat=path.stat(); return _read_manifest_cached(path.as_posix(),stat.st_mtime_ns,stat.st_size)

def artifact_version(manifest_path: Path, artifact_name: str) -> str:
    manifest=read_manifest(manifest_path); build_id=str(manifest.get("build_id","")).strip()
    artifacts=manifest.get("artifacts",[])
    for artifact in artifacts if isinstance(artifacts,list) else []:
        if isinstance(artifact,dict) and artifact.get("name")==artifact_name:
            digest=str(artifact.get("sha256","")).strip()
            if digest: return f"{build_id}:{digest}"
    raise DataArtifactReadError(f"Artifact {artifact_name!r} ontbreekt in buildmanifest.",path=Path(manifest_path))
