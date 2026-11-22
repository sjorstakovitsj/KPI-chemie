"""Central file-system configuration for the chemistry application.

Canonical data locations live below ``data/``. During the migration, existing
root-level files remain supported so the current ``utils.py`` keeps working.
Remove the legacy fallback after all data files have moved to the new layout.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
SOURCE_DIR = DATA_DIR / "source"
REFERENCE_DIR = DATA_DIR / "reference"
PARQUET_DIR = DATA_DIR / "parquets"


def _canonical_or_legacy(canonical: Path, legacy: Path) -> Path:
    """Return canonical path, falling back to an existing legacy file.

    The fallback is temporary and preserves compatibility during migration.
    New builds should always write to the canonical path.
    """
    return canonical if canonical.exists() or not legacy.exists() else legacy


DATA_FILE_PATH = _canonical_or_legacy(
    SOURCE_DIR / "IJG Chemie.csv",
    BASE_DIR / "IJG Chemie.csv",
)
NORMEN_FILE_PATH = _canonical_or_legacy(
    REFERENCE_DIR / "KRW stoffen koppeltabel.csv",
    BASE_DIR / "KRW stoffen koppeltabel.csv",
)
PFAS_FILE_PATH = _canonical_or_legacy(
    REFERENCE_DIR / "PFAS PEQ koppeltabel.csv",
    BASE_DIR / "PFAS PEQ koppeltabel.csv",
)
SIGNALERINGSWAARDEN_FILE_PATH = _canonical_or_legacy(
    REFERENCE_DIR / "Signaleringswaarden koppeltabel.csv",
    BASE_DIR / "Signaleringswaarden koppeltabel.csv",
)
ACHTERGRONDCORRECTIE_FILE_PATH = _canonical_or_legacy(
    REFERENCE_DIR / "Achtergrondcorrectie koppeltabel.csv",
    BASE_DIR / "Achtergrondcorrectie koppeltabel.csv",
)
PARQUET_FILE_PATH = _canonical_or_legacy(
    PARQUET_DIR / "metingen.parquet",
    DATA_DIR / "metingen.parquet",
)
METADATA_FILE_PATH = _canonical_or_legacy(
    PARQUET_DIR / "metadata.json",
    DATA_DIR / "metadata.json",
)


MEASUREMENTS_PARQUET_PATH = PARQUET_DIR / "chemistry_measurements.parquet"
STANDARDS_PARQUET_PATH = PARQUET_DIR / "chemistry_standards.parquet"
REFERENCE_PARQUET_PATH = PARQUET_DIR / "chemistry_reference.parquet"
BUILD_MANIFEST_PATH = PARQUET_DIR / "build_manifest.json"

def ensure_data_directories() -> None:
    """Create canonical data directories for build scripts."""
    for directory in (SOURCE_DIR, REFERENCE_DIR, PARQUET_DIR):
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "BASE_DIR", "DATA_DIR", "SOURCE_DIR", "REFERENCE_DIR", "PARQUET_DIR",
    "DATA_FILE_PATH", "NORMEN_FILE_PATH", "PFAS_FILE_PATH",
    "SIGNALERINGSWAARDEN_FILE_PATH", "ACHTERGRONDCORRECTIE_FILE_PATH",
    "PARQUET_FILE_PATH", "METADATA_FILE_PATH", "MEASUREMENTS_PARQUET_PATH",
    "STANDARDS_PARQUET_PATH", "REFERENCE_PARQUET_PATH", "BUILD_MANIFEST_PATH",
    "ensure_data_directories",
]
