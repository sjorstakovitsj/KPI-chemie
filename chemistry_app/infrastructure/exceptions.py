"""Application-specific exception hierarchy."""
from __future__ import annotations
from pathlib import Path

class ChemistryAppError(Exception):
    """Base exception for expected chemistry application failures."""

class ConfigurationError(ChemistryAppError):
    """Raised when application configuration is invalid."""

class DataArtifactError(ChemistryAppError):
    """Base class for data-artifact failures."""
    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)

class DataArtifactNotFoundError(DataArtifactError, FileNotFoundError):
    """Raised when a required CSV, Parquet, or manifest is unavailable."""

class DataArtifactReadError(DataArtifactError):
    """Raised when an artifact exists but cannot be read safely."""

class ReferenceTableValidationError(ChemistryAppError, ValueError):
    """Raised when a reference table violates its data contract."""
    def __init__(self, message: str, *, report: object | None = None) -> None:
        self.report = report
        super().__init__(message)

class MissingColumnsError(ReferenceTableValidationError):
    """Raised when required columns are absent."""

class DuplicateReferenceError(ReferenceTableValidationError):
    """Raised when a reference key is not unique."""

class InvalidReferenceValueError(ReferenceTableValidationError):
    """Raised when one or more reference values are invalid."""

class MeasurementSchemaError(ChemistryAppError, ValueError):
    """Raised when measurement data violates the expected schema."""

class ParquetSchemaError(ChemistryAppError, ValueError):
    """Raised when a Parquet artifact violates the expected schema."""

class UnknownPeriodError(ChemistryAppError, ValueError):
    """Raised when an unsupported period name is requested."""
