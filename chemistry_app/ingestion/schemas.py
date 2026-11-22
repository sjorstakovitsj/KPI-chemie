"""CSV and Parquet data contracts plus validation-report models."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

class Severity(StrEnum):
    INFO="info"
    WARNING="warning"
    ERROR="error"

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: Severity
    rows: tuple[int, ...]=()
    columns: tuple[str, ...]=()

@dataclass(slots=True)
class ValidationReport:
    source: str
    row_count: int=0
    issues: list[ValidationIssue]=field(default_factory=list)
    def add(self, code: str, message: str, severity: Severity, *, rows: tuple[int,...]=(), columns: tuple[str,...]=()) -> None:
        self.issues.append(ValidationIssue(code,message,severity,rows,columns))
    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == Severity.ERROR for issue in self.issues)
    @property
    def errors(self) -> tuple[ValidationIssue,...]:
        return tuple(i for i in self.issues if i.severity == Severity.ERROR)
    @property
    def warnings(self) -> tuple[ValidationIssue,...]:
        return tuple(i for i in self.issues if i.severity == Severity.WARNING)
    def as_dict(self) -> dict[str, object]:
        return {"source":self.source,"row_count":self.row_count,"is_valid":self.is_valid,
                "issues":[{"code":i.code,"message":i.message,"severity":i.severity.value,"rows":list(i.rows),"columns":list(i.columns)} for i in self.issues]}

@dataclass(frozen=True, slots=True)
class CsvSchema:
    name: str
    required_columns: frozenset[str]
    key_columns: tuple[str,...]
    delimiter: str=","
    encoding: str="utf-8-sig"

SIGNALERINGSWAARDEN_SCHEMA: Final = CsvSchema("signaleringswaarden",frozenset({"stofnaam","signaleringswaarde"}),("stofnaam",))
ACHTERGRONDCORRECTIE_SCHEMA: Final = CsvSchema("achtergrondcorrecties",frozenset({"stofnaam","achtergrondconcentratie","eenheid"}),("stofnaam","eenheid"))
PFAS_SCHEMA: Final = CsvSchema("pfas_referentie",frozenset({"RPF","RBF"}),())
# Minimum legacy contract. Additional norm columns remain available.
NORMEN_SCHEMA: Final = CsvSchema("krw_normen",frozenset({"Stofnaam","Norm.code","Waarde","Eenheid"}),("Stofnaam","Norm.code","Compartiment.code","Hoedanigheid.code","Eenheid"))

@dataclass(frozen=True, slots=True)
class ParquetField:
    name: str
    dtype: str
    nullable: bool=True

MEASUREMENTS_PARQUET_SCHEMA: Final[tuple[ParquetField,...]] = (
    ParquetField("measurement_id","string",False), ParquetField("location_code","string",False),
    ParquetField("sample_date","date32",False), ParquetField("parameter_code","string",False),
    ParquetField("parameter_name","string",False), ParquetField("condition_code","string"),
    ParquetField("compartment_code","string"), ParquetField("unit_code","string"),
    ParquetField("numeric_value_reported","float64"), ParquetField("numeric_value_analysis","float64"),
    ParquetField("limit_symbol","string"), ParquetField("is_censored","bool",False),
    ParquetField("quality_code","string"), ParquetField("substance_group","string"),
)
STANDARDS_PARQUET_SCHEMA: Final[tuple[ParquetField,...]] = (
    ParquetField("parameter_code","string",False), ParquetField("compartment_code","string"),
    ParquetField("condition_code","string"), ParquetField("unit_code","string",False),
    ParquetField("standard_type","string",False), ParquetField("standard_value","float64",False),
    ParquetField("valid_from","date32"), ParquetField("valid_to","date32"),
)
REFERENCE_PARQUET_SCHEMA: Final[tuple[ParquetField,...]] = (
    ParquetField("parameter_code","string",False), ParquetField("normalized_name","string",False),
    ParquetField("substance_group","string"), ParquetField("is_pfas","bool",False),
    ParquetField("rpf","float64"), ParquetField("rbf","float64"),
)
