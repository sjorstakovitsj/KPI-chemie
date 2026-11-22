"""Explicit result schemas shared by chemistry domain calculations."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class StandardType(StrEnum):
    JG_MKN = "JG-MKN"
    MAC_MKN = "MAC-MKN"

class ComplianceStatus(StrEnum):
    COMPLIANT = "voldoet"
    EXCEEDS = "overschrijdt"
    NO_STANDARD = "geen_norm"
    NO_VALUE = "geen_waarde"

class TrendDirection(StrEnum):
    RISING = "stijgend"
    STABLE = "vlak"
    DECLINING = "dalend"
    INSUFFICIENT_DATA = "onvoldoende_data"

@dataclass(frozen=True, slots=True)
class ComplianceResult:
    location: str
    year: int
    substance: str
    standard_type: StandardType
    assessment_value: float | None
    standard_value: float | None
    status: ComplianceStatus
    exceedance_factor: float | None
    measurement_count: int

@dataclass(frozen=True, slots=True)
class TrendResult:
    location: str
    substance: str
    first_year: int
    last_year: int
    start_value: float
    end_value: float
    slope: float
    intercept: float
    direction: TrendDirection
    year_count: int
    uncensored_measurement_count: int
    jg_mkn: float | None
    latest_exceeds_jg_mkn: bool | None
    years_to_jg_mkn: float | None

COMPLIANCE_RESULT_COLUMNS = (
    "Meetpunt", "Jaar", "Stof", "Normtype", "Toetswaarde", "Normwaarde",
    "Status", "Overschrijdingsfactor", "Aantal_metingen",
)
TREND_RESULT_COLUMNS = (
    "Meetpunt", "Stof", "Eerste_jaar", "Laatste_jaar", "Startwaarde",
    "Eindwaarde", "Trendscore", "Intercept", "Trendrichting", "Aantal_jaren",
    "n_metingen_boven_rg", "JG_MKN", "Laatste_jaar_overschrijdt_JG_MKN",
    "Tijd_tot_JG_MKN",
)
