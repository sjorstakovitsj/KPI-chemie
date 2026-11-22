"""Application service for central chemical compliance assessment."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.domain.censored_values import CensoredValueStrategy
from chemistry_app.domain.compliance import calculate_compliance, compliance_failures

@dataclass(frozen=True, slots=True)
class ComplianceSummary:
    assessable_count: int
    exceedance_count: int
    exceedance_percentage: float

def assess_compliance(measurements: pd.DataFrame, *, strategy: CensoredValueStrategy | str=CensoredValueStrategy.EXCLUDE) -> pd.DataFrame:
    return calculate_compliance(measurements,strategy=strategy)

def summarize_compliance(results: pd.DataFrame) -> ComplianceSummary:
    if results.empty: return ComplianceSummary(0,0,0.0)
    assessable=results[results["Status"].isin(["voldoet","overschrijdt"])]
    exceedances=int(assessable["Status"].eq("overschrijdt").sum()); total=len(assessable)
    return ComplianceSummary(total,exceedances,(exceedances/total*100.0) if total else 0.0)

def compliance_details(measurements: pd.DataFrame, *, strategy: CensoredValueStrategy | str=CensoredValueStrategy.EXCLUDE) -> pd.DataFrame:
    results=assess_compliance(measurements,strategy=strategy)
    failures=compliance_failures(results)
    if failures.empty: return pd.DataFrame(columns=["Meetpunt","Jaar","Stof","Normtype","Factor"])
    return failures.rename(columns={"Overschrijdingsfactor":"Factor"})[["Meetpunt","Jaar","Stof","Normtype","Factor"]]
