"""Application service for generic chemistry trends."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.domain.censored_values import CensoredValueStrategy
from chemistry_app.domain.trends import calculate_trends, declining_exceedances, rising_trends
@dataclass(frozen=True, slots=True)
class TrendAnalysis:
    all_trends: pd.DataFrame
    rising: pd.DataFrame
    declining_above_standard: pd.DataFrame

def analyse_trends(measurements: pd.DataFrame, standards: pd.DataFrame | None=None, *, strategy: CensoredValueStrategy | str=CensoredValueStrategy.EXCLUDE, minimum_years: int=3) -> TrendAnalysis:
    all_results=calculate_trends(measurements,standards,strategy=strategy,minimum_years=minimum_years)
    return TrendAnalysis(all_results,rising_trends(all_results),declining_exceedances(all_results))
