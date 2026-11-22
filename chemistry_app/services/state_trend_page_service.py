from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from chemistry_app.config.periods import PERIODES
from chemistry_app.services.page_data_service import get_filter_options, query_data
PAGE_COLUMNS=("Datum","Meetpunt","Locatiecode","Stof","Stofgroep","Waarde","Eenheid","Limietsymbool","JG_MKN","MAC_MKN")
@dataclass(frozen=True)
class StateTrendData:
 measurements: pd.DataFrame
 substance_lookup: pd.DataFrame
 available_years: tuple[int,...]
def get_available_years(): return tuple(sorted({int(y) for y in get_filter_options().get("jaren",())},reverse=True))
def selected_months(periods): return frozenset(m for period in periods for m in PERIODES[period])
def use_seasonal_mann_kendall(periods): return bool(periods) and selected_months(periods)!=frozenset(range(1,13))
def load_state_trend_data(*,years,periods):
 measurements=query_data(jaren=years,periodes=periods,kolommen=PAGE_COLUMNS)
 lookup=query_data(kolommen=("Stof","Stofgroep","JG_MKN","MAC_MKN")).drop_duplicates()
 return StateTrendData(measurements,lookup,get_available_years())
