"""Legacy API adapters used exclusively by the temporary root ``utils.py``."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd
import streamlit as st
from chemistry_app.config.settings import BUILD_MANIFEST_PATH,MEASUREMENTS_PARQUET_PATH
from chemistry_app.domain.background_correction import apply_background_correction
from chemistry_app.domain.censored_values import parse_censored_strategy
from chemistry_app.domain.substance_groups import STOFGROEPEN_MAPPING,UITGESLOTEN_ELEMENTEN,match_stofgroep_optimized
from chemistry_app.ingestion.reference_csv import load_achtergrondcorrecties,load_pfas_ref,load_signaleringswaarden
from chemistry_app.infrastructure.legacy_adapter import LEGACY_MAP, query_legacy
from chemistry_app.infrastructure.measurements_repository import MeasurementsRepository
from chemistry_app.infrastructure.models import MeasurementFilters
from chemistry_app.services.compliance_service import compliance_details,summarize_compliance,assess_compliance
from chemistry_app.services.filter_service import get_filter_options as get_typed_filter_options
from chemistry_app.services.trend_service import analyse_trends
from chemistry_app.services.visualization_data_service import prepare_heatmap_data as _prepare_heatmap,prepare_sunburst_data as _prepare_sunburst
from chemistry_app.ui.charts import create_gauge
from chemistry_app.ui.sidebar_filters import render_shared_sidebar

@lru_cache(maxsize=4)
def _repository(parquet_path: str=MEASUREMENTS_PARQUET_PATH.as_posix(),manifest_path: str=BUILD_MANIFEST_PATH.as_posix()) -> MeasurementsRepository:
    return MeasurementsRepository(Path(parquet_path),Path(manifest_path))

def query_data(*,jaren=(),periodes=(),stoffen=(),meetpunten=(),stofgroepen=(),eenheden=(),kolommen=(),datum_vanaf=None,datum_tot=None,compartimenten=(),hoedanigheden=(),kwaliteitsoordelen=(),statussen=()) -> pd.DataFrame:
    filters=MeasurementFilters(years=tuple(jaren),periods=tuple(periodes),substances=tuple(stoffen),locations=tuple(meetpunten),substance_groups=tuple(stofgroepen),units=tuple(eenheden),compartments=tuple(compartimenten),conditions=tuple(hoedanigheden),quality_codes=tuple(kwaliteitsoordelen),statuses=tuple(statussen),date_from=None if datum_vanaf is None else pd.Timestamp(datum_vanaf).date(),date_to=None if datum_tot is None else pd.Timestamp(datum_tot).date())
    requested=tuple(kolommen) if kolommen else tuple(LEGACY_MAP.values())
    return query_legacy(_repository(),filters,columns=requested)

def load_data() -> pd.DataFrame:
    return query_data()

def get_filter_options() -> dict[str,tuple]:
    options=get_typed_filter_options(_repository())
    return {"jaren":options.years,"meetpunten":options.locations,"stoffen":options.substances,"stofgroepen":options.substance_groups,"eenheden":options.units,"compartimenten":options.compartments,"hoedanigheden":options.conditions,"kwaliteitsoordelen":options.quality_codes,"statussen":options.statuses}

def apply_achtergrondcorrectie(frame: pd.DataFrame) -> pd.DataFrame:
    return apply_background_correction(frame,load_achtergrondcorrecties())

def calculate_compliance(frame: pd.DataFrame) -> pd.DataFrame:
    return assess_compliance(frame)

def calculate_metrics(frame: pd.DataFrame,is_period_average: bool=False):
    results=assess_compliance(frame); summary=summarize_compliance(results)
    if is_period_average and not results.empty:
        years=max(int(results["Jaar"].nunique()),1); return summary.assessable_count/years,summary.exceedance_count/years,summary.exceedance_percentage
    return float(summary.assessable_count),float(summary.exceedance_count),summary.exceedance_percentage

def calculate_compliance_details(frame: pd.DataFrame) -> pd.DataFrame:
    return compliance_details(frame)

def calculate_trends(frame: pd.DataFrame,norm_lookup_df: pd.DataFrame|None=None,lt_optie: str="Sluit uit van berekening") -> pd.DataFrame:
    return analyse_trends(frame,norm_lookup_df,strategy=parse_censored_strategy(lt_optie),minimum_years=3).all_trends

def calculate_trends_optimized(frame: pd.DataFrame,lt_optie: str,norm_lookup_df: pd.DataFrame) -> pd.DataFrame:
    result=analyse_trends(frame,norm_lookup_df,strategy=parse_censored_strategy(lt_optie),minimum_years=3).rising.copy()
    return result.rename(columns={"Tijd_tot_JG_MKN":"Tijd_tot_JG_normoverschrijding"})

def calculate_declining_exceedances_optimized(frame: pd.DataFrame,lt_optie: str,norm_lookup_df: pd.DataFrame) -> pd.DataFrame:
    result=analyse_trends(frame,norm_lookup_df,strategy=parse_censored_strategy(lt_optie),minimum_years=3).declining_above_standard.copy()
    return result.rename(columns={"Tijd_tot_JG_MKN":"Tijd_tot_onder_norm"})

def prepare_heatmap_data(frame: pd.DataFrame):
    data=_prepare_heatmap(frame)
    if data.factors.empty: return None,None,[],[]
    return data.factors,data.labels,list(data.substances),list(data.locations)

def prepare_sunburst_data(failures: pd.DataFrame) -> pd.DataFrame:
    return _prepare_sunburst(failures).rows

def get_shared_sidebar_filters(*,toon_aggregatiemethode: bool=True):
    selection=render_shared_sidebar(_repository(),show_aggregation=toon_aggregatiemethode)
    if selection is None: return (),()
    return selection.filters.years,selection.filters.periods

def get_shared_sidebar(df_main: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.warning("Deze legacy-sidebar vervalt na migratie van de pages.")
    years=tuple(sorted(df_main["Datum"].dt.year.dropna().astype(int).unique(),reverse=True)) if not df_main.empty and "Datum" in df_main else ()
    selected=st.sidebar.multiselect("Selecteer gewenste jaren:",options=years,default=years,key="shared_jaren_filter")
    return df_main[df_main["Datum"].dt.year.isin(selected)].copy() if selected and "Datum" in df_main else df_main.copy()

