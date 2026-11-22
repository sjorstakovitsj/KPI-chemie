"""Generic annual-mean trend calculation for chemistry measurements."""
from __future__ import annotations
import numpy as np
import pandas as pd
from chemistry_app.domain.censored_values import CensoredValueStrategy, apply_censored_value_strategy, normalize_missing_measurement_values
from chemistry_app.domain.schemas import TREND_RESULT_COLUMNS, TrendDirection

def calculate_trends(
    measurements: pd.DataFrame, standards: pd.DataFrame | None=None, *,
    strategy: CensoredValueStrategy | str=CensoredValueStrategy.EXCLUDE,
    minimum_years: int=3, stable_tolerance: float=1e-12,
    location_column: str="Meetpunt", date_column: str="Datum", substance_column: str="Stof",
    value_column: str="Waarde", limit_symbol_column: str="Limietsymbool",
    standard_substance_column: str="Stof", jg_column: str="JG_MKN", max_projection_years: float=50.0,
) -> pd.DataFrame:
    """Calculate one slope per location/substance from annual arithmetic means."""
    if minimum_years<2: raise ValueError("minimum_years moet minimaal 2 zijn.")
    required={location_column,date_column,substance_column,value_column}; missing=sorted(required-set(measurements))
    if missing: raise KeyError("Ontbrekende trendkolommen: "+", ".join(missing))
    if measurements.empty: return pd.DataFrame(columns=TREND_RESULT_COLUMNS)
    calc=measurements.copy(); calc[date_column]=pd.to_datetime(calc[date_column],errors="coerce")
    if calc[date_column].isna().any(): raise ValueError("Trendanalyse bevat ongeldige of ontbrekende datums.")
    calc[value_column]=normalize_missing_measurement_values(calc[value_column])
    calc=apply_censored_value_strategy(calc,strategy,value_column=value_column,limit_symbol_column=limit_symbol_column,output_column="__analysis")
    calc["__year"]=calc[date_column].dt.year.astype(int); uncensored=(~calc["Is_Gecensureerd"]).groupby([calc[location_column],calc[substance_column]],observed=True).sum().to_dict()
    annual=calc.groupby([location_column,substance_column,"__year"],observed=True)["__analysis"].mean().dropna().reset_index()
    norm_map={}
    if standards is not None and not standards.empty:
        if not {standard_substance_column,jg_column}.issubset(standards): raise KeyError("Normtabel mist Stof of JG_MKN.")
        norm_map=standards.drop_duplicates(standard_substance_column).set_index(standard_substance_column)[jg_column].to_dict()
    elif jg_column in calc:
        norm_map=calc.drop_duplicates(substance_column).set_index(substance_column)[jg_column].to_dict()
    rows=[]
    for (location,substance),group in annual.groupby([location_column,substance_column],observed=True):
        group=group.sort_values("__year"); n=len(group)
        if n<minimum_years: continue
        x=group["__year"].to_numpy(dtype=float); y=group["__analysis"].to_numpy(dtype=float); slope,intercept=np.polyfit(x,y,1)
        direction=TrendDirection.STABLE if abs(slope)<=stable_tolerance else (TrendDirection.RISING if slope>0 else TrendDirection.DECLINING)
        norm=norm_map.get(substance,pd.NA); norm_value=None if pd.isna(norm) else float(norm); latest=float(y[-1]); exceeds=None if norm_value is None or norm_value<=0 else latest>norm_value
        years_to=None
        if norm_value is not None and norm_value>0:
            if latest>=norm_value and slope>0: years_to=0.0
            elif latest<norm_value and slope>stable_tolerance: years_to=max(0.0,(norm_value-latest)/slope)
            elif latest>norm_value and slope<-stable_tolerance: years_to=max(0.0,(norm_value-latest)/slope)
            if years_to is not None and years_to>max_projection_years: years_to=None
        rows.append({"Meetpunt":location,"Stof":substance,"Eerste_jaar":int(x[0]),"Laatste_jaar":int(x[-1]),"Startwaarde":float(y[0]),"Eindwaarde":latest,"Trendscore":float(slope),"Intercept":float(intercept),"Trendrichting":direction.value,"Aantal_jaren":n,"n_metingen_boven_rg":int(uncensored.get((location,substance),0)),"JG_MKN":norm_value,"Laatste_jaar_overschrijdt_JG_MKN":exceeds,"Tijd_tot_JG_MKN":years_to})
    return pd.DataFrame(rows,columns=TREND_RESULT_COLUMNS)

def rising_trends(results: pd.DataFrame) -> pd.DataFrame:
    return results.loc[results["Trendrichting"].eq(TrendDirection.RISING.value)].copy() if not results.empty else results.copy()

def declining_exceedances(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty: return results.copy()
    return results.loc[results["Trendrichting"].isin([TrendDirection.DECLINING.value,TrendDirection.STABLE.value]) & results["Laatste_jaar_overschrijdt_JG_MKN"].eq(True)].copy()
