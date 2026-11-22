"""Page-facing chemistry data assembly from read-only repositories.

This is the temporary boundary for pages that still use Dutch display columns.
It performs repository queries, norm enrichment and background correction. It
contains no UI framework calls and never reads CSV files.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
import pandas as pd
from chemistry_app.config.settings import BUILD_MANIFEST_PATH,MEASUREMENTS_PARQUET_PATH,REFERENCE_PARQUET_PATH,STANDARDS_PARQUET_PATH
from chemistry_app.domain.background_correction import apply_background_correction
from chemistry_app.infrastructure.models import MeasurementFilters
from chemistry_app.infrastructure.measurements_repository import MeasurementsRepository
from chemistry_app.infrastructure.reference_repository import ReferenceRepository
from chemistry_app.infrastructure.standards_repository import StandardsRepository
from chemistry_app.services.filter_service import get_filter_options as _typed_filter_options

LEGACY_TO_CANONICAL={
    "Datum":"sample_date","Meetpunt":"location_name","Locatiecode":"location_code",
    "Stof":"parameter_name","Parametercode":"parameter_code","Stofgroep":"substance_group",
    "Waarde":"numeric_value_reported","Eenheid":"unit_code","Limietsymbool":"limit_symbol",
    "Latitude":"latitude","Longitude":"longitude","Compartiment":"compartment_code",
    "Hoedanigheid":"condition_code","Kwaliteitsoordeel":"quality_code","Status":"status",
}
CANONICAL_TO_LEGACY={value:key for key,value in LEGACY_TO_CANONICAL.items()}
DERIVED_COLUMNS={"Waarde_Origineel","JG_MKN","MAC_MKN","Achtergrondconcentratie","Achtergrondcorrectie_Toegepast"}
BASE_CONTEXT=("parameter_code","parameter_name","compartment_code","condition_code","unit_code")

@lru_cache(maxsize=2)
def repositories(
    measurements_path: str=MEASUREMENTS_PARQUET_PATH.as_posix(),
    standards_path: str=STANDARDS_PARQUET_PATH.as_posix(),
    reference_path: str=REFERENCE_PARQUET_PATH.as_posix(),
    manifest_path: str=BUILD_MANIFEST_PATH.as_posix(),
):
    manifest=Path(manifest_path)
    return (
        MeasurementsRepository(Path(measurements_path),manifest),
        StandardsRepository(Path(standards_path),manifest),
        ReferenceRepository(Path(reference_path),manifest),
    )

def get_filter_options() -> dict[str,tuple]:
    measurement_repo,_,_ = repositories(); options=_typed_filter_options(measurement_repo)
    return {"jaren":options.years,"meetpunten":options.locations,"stoffen":options.substances,"stofgroepen":options.substance_groups,"eenheden":options.units,"compartimenten":options.compartments,"hoedanigheden":options.conditions,"kwaliteitsoordelen":options.quality_codes,"statussen":options.statuses}

def _standard_table(repository: StandardsRepository) -> pd.DataFrame:
    available=repository.columns
    wanted=tuple(c for c in ("parameter_code","substance","standard_type","standard_value","unit_code","compartment_code","condition_code","valid_from","valid_to") if c in available)
    standards=repository.query(standard_types=("JG-MKN","MAC-MKN"),columns=wanted)
    return standards

def _attach_standards(frame: pd.DataFrame,repository: StandardsRepository) -> pd.DataFrame:
    standards=_standard_table(repository)
    result=frame.copy()
    if standards.empty:
        result["JG_MKN"]=pd.NA; result["MAC_MKN"]=pd.NA; return result
    # Normalize empty keys before choosing the identity field.
    for column in ("parameter_code", "substance", "compartment_code", "condition_code", "unit_code"):
        if column in standards.columns:
            standards[column] = standards[column].replace(r"^\s*$", pd.NA, regex=True)

    def has_usable_values(data: pd.DataFrame, column: str) -> bool:
        return column in data.columns and data[column].notna().any()

    # Prefer substance/parameter_name. The standards table contains a mixture of
    # filled and empty parameter codes, so choosing parameter_code when any code
    # exists would group all code-less substances into one false conflict.
    if has_usable_values(result, "parameter_name") and has_usable_values(standards, "substance"):
        standards = standards.rename(columns={"substance": "parameter_name"})
        identity_key = "parameter_name"
    elif has_usable_values(result, "parameter_code") and has_usable_values(standards, "parameter_code"):
        identity_key = "parameter_code"
    else:
        raise ValueError(
            "Normen kunnen niet aan metingen worden gekoppeld: er is geen bruikbare "
            "combinatie substance/parameter_name en ook geen parameter_code."
        )

    keys = [identity_key]
    keys.extend(
        column
        for column in ("compartment_code", "condition_code", "unit_code")
        if column in result.columns
        and column in standards.columns
        and standards[column].notna().any()
    )
    standards=standards.sort_values([c for c in ("valid_from","valid_to") if c in standards],na_position="first")
    # Exact duplicates with the same value are harmless. Report only contexts
    # where more than one distinct standard value remains.
    group_columns = keys + ["standard_type"]
    value_counts = (
        standards.groupby(group_columns, dropna=False)["standard_value"]
        .nunique(dropna=True)
    )
    conflict_keys = (
        value_counts[value_counts.gt(1)]
        .rename("number_of_values")
        .reset_index()
    )

    if not conflict_keys.empty:
        conflict_details = standards.merge(
            conflict_keys[group_columns],
            on=group_columns,
            how="inner",
            validate="many_to_many",
        )
        detail_columns = [
            column
            for column in (
                *group_columns,
                "standard_value",
                "valid_from",
                "valid_to",
                "substance",
            )
            if column in conflict_details.columns
        ]
        sort_columns = group_columns + [
            column
            for column in ("valid_from", "valid_to", "standard_value")
            if column in detail_columns
        ]
        conflict_details = (
            conflict_details.loc[:, detail_columns]
            .sort_values(sort_columns, na_position="first")
            .reset_index(drop=True)
        )

        conflict_path = Path("standards_conflicts.csv")
        conflict_details.to_csv(
            conflict_path,
            index=False,
            encoding="utf-8-sig",
        )

        print("\n--- CONFLICTERENDE NORMKOPPELINGEN ---")
        print(conflict_details.to_string(index=False))
        print(f"Details opgeslagen in: {conflict_path.resolve()}")
        print("--- EINDE CONFLICTEN ---\n")

        raise ValueError(
            "Meerdere verschillende normwaarden passen op dezelfde meetcontext. "
            f"Details zijn opgeslagen in {conflict_path}."
        )
    compact=standards.drop_duplicates(keys+["standard_type"],keep="last")
    pivot=compact.pivot(index=keys,columns="standard_type",values="standard_value").reset_index().rename(columns={"JG-MKN":"JG_MKN","MAC-MKN":"MAC_MKN"})
    result=result.merge(pivot,on=keys,how="left",validate="many_to_one")
    for column in ("JG_MKN","MAC_MKN"):
        if column not in result: result[column]=pd.NA
    return result

def _apply_background(
    frame: pd.DataFrame,
    repository: ReferenceRepository,
) -> pd.DataFrame:
    """Apply background correction using the available reference-name column.

    DuckDB treats ``Stofnaam`` and ``stofnaam`` case-insensitively. Because the
    combined reference Parquet contains both PFAS and background reference
    schemas, DuckDB exposes the latter as ``stofnaam_1``. Normalize that
    infrastructure detail back to the domain contract ``stofnaam`` here.
    """
    columns = repository.columns
    substance_column = (
        "stofnaam"
        if "stofnaam" in columns
        else "stofnaam_1"
        if "stofnaam_1" in columns
        else None
    )
    required_columns = {
        "reference_type",
        "eenheid",
        "achtergrondconcentratie",
    }

    if substance_column is None or not required_columns.issubset(columns):
        result = frame.copy()
        result["Waarde_Origineel"] = result["Waarde"]
        result["Achtergrondconcentratie"] = pd.NA
        result["Achtergrondcorrectie_Toegepast"] = False
        return result

    refs = repository.query(
        reference_types=("ACHTERGROND",),
        columns=(
            "reference_type",
            substance_column,
            "eenheid",
            "achtergrondconcentratie",
        ),
    ).rename(columns={substance_column: "stofnaam"})

    if refs.empty:
        result = frame.copy()
        result["Waarde_Origineel"] = result["Waarde"]
        result["Achtergrondconcentratie"] = pd.NA
        result["Achtergrondcorrectie_Toegepast"] = False
        return result

    return apply_background_correction(
        frame,
        refs,
        substance_column="Stof",
        unit_column="Eenheid",
        value_column="Waarde",
    )


def _attach_display_location(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing display names from location codes without changing row count."""
    result = frame.copy()

    if "Meetpunt" not in result.columns:
        return result

    result["Meetpunt"] = result["Meetpunt"].replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

    if "Locatiecode" in result.columns:
        result["Meetpunt"] = result["Meetpunt"].fillna(
            result["Locatiecode"]
        )

    return result

def query_data(*,jaren=(),periodes=(),stoffen=(),meetpunten=(),stofgroepen=(),eenheden=(),kolommen=(),datum_vanaf=None,datum_tot=None,compartimenten=(),hoedanigheden=(),kwaliteitsoordelen=(),statussen=()) -> pd.DataFrame:
    measurement_repo,standards_repo,reference_repo=repositories()
    requested=tuple(kolommen)
    canonical_requested={LEGACY_TO_CANONICAL[c] for c in requested if c in LEGACY_TO_CANONICAL}
    canonical_requested.update(BASE_CONTEXT)
    if any(c in DERIVED_COLUMNS for c in requested): canonical_requested.update(("numeric_value_reported","parameter_name","unit_code"))
    filters=MeasurementFilters(years=tuple(jaren),periods=tuple(periodes),substances=tuple(stoffen),locations=tuple(meetpunten),substance_groups=tuple(stofgroepen),units=tuple(eenheden),compartments=tuple(compartimenten),conditions=tuple(hoedanigheden),quality_codes=tuple(kwaliteitsoordelen),statuses=tuple(statussen),date_from=None if datum_vanaf is None else pd.Timestamp(datum_vanaf).date(),date_to=None if datum_tot is None else pd.Timestamp(datum_tot).date())
    frame=measurement_repo.query(filters,columns=tuple(sorted(canonical_requested)))
    frame=_attach_standards(frame,standards_repo)
    frame=frame.rename(columns=CANONICAL_TO_LEGACY)
    frame=_attach_display_location(frame)
    if "Waarde" in frame:
        frame=_apply_background(frame,reference_repo)
    if requested:
        for column in requested:
            if column not in frame: frame[column]=pd.NA
        frame=frame.loc[:,list(requested)]
    return frame

