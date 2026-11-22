"""Streamlit-independent loaders for chemistry reference CSV files."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from chemistry_app.config.settings import (ACHTERGRONDCORRECTIE_FILE_PATH,NORMEN_FILE_PATH,PFAS_FILE_PATH,SIGNALERINGSWAARDEN_FILE_PATH)
from chemistry_app.infrastructure.exceptions import (DataArtifactNotFoundError,DataArtifactReadError,DuplicateReferenceError,InvalidReferenceValueError,MissingColumnsError)
from chemistry_app.ingestion.normalization import normalize_columns,normalize_decimal,normalize_text_series,normalize_unit_series
from chemistry_app.ingestion.schemas import (ACHTERGRONDCORRECTIE_SCHEMA,NORMEN_SCHEMA,PFAS_SCHEMA,SIGNALERINGSWAARDEN_SCHEMA,CsvSchema,Severity,ValidationReport)


def _canonical_header(value: object) -> str:
    """Map common Dutch/legacy header spellings to canonical loader names."""
    text = str(value).strip().lstrip("\ufeff")
    key = (
        text.casefold()
        .replace("-", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )
    while "__" in key:
        key = key.replace("__", "_")
    aliases = {
        "stof": "Stof",
        "stofnaam": "Stofnaam",
        "jg_mkn": "JG_MKN",
        "jgmkn": "JG_MKN",
        "jaargemiddelde_mkn": "JG_MKN",
        "mac_mkn": "MAC_MKN",
        "macmkn": "MAC_MKN",
        "maximaal_aanvaardbare_concentratie": "MAC_MKN",
        "rpf": "RPF",
        "rbf": "RBF",
        "signaleringswaarde": "signaleringswaarde",
        "achtergrondconcentratie": "achtergrondconcentratie",
        "eenheid": "Eenheid",
    }
    return aliases.get(key, text)

def _read_csv(path: Path, schema: CsvSchema, *, normalize_headers: bool=True) -> tuple[pd.DataFrame,ValidationReport]:
    path=Path(path); report=ValidationReport(source=str(path))
    encodings = tuple(dict.fromkeys((schema.encoding, "utf-8", "cp1252")))
    frame = None
    decode_errors: list[str] = []
    try:
        for encoding in encodings:
            try:
                frame = pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    encoding=encoding,
                    dtype="string",
                )
                if encoding != schema.encoding:
                    report.add(
                        "encoding_fallback",
                        f"Bestand is gelezen met {encoding} in plaats van {schema.encoding}.",
                        Severity.WARNING,
                    )
                break
            except UnicodeDecodeError as exc:
                decode_errors.append(f"{encoding}: {exc}")
    except FileNotFoundError as exc:
        raise DataArtifactNotFoundError(f"Referentiebestand niet gevonden: {path}",path=path) from exc
    except (pd.errors.ParserError,OSError) as exc:
        raise DataArtifactReadError(f"Referentiebestand kan niet worden gelezen: {path}: {exc}",path=path) from exc
    if frame is None:
        details = " | ".join(decode_errors)
        raise DataArtifactReadError(
            f"Referentiebestand heeft geen ondersteunde tekstcodering: {path}. {details}",
            path=path,
        )
    frame.columns = pd.Index(_canonical_header(column) for column in frame.columns)
    report.row_count=len(frame)
    if normalize_headers: frame.columns=normalize_columns(frame.columns)
    required={c.casefold().replace("-","_").replace(" ","_") for c in schema.required_columns} if normalize_headers else set(schema.required_columns)
    missing=sorted(required-set(frame.columns))
    if missing:
        report.add("missing_columns","Ontbrekende kolommen: "+", ".join(missing),Severity.ERROR,columns=tuple(missing))
        raise MissingColumnsError(f"Ontbrekende kolommen in {schema.name}: {', '.join(missing)}",report=report)
    return frame,report

def _raise_invalid(report: ValidationReport, message: str) -> None:
    raise InvalidReferenceValueError(message,report=report)

def _validate_duplicates(frame: pd.DataFrame, keys: tuple[str,...], report: ValidationReport) -> None:
    if not keys: return
    mask=frame.duplicated(subset=list(keys),keep=False)
    if mask.any():
        rows=tuple((frame.index[mask]+2).astype(int).tolist())
        report.add("duplicate_key",f"Dubbele sleutel voor {', '.join(keys)}",Severity.ERROR,rows=rows,columns=keys)
        raise DuplicateReferenceError(f"Dubbele sleutel in {report.source} op CSV-regels: {', '.join(map(str,rows))}",report=report)

def load_signaleringswaarden(path: Path=SIGNALERINGSWAARDEN_FILE_PATH) -> pd.DataFrame:
    frame,report=_read_csv(path,SIGNALERINGSWAARDEN_SCHEMA)
    frame=frame[["stofnaam","signaleringswaarde"]].copy()
    frame["stofnaam"]=normalize_text_series(frame["stofnaam"])
    frame["signaleringswaarde"]=normalize_decimal(frame["signaleringswaarde"])
    invalid=frame["stofnaam"].isna()|frame["signaleringswaarde"].isna()|frame["signaleringswaarde"].le(0)
    if invalid.any():
        rows=tuple((frame.index[invalid]+2).astype(int).tolist()); report.add("invalid_value","Stofnaam ontbreekt of signaleringswaarde is niet positief.",Severity.ERROR,rows=rows)
        _raise_invalid(report,f"Ongeldige signaleringswaarde op CSV-regels: {', '.join(map(str,rows))}")
    _validate_duplicates(frame,SIGNALERINGSWAARDEN_SCHEMA.key_columns,report)
    frame.attrs["validation_report"]=report
    return frame

def load_achtergrondcorrecties(path: Path=ACHTERGRONDCORRECTIE_FILE_PATH) -> pd.DataFrame:
    frame,report=_read_csv(path,ACHTERGRONDCORRECTIE_SCHEMA)
    cols=["stofnaam","achtergrondconcentratie","eenheid"]; frame=frame[cols].copy()
    frame["stofnaam"]=normalize_text_series(frame["stofnaam"]); frame["eenheid"]=normalize_unit_series(frame["eenheid"])
    frame["achtergrondconcentratie"]=normalize_decimal(frame["achtergrondconcentratie"])
    invalid=frame["stofnaam"].isna()|frame["eenheid"].isna()|frame["achtergrondconcentratie"].isna()|frame["achtergrondconcentratie"].lt(0)
    if invalid.any():
        rows=tuple((frame.index[invalid]+2).astype(int).tolist()); report.add("invalid_value","Ongeldige achtergrondcorrectie.",Severity.ERROR,rows=rows)
        _raise_invalid(report,f"Ongeldige achtergrondcorrectie op CSV-regels: {', '.join(map(str,rows))}")
    _validate_duplicates(frame,ACHTERGRONDCORRECTIE_SCHEMA.key_columns,report)
    frame.attrs["validation_report"]=report
    return frame

def load_pfas_ref(path: Path=PFAS_FILE_PATH) -> pd.DataFrame:
    # Preserve original PFAS column names because existing pages may depend on them.
    frame,report=_read_csv(path,PFAS_SCHEMA,normalize_headers=False)
    frame.columns=frame.columns.astype(str).str.strip()
    missing=sorted(PFAS_SCHEMA.required_columns-set(frame.columns))
    if missing:
        report.add("missing_columns","Ontbrekende PFAS-kolommen: "+", ".join(missing),Severity.ERROR,columns=tuple(missing))
        raise MissingColumnsError(f"Ontbrekende PFAS-kolommen: {', '.join(missing)}",report=report)
    for column in ("RPF","RBF"):
        raw=frame[column].astype("string")
        frame[column]=normalize_decimal(raw)
        invalid=raw.notna() & raw.str.strip().ne("") & frame[column].isna()
        if invalid.any():
            rows=tuple((frame.index[invalid]+2).astype(int).tolist()); report.add("invalid_numeric",f"Ongeldige {column}-waarde.",Severity.ERROR,rows=rows,columns=(column,))
            _raise_invalid(report,f"Ongeldige {column}-waarde op CSV-regels: {', '.join(map(str,rows))}")
        missing_values=frame[column].isna()
        if missing_values.any():
            rows=tuple((frame.index[missing_values]+2).astype(int).tolist()); report.add("missing_numeric",f"Ontbrekende {column} is als 0 geïnterpreteerd voor legacycompatibiliteit.",Severity.WARNING,rows=rows,columns=(column,)); frame.loc[missing_values,column]=0.0
    frame.attrs["validation_report"]=report
    return frame

def load_normen(path: Path=NORMEN_FILE_PATH) -> pd.DataFrame:
    """Load the actual long-format KRW standards table.

    The source contains one row per standard, with ``Norm.code`` identifying
    JG-MKN or MAC and ``Waarde`` containing the numeric standard value.
    Context fields are retained so later matching can use parameter, compartment,
    condition and unit instead of substance name alone.
    """
    frame, report = _read_csv(path, NORMEN_SCHEMA, normalize_headers=False)
    frame.columns = frame.columns.astype(str).str.strip()

    required = {"Stofnaam", "Norm.code", "Waarde", "Eenheid"}
    missing = sorted(required - set(frame.columns))
    if missing:
        report.add(
            "missing_columns",
            "Ontbrekende normkolommen: " + ", ".join(missing),
            Severity.ERROR,
            columns=tuple(missing),
        )
        raise MissingColumnsError(
            f"Ontbrekende normkolommen: {', '.join(missing)}",
            report=report,
        )

    output = pd.DataFrame(index=frame.index)
    output["substance"] = normalize_text_series(frame["Stofnaam"])
    output["parameter_code"] = frame.get("Aquo-code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["standard_type"] = frame["Norm.code"].astype("string").str.strip().str.upper()
    output["standard_value"] = normalize_decimal(frame["Waarde"])
    output["unit_code"] = normalize_unit_series(frame["Eenheid"])
    output["compartment_code"] = frame.get("Compartiment.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["condition_code"] = frame.get("Hoedanigheid.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["quantity_code"] = frame.get("Grootheid.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["value_processing_code"] = frame.get("Waardebewerkingsmethode.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["category"] = frame.get("Categorie", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["norm_subgroup_code"] = frame.get("Normsubgroup.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["compartment_subgroup_code"] = frame.get("Compartimentsubgroep.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["krw_status"] = frame.get("KRW status", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["source_row_number"] = pd.Series(frame.index + 2, index=frame.index, dtype="int64")
    output["valid_from"] = pd.to_datetime(frame.get("Geldigheid.begindatum", pd.Series(pd.NA, index=frame.index)), errors="coerce", dayfirst=True)
    output["valid_to"] = pd.to_datetime(frame.get("Geldigheid.einddatum", pd.Series(pd.NA, index=frame.index)), errors="coerce", dayfirst=True)
    output["standard_status_code"] = frame.get("Normwaardestatus.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["background_inclusion_code"] = frame.get("Achtergrondconcentratie.code", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["standard_description"] = frame.get("Norm.omschrijving", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()
    output["source_standard_name"] = frame.get("Norm", pd.Series(pd.NA, index=frame.index)).astype("string").str.strip()

    # Only the two KRW norm types used by the application are published.
    output = output.loc[output["standard_type"].isin(["JG-MKN", "MAC"])].copy()
    output["standard_type"] = output["standard_type"].replace({"MAC": "MAC-MKN"})

    invalid = (
        output["substance"].isna()
        | output["standard_value"].isna()
        | output["standard_value"].lt(0)
        | output["unit_code"].isna()
    )
    if invalid.any():
        rows = tuple((output.index[invalid] + 2).astype(int).tolist())
        report.add(
            "invalid_standard",
            "Norm mist stof, waarde of eenheid, of heeft een negatieve waarde.",
            Severity.ERROR,
            rows=rows,
        )
        _raise_invalid(
            report,
            "Ongeldige KRW-norm op CSV-regels: " + ", ".join(map(str, rows[:20])),
        )

    # KRW Beheer kent legitiem meerdere normen per stof en normtype, onder meer
    # door normsubgroep, compartimentsubgroep, achtergrondstatus en geldigheid.
    # Daarom is de bronregel de technische sleutel. Alleen volledig identieke
    # semantische regels worden als duplicaat beschouwd en eenmaal gepubliceerd.
    semantic_columns = [
        column
        for column in output.columns
        if column not in {"source_row_number"}
    ]
    exact_duplicate_mask = output.duplicated(
        subset=semantic_columns,
        keep="first",
    )
    if exact_duplicate_mask.any():
        rows = tuple(output.loc[exact_duplicate_mask, "source_row_number"].astype(int))
        report.add(
            "exact_duplicate_standard",
            "Volledig identieke normregels zijn eenmaal gepubliceerd.",
            Severity.WARNING,
            rows=rows,
            columns=tuple(semantic_columns),
        )
        output = output.loc[~exact_duplicate_mask].copy()

    # Een stabiele unieke technische ID bewaart alle legitieme contextvarianten.
    output["standard_id"] = output["source_row_number"].map(
        lambda row_number: f"krw_{int(row_number):07d}"
    ).astype("string")

    output = output.reset_index(drop=True)
    output.attrs["validation_report"] = report
    return output

__all__=["load_signaleringswaarden","load_achtergrondcorrecties","load_pfas_ref","load_normen"]
