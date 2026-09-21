# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- CONSTANTEN ---
DATA_FILE_PATH = 'IJG Chemie.csv'
NORMEN_FILE_PATH = 'KRW stoffen koppeltabel.csv'
PFAS_FILE_PATH = 'PFAS PEQ koppeltabel.csv'
SIGNALERINGSWAARDEN_FILE_PATH = 'Signaleringswaarden koppeltabel.csv'
ACHTERGRONDCORRECTIE_FILE_PATH = 'Achtergrondcorrectie koppeltabel.csv'

STOFGROEPEN_MAPPING = {
    'PFAS': [
        'perfluor', 'genx', 'adona', 'pfhpa', 'fluortelomeer', 'pfas', '9-chloorhexadecaanfluor-3-oxanon-1-sulfonzuur',
        'trifluor', 'tridecafluor', '10:2', '8:2', '6:2', '4:2', '11-chlooreicosafluor-3-oxaundecaan-1-sulfonzuur',
        'som hexadecafluor-2-deceenzuur-isomerenâ', "2,3,3,3-tetrafluorpropaanzuur", "ammonium 2,3,3,3-tetrafluor-2-(heptafluorpropoxy)-propanoaat"
    ],
    'PAKs/PCBs/PBDEs': [
        'naftaleen', 'antraceen', 'fenantreen', 'fluorantheen', 'benzo(a)', 'benzo(g', 'benzo(k',
        'chryseen', 'pyreen', 'dibenzo', 'indeno', 'benzo(b)', 'pcb', 'broomdiphenylether',
        'broomdifenylether', 'chloorbifenyl', 'acenaftyleen', 'acenafteen', 'fluoreen', "3,4,4',5-tetrachlorobifenyl",
        "2,3,3',4,4',5,5 '-heptachlorobifenyl", 'som 29 dioxines en dioxineachtige verbindingen', "som pbb153 en pbde154",
        "som hbcd (technisch mengsel, niet-gespecif. broom-posities)"
    ],
    'Bestrijdingsmiddelen': [
        'glyfosaat', 'ampa', 'metolachloor', 'imidacloprid', 'mcpa', 'mecoprop', 'terbutylazine',
        'abamectine', 'aclonifen', 'alachloor', 'aldrin', 'endosulfan', 'hexachloor', 'atrazine',
        'bentazon', 'bifenox', 'chloorfenvinfos', 'chloortoluron', 'chloridazon', 'heptachloor',
        'cumafos', 'cypermethrin', 'diazinon', 'dichloorvos', 'dicofol', 'dieldrin', 'dimethenamid',
        'dimethoaat', 'dinoseb', 'diuron', 'dodine', 'endrin', 'azinfos', 'chloorpyrifos',
        'fenamifos', 'fenoxycarb', 'irgarol', 'isodrin', 'isoproturon', 'linuron', 'malathion',
        'methabenzthiazuron', 'metazachloor', 'metsulfuron', 'pirimifos', 'mevinfos', 'monolinuron',
        'pirimicarb', 'propazine', 'propiconazol', 'pyrazofos', 'pyridaben', 'pyriproxyfen',
        'quinoxyfen', 'simazine', 'teflubenzuron', 'terbutrin', 'thiacloprid', 'tolclofos',
        'triazofos', 'fenoxyazijnzuur', 'fenoxyboterzuur', 'fenoxypropionzuur', 'ddt', 'ddd', 'dde',
        'dinitro-o-cresol', 'aminomethyl', 'amisulpride', 'deltamethrin', 'diflufenican',
        'esfenvaleraat', 'parathion', 'fenitrothion', 'fenthion', 'fipronil', 'fluconazol',
        'glufosinaat', 'cyhalothrin', 'trifluraline', '3-(hydroxymethylfosfinoyl)propionzuur',
        'dinoterb', 'heptenofos', 'metabenzthiazuron', "4,4'-dichloordifenyltrichloorethaan",
        "2,4'-dichloordifenyltrichloorethaan", 'penconazool', 'acetamiprid', 'thiamethoxam', 'methiocarb',
         'cyprodinil', 'bromacil', 'diethyltoluamide', 'chloorprofam', 'propyzamide', 'propamocarb', 'propoxur',
         'prosulfocarb', 'etridiazol', 'ethofumesaat', 'carbendazim', 'clomazon', '2,6-dichloorbenzamide',
         'aldicarb', 'triallaat', 'triclosan', 'imazalil', 'ipconazole', 'prochloraz', 'tebuconazol', 
         'tetraconazool', 'metaflumizon', 'dimoxystrobine', 'metconazool', 'clothianidine',
         'cyazofamide', 'bromuconazool', 'difenoconazool', 'amisulbrom', 'etoxazool', 'ketoconazol',
         'epoxiconazool', 'mefentrifluconazool', 'triticonazool', 'dichlobenil', '1,2,4-triazool'
    ],
    'Geneesmiddelen': [
        'diclofenac', 'carbamazepine', 'metformine', 'tramadol', 'paracetamol', 'gadobutrol',
        'gadopentetaat', 'gadoteraat', 'gadoteridol', 'ibuprofen', 'amidotrizo', 'amoxicilline',
        'atenolol', 'azitromycine', 'azoxystrobin', 'bezafibraat', 'ciprofloxacine', 'claritromycine',
        'clindamycine', 'clofibraat', 'clofibrinezuur', 'clozapine', 'desvenlafaxine', 'dimetridazol',
        'dipyridamol', 'erytromycine', 'fenazon', 'fenofibraat', 'furosemide', 'gemfibrozil',
        'hydrochloorthiazide', 'ifosfamide', 'irbesartan', 'johexol', 'jomeprol', 'jopamidol',
        'jopromide', 'joxitalaminezuur', 'ketoprofen', 'levonorgestrel', 'lidoca', 'lincomycine',
        'losartan', 'metoprolol', 'miconazol', 'naproxen', 'norethisteron', 'ofloxacine', 'oxazepam',
        'oxybenzone', 'pentoxifylline', 'pipamperon', 'primidon', 'propranolol', 'sotalol',
        'sulfadiazine', 'sulfadimidine', 'sulfamethoxazol', 'sulfapyridine', 'sulfaquinoxaline',
        'tiamuline', 'trimethoprim', 'valsartan', 'venlafaxine', 'chlooramfenicol', 'cyclofosfamide',
        'guanylureum', 'avobenzone', 'octocrilene', 'paroxetine', 'fluoxetine', 'fenofibrinezuur',
        'gabapentine', 'gadoversetamide', 'gadodiamide', 'citalopram', 'valium', 'candesartan',
        'bisoprolol', 'gadoteerzuur', 'gadoxeetzuur', 'gadobeenzuur', 'famoxadone', 'acetylcedreen',
        'sandacanol', 'oxytetracycline', 'clotrimazol', '17beta-estradiol', 'oestron', 'ethinylestradiol',
        'amiodaron', 'climbazole', 'norfloxacine', 'tetracycline', 'tylosine', 'gadopentetinezuur'
    ],

    'Vluchtige organische stoffen': [
        'benzeen', 'tolueen', 'etylbenzeen', 'xyleen', 'styreen', 'chloorbenzeen', 'chlooretheen',
        'dichloorethaan', 'dichloorpropeen', 'cumeen', 'cyclohexaan', 'methaan', 'dicyclopentadieen',
        'ether', 'disulfide', 'hydrine', 'etheen', 'chloortolueen', 'propylbenzeen', 'tetrahydrofuran',
        '1,2-dimethoxyethaan', '1,1,2,2-tetrachloorethaan', '1,3-dichloorpropaan', '1,2-dichloorpropaan',
        '3-chloorpropeen', '1,2,3-trichloorpropaan', '1,1,1-trichloorethaan', '1,1,2-trichloorethaan',
        'tetrachloorethaan', 'propanol'
    ],
    'Industrie & overigen': [
        'bisfenol', 'chloorbenzeen', 'chloorfenol', 'nitrofenol', 'dtpa', 'methacrylaat', 'nitrilotriazijnzuur',
        'edta', 'pyrazol', 'melamine', 'difenol', 'cyaanguanidine', 'cyanuurzuur', 'urotropine',
        'ftalaat', 'acesulfaam', 'cyclamaat', 'saccharine', 'sucralose', 'tributylfosfaat', 'vinylchloride',
        '4-tertiair-octylfenol', 'som 4-nonylfenol-isomeren (vertakt)', 'melamine', 'trifenylfosfaat',
        'trifenylfosfineoxide', '5-acetyl-1,1,2,3,3,6-hexamethylindaan', 'verdyl acetaat', 'tonalide',
        'traseolide', 'verdox', 'celestolide', 'galaxolide', 'antrachinon', 'cashmeran', 'isoforon',
        '1,4-dioxaan', 'chloorxylenol', 'amberonne', 'hexahydrohexamethylcyclopentabenzopyran (hhcb)',
        'triethylfosfaat', 'tris(2-butoxyethyl)fosfaat', '1,2,3-benzotriazool', 'som 4- en 5-methyl-1h-benzotriazool'
    ],
    'Nutriënten & algemeen': [
        'fluoride', 'zuurstof', 'chlorofyl', 'silicium', 'sulfaat', 'koolstof', 'stikstof',
        'nitraat', 'nitriet', 'ammonium', "fosfor", "fosfaat", 'chloride', 'zwevende stof',
        'hardheid', 'temperatuur', 'zuurgraad', 'geleidbaarheid', 'gloeirest', 'onopgeloste',
        'doorzicht', 'saliniteit', 'troebelheid', 'cyanide', 'bicarbonaat', 'waterstofcarbonaat',
        'extinctie', 'kleur', 'geur', 'olie', 'schuim', 'vuil', 'escherichia coli', 'intestinale enterococcen'
    ],
    'Metalen & elementen': [
        'aluminium', 'antimoon', 'arseen', 'barium', 'beryllium', 'boor', 'cadmium', 'calcium',
        'cerium', 'cesium', 'chroom', 'cobalt', 'kobalt', 'dysprosium', 'erbium', 'europium',
        'kalium', 'koper', 'kwik', 'lood', 'gadolinium', 'gallium', 'hafnium', 'magnesium',
        'mangaan', 'molybdeen', 'natrium', 'nikkel', 'seleen', 'strontium', 'thallium', 'tin',
        'titanium', 'uranium', 'vanadium', 'ijzer', 'zilver', 'zink', 'holmium', 'indium',
        'lanthaan', 'lithium', 'lutetium', 'neodymium', 'niobium', 'platina', 'praseodymium',
        'rubidium', 'samarium', 'tantalium', 'tellurium', 'terbium', 'thorium', 'thulium',
        'wolfraam', 'ytterbium', 'yttrium', 'zirkonium', 'titaan', 'scandium', 'arsenaat',
        'arseniet', 'selenaat', 'seleniet', 'totaal beta', 'rest beta', 'totaal alfa', 'tritium'
    ],
}

UITGESLOTEN_ELEMENTEN = [
    'aluminium', 'ammonium', 'antimoon', 'barium', 'boor', 'rubidium',
    'cerium', 'chloride', 'calcium','magnesium', 'mangaan', 'molybdeen', 'natrium',
    'kalium', 'silicium', 'strontium', 'gadolinium', 'titanium', 'ijzer', 'zink', 'koolstof organisch',
    'lithium', 'nitraat', 'nitriet', 'siliciumdioxide', 'sulfaat', 'titaan', 'scandium', 'chlorofyl-a',
    'som extraheerbare organische halogeenverbindingen', 'dysprosium', 'cesium', 'ytterbium'
]

# --- FUNCTIES ---

@st.cache_data
def load_signaleringswaarden() -> pd.DataFrame:
    """Laadt en valideert stofspecifieke signaleringswaarden in ug/l."""
    vereiste_kolommen = {'stofnaam', 'signaleringswaarde'}

    try:
        df_signalering = pd.read_csv(
            SIGNALERINGSWAARDEN_FILE_PATH,
            delimiter=',',
            encoding='utf-8-sig',
            dtype='string',
            low_memory=False,
        )
    except FileNotFoundError:
        st.error(
            "Signaleringswaardentabel niet gevonden op pad: "
            f"{SIGNALERINGSWAARDEN_FILE_PATH}. Voor niet-uitgesloten stoffen "
            "zonder JG-MKN wordt daarom de generieke waarde van 0,1 ug/l gebruikt."
        )
        return pd.DataFrame(columns=['stofnaam', 'signaleringswaarde'])
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Signaleringswaardentabel kan niet als CSV worden gelezen. "
            "Controleer komma's en aanhalingstekens."
        ) from exc

    df_signalering.columns = (
        df_signalering.columns.astype(str).str.strip().str.casefold()
    )
    ontbrekende_kolommen = vereiste_kolommen - set(df_signalering.columns)
    if ontbrekende_kolommen:
        raise ValueError(
            "Ontbrekende kolommen in de signaleringswaardentabel: "
            + ", ".join(sorted(ontbrekende_kolommen))
        )

    df_signalering = df_signalering[
        ['stofnaam', 'signaleringswaarde']
    ].copy()
    df_signalering['stofnaam'] = (
        df_signalering['stofnaam'].astype('string').str.strip().str.casefold()
    )
    df_signalering['signaleringswaarde'] = (
        df_signalering['signaleringswaarde']
        .astype('string')
        .str.strip()
        .str.replace(',', '.', regex=False)
    )
    df_signalering['signaleringswaarde'] = pd.to_numeric(
        df_signalering['signaleringswaarde'], errors='coerce'
    )

    ongeldige_rijen = (
        df_signalering['stofnaam'].isna()
        | df_signalering['stofnaam'].eq('')
        | df_signalering['signaleringswaarde'].isna()
        | df_signalering['signaleringswaarde'].le(0)
    )
    if ongeldige_rijen.any():
        regelnummers = (df_signalering.index[ongeldige_rijen] + 2).tolist()
        raise ValueError(
            "Ongeldige stofnaam of signaleringswaarde op CSV-regel(s): "
            + ", ".join(map(str, regelnummers))
        )

    dubbele_stoffen = df_signalering.loc[
        df_signalering['stofnaam'].duplicated(keep=False), 'stofnaam'
    ].unique()
    if len(dubbele_stoffen) > 0:
        raise ValueError(
            "Dubbele stoffen in de signaleringswaardentabel: "
            + ", ".join(sorted(map(str, dubbele_stoffen)))
        )

    return df_signalering

@st.cache_data
def load_achtergrondcorrecties() -> pd.DataFrame:
    """Laadt en valideert natuurlijke achtergrondconcentraties per stof."""
    kolommen = ['stofnaam', 'achtergrondconcentratie', 'eenheid']
    vereiste_kolommen = set(kolommen)

    try:
        df_achtergrond = pd.read_csv(
            ACHTERGRONDCORRECTIE_FILE_PATH,
            delimiter=',',
            encoding='utf-8-sig',
            dtype='string',
            low_memory=False,
        )
    except FileNotFoundError:
        st.warning(
            "Achtergrondcorrectietabel niet gevonden op pad: "
            f"{ACHTERGRONDCORRECTIE_FILE_PATH}. Er wordt geen "
            "achtergrondcorrectie toegepast."
        )
        return pd.DataFrame(columns=kolommen)
    except pd.errors.ParserError as exc:
        raise ValueError(
            "Achtergrondcorrectietabel kan niet als CSV worden gelezen. "
            "Controleer komma's en aanhalingstekens."
        ) from exc

    df_achtergrond.columns = (
        df_achtergrond.columns.astype(str).str.strip().str.casefold()
    )
    ontbrekende_kolommen = vereiste_kolommen - set(df_achtergrond.columns)
    if ontbrekende_kolommen:
        raise ValueError(
            "Ontbrekende kolommen in de achtergrondcorrectietabel: "
            + ", ".join(sorted(ontbrekende_kolommen))
        )

    df_achtergrond = df_achtergrond[kolommen].copy()
    df_achtergrond['stofnaam'] = (
        df_achtergrond['stofnaam'].astype('string').str.strip().str.casefold()
    )
    df_achtergrond['eenheid'] = (
        df_achtergrond['eenheid']
        .astype('string')
        .str.strip()
        .str.casefold()
        .str.replace('µ', 'u', regex=False)
        .str.replace('μ', 'u', regex=False)
    )
    df_achtergrond['achtergrondconcentratie'] = (
        df_achtergrond['achtergrondconcentratie']
        .astype('string')
        .str.strip()
        .str.replace(',', '.', regex=False)
    )
    df_achtergrond['achtergrondconcentratie'] = pd.to_numeric(
        df_achtergrond['achtergrondconcentratie'], errors='coerce'
    )

    ongeldige_rijen = (
        df_achtergrond['stofnaam'].isna()
        | df_achtergrond['stofnaam'].eq('')
        | df_achtergrond['eenheid'].isna()
        | df_achtergrond['eenheid'].eq('')
        | df_achtergrond['achtergrondconcentratie'].isna()
        | df_achtergrond['achtergrondconcentratie'].lt(0)
    )
    if ongeldige_rijen.any():
        regelnummers = (df_achtergrond.index[ongeldige_rijen] + 2).tolist()
        raise ValueError(
            "Ongeldige achtergrondcorrectie op regel(s): "
            + ", ".join(map(str, regelnummers))
        )

    dubbele_sleutels = df_achtergrond.duplicated(
        subset=['stofnaam', 'eenheid'], keep=False
    )
    if dubbele_sleutels.any():
        dubbele_combinaties = (
            df_achtergrond.loc[dubbele_sleutels, ['stofnaam', 'eenheid']]
            .astype(str)
            .agg(' / '.join, axis=1)
            .unique()
        )
        raise ValueError(
            "Dubbele stof/eенheid-combinaties in de achtergrondcorrectietabel: "
            + ", ".join(sorted(dubbele_combinaties))
        )

    return df_achtergrond


def apply_achtergrondcorrectie(df: pd.DataFrame) -> pd.DataFrame:
    """Past achtergrondcorrectie toe en bewaart de oorspronkelijke meetwaarde."""
    resultaat = df.copy()
    resultaat['Waarde_Origineel'] = resultaat['Waarde'].copy()
    resultaat['Achtergrondconcentratie'] = np.nan
    resultaat['Achtergrondcorrectie_Toegepast'] = False

    df_achtergrond = load_achtergrondcorrecties()
    if df_achtergrond.empty:
        return resultaat

    basisstofnaam = (
        resultaat['Stof']
        .astype(str)
        .str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True)
        .str.strip()
        .str.casefold()
    )
    eenheid_norm = (
        resultaat['Eenheid']
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace('µ', 'u', regex=False)
        .str.replace('μ', 'u', regex=False)
    )

    achtergrond_map = df_achtergrond.set_index(
        ['stofnaam', 'eenheid']
    )['achtergrondconcentratie']
    koppelsleutel = pd.MultiIndex.from_arrays(
        [basisstofnaam, eenheid_norm], names=['stofnaam', 'eenheid']
    )
    gekoppelde_achtergrond = pd.Series(
        achtergrond_map.reindex(koppelsleutel).to_numpy(),
        index=resultaat.index,
        dtype='float64',
    )
    masker = gekoppelde_achtergrond.notna()

    resultaat.loc[masker, 'Achtergrondconcentratie'] = (
        gekoppelde_achtergrond.loc[masker]
    )
    resultaat.loc[masker, 'Waarde'] = (
        resultaat.loc[masker, 'Waarde_Origineel']
        - gekoppelde_achtergrond.loc[masker]
    ).clip(lower=0)
    resultaat.loc[masker, 'Achtergrondcorrectie_Toegepast'] = True
    return resultaat


def match_stofgroep_optimized(unieke_stoffen):
    mapping = {}
    for stof in unieke_stoffen:
        s_lower = stof.lower()
        gevonden = False
        for groep, keywords in STOFGROEPEN_MAPPING.items():
            if any(k in s_lower for k in keywords):
                mapping[stof] = groep
                gevonden = True
                break
        if not gevonden:
            mapping[stof] = 'Onbekend'
    return mapping

PARQUET_FILE_PATH = 'data/metingen.parquet'
METADATA_FILE_PATH = 'data/metadata.json'

# Centrale periode-indeling voor alle dashboardpagina's.
# Bij gecombineerde selecties wordt de unie van de bijbehorende maanden gebruikt.
PERIODES = {
    "Winter": (12, 1, 2),
    "Voorjaar": (3, 4, 5),
    "Zomer": (6, 7, 8),
    "Herfst": (9, 10, 11),
    "Zomerhalfjaar": (4, 5, 6, 7, 8, 9),
    "Winterhalfjaar": (10, 11, 12, 1, 2, 3),
}
PERIODE_VOLGORDE = tuple(PERIODES)


def _data_artifact_version() -> tuple[int, int]:
    """Geeft een stabiele cacheversie op basis van mtime en bestandsgrootte."""
    from pathlib import Path

    pad = Path(PARQUET_FILE_PATH)
    if not pad.is_file():
        return (0, 0)
    stat = pad.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _normaliseer_queryselectie(waarden) -> tuple:
    """Maakt widgetselecties stabiel en hashbaar voor de Streamlit-cache."""
    if waarden is None:
        return ()
    if isinstance(waarden, (str, bytes)):
        waarden = (waarden,)
    return tuple(sorted(set(waarden), key=lambda waarde: str(waarde)))


def _quote_identifier(kolomnaam: str) -> str:
    """Quote een reeds gevalideerde DuckDB-identifier veilig."""
    return '"' + kolomnaam.replace('"', '""') + '"'


@st.cache_data(show_spinner=False, max_entries=8)
def _get_parquet_columns_cached(
    artifact_version: tuple[int, int],
) -> tuple[str, ...]:
    """Leest alleen het Parquet-schema; artifact_version invalideert de cache."""
    from pathlib import Path

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is niet geinstalleerd. Voeg `duckdb` toe aan requirements.txt."
        ) from exc

    pad = Path(PARQUET_FILE_PATH).resolve()
    if not pad.is_file():
        raise FileNotFoundError(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}."
        )

    try:
        with duckdb.connect(database=":memory:") as con:
            beschrijving = con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)",
                [pad.as_posix()],
            ).fetchall()
    except duckdb.Error as exc:
        raise RuntimeError(
            f"Parquetschema kan niet via DuckDB worden gelezen: {exc}"
        ) from exc

    return tuple(rij[0] for rij in beschrijving)



@st.cache_data(show_spinner=False, max_entries=8)
def _get_filter_options_cached(
    artifact_version: tuple[int, int],
) -> dict[str, tuple]:
    """Haalt unieke filterwaarden read-only op uit Parquet via DuckDB.

    artifact_version is bewust onderdeel van de functieparameters, zodat een
    nieuw Parquet-artifact automatisch een nieuwe Streamlit-cache-entry krijgt.
    """
    from pathlib import Path

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "DuckDB is niet geinstalleerd. Voeg `duckdb` toe aan "
            "requirements.txt."
        ) from exc

    pad = Path(PARQUET_FILE_PATH).resolve()
    if not pad.is_file():
        raise FileNotFoundError(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}."
        )

    sql = """
        SELECT
            list_sort(list(DISTINCT year("Datum"))) AS jaren,
            list_sort(list(DISTINCT "Meetpunt")) AS meetpunten,
            list_sort(list(DISTINCT "Stof")) AS stoffen,
            list_sort(list(DISTINCT "Stofgroep")) AS stofgroepen,
            list_sort(list(DISTINCT "Eenheid")) AS eenheden
        FROM read_parquet(?)
    """

    try:
        with duckdb.connect(database=":memory:") as con:
            rij = con.execute(sql, [pad.as_posix()]).fetchone()
    except duckdb.Error as exc:
        raise RuntimeError(
            f"Filteropties kunnen niet via DuckDB worden opgehaald: {exc}"
        ) from exc

    if rij is None:
        return _lege_filteropties()

    return {
        "jaren": tuple(
            int(jaar) for jaar in (rij[0] or []) if jaar is not None
        ),
        "meetpunten": tuple(
            waarde for waarde in (rij[1] or []) if waarde is not None
        ),
        "stoffen": tuple(
            waarde for waarde in (rij[2] or []) if waarde is not None
        ),
        "stofgroepen": tuple(
            waarde for waarde in (rij[3] or []) if waarde is not None
        ),
        "eenheden": tuple(
            waarde for waarde in (rij[4] or []) if waarde is not None
        ),
    }


def _lege_filteropties() -> dict[str, tuple]:
    """Levert een nieuwe, lege optiestructuur voor foutafhandeling."""
    return {
        "jaren": (),
        "meetpunten": (),
        "stoffen": (),
        "stofgroepen": (),
        "eenheden": (),
    }


def get_filter_options() -> dict[str, tuple]:
    """Publieke, gecachete filteroptielaag voor Streamlit-pagina's."""
    artifact_version = _data_artifact_version()

    if artifact_version == (0, 0):
        st.error(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}. "
            "Voer eerst `python build_data.py` uit."
        )
        return _lege_filteropties()

    try:
        return _get_filter_options_cached(
            artifact_version=artifact_version,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return _lege_filteropties()


@st.cache_data(
    show_spinner="Meetgegevens selecteren...",
    max_entries=128,
)
def _query_data_cached(
    artifact_version: tuple[int, int],
    jaren: tuple[int, ...],
    periodes: tuple[str, ...],
    stoffen: tuple[str, ...],
    meetpunten: tuple[str, ...],
    stofgroepen: tuple[str, ...],
    eenheden: tuple[str, ...],
    kolommen: tuple[str, ...],
    datum_vanaf: str | None,
    datum_tot: str | None,
) -> pd.DataFrame:
    """Voert een read-only DuckDB-query met pushdown uit."""
    from datetime import datetime
    from pathlib import Path

    try:
        import duckdb
    except ImportError:
        st.error(
            "DuckDB is niet geinstalleerd. Voeg `duckdb` toe aan "
            "requirements.txt en installeer de requirements opnieuw."
        )
        return pd.DataFrame()

    pad = Path(PARQUET_FILE_PATH).resolve()
    if not pad.is_file():
        st.error(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}. "
            "Voer eerst `python build_data.py` uit."
        )
        return pd.DataFrame()

    try:
        beschikbare_kolommen = _get_parquet_columns_cached(artifact_version)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        return pd.DataFrame()

    beschikbare_set = set(beschikbare_kolommen)
    geselecteerde_kolommen = kolommen or beschikbare_kolommen
    onbekende_kolommen = sorted(set(geselecteerde_kolommen) - beschikbare_set)
    if onbekende_kolommen:
        raise ValueError(
            "Onbekende of niet-toegestane Parquetkolommen: "
            + ", ".join(onbekende_kolommen)
        )

    select_sql = ", ".join(
        _quote_identifier(kolom) for kolom in geselecteerde_kolommen
    )
    voorwaarden: list[str] = []
    parameters: list[object] = [pad.as_posix()]

    def voeg_in_filter_toe(kolom: str, waarden: tuple) -> None:
        if not waarden:
            return
        if kolom not in beschikbare_set:
            raise ValueError(
                f"Filterkolom '{kolom}' ontbreekt in het Parquetbestand."
            )
        placeholders = ", ".join("?" for _ in waarden)
        voorwaarden.append(
            f"{_quote_identifier(kolom)} IN ({placeholders})"
        )
        parameters.extend(waarden)

    if jaren:
        if "Datum" not in beschikbare_set:
            raise ValueError("Filterkolom 'Datum' ontbreekt in het Parquetbestand.")
        placeholders = ", ".join("?" for _ in jaren)
        voorwaarden.append(
            f"year({_quote_identifier('Datum')}) IN ({placeholders})"
        )
        parameters.extend(jaren)

    if periodes:
        if "Datum" not in beschikbare_set:
            raise ValueError("Filterkolom 'Datum' ontbreekt in het Parquetbestand.")

        onbekende_periodes = sorted(set(periodes) - set(PERIODES))
        if onbekende_periodes:
            raise ValueError(
                "Onbekende perioden: " + ", ".join(onbekende_periodes)
            )

        maanden = tuple(sorted({
            maand
            for periode in periodes
            for maand in PERIODES[periode]
        }))
        placeholders = ", ".join("?" for _ in maanden)
        voorwaarden.append(
            f"month({_quote_identifier('Datum')}) IN ({placeholders})"
        )
        parameters.extend(maanden)

    voeg_in_filter_toe("Stof", stoffen)
    voeg_in_filter_toe("Meetpunt", meetpunten)
    voeg_in_filter_toe("Stofgroep", stofgroepen)
    voeg_in_filter_toe("Eenheid", eenheden)

    if datum_vanaf is not None:
        voorwaarden.append(f"{_quote_identifier('Datum')} >= CAST(? AS DATE)")
        parameters.append(datum_vanaf)
    if datum_tot is not None:
        voorwaarden.append(f"{_quote_identifier('Datum')} <= CAST(? AS DATE)")
        parameters.append(datum_tot)

    where_sql = ""
    if voorwaarden:
        where_sql = " WHERE " + " AND ".join(voorwaarden)

    sql = f"SELECT {select_sql} FROM read_parquet(?){where_sql}"

    try:
        # Private in-memory connectie per cache-miss: geen gedeelde state en
        # geen persistent DuckDB-bestand. read_parquet wordt uitsluitend gelezen.
        with duckdb.connect(database=":memory:") as con:
            df = con.execute(sql, parameters).fetchdf()
    except duckdb.Error as exc:
        st.error(f"DuckDB-query op Parquet is mislukt: {exc}")
        return pd.DataFrame(columns=list(geselecteerde_kolommen))

    if "Datum" in df.columns:
        df["Datum"] = pd.to_datetime(df["Datum"], errors="coerce")
    for kolom in ("Meetpunt", "Eenheid", "Stofgroep"):
        if kolom in df.columns and not isinstance(
            df[kolom].dtype, pd.CategoricalDtype
        ):
            df[kolom] = df[kolom].astype("category")

    return df


def query_data(
    *,
    jaren=(),
    periodes=(),
    stoffen=(),
    meetpunten=(),
    stofgroepen=(),
    eenheden=(),
    kolommen=(),
    datum_vanaf=None,
    datum_tot=None,
) -> pd.DataFrame:
    """Publieke querylaag met veilige pushdown en genormaliseerde cachesleutels.

    Lege filters betekenen 'geen beperking'. Kolommen=() retourneert alle
    kolommen en houdt daarmee de bestaande load_data-interface intact.
    """
    artifact_version = _data_artifact_version()
    if artifact_version == (0, 0):
        st.error(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}. "
            "Voer eerst `python build_data.py` uit."
        )
        return pd.DataFrame()

    genormaliseerde_kolommen = tuple(kolommen) if kolommen else ()
    vanaf = None if datum_vanaf is None else str(pd.Timestamp(datum_vanaf).date())
    tot = None if datum_tot is None else str(pd.Timestamp(datum_tot).date())

    return _query_data_cached(
        artifact_version=artifact_version,
        jaren=tuple(int(jaar) for jaar in _normaliseer_queryselectie(jaren)),
        periodes=_normaliseer_queryselectie(periodes),
        stoffen=_normaliseer_queryselectie(stoffen),
        meetpunten=_normaliseer_queryselectie(meetpunten),
        stofgroepen=_normaliseer_queryselectie(stofgroepen),
        eenheden=_normaliseer_queryselectie(eenheden),
        kolommen=genormaliseerde_kolommen,
        datum_vanaf=vanaf,
        datum_tot=tot,
    )


# Achterwaartse compatibiliteit voor eventuele rechtstreekse interne aanroepen.
def _load_parquet_cached(artifact_version: tuple[int, int]) -> pd.DataFrame:
    return _query_data_cached(
        artifact_version=artifact_version,
        jaren=(),
        periodes=(),
        stoffen=(),
        meetpunten=(),
        stofgroepen=(),
        eenheden=(),
        kolommen=(),
        datum_vanaf=None,
        datum_tot=None,
    )
def _set_last_update_from_metadata() -> None:
    """Behoudt last_update, maar gebruikt voortaan het echte buildmoment."""
    import json
    from pathlib import Path

    metadata_pad = Path(METADATA_FILE_PATH)
    waarde = datetime.fromtimestamp(
        Path(PARQUET_FILE_PATH).stat().st_mtime
    ).strftime("%Y-%m-%d %H:%M:%S")

    if metadata_pad.is_file():
        try:
            metadata = json.loads(metadata_pad.read_text(encoding="utf-8"))
            waarde = metadata.get("built_at_local", waarde)
        except (OSError, ValueError, TypeError):
            pass

    st.session_state.last_update = waarde


def load_data() -> pd.DataFrame:
    """Publieke compatibiliteitslaag: leest de vooraf gebouwde Parquet-dataset."""
    versie = _data_artifact_version()
    if versie == (0, 0):
        st.error(
            f"Parquetbestand niet gevonden op pad: {PARQUET_FILE_PATH}. "
            "Voer eerst `python build_data.py` uit."
        )
        return pd.DataFrame()

    df = query_data()
    if not df.empty:
        _set_last_update_from_metadata()
    return df

 
@st.cache_data
def load_pfas_ref() -> pd.DataFrame:
    try:
        df_pfas = pd.read_csv(PFAS_FILE_PATH, dtype=str)
        df_pfas.columns = df_pfas.columns.str.strip()

        cols_to_fix = ['RPF', 'RBF']
        for col in cols_to_fix:
            if col in df_pfas.columns:
                df_pfas[col] = df_pfas[col].str.replace(',', '.', regex=False)
                df_pfas[col] = pd.to_numeric(df_pfas[col], errors='coerce').fillna(0)

        return df_pfas
    except FileNotFoundError:
        st.error(f"PFAS bestand niet gevonden: {PFAS_FILE_PATH}")
        return pd.DataFrame()


def create_gauge(percentage: float, title_text: str = "Metingen onder Norm", drempel: int = 95) -> go.Figure:
    if pd.isna(percentage):
        percentage = 0
        
    bar_color = "green" if percentage >= drempel else "red"
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title_text, 'font': {'size': 14}},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': bar_color},
                 'steps' : [
                     {'range': [0, drempel], 'color': "lightgray"},
                     {'range': [drempel, 100], 'color': "green"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': drempel}}
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=10))
    return fig

def get_shared_sidebar_filters(
    *,
    toon_aggregatiemethode: bool = True,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Toont gedeelde jaar- en periodefilters voor DuckDB-querypagina's.

    Lege selecties betekenen, net als bij query_data, geen beperking. De vaste
    widgetkeys zorgen dat de selectie tussen Streamlit-pagina's behouden blijft.
    """
    filter_options = get_filter_options()
    beschikbare_jaren = sorted(filter_options["jaren"], reverse=True)

    st.sidebar.header("📅 Periodefilters")
    geselecteerde_jaren = st.sidebar.multiselect(
        "Selecteer gewenste jaren:",
        options=beschikbare_jaren,
        default=beschikbare_jaren,
        key="shared_jaren_filter",
    )
    geselecteerde_periodes = st.sidebar.multiselect(
        "Selecteer gewenste seizoenen of halfjaren:",
        options=list(PERIODE_VOLGORDE),
        default=list(PERIODE_VOLGORDE),
        key="shared_periodes_filter",
        help=(
            "Winter: december t/m februari; voorjaar: maart t/m mei; "
            "zomer: juni t/m augustus; herfst: september t/m november; "
            "zomerhalfjaar: april t/m september; "
            "winterhalfjaar: oktober t/m maart. "
            "Bij meerdere keuzes worden de maanden gecombineerd."
        ),
    )

    if toon_aggregatiemethode:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Aggregatiemethode")
        st.sidebar.radio(
            "Bereken ruimtelijke waarden als:",
            options=["Gemiddelde", "Mediaan"],
            index=0,
            key="ruimtelijke_aggregatiemethode",
            help=(
                "Deze keuze wordt gebruikt in de Ruimtelijke analyse. "
                "De mediaan is minder gevoelig voor uitschieters dan het gemiddelde."
            ),
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Navigeer via het menu hierboven naar de verschillende analyses."
    )
    return tuple(geselecteerde_jaren), tuple(geselecteerde_periodes)


def get_shared_sidebar(df_main):
    """Achterwaarts compatibele Pandas-sidebar met jaar- en periodefilter."""
    st.sidebar.header("📅 Periodefilters")

    if not df_main.empty and 'Datum' in df_main.columns:
        beschikbare_jaren = sorted(
            df_main['Datum'].dt.year.dropna().unique(), reverse=True
        )
    else:
        beschikbare_jaren = []

    geselecteerde_jaren = st.sidebar.multiselect(
        "Selecteer gewenste jaren:",
        options=beschikbare_jaren,
        default=beschikbare_jaren,
        key="shared_jaren_filter",
    )
    geselecteerde_periodes = st.sidebar.multiselect(
        "Selecteer gewenste seizoenen of halfjaren:",
        options=list(PERIODE_VOLGORDE),
        default=list(PERIODE_VOLGORDE),
        key="shared_periodes_filter",
        help=(
            "Winter: december t/m februari; voorjaar: maart t/m mei; "
            "zomer: juni t/m augustus; herfst: september t/m november; "
            "zomerhalfjaar: april t/m september; "
            "winterhalfjaar: oktober t/m maart. "
            "Bij meerdere keuzes worden de maanden gecombineerd."
        ),
    )

    df_filtered = df_main.copy()
    if geselecteerde_jaren and not df_filtered.empty:
        df_filtered = df_filtered[
            df_filtered['Datum'].dt.year.isin(geselecteerde_jaren)
        ].copy()

    if geselecteerde_periodes and not df_filtered.empty:
        geselecteerde_maanden = {
            maand
            for periode in geselecteerde_periodes
            for maand in PERIODES[periode]
        }
        df_filtered = df_filtered[
            df_filtered['Datum'].dt.month.isin(geselecteerde_maanden)
        ].copy()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Aggregatiemethode")
    st.sidebar.radio(
        "Bereken ruimtelijke waarden als:",
        options=["Gemiddelde", "Mediaan"],
        index=0,
        key="ruimtelijke_aggregatiemethode",
        help=(
            "Deze keuze wordt gebruikt in de Ruimtelijke analyse. "
            "De mediaan is minder gevoelig voor uitschieters dan het gemiddelde."
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Navigeer via het menu hierboven naar de verschillende analyses."
    )
    return df_filtered

@st.cache_data
def calculate_trends_optimized(df_in: pd.DataFrame, lt_optie: str, norm_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berekent opwaartse trends (helling > 0).
    """
    if df_in.empty:
        return pd.DataFrame()

    # 1. Bereken Aantal metingen > RG per Meetpunt/Stof
    # Dit doen we op df_in VOORDAT we groeperen naar jaren.
    # Een meting is > RG als Limietsymbool NIET '<' bevat.
    mask_boven_rg = ~df_in['Limietsymbool'].astype(str).str.contains('<', na=False)
    
    # Maak een dictionary voor snelle lookup: {(Meetpunt, Stof): count}
    counts_boven_rg = df_in[mask_boven_rg].groupby(['Meetpunt', 'Stof'], observed=True).size().to_dict()

    df_calc = df_in.copy()
    
    if lt_optie == "Sluit uit van berekening":
        mask_lt = df_calc['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_calc.loc[mask_lt, 'Waarde'] = np.nan

    df_calc['Jaar'] = df_calc['Datum'].dt.year
    df_yearly = df_calc.groupby(['Meetpunt', 'Stof', 'Jaar'], observed=True)['Waarde'].mean().reset_index()
    
    norm_map = norm_lookup_df.set_index('Stof')['JG_MKN'].to_dict()

    trend_results = []
    grouped = df_yearly.dropna(subset=['Waarde']).groupby(['Meetpunt', 'Stof'], observed=True)

    for (meetpunt, stof), group in grouped:
        if len(group) < 2:
            continue
            
        group = group.sort_values('Jaar')
        x = group['Jaar'].values
        y = group['Waarde'].values

        slope, intercept = np.polyfit(x, y, 1)

        # Alleen stijgende trends
        if slope > 0:
            laatste_jaargemiddelde = y[-1]
            jg_norm = norm_map.get(stof, np.nan)
            
            if pd.notna(jg_norm) and laatste_jaargemiddelde >= jg_norm:
                tijd_tot_norm = 0.0
            elif pd.notna(jg_norm) and slope > 0:
                tijd_tot_norm = (jg_norm - laatste_jaargemiddelde) / slope
            else:
                tijd_tot_norm = np.inf
            
            # Haal het aantal metingen > RG op uit de dictionary
            n_metingen = counts_boven_rg.get((meetpunt, stof), 0)

            trend_results.append({
                'Meetpunt': meetpunt,
                'Stof': stof,
                'Startwaarde': y[0],
                'Eindwaarde': laatste_jaargemiddelde,
                'JG_MKN': jg_norm, 
                'Trendscore': slope,
                'Tijd_tot_JG_normoverschrijding': tijd_tot_norm,
                'Aantal_jaren': len(group),
                'n_metingen_boven_rg': n_metingen
            })

    return pd.DataFrame(trend_results)

@st.cache_data
def calculate_declining_exceedances_optimized(df_in: pd.DataFrame, lt_optie: str, norm_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Berekent trends voor stoffen die de norm overschrijden, maar een vlakke of dalende trend hebben (helling <= 0).
    """
    if df_in.empty:
        return pd.DataFrame()

    # 1. Bereken Aantal metingen > RG per Meetpunt/Stof
    mask_boven_rg = ~df_in['Limietsymbool'].astype(str).str.contains('<', na=False)
    counts_boven_rg = df_in[mask_boven_rg].groupby(['Meetpunt', 'Stof'], observed=True).size().to_dict()

    df_calc = df_in.copy()
    
    if lt_optie == "Sluit uit van berekening":
        mask_lt = df_calc['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_calc.loc[mask_lt, 'Waarde'] = np.nan

    df_calc['Jaar'] = df_calc['Datum'].dt.year
    df_yearly = df_calc.groupby(['Meetpunt', 'Stof', 'Jaar'], observed=True)['Waarde'].mean().reset_index()
    
    norm_map = norm_lookup_df.set_index('Stof')['JG_MKN'].to_dict()

    trend_results = []
    grouped = df_yearly.dropna(subset=['Waarde']).groupby(['Meetpunt', 'Stof'], observed=True)

    for (meetpunt, stof), group in grouped:
        if len(group) < 2:
            continue
            
        group = group.sort_values('Jaar')
        x = group['Jaar'].values
        y = group['Waarde'].values

        slope, intercept = np.polyfit(x, y, 1)
        laatste_jaargemiddelde = y[-1]
        jg_norm = norm_map.get(stof, np.nan)

        # CRITERIA: Normoverschrijding (laatste jaar) EN Helling <= 0 (niet stijgend)
        if pd.notna(jg_norm) and laatste_jaargemiddelde > jg_norm and slope <= 0:
            
            # Berekening tijd tot onder norm
            if slope < 0:
                tijd_tot_onder = (jg_norm - laatste_jaargemiddelde) / slope
            else:
                tijd_tot_onder = np.inf # Stagnant boven norm, gaat nooit onder norm komen
            
            n_metingen = counts_boven_rg.get((meetpunt, stof), 0)
                
            trend_results.append({
                'Meetpunt': meetpunt,
                'Stof': stof,
                'Startwaarde': y[0],
                'Eindwaarde': laatste_jaargemiddelde,
                'Trendscore': slope,
                'JG_MKN': jg_norm,
                'Tijd_tot_onder_norm': tijd_tot_onder,
                'Aantal_jaren': len(group),
                'n_metingen_boven_rg': n_metingen
            })

    return pd.DataFrame(trend_results)

def calculate_metrics(df_in: pd.DataFrame, is_period_average: bool = False):
    if df_in.empty:
        return 0, 0, 0.0
    
    df_calc = df_in.copy()
    if 'Jaar' not in df_calc.columns:
        df_calc['Jaar'] = df_calc['Datum'].dt.year
    
    unique_years = df_calc['Jaar'].nunique()

    mask_jg_over = (df_calc['Waarde'] > df_calc['JG_MKN'])
    mask_mac_over = (df_calc['Waarde'] > df_calc['MAC_MKN'])
    
    total_count = len(df_calc)
    total_viol_count = (mask_jg_over | mask_mac_over).sum()
    
    if is_period_average and unique_years > 0:
        avg_count = total_count / unique_years
        avg_viol_count = total_viol_count / unique_years
        pct_viol = (total_viol_count / total_count * 100)
        return avg_count, avg_viol_count, pct_viol
    else:
        pct_viol = (total_viol_count / total_count * 100) if total_count > 0 else 0.0
        return total_count, total_viol_count, pct_viol

def calculate_compliance_details(df_in: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['Meetpunt', 'Datum', 'Stof', 'Waarde', 'JG_MKN', 'MAC_MKN']
    if df_in.empty or not all(col in df_in.columns for col in required_cols):
        return pd.DataFrame()

    df_calc = df_in.copy()
    
    if 'Limietsymbool' in df_calc.columns:
        df_calc = df_calc[df_calc['Limietsymbool'] != '<']
        
    df_calc['Jaar'] = df_calc['Datum'].dt.year

    jg_means = df_calc.groupby(['Meetpunt', 'Jaar', 'Stof', 'JG_MKN'], observed=True)['Waarde'].mean().reset_index()
    jg_failures = jg_means[jg_means['Waarde'] > jg_means['JG_MKN']].copy()
    jg_failures['Normtype'] = 'JG-MKN'
    jg_failures['Factor'] = jg_failures['Waarde'] / jg_failures['JG_MKN']

    mac_maxs = df_calc.groupby(['Meetpunt', 'Jaar', 'Stof', 'MAC_MKN'], observed=True)['Waarde'].max().reset_index()
    mac_failures = mac_maxs[mac_maxs['Waarde'] > mac_maxs['MAC_MKN']].copy()
    mac_failures['Normtype'] = 'MAC-MKN'
    mac_failures['Factor'] = mac_failures['Waarde'] / mac_failures['MAC_MKN']

    combined = pd.concat([
        jg_failures[['Meetpunt', 'Jaar', 'Stof', 'Normtype', 'Factor']], 
        mac_failures[['Meetpunt', 'Jaar', 'Stof', 'Normtype', 'Factor']]
    ])
    
    return combined

def prepare_heatmap_data(df_filtered: pd.DataFrame):
    df_source = df_filtered.copy()
    if 'Limietsymbool' in df_source.columns:
        df_source = df_source[df_source['Limietsymbool'] != '<']

    if df_source.empty:
        return None, None, [], []

    if 'Jaar' not in df_source.columns:
        df_source['Jaar'] = df_source['Datum'].dt.year

    s_factor_jg = df_source['Waarde'].div(df_source['JG_MKN']).fillna(0)
    s_factor_mac = df_source['Waarde'].div(df_source['MAC_MKN']).fillna(0)
    
    df_source['MaxFactor_Meting'] = np.maximum(s_factor_jg, s_factor_mac)
    df_source.loc[df_source['MaxFactor_Meting'] <= 1.0, 'MaxFactor_Meting'] = 0.0

    # Bepaal overschrijdingen uitsluitend wanneer zowel meetwaarde als norm
    # bruikbaar zijn. Ontbrekende waarden/normen zijn geen vastgestelde
    # overschrijding en worden daarom expliciet False. Dit voorkomt tevens
    # nullable booleans (pd.NA/NaN) in de aggregatie hieronder.
    is_jg_over = (
        df_source['Waarde'].notna()
        & df_source['JG_MKN'].notna()
        & df_source['JG_MKN'].gt(0)
        & df_source['Waarde'].gt(df_source['JG_MKN'])
    ).fillna(False).astype(bool)
    is_mac_over = (
        df_source['Waarde'].notna()
        & df_source['MAC_MKN'].notna()
        & df_source['MAC_MKN'].gt(0)
        & df_source['Waarde'].gt(df_source['MAC_MKN'])
    ).fillna(False).astype(bool)

    df_annual = df_source.groupby(
        ['Stof', 'Meetpunt', 'Jaar'], observed=True
    ).agg(
        Fail_JG=('Waarde', lambda x: bool(is_jg_over.loc[x.index].any(skipna=True))),
        Fail_MAC=('Waarde', lambda x: bool(is_mac_over.loc[x.index].any(skipna=True)))
    ).reset_index()

    # Verdedigende normalisatie voor lege/categorische groepen en toekomstige
    # pandas-versies: de statuskolommen bevatten altijd uitsluitend bools.
    for fail_col in ('Fail_JG', 'Fail_MAC'):
        if fail_col not in df_annual.columns:
            df_annual[fail_col] = False
        else:
            df_annual[fail_col] = df_annual[fail_col].fillna(False).astype(bool)

    conditions = [
        (df_annual['Fail_JG'] & df_annual['Fail_MAC']),
        (df_annual['Fail_JG']),
        (df_annual['Fail_MAC'])
    ]
    choices = ['JG+MAC', 'JG', 'MAC']
    df_annual['StatusType'] = np.select(conditions, choices, default='OK')

    df_annual_fails = df_annual[df_annual['StatusType'] != 'OK'].copy()

    if df_annual_fails.empty:
        return None, None, [], []

    df_text_parts = df_annual_fails.groupby(['Stof', 'Meetpunt', 'StatusType'],observed=True)['Jaar'].apply(
        lambda x: ", ".join(map(str, sorted(x.dropna().astype(int).unique())))
    ).reset_index(name='JarenStr')

    df_text_parts['FullText'] = df_text_parts['StatusType'] + " (" + df_text_parts['JarenStr'] + ")"
    
    status_order = pd.CategoricalDtype(['JG', 'MAC', 'JG+MAC'], ordered=True)
    df_text_parts['StatusType'] = df_text_parts['StatusType'].astype(status_order)
    df_text_parts = df_text_parts.sort_values(['Stof', 'Meetpunt', 'StatusType'])

    df_viz_text = df_text_parts.groupby(['Stof', 'Meetpunt'], observed=True)['FullText'].apply(
        lambda x: '<br>'.join(x.dropna().astype(str))
    ).reset_index(name='CellText')

    df_viz_factor = df_source.groupby(['Stof', 'Meetpunt'], observed=True)['MaxFactor_Meting'].max().reset_index(name='MaxFactor')

    violating_keys = df_viz_factor[df_viz_factor['MaxFactor'] > 1.0][['Stof', 'Meetpunt']]
    if violating_keys.empty:
        return None, None, [], []

    all_violating_stof = sorted(violating_keys['Stof'].unique())
    all_violating_meetpunt = sorted(violating_keys['Meetpunt'].unique())

    df_viz_heatmap = pd.merge(df_viz_factor, df_viz_text, on=['Stof', 'Meetpunt'], how='left')
    df_viz_heatmap['CellText'] = df_viz_heatmap['CellText'].fillna(' ')
    df_viz_heatmap['MaxFactor'] = df_viz_heatmap['MaxFactor'].fillna(0.0)
    
    df_viz_heatmap.loc[df_viz_heatmap['MaxFactor'] <= 1.0, 'CellText'] = ' '

    df_final = df_viz_heatmap[
        df_viz_heatmap['Stof'].isin(all_violating_stof) & 
        df_viz_heatmap['Meetpunt'].isin(all_violating_meetpunt)
    ]

    factor_matrix = df_final.pivot(index='Stof', columns='Meetpunt', values='MaxFactor')
    text_matrix = df_final.pivot(index='Stof', columns='Meetpunt', values='CellText')

    factor_matrix = factor_matrix.reindex(index=all_violating_stof, columns=all_violating_meetpunt).fillna(0.0)
    text_matrix = text_matrix.reindex(index=all_violating_stof, columns=all_violating_meetpunt).fillna(' ')

    return factor_matrix, text_matrix, all_violating_stof, all_violating_meetpunt

def prepare_sunburst_data(df_mp_fail):
    stof_summary = []
    
    for stof, group in df_mp_fail.groupby('Stof'):
        types = group['Normtype'].unique()
        max_factor = group['Factor'].max()
        
        if 'JG-MKN' in types and 'MAC-MKN' in types:
            cat = "JG + MAC"
            base_color_scale = 'Reds'
        elif 'JG-MKN' in types:
            cat = "JG (Gemiddelde)"
            base_color_scale = 'Oranges'
        else:
            cat = "MAC (Piek)"
            base_color_scale = 'Purples'
            
        norm_val = min((max_factor - 1) / 4, 1.0) 
        color_val = 0.3 + (norm_val * 0.7) 
        hex_color = px.colors.sample_colorscale(base_color_scale, [color_val])[0]
        
        stof_summary.append({
            'Stof': stof,
            'Categorie': cat,
            'Factor': max_factor,
            'Color': hex_color
        })
        
    return pd.DataFrame(stof_summary)
