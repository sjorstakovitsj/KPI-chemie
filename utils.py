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
        'som hexadecafluor-2-deceenzuur-isomerenâ', "2,3,3,3-tetrafluorpropaanzuur"
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
        'glufosinaat', 'cyhalothrin', 'trifluraline', 'triazool', '3-(hydroxymethylfosfinoyl)propionzuur',
        'dinoterb', 'heptenofos', 'metabenzthiazuron', "4,4'-dichloordifenyltrichloorethaan",
        "2,4'-dichloordifenyltrichloorethaan", 'penconazool', 'acetamiprid', 'thiamethoxam', 'methiocarb',
         'cyprodinil', 'bromacil', 'diethyltoluamide', 'chloorprofam', 'propyzamide', 'propamocarb', 'propoxur',
         'prosulfocarb', 'etridiazol', 'ethofumesaat', 'carbendazim', 'clomazon', '2,6-dichloorbenzamide',
         'aldicarb', 'triallaat', 'triclosan', 'imazalil', 'ipconazole', 'prochloraz', 'tebuconazol', 
         'tetraconazool', 'metaflumizon', 'dimoxystrobine', 'metconazool', 'clothianidine',
         'cyazofamide', 'bromuconazool', 'difenoconazool', 'amisulbrom', 'etoxazool', 'ketoconazol',
         'epoxiconazool', 'mefentrifluconazool', 'triticonazool', 'dichlobenil'
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
        'amiodaron', 'climbazole', 'norfloxacine', 'tetracycline', 'tylosine'
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
        '1,4-dioxaan', 'chloorxylenol', 'amberonne', 'hexahydrohexamethylcyclopentabenzopyran (hhcb)'
    ],
    'Nutriënten & algemeen': [
        'fluoride', 'zuurstof', 'chlorofyl', 'silicium', 'sulfaat', 'koolstof', 'stikstof',
        'nitraat', 'nitriet', 'ammonium', 'fosfor', 'fosfaat', 'chloride', 'zwevende stof',
        'hardheid', 'temperatuur', 'zuurgraad', 'geleidbaarheid', 'gloeirest', 'onopgeloste',
        'doorzicht', 'saliniteit', 'troebelheid', 'cyanide', 'bicarbonaat', 'waterstofcarbonaat',
        'extinctie', 'kleur', 'geur', 'olie', 'schuim', 'vuil', 'escherichia coli'
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

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILE_PATH, delimiter=';', low_memory=False, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Bestand niet gevonden op pad: {DATA_FILE_PATH}.")
        return pd.DataFrame()
    
    # Normaliseer en koppel ALLE benodigde bronkolommen aan vaste interne namen.
    # De vergelijking is ongevoelig voor hoofdletters, BOM-tekens, spaties,
    # koppeltekens, punten en underscores. De originele gegevenswaarden blijven behouden.
    def _normaliseer_kolomnaam(naam):
        return (
            str(naam)
            .replace('\ufeff', '')
            .strip()
            .casefold()
            .replace('ë', 'e')
            .replace('é', 'e')
            .replace('ï', 'i')
            .replace('ö', 'o')
            .replace('ü', 'u')
        )

    def _kolomsleutel(naam):
        # Maak equivalente schrijfwijzen gelijk, bijvoorbeeld:
        # 'Event Waarde', 'event-waarde' en 'event_waarde'.
        return ''.join(
            teken for teken in _normaliseer_kolomnaam(naam)
            if teken.isalnum()
        )

    column_aliases = {
        'Datum': [
            'datum', 'eventdatum', 'event_datum', 'meetdatum', 'monsterdatum'
        ],
        'Meetpunt': [
            'meetpunt', 'locatie_code', 'locatiecode', 'meetpunt_code',
            'meetpuntcode', 'locatie'
        ],
        'Stof': [
            'stof', 'parameter_omschrijving', 'parameteromschrijving',
            'parameter', 'stofnaam'
        ],
        'Waarde': [
            'waarde', 'event_waarde', 'eventwaarde', 'meetwaarde',
            'resultaat', 'numerieke_waarde'
        ],
        'Eenheid': [
            'eenheid', 'eenheid_code', 'eenheidcode', 'unit'
        ],
        'Limietsymbool': [
            'limietsymbool', 'limiet_symbool', 'event_waarde_limietsymbool',
            'eventwaardelimietsymbool', 'event_waarde_limiet_symbool',
            'grenssymbool', 'detectielimietsymbool'
        ],
        'hoedanigheid': [
            'hoedanigheid', 'hoedanigheid_code', 'hoedanigheidcode'
        ],
        'Latitude': [
            'latitude', 'lat', 'locatie_lat_etrs89', 'locatielatetrs89',
            'breedtegraad'
        ],
        'Longitude': [
            'longitude', 'lon', 'lng', 'locatie_lon_etrs89',
            'locatielonetrs89', 'lengtegraad'
        ],
        'Hoedanigheid_Omschr': [
            'hoedanigheid_omschr', 'hoedanigheid_omschrijving',
            'hoedanigheidomschrijving'
        ],
        'Eenheid_Omschr': [
            'eenheid_omschr', 'eenheid_omschrijving', 'eenheidomschrijving'
        ],
    }

    # Bouw één lookup op voor alle toegestane schrijfwijzen.
    alias_lookup = {}
    for target, aliases in column_aliases.items():
        for naam in [target, *aliases]:
            sleutel = _kolomsleutel(naam)
            bestaand_target = alias_lookup.get(sleutel)
            if bestaand_target is not None and bestaand_target != target:
                raise ValueError(
                    f"Dubbelzinnige kolomalias '{naam}' voor "
                    f"'{bestaand_target}' en '{target}'."
                )
            alias_lookup[sleutel] = target

    # Verzamel alle gevonden varianten per interne doelkolom.
    gevonden_kolommen = {target: [] for target in column_aliases}
    for kolom in df.columns:
        target = alias_lookup.get(_kolomsleutel(kolom))
        if target is not None:
            gevonden_kolommen[target].append(kolom)

    # Combineer dubbele/alternatieve kolommen zonder niet-lege waarden te verliezen.
    # De interne doelnaam krijgt voorrang als die al aanwezig is.
    for target, bronkolommen in gevonden_kolommen.items():
        if not bronkolommen:
            continue

        bronkolommen.sort(
            key=lambda kolom: _kolomsleutel(kolom) != _kolomsleutel(target)
        )
        gecombineerd = df[bronkolommen[0]].copy()
        for kolom in bronkolommen[1:]:
            gecombineerd = gecombineerd.combine_first(df[kolom])

        df[target] = gecombineerd
        verwijderen = [kolom for kolom in bronkolommen if kolom != target]
        if verwijderen:
            df = df.drop(columns=verwijderen)

    # Stop met een duidelijke diagnose als een verplichte bronkolom echt ontbreekt.
    verplichte_kolommen = [
        'Datum', 'Meetpunt', 'Stof', 'Waarde', 'Eenheid', 'Limietsymbool',
        'hoedanigheid', 'Latitude', 'Longitude',
        'Hoedanigheid_Omschr', 'Eenheid_Omschr'
    ]
    ontbrekende_kolommen = [
        kolom for kolom in verplichte_kolommen if kolom not in df.columns
    ]
    if ontbrekende_kolommen:
        raise ValueError(
            "De dataset mist verplichte kolommen na normalisatie: "
            f"{ontbrekende_kolommen}. Aangetroffen bronkolommen: "
            f"{list(df.columns)}"
        )

    df['hoedanigheid'] = df['hoedanigheid'].astype(str).str.strip().str.lower()
    df['Stof'] = df['Stof'].astype(str).str.strip()
    df['Limietsymbool'] = (
        df['Limietsymbool']
        .fillna('')
        .astype(str)
        .str.strip()
        .replace({'nan': '', 'None': '', '<NA>': ''})
    )
    
    df['Hoedanigheid_Omschr_lower'] = df['Hoedanigheid_Omschr'].astype(str).str.strip().str.lower()
    df['Eenheid_Omschr_lower'] = df['Eenheid_Omschr'].astype(str).str.strip().str.lower()
    
    mask_nvt = df['Stof'] == 'NVT'
    
    mask_gadolinium_antropogeen = (
        (df['Stof'].str.lower() == 'gadolinium') & 
        (df['Eenheid_Omschr_lower'] == 'dimensieloos')
    )
    
    conditions = [
        mask_nvt & (df['Hoedanigheid_Omschr_lower'].str.contains('calciumcarbonaat', na=False)), 
        mask_nvt & (df['Hoedanigheid_Omschr_lower'].str.contains('t.o.v. 20 graden celsius', na=False)), 
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'decimeter'), 
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'dimensieloos') & (df['Waarde'] < 3),
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'dimensieloos') & (df['Waarde'] > 3),
        mask_nvt & (df['Eenheid_Omschr_lower'].str.contains('formazine nephelometric unit', na=False)), 
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'graad celsius'),
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'per meter'),
        mask_gadolinium_antropogeen
    ]
    
    new_values = [
        'hardheid',
        'geleidbaarheid',
        'doorzicht',
        'saliniteit',
        'zuurgraad',
        'troebelheid',
        'temperatuur',
        'extinctie',
        'gadolinium (antropogeen)'
    ]
    
    df['Stof'] = np.select(conditions, new_values, default=df['Stof'])
    
    df = df.drop(columns=[
        'Hoedanigheid_Omschr', 
        'Eenheid_Omschr', 
        'Hoedanigheid_Omschr_lower', 
        'Eenheid_Omschr_lower'
    ])

    cond_opgelost = df['hoedanigheid'].str.contains('nf|filtratie|opgeloste', na=False)
    suffix = " (totaal)"
    df['suffix'] = np.where(cond_opgelost, " (opgelost)", suffix)
    df['Stof'] = df['Stof'] + df['suffix']
    df = df.drop(columns=['suffix'])
    
    df['Stof'] = df['Stof'].str.lower()
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d', errors='coerce')
    # Ondersteun numerieke waarden met zowel een decimale punt als decimale komma.
    for numerieke_kolom in ['Waarde', 'Latitude', 'Longitude']:
        df[numerieke_kolom] = (
            df[numerieke_kolom]
            .astype('string')
            .str.strip()
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        df[numerieke_kolom] = pd.to_numeric(
            df[numerieke_kolom], errors='coerce'
        )

    df = df[df['Waarde'] != 999999999999]
    df = df.dropna(subset=['Waarde', 'Datum', 'Meetpunt', 'Stof']).copy()

    for col in ['Meetpunt', 'Eenheid']:
        df[col] = df[col].astype('category')

    try:
        df_normen = pd.read_csv(NORMEN_FILE_PATH, delimiter=',', low_memory=False, encoding='latin-1')
        df_normen = df_normen.rename(columns={
            'Stofnaam': 'Stof',
            'Norm': 'NormType',
            'Waarde': 'NormWaarde'
        })

        df_normen['Stof'] = df_normen['Stof'].astype(str).str.strip()
        
        norm_type_str = df_normen['NormType'].astype(str).str.lower()
        cond_norm_opgelost = norm_type_str.str.contains('opgelost')
        cond_norm_totaal = norm_type_str.str.contains('totaal')
        
        df_normen['suffix'] = ''
        df_normen.loc[cond_norm_opgelost, 'suffix'] = ' (opgelost)'
        df_normen.loc[cond_norm_totaal, 'suffix'] = ' (totaal)'
        
        df_normen['Stof'] = (df_normen['Stof'] + df_normen['suffix']).str.lower()

        cond_jg = norm_type_str.str.contains('jg-mkn|jaargemiddelde')
        cond_mac = norm_type_str.str.contains('mac-mkn|maximaal')
        
        df_normen['NormCode'] = None
        df_normen.loc[cond_jg, 'NormCode'] = 'JG_MKN'
        df_normen.loc[cond_mac, 'NormCode'] = 'MAC_MKN'

        df_normen_filtered = df_normen.dropna(subset=['NormCode']).copy()

        df_normen_filtered['NormWaarde'] = (
            df_normen_filtered['NormWaarde']
            .astype(str)
            .str.replace(',', '.', regex=False)
        )
        df_normen_filtered['NormWaarde'] = pd.to_numeric(df_normen_filtered['NormWaarde'], errors='coerce')

        df_normen_pivot = df_normen_filtered.pivot_table(
            index='Stof',
            columns='NormCode',
            values='NormWaarde',
            aggfunc='first'
        ).reset_index()

        df = pd.merge(
            df,
            df_normen_pivot,
            on='Stof',
            how='left'
        )

    except FileNotFoundError:
        st.error(f"Koppeltabel KRW-normen niet gevonden op pad: {NORMEN_FILE_PATH}.")
        df['JG_MKN'] = np.nan
        df['MAC_MKN'] = np.nan

    if 'JG_MKN' not in df.columns: df['JG_MKN'] = np.nan
    
    df['KRW_Norm'] = df['JG_MKN']

    # Pas de achtergrondcorrectie centraal toe. Alle vervolgfuncties en pagina's
    # gebruiken daarna de gecorrigeerde kolom 'Waarde'; de bronwaarde blijft
    # beschikbaar in 'Waarde_Origineel'.
    df = apply_achtergrondcorrectie(df)

    df['Signaleringswaarde'] = np.nan

    # Koppel op de basisstofnaam, zodat '(totaal)' en '(opgelost)'
    # automatisch dezelfde stofspecifieke signaleringswaarde krijgen.
    base_stofnaam = (
        df['Stof']
        .astype(str)
        .str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True)
        .str.strip()
        .str.casefold()
    )
    eenheid_is_ug_l = (
        df['Eenheid'].astype(str).str.strip().str.casefold().eq('ug/l')
    )
    heeft_geen_jg_mkn = df['JG_MKN'].isna()
    uitgesloten_elementen = {
        str(stof).strip().casefold()
        for stof in UITGESLOTEN_ELEMENTEN
    }

    df_signalering = load_signaleringswaarden()
    signaleringswaarde_map = df_signalering.set_index(
        'stofnaam'
    )['signaleringswaarde']
    gekoppelde_signaleringswaarde = base_stofnaam.map(
        signaleringswaarde_map
    )

    # Een match uit de koppeltabel heeft voorrang, met behoud van de bestaande
    # voorwaarden: alleen ug/l en alleen wanneer geen JG-MKN beschikbaar is.
    masker_gekoppeld = (
        heeft_geen_jg_mkn
        & eenheid_is_ug_l
        & gekoppelde_signaleringswaarde.notna()
    )
    df.loc[masker_gekoppeld, 'Signaleringswaarde'] = (
        gekoppelde_signaleringswaarde.loc[masker_gekoppeld]
    )

    # Bij geen match blijft de generieke 0,1 ug/l gelden, behalve voor de
    # stoffen die expliciet in UITGESLOTEN_ELEMENTEN staan.
    masker_generiek = (
        heeft_geen_jg_mkn
        & eenheid_is_ug_l
        & gekoppelde_signaleringswaarde.isna()
        & ~base_stofnaam.isin(uitgesloten_elementen)
    )
    df.loc[masker_generiek, 'Signaleringswaarde'] = 0.1

    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    unieke_stoffen = df['Stof'].unique()
    stof_map = match_stofgroep_optimized(unieke_stoffen)
    df['Stofgroep'] = df['Stof'].map(stof_map).astype('category')

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

def get_shared_sidebar(df_main):
    st.sidebar.header("📅 Filter op jaren")
    
    if not df_main.empty and 'Datum' in df_main.columns:
        beschikbare_jaren = sorted(df_main['Datum'].dt.year.dropna().unique(), reverse=True)
    else:
        beschikbare_jaren = []
    
    geselecteerde_jaren = st.sidebar.multiselect(
        "Selecteer gewenste jaren:",
        options=beschikbare_jaren,
        default=beschikbare_jaren
    )

    if geselecteerde_jaren and not df_main.empty:
        df_filtered = df_main[df_main['Datum'].dt.year.isin(geselecteerde_jaren)].copy()
    else:
        df_filtered = df_main.copy()
        
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
    st.sidebar.info("Navigeer via het menu hierboven naar de verschillende analyses.")
    
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