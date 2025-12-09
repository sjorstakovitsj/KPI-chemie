import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# --- CONFIGURATIE EN CONSTANTEN ---

DATA_FILE_PATH = 'IJG Chemie.csv'
NORMEN_FILE_PATH = 'KRW stoffen koppeltabel.csv'
PFAS_FILE_PATH = 'PFAS PEQ koppeltabel.csv'

# Definieer zoeklijsten eenmalig buiten de functies (Constanten)
STOFGROEPEN_MAPPING = {
    'PFAS': [
        'perfluor', 'genx', 'adona', 'pfhpa', 'fluortelomeer', 'pfas', '9-chloorhexadecaanfluor-3-oxanon-1-sulfonzuur',
        'trifluor', 'tridecafluor', '10:2', '8:2', '6:2', '4:2', '11-chlooreicosafluor-3-oxaundecaan-1-sulfonzuur',
        'som hexadecafluor-2-deceenzuur-isomerenâ'
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
        "2,4'-dichloordifenyltrichloorethaan" 
        
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
        'guanylureum', 'avobenzone', 'octocrilene', 'paroxetine', 'fluoxetine', 'fenofibrinezuur'
    ],
    'PAKs/PCBs/PBDEs': [
        'naftaleen', 'antraceen', 'fenantreen', 'fluorantheen', 'benzo(a)', 'benzo(g', 'benzo(k',
        'chryseen', 'pyreen', 'dibenzo', 'indeno', 'benzo(b)', 'pcb', 'broomdiphenylether',
        'broomdifenylether', 'chloorbifenyl', 'acenaftyleen'
    ],
    'Vluchtige organische stoffen': [
        'benzeen', 'tolueen', 'etylbenzeen', 'xyleen', 'styreen', 'chloorbenzeen', 'chlooretheen',
        'dichloorethaan', 'dichloorpropeen', 'cumeen', 'cyclohexaan', 'methaan', 'dicyclopentadieen',
        'ether', 'disulfide', 'hydrine', 'etheen', 'chloortolueen', 'propylbenzeen', 'tetrahydrofuran',
        '1,2-dimethoxyethaan', '1,1,2,2-tetrachloorethaan', '1,3-dichloorpropaan', '1,2-dichloorpropaan',
        '3-chloorpropeen', '1,2,3-trichloorpropaan', '1,1,1-trichloorethaan', '1,1,2-trichloorethaan',
        'tetrachloorethaan'
    ],
    'Industrie & overigen': [
        'bisfenol', 'chloorbenzeen', 'chloorfenol', 'nitrofenol', 'dtpa', 'methacrylaat', 'nta',
        'edta', 'pyrazol', 'melamine', 'difenol', 'cyaanguanidine', 'cyanuurzuur', 'urotropine',
        'ftalaat', 'acesulfaam', 'cyclamaat', 'saccharine', 'sucralose', 'fosfaat', 'vinylchloride',
        '4-tertiair-octylfenol', 'som 4-nonylfenol-isomeren (vertakt)', 'melamine'
    ],
    'Nutriënten & algemeen': [
        'fluoride', 'zuurstof', 'chlorofyl', 'silicium', 'sulfaat', 'koolstof', 'stikstof',
        'nitraat', 'nitriet', 'ammonium', 'fosfor', 'fosfaat', 'chloride', 'zwevende stof',
        'hardheid', 'temperatuur', 'zuurgraad', 'geleidbaarheid', 'gloeirest', 'onopgeloste',
        'doorzicht', 'saliniteit', 'troebelheid', 'cyanide', 'bicarbonaat', 'waterstofcarbonaat',
        'extinctie'
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
        'arseniet', 'selenaat', 'seleniet'
    ],

}

UITGESLOTEN_ELEMENTEN = [
    'aluminium', 'ammonium', 'antimoon', 'arseen', 'barium', 'beryllium', 'boor',
    'cadmium', 'cerium', 'cesium', 'chloride', 'chroom', 'calcium', 'cobalt', 'kobalt', 'koper', 'kwik',
    'lood', 'magnesium', 'mangaan', 'molybdeen', 'natrium', 'nikkel', 'dysprosium', 'erbium', 'europium',
    'kalium', 'seleen', 'silicium', 'strontium', 'thallium', 'tin', 'gadolinium', 'gallium', 'hafnium',
    'titanium', 'uranium', 'vanadium', 'ijzer', 'zilver', 'zink', 'holmium', 'indium', 'koolstof organisch',
    'lanthaan', 'lithium', 'lutetium', 'neodymium', 'niobium', 'nitraat', 'nitriet', 'platina', 'praseodymium',
    'rubidium', 'samarium', 'siliciumdioxide', 'sulfaat', 'tantalium', 'tellurium',
    'terbium', 'thallium', 'thorium', 'thulium', 'wolfraam', 'ytterbium', 'yttrium', 'zirkonium',
    'titaan', 'scandium', 'chlorofyl-a'
]


# --- FUNCTIES ---

def match_stofgroep_optimized(unieke_stoffen):
    """
    Maakt een dictionary mapping aan voor stofnaam -> stofgroep.
    Dit is veel sneller dan per rij checken.
    """
    mapping = {}
    # Volgorde in STOFGROEPEN_MAPPING is belangrijk; eerder gematcht = definitief
    # (Tenzij je specifieke prioriteit wilt, pas dan de volgorde in de dict aan)
    
    for stof in unieke_stoffen:
        s_lower = stof.lower()
        gevonden = False
        for groep, keywords in STOFGROEPEN_MAPPING.items():
            # Gebruik any() in een generator expression (snel)
            if any(k in s_lower for k in keywords):
                mapping[stof] = groep
                gevonden = True
                break # Stop bij eerste match
        
        if not gevonden:
            mapping[stof] = 'Onbekend'
            
    return mapping


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    # 1. Laden Meetdata
    try:
        df = pd.read_csv(file_path, delimiter=';', low_memory=False, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Bestand niet gevonden op pad: {file_path}.")
        return pd.DataFrame()

    df = df.rename(columns={
        'eventdatum': 'Datum',
        'locatie_code': 'Meetpunt',
        'parameter_omschrijving': 'Stof',
        'event_waarde': 'Waarde',
        'eenheid_code': 'Eenheid',
        'event_waarde_limietsymbool': 'Limietsymbool',
        'hoedanigheid_code': 'hoedanigheid',
        'locatie_lat_etrs89': 'Latitude',
        'locatie_lon_etrs89': 'Longitude',
        # TOEVOEGEN: Kolommen voor de NVT-transformatie
        'hoedanigheid_omschrijving': 'Hoedanigheid_Omschr', 
        'eenheid_omschrijving': 'Eenheid_Omschr',
    })

    # Optimalisatie: Vectorized string operations
    df['hoedanigheid'] = df['hoedanigheid'].astype(str).str.strip().str.lower()
    df['Stof'] = df['Stof'].astype(str).str.strip()
    df['Limietsymbool'] = df['Limietsymbool'].astype(str).replace('nan', '').fillna('')
    
    # ----------------------------------------------------------------------------------
    # --- NIEUWE LOGICA: NVT-STOFFEN DETECTIE (Zoeken op Omschrijving en Waarde) ---
    # ----------------------------------------------------------------------------------
    
    # 1. Standaardiseer de tekstkolommen voor filtering
    df['Hoedanigheid_Omschr_lower'] = df['Hoedanigheid_Omschr'].astype(str).str.strip().str.lower()
    df['Eenheid_Omschr_lower'] = df['Eenheid_Omschr'].astype(str).str.strip().str.lower()
    
    # 2. Definieer de condities
    mask_nvt = df['Stof'] == 'NVT' # Basismarkering voor NVT
    
    # NIEUWE CONDITIE: mask_gadolinium
    mask_gadolinium = df['Stof'].str.lower() == 'gadolinium'
    
    conditions = [
        # 1. Hardheid
        mask_nvt & (df['Hoedanigheid_Omschr_lower'].str.contains('calciumcarbonaat', na=False)), 
        # 2. Geleidbaarheid
        mask_nvt & (df['Hoedanigheid_Omschr_lower'].str.contains('t.o.v. 20 graden celsius', na=False)), 
        # 3. Doorzicht
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'decimeter'), 
        # 4. Saliniteit: dimensieloos EN Waarde < 3
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'dimensieloos') & (df['Waarde'] < 3),
        # 5. Zuurgraad (pH): dimensieloos EN Waarde > 3
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'dimensieloos') & (df['Waarde'] > 3),
        # 6. Troebelheid
        mask_nvt & (df['Eenheid_Omschr_lower'].str.contains('formazine nephelometric unit', na=False)), 
        # 7. Temperatuur
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'graad celsius'),
        # 8. Extinctie
        mask_nvt & (df['Eenheid_Omschr_lower'] == 'per meter'),
        # 9. Gadolinium (Antropogeen)
        mask_gadolinium & (df['Hoedanigheid_Omschr_lower'].str.contains('antropogeen / opgeloste fractie', na=False)),
    ]
    
    # 3. Corresponderende waarden
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
    
    # 4. Voer de transformatie uit met np.select (houd de oude 'Stof' als default)
    df['Stof'] = np.select(conditions, new_values, default=df['Stof'])
    
    # 5. Verwijder de tijdelijke en nu onnodige omschrijvingskolommen voor geheugenefficiëntie
    df = df.drop(columns=[
        'Hoedanigheid_Omschr', 
        'Eenheid_Omschr', 
        'Hoedanigheid_Omschr_lower', 
        'Eenheid_Omschr_lower'
    ])

    # Optimalisatie: Vervang 'specificeer_stofnaam' apply door vectorized numpy/pandas logic
    cond_opgelost = df['hoedanigheid'].str.contains('nf|filtratie|opgeloste', na=False)
    # Standaard suffix
    suffix = " (totaal)"
    # Update suffix waar nodig
    df['suffix'] = np.where(cond_opgelost, " (opgelost)", suffix)
    df['Stof'] = df['Stof'] + df['suffix']
    df = df.drop(columns=['suffix']) # Opruimen
    
    df['Stof'] = df['Stof'].str.lower()
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d', errors='coerce')
    df['Waarde'] = pd.to_numeric(df['Waarde'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # Filtering
    df = df[df['Waarde'] != 999999999999]
    df = df.dropna(subset=['Waarde', 'Datum', 'Meetpunt', 'Stof']).copy()

    # Optimalisatie: Geheugenbesparing door categories
    for col in ['Meetpunt', 'Eenheid']:
        df[col] = df[col].astype('category')

    # 2. KRW Normen Inlezen en Koppelen
    try:
        df_normen = pd.read_csv(NORMEN_FILE_PATH, delimiter=',', low_memory=False, encoding='latin-1')
        df_normen = df_normen.rename(columns={
            'Stofnaam': 'Stof',
            'Norm': 'NormType',
            'Waarde': 'NormWaarde'
        })

        df_normen['Stof'] = df_normen['Stof'].astype(str).str.strip()
        
        # Vectorized norm matching logic
        norm_type_str = df_normen['NormType'].astype(str).str.lower()
        cond_norm_opgelost = norm_type_str.str.contains('opgelost')
        cond_norm_totaal = norm_type_str.str.contains('totaal')
        
        df_normen['suffix'] = ''
        df_normen.loc[cond_norm_opgelost, 'suffix'] = ' (opgelost)'
        df_normen.loc[cond_norm_totaal, 'suffix'] = ' (totaal)'
        # Als er geen match is, blijft suffix leeg (zoals origineel 'return stof')
        
        df_normen['Stof'] = (df_normen['Stof'] + df_normen['suffix']).str.lower()

        # NormCode logic mapping
        # JG_MKN
        cond_jg = norm_type_str.str.contains('jg-mkn|jaargemiddelde')
        # MAC_MKN
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

    # 3. Signaleringswaarden Berekenen
    # Zorg dat de kolommen bestaan
    if 'JG_MKN' not in df.columns: df['JG_MKN'] = np.nan
    
    df['KRW_Norm'] = df['JG_MKN']
    df['Signaleringswaarde'] = np.nan

    # Bereken masker voor signaleringswaarde
    # Eerst basis stofnaam isoleren (vectorized)
    base_stofnaam = df['Stof'].str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True).str.strip()
    is_metaal_of_element = base_stofnaam.isin(UITGESLOTEN_ELEMENTEN)

    masker = (
        df['JG_MKN'].isna() &
        (df['Eenheid'].astype(str).str.lower() == 'ug/l') &
        (~is_metaal_of_element)
    )

    df.loc[masker, 'Signaleringswaarde'] = 0.1

    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Optimalisatie Stofgroep Bepaling ---
    # Haal unieke stoffen op (veel minder dan aantal rijen)
    unieke_stoffen = df['Stof'].unique()
    # Maak een map
    stof_map = match_stofgroep_optimized(unieke_stoffen)
    # Map terug naar de dataframe
    df['Stofgroep'] = df['Stof'].map(stof_map).astype('category')

    return df


@st.cache_data
def load_pfas_ref(file_path: str) -> pd.DataFrame:
    try:
        df_pfas = pd.read_csv(file_path, dtype=str)
        df_pfas.columns = df_pfas.columns.str.strip()

        cols_to_fix = ['RPF', 'RBF']
        for col in cols_to_fix:
            if col in df_pfas.columns:
                df_pfas[col] = df_pfas[col].str.replace(',', '.', regex=False)
                df_pfas[col] = pd.to_numeric(df_pfas[col], errors='coerce').fillna(0)

        return df_pfas
    except FileNotFoundError:
        st.error(f"PFAS bestand niet gevonden: {file_path}")
        return pd.DataFrame()


def create_gauge(percentage: float, title_text: str = "Metingen onder Norm", drempel: int = 95) -> go.Figure:
    # Veilige afhandeling van NaN of None
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


def main():

    st.set_page_config(layout="wide", page_title="Waterkwaliteit KPI Dashboard")

    st.title("💧 Dashboard chemische waterkwaliteit MN")
    if 'last_update' not in st.session_state:
         st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"**Laatste update:** {st.session_state.last_update}")

    df_main = load_data(DATA_FILE_PATH)

    if df_main.empty:
        st.error("🚨 Kritieke Fout: De geladen en opgeschoonde data is leeg.")
        st.stop()
        
    st.markdown("---")
    st.subheader("📅 Filter op jaren")
    
    # Jaren ophalen uit de dataset
    beschikbare_jaren = sorted(df_main['Datum'].dt.year.dropna().unique(), reverse=True)
    
    # Multiselectbox
    geselecteerde_jaren = st.multiselect(
        "Selecteer gewenste jaren:",
        options=beschikbare_jaren,
        default=beschikbare_jaren
    )

    # Data filteren op basis van selectie
    if geselecteerde_jaren:
        df_filtered = df_main[df_main['Datum'].dt.year.isin(geselecteerde_jaren)].copy()
    else:
        df_filtered = df_main.copy()
    
    st.markdown("---")    

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Toestand & Trend",
        "✅ KRW normcheck",
        "📊 Overzicht Toestand & Kaart",
        "⚠️ Risicoanalyse (Signaleringswaarden)",
        "Effectbeoordeling PFAS",
        "Herkomst en verdeling van stoffen",
        "Ruimtelijke analyse"
    ])

    # --- TAB 1: Toestand & Trend Ontwikkeling ---
    with tab1:
        st.header("📈 Toestand en trendontwikkeling")
        st.info("Voor enkele stoffen is correctie met achtergrondwaardes van toepassing voor de KRW. In deze tool is dat niet meegenomen.")

        st.write("### 🔍 Selectie filters")
        stof_filter_type = st.radio(
            "Welk type stoffen wil je selecteren?",
            options=["Alle stoffen", "KRW-stoffen", "Niet-genormeerde stoffen"],
            index=0,
            horizontal=True
        )

        # Pre-calculatie unieke stoffen op basis van radio button
        unique_stoffen_series = df_main['Stof'].unique()
        
        if "KRW-stoffen" in stof_filter_type:
            # Filter op rows die wel normen hebben
            # We gebruiken df_main hier om alle mogelijke stoffen te tonen, of df_filtered? Meestal filtered.
            # Laten we df_main gebruiken voor de optielijst om 'lege' selecties in gefilterde jaren te voorkomen,
            # of juist df_filtered zodat je alleen kiest wat er is. De originele code gebruikte df_main voor de opties.
            
            # Check welke stoffen normen hebben
            mask_norm = df_main['JG_MKN'].notna() | df_main['MAC_MKN'].notna()
            beschikbare_stoffen = df_main.loc[mask_norm, 'Stof'].unique()
            
        elif "Niet-genormeerde stoffen" in stof_filter_type:
            mask_norm = df_main['JG_MKN'].notna() | df_main['MAC_MKN'].notna()
            beschikbare_stoffen = df_main.loc[~mask_norm, 'Stof'].unique()
        else:
            beschikbare_stoffen = unique_stoffen_series

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            sorted_meetpunten = sorted(df_filtered['Meetpunt'].unique())
            default_meetpunten = st.session_state.get("tab1_meetpunten_select", sorted_meetpunten[:1] if sorted_meetpunten else [])
            selected_meetpunten = st.multiselect(
                "Selecteer Meetpunt(en)",
                options=sorted_meetpunten,
                default=default_meetpunten,
                key="tab1_meetpunten_select"
            )

        with col2:
            stofgroep_opties = sorted(df_filtered['Stofgroep'].unique())
            stofgroep_selected = st.multiselect(
                "Stofgroep",
                options=stofgroep_opties,
                default=stofgroep_opties[0] if stofgroep_opties else None,
                key="tab1_stofgroep_select"
            )

        with col3:
            # Filter beschikbare stoffen op basis van geselecteerde stofgroep en eerdere filter
            if stofgroep_selected:
                # Eerst filteren op groep
                stoffen_in_groep = df_filtered[df_filtered['Stofgroep'].isin(stofgroep_selected)]['Stof'].unique()
                # Intersectie met radio button filter
                beschikbare_stoffen = np.intersect1d(beschikbare_stoffen, stoffen_in_groep)
            
            beschikbare_stoffen = sorted(beschikbare_stoffen)

            current_selection = st.session_state.get("tab1_stoffen_select", [])
            valid_default = [s for s in current_selection if s in beschikbare_stoffen]

            if not valid_default and len(beschikbare_stoffen) > 0:
                valid_default = [beschikbare_stoffen[0]]

            selected_stoffen = st.multiselect(
                "Selecteer Stof(fen)",
                options=beschikbare_stoffen,
                default=valid_default,
                key="tab1_stoffen_select"
            )

        if selected_meetpunten and selected_stoffen:
            df_trend = df_filtered[
                (df_filtered['Meetpunt'].isin(selected_meetpunten)) &
                (df_filtered['Stof'].isin(selected_stoffen))
            ].copy()

            if not df_trend.empty:
                df_trend['Meting Type'] = np.where(
                    df_trend['Limietsymbool'].astype(str).str.contains('<'),
                    '< Onder rapportagegrens',
                    'Gemeten waarde'
                )
                
                unique_stoffen = sorted(df_trend['Stof'].unique())
                
                # Optimalisatie: Groupby aggregeren is sneller dan dict comprehension over unique
                # Maar we hebben JG/MAC MKN nodig per stof. Dat is constant per stof.
                # We pakken de eerste niet-nan waarde.
                stof_info_df = df_trend.groupby('Stof', observed=True)[['Eenheid', 'JG_MKN', 'MAC_MKN']].first()
                
                fig = px.scatter(
                    df_trend,
                    x='Datum',
                    y='Waarde',
                    color='Meetpunt',
                    symbol='Meting Type',
                    symbol_map={
                        '< Onder rapportagegrens': 'x',
                        'Gemeten waarde': 'circle'
                    },
                    facet_row='Stof',
                    title='Geselecteerde metingen over tijd',
                    hover_data={'Datum': True, 'Waarde': ':.4f', 'Eenheid': True, 'Meetpunt': True},
                    category_orders={"Stof": unique_stoffen},
                    height=350 * len(unique_stoffen)
                )

                fig.update_xaxes(matches='x')
                fig.update_yaxes(matches=None)

                for i, stof_naam in enumerate(unique_stoffen):
                    row_index = len(unique_stoffen) - i
                    
                    if stof_naam in stof_info_df.index:
                        info = stof_info_df.loc[stof_naam]
                        eenheid = info['Eenheid']
                        jg_norm = info['JG_MKN']
                        mac_norm = info['MAC_MKN']
                        
                        # Max waarde voor scaling
                        max_waarde_data = df_trend[df_trend['Stof'] == stof_naam]['Waarde'].max()
                        current_max = max_waarde_data if pd.notna(max_waarde_data) else 0

                        vergelijkings_waarden = [current_max]
                        if pd.notna(jg_norm): vergelijkings_waarden.append(jg_norm)
                        if pd.notna(mac_norm): vergelijkings_waarden.append(mac_norm)
                        
                        target_top = max(vergelijkings_waarden) * 1.15 if vergelijkings_waarden else 10

                        fig.update_yaxes(
                            title_text=f"Waarde ({eenheid})",
                            range=[0, target_top],
                            row=row_index,
                            col=1,
                            showticklabels=True
                        )

                        if pd.notna(jg_norm):
                            fig.add_hline(
                                y=jg_norm, line_dash="dash", line_color="darkorange", line_width=2,
                                row=row_index, col=1,
                                annotation_text=f"JG: {jg_norm:.4f}", annotation_position="top right"
                            )

                        if pd.notna(mac_norm):
                            fig.add_hline(
                                y=mac_norm, line_dash="dot", line_color="red", line_width=2,
                                row=row_index, col=1,
                                annotation_text=f"MAC: {mac_norm:.2f}", annotation_position="top left"
                            )

                fig.update_layout(margin=dict(l=80, r=80))
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("Geen data gevonden voor de geselecteerde combinatie.")
        else:
            st.info("Selecteer minstens één meetpunt en één stof om de trend te zien.")

    # --- TAB 2: Normtoetsing met KRW (JG-MKN en MAC-MKN) ---
    with tab2:
        st.header("✅ KRW normcheck per meetpunt")
        
        col_filter_1, col_filter_2, col_filter_3 = st.columns(3)

        with col_filter_1:
            all_meetpunten_norm = sorted(df_main['Meetpunt'].unique())
            default_meetpunten_norm = all_meetpunten_norm[:3] if all_meetpunten_norm else []
            selected_meetpunten_norm = st.multiselect(
                "📍 Selecteer meetpunt(en)",
                options=all_meetpunten_norm,
                default=default_meetpunten_norm,
                key="tab2_meetpunten_select"
            )

        with col_filter_2:
            stofgroep_opties = sorted(df_filtered['Stofgroep'].unique())
            stofgroep_selected_tab2 = st.multiselect(
                "Selecteer stofgroep",
                options=stofgroep_opties,
                default=stofgroep_opties[0] if stofgroep_opties else None,
                key="tab2_stofgroep_select"
            )

        with col_filter_3:
            if stofgroep_selected_tab2:
                filtered_stoffen_df = df_filtered[df_filtered['Stofgroep'].isin(stofgroep_selected_tab2)]
                all_stoffen_norm = sorted(filtered_stoffen_df['Stof'].unique())
            else:
                all_stoffen_norm = sorted(df_filtered['Stof'].unique())

            # Slimme defaults: stoffen met norm
            stoffen_met_norm = df_filtered.dropna(subset=['JG_MKN', 'MAC_MKN'])['Stof'].unique()
            mogelijke_defaults = np.intersect1d(stoffen_met_norm, all_stoffen_norm)
            
            default_stoffen_norm = sorted(mogelijke_defaults)[:3] if len(mogelijke_defaults) > 0 else []

            selected_stoffen_norm = st.multiselect(
                "🔎 Selecteer stof(fen)",
                options=all_stoffen_norm,
                default=default_stoffen_norm,
                key="tab2_stoffen_select"
            )

        if not selected_stoffen_norm or not selected_meetpunten_norm:
            st.warning("Selecteer minstens één stof en één meetpunt.")
        else:
            # JG-MKN Toetsing
            st.subheader("Jaargemiddelde toetsing (JG-MKN)")
            
            # Filter eerst, daarna berekenen
            df_jg = df_filtered[
                (df_filtered['Stof'].isin(selected_stoffen_norm)) &
                (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
                (df_filtered['JG_MKN'].notna())
            ].copy()

            if not df_jg.empty:
                df_jg['Jaar'] = df_jg['Datum'].dt.year
                
                # Bereken jaargemiddelde
                # observed=True is belangrijk bij categorical data voor performance
                df_gemiddelde = df_jg.groupby(['Jaar', 'Meetpunt', 'Stof'], observed=True)['Waarde'].mean().reset_index()
                
                # Voeg norm toe (is constant per stof)
                norm_lookup = df_jg[['Stof', 'JG_MKN']].drop_duplicates()
                df_gemiddelde = pd.merge(df_gemiddelde, norm_lookup, on='Stof', how='left')

                df_gemiddelde['Overschrijding'] = df_gemiddelde['Waarde'] > df_gemiddelde['JG_MKN']

                jg_overschrijding_pct = df_gemiddelde.groupby(['Meetpunt', 'Stof'], observed=True)['Overschrijding'].mean().reset_index()
                jg_overschrijding_pct['Overschrijding (%)'] = (jg_overschrijding_pct['Overschrijding'] * 100).round(2)

                # Convert categories to str for plotting labels
                jg_overschrijding_pct['PlotLabel'] = jg_overschrijding_pct['Meetpunt'].astype(str) + ' (' + jg_overschrijding_pct['Stof'].astype(str) + ')'

                fig_jg = px.bar(
                    jg_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
                    x='PlotLabel',
                    y='Overschrijding (%)',
                    title='Percentage jaren boven JG-MKN norm',
                    color='Meetpunt',
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig_jg, use_container_width=True)
            else:
                st.info("Geen JG-MKN data voor selectie.")

            st.markdown("---")

            # MAC-MKN Toetsing
            st.subheader("Maximale aanvaardbare concentratie (MAC-MKN)")
            
            df_mac = df_filtered[
                (df_filtered['Stof'].isin(selected_stoffen_norm)) &
                (df_filtered['Meetpunt'].isin(selected_meetpunten_norm)) &
                (df_filtered['MAC_MKN'].notna())
            ].copy()

            if not df_mac.empty:
                df_mac['Overschrijding'] = df_mac['Waarde'] > df_mac['MAC_MKN']

                mac_overschrijding_pct = df_mac.groupby(['Meetpunt', 'Stof'], observed=True)['Overschrijding'].mean().reset_index()
                mac_overschrijding_pct['Overschrijding (%)'] = (mac_overschrijding_pct['Overschrijding'] * 100).round(2)

                mac_overschrijding_pct['PlotLabel'] = mac_overschrijding_pct['Meetpunt'].astype(str) + ' (' + mac_overschrijding_pct['Stof'].astype(str) + ')'

                fig_mac = px.bar(
                    mac_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
                    x='PlotLabel', y='Overschrijding (%)',
                    title='Percentage metingen boven MAC-MKN',
                    color='Meetpunt',
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig_mac, use_container_width=True)
            else:
                st.info("Geen MAC-MKN data voor selectie.")

    # --- TAB 3: In één Oogopslag Toestand & Kaart ---
    with tab3:
        st.header("📊 Normoverschrijdingen KPI")
        
        # Berekeningen over totale gefilterde set
        col_kpi_info, col_kpi_gauges, col_map = st.columns([1, 2, 2])

        # JG Overall
        df_jg_overall = df_filtered.dropna(subset=['JG_MKN'])
        pct_jg_total = 0
        if not df_jg_overall.empty:
            voldoet = (df_jg_overall['Waarde'] <= df_jg_overall['JG_MKN']).sum()
            totaal = len(df_jg_overall)
            pct_jg_total = round((voldoet / totaal * 100), 1)

        # MAC Overall
        df_mac_overall = df_filtered.dropna(subset=['MAC_MKN'])
        pct_mac_total = 0
        if not df_mac_overall.empty:
            voldoet = (df_mac_overall['Waarde'] <= df_mac_overall['MAC_MKN']).sum()
            totaal = len(df_mac_overall)
            pct_mac_total = round((voldoet / totaal * 100), 1)

        with col_kpi_info:
            st.metric(label="Unieke meetpunten", value=df_main['Meetpunt'].nunique())
            st.metric(label="Metingen met JG-norm", value=len(df_jg_overall))
            st.metric(label="Metingen met MAC-norm", value=len(df_mac_overall))

        with col_kpi_gauges:
            sub1, sub2 = st.columns(2)
            sub1.plotly_chart(create_gauge(pct_jg_total, "Totaal: voldoet JG (%)"), use_container_width=True)
            sub2.plotly_chart(create_gauge(pct_mac_total, "Totaal: voldoet MAC (%)"), use_container_width=True)

        with col_map:
            df_map = df_filtered[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()
            if not df_map.empty:
                center_lat = df_map['Latitude'].mean()
                center_lon = df_map['Longitude'].mean()
                
                fig_map = px.scatter_mapbox(
                    df_map, lat='Latitude', lon='Longitude', hover_name='Meetpunt',
                    size_max=15, zoom=8, mapbox_style="open-street-map"
                )
                fig_map.update_traces(marker=dict(size=12, color='red'))
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, mapbox_center={"lat": center_lat, "lon": center_lon})
                st.plotly_chart(fig_map, use_container_width=True)

        st.divider()

        # KPI per specifiek meetpunt (Expanders zijn zwaar als het er veel zijn, beperk evt)
        unieke_meetpunten = sorted(df_filtered['Meetpunt'].unique())
        
        # We berekenen alles vooraf in één keer i.p.v. per loop
        # JG Stats per punt
        jg_stats = df_jg_overall.groupby('Meetpunt', observed=True).apply(
            lambda x: (x['Waarde'] <= x['JG_MKN']).mean() * 100
        )
        # MAC Stats per punt
        mac_stats = df_mac_overall.groupby('Meetpunt', observed=True).apply(
            lambda x: (x['Waarde'] <= x['MAC_MKN']).mean() * 100
        )

        with st.expander(f"Toon detailmeters voor alle {len(unieke_meetpunten)} meetpunten", expanded=True):
            for meetpunt in unieke_meetpunten:
                pct_jg_mp = jg_stats.get(meetpunt, None)
                pct_mac_mp = mac_stats.get(meetpunt, None)
                
                # Check of er data is voor dit punt in de gefilterde set (soms is punt er wel, maar geen normdata)
                # Als beide None zijn, skip visueel of toon NVT
                if pd.isna(pct_jg_mp) and pd.isna(pct_mac_mp):
                     continue

                st.markdown(f"**{meetpunt}**")
                c1, c2 = st.columns(2)
                
                if pd.notna(pct_jg_mp):
                    c1.plotly_chart(create_gauge(pct_jg_mp, f"JG: {meetpunt}", 95), use_container_width=True, key=f"g_jg_{meetpunt}")
                else:
                    c1.info("Geen JG data")
                    
                if pd.notna(pct_mac_mp):
                    c2.plotly_chart(create_gauge(pct_mac_mp, f"MAC: {meetpunt}", 95), use_container_width=True, key=f"g_mac_{meetpunt}")
                else:
                    c2.info("Geen MAC data")
                
                st.markdown("---")

        st.subheader("⚠️ Meest recente overschrijdingen")
        
        # Vectorized check
        mask_jg_over = (df_filtered['Waarde'] > df_filtered['JG_MKN'])
        mask_mac_over = (df_filtered['Waarde'] > df_filtered['MAC_MKN'])
        
        df_violations = df_filtered[mask_jg_over | mask_mac_over].copy()
        
        if not df_violations.empty:
            df_violations['Type'] = np.where(
                (df_violations['Waarde'] > df_violations['JG_MKN']) & (df_violations['Waarde'] > df_violations['MAC_MKN']), "JG+MAC",
                np.where(df_violations['Waarde'] > df_violations['JG_MKN'], "JG", "MAC")
            )
            
            st.dataframe(
                df_violations[['Datum', 'Meetpunt', 'Stof', 'Waarde', 'Eenheid', 'JG_MKN', 'MAC_MKN', 'Type']]
                .sort_values('Datum', ascending=False).head(15),
                use_container_width=True
            )
        else:
            st.success("Geen overschrijdingen gevonden.")

# --- TAB 4: Risicoanalyse (Signaleringswaarden) ---
    with tab4:
        st.header("⚠️ Risicoanalyse (Signaleringswaarden)")
        st.markdown("Deze analyse toont stoffen waarvan **normen** bekend zijn. Deze stoffen wordt getoetst aan een generieke signaleringswaarde van **0.1 ug/l**.")
        
        # 1. Basis data voorbereiden
        df_risico_basis = df_filtered.dropna(subset=['Signaleringswaarde']).copy()
        # Filter < rapportagegrens weg (Vectorized is sneller)
        mask_detectie_risico = ~df_risico_basis['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_risico_basis = df_risico_basis[mask_detectie_risico]

        col_risico_1, col_risico_2, col_risico_3 = st.columns(3)
        
        # Filters
        opts_mp = sorted(df_risico_basis['Meetpunt'].unique())
        sel_mp = col_risico_1.multiselect("📍 Selecteer Meetpunt(en)", opts_mp, default=opts_mp, key="tab4_meetpunt")
        
        opts_grp = sorted(df_risico_basis['Stofgroep'].unique())
        sel_grp = col_risico_2.multiselect("📂 Selecteer Stofgroep", opts_grp, default=opts_grp, key="tab4_stofgroep")
        
        # Filter tussentijds om stof-opties te beperken
        df_risico = df_risico_basis[
            df_risico_basis['Meetpunt'].isin(sel_mp) & 
            df_risico_basis['Stofgroep'].isin(sel_grp)
        ].copy()
        
        opts_stof = sorted(df_risico['Stof'].unique())
        sel_stof = col_risico_3.multiselect("🔎 Selecteer Stof(fen)", opts_stof, default=opts_stof, key="tab4_stof")
        
        # Laatste filter
        df_risico = df_risico[df_risico['Stof'].isin(sel_stof)]

        st.markdown("---")

        if not df_risico.empty:
            # Vectorized berekening van overschrijding
            df_risico['Boven_Signalering'] = df_risico['Waarde'] > df_risico['Signaleringswaarde']

            # Aggregeren (observed=True is cruciaal voor snelheid bij categorical data)
            risico_summary = df_risico.groupby('Stof', observed=True).agg(
                Aantal_Metingen=('Waarde', 'count'),
                Aantal_Boven_Signalering=('Boven_Signalering', 'sum')
            ).reset_index()

            risico_summary['Risico_Score'] = (risico_summary['Aantal_Boven_Signalering'] / risico_summary['Aantal_Metingen'] * 100).round(2)
            risico_summary_filtered = risico_summary[risico_summary['Aantal_Boven_Signalering'] > 0].copy()

            if not risico_summary_filtered.empty:
                fig_risico = px.bar(
                    risico_summary_filtered.sort_values('Risico_Score', ascending=False),
                    x='Stof',
                    y='Risico_Score',
                    title='Percentage Aangetroffen Metingen Boven Signaleringswaarde (0.1 ug/l)',
                    labels={'Risico_Score': 'Risico Score (%)'},
                    color_continuous_scale=px.colors.sequential.Reds,
                    color='Risico_Score'
                )
                st.plotly_chart(fig_risico, use_container_width=True)
            else:
                st.success("Geen overschrijdingen van de signaleringswaarde gevonden in de aangetroffen stoffen.")

            st.markdown("---")
            st.subheader("Trends in overschrijdingen per jaar")
            st.info("Meetpunten waar sprake is van signaleringswaardeoverschrijdingen worden hieronder weergegeven.")
            
            # Zorg dat er een jaarkolom is (Vectorized via .dt accessor is snel)
            df_risico['Jaar'] = df_risico['Datum'].dt.year
            
            # Filter alleen de regels die daadwerkelijk een overschrijding zijn
            df_trends = df_risico[df_risico['Boven_Signalering']].copy()

            if not df_trends.empty:
                # Stap 1: Tel overschrijdingen per Jaar én per Meetpunt
                # observed=True toegevoegd voor optimalisatie
                counts_per_mp = df_trends.groupby(['Jaar', 'Meetpunt'], observed=True).size().reset_index(name='Aantal_Overschrijdingen')
                
                # Stap 2: Bereken het gemiddelde aantal overschrijdingen per jaar
                avg_per_year = counts_per_mp.groupby('Jaar')['Aantal_Overschrijdingen'].mean().reset_index(name='Gemiddeld_Aantal')

                col_trend_1, col_trend_2 = st.columns(2)

                with col_trend_1:
                    st.markdown("**Gemiddelde van alle meetpunten**")
                    fig_avg = px.line(
                        avg_per_year, x='Jaar', y='Gemiddeld_Aantal', markers=True,
                        title="Gemiddeld aantal overschrijdingen (van locaties met overschrijding)",
                        labels={'Gemiddeld_Aantal': 'Gemiddeld aantal', 'Jaar': 'Jaar'}
                    )
                    fig_avg.update_xaxes(type='category', tickformat='d')
                    st.plotly_chart(fig_avg, use_container_width=True)

                with col_trend_2:
                    st.markdown("**Per individueel meetpunt**")
                    fig_indiv = px.line(
                        counts_per_mp, x='Jaar', y='Aantal_Overschrijdingen', color='Meetpunt', markers=True,
                        title="Totaal aantal overschrijdingen per meetpunt",
                        labels={'Aantal_Overschrijdingen': 'Aantal overschrijdingen', 'Jaar': 'Jaar'}
                    )
                    fig_indiv.update_xaxes(type='category', tickformat='d')
                    st.plotly_chart(fig_indiv, use_container_width=True)
            else:
                st.info("Onvoldoende data om een trendgrafiek van overschrijdingen te maken.")
            
            # SPIDER CHARTS    
            st.markdown("---")
            st.subheader("🕸️ Seizoenspatroon: signaleringswaardeoverschrijdingen per maand en jaar")
            st.info("Deze grafiek toont in welke maanden de meeste overschrijdingen plaatsvinden. Elke lijn vertegenwoordigt een jaar.")

            if not df_trends.empty:
                # 1. Maak de locatiefilter (gebruik sorted unique voor snelheid)
                alle_meetpunten = sorted(df_trends['Meetpunt'].unique())
                
                selected_meetpunten_spider = st.multiselect(
                    "📍 Selecteer meetpunt(en) voor seizoensanalyse:",
                    options=alle_meetpunten,
                    default=alle_meetpunten,
                    key="tab4_spider_meetpunt"
                )

                # Filter de data op basis van de selectie
                df_filtered_spider = df_trends[df_trends['Meetpunt'].isin(selected_meetpunten_spider)].copy()

                if not df_filtered_spider.empty:
                    
                    # 2. Data voorbereiden: Maand en Jaar toevoegen
                    df_filtered_spider['MaandNr'] = df_filtered_spider['Datum'].dt.month
                    
                    # Meetpunt is category, dus .astype(str) is nodig voor concatenatie
                    df_filtered_spider['Analyse_Groep'] = df_filtered_spider['Meetpunt'].astype(str) + ' (' + df_filtered_spider['Jaar'].astype(str) + ')'

                    # 3. Aggregeren: Tel overschrijdingen per Analyse_Groep en Maand
                    seasonal_counts = df_filtered_spider.groupby(['Analyse_Groep', 'MaandNr']).size().reset_index(name='Aantal')
                    
                    # 4. Zorg dat ALLE maanden (1-12) bestaan voor elke Analyse_Groep (via MultiIndex)
                    unieke_groepen = seasonal_counts['Analyse_Groep'].unique()
                    full_index = pd.MultiIndex.from_product([unieke_groepen, range(1, 13)], names=['Analyse_Groep', 'MaandNr']).to_frame(index=False)
                    
                    # Merge de tellingen hierin, vul NaN op met 0
                    df_radar = pd.merge(full_index, seasonal_counts, on=['Analyse_Groep', 'MaandNr'], how='left').fillna(0)
                    
                    # 5. Maandnummers omzetten naar Namen voor de grafiek
                    maand_namen = {
                        1: 'Jan', 2: 'Feb', 3: 'Mrt', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
                    }
                    df_radar['MaandNaam'] = df_radar['MaandNr'].map(maand_namen)
                    
                    # 6. Plotly Line Polar (Spider Chart)
                    fig_radar = px.line_polar(
                        df_radar, r='Aantal', theta='MaandNaam', color='Analyse_Groep',
                        line_close=True, markers=True,
                        title=f"Aantal signaleringsoverschrijdingen per maand",
                        category_orders={"MaandNaam": ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]}
                    )
                    
                    fig_radar.update_traces(fill='toself', opacity=0.3) 
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, df_radar['Aantal'].max() * 1.1])),
                        legend_title_text='Meetpunt (Jaar)'
                    )
                    
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.warning("Selecteer één of meer meetpunten om de seizoensanalyse te zien.")
            else:
                 st.info("Geen overschrijdingen gevonden om een seizoenspatroon te tonen.")

            # ---------------------------------------------
            st.subheader("Actuele signaleringswaarde overschrijdingen")
            
            df_overtredingen = df_risico[df_risico['Boven_Signalering']].sort_values('Datum', ascending=False)

            if not df_overtredingen.empty:
                st.dataframe(
                    df_overtredingen[['Datum', 'Meetpunt', 'Stof', 'Waarde', 'Signaleringswaarde', 'Eenheid']],
                    use_container_width=True
                )
            else:
                st.success("Er zijn stoffen zonder norm aangetroffen, maar geen enkele meting kwam boven de 0.1 ug/l uit.")
        else:
            st.warning("Geen aangetroffen metingen (boven rapportagegrens) gevonden voor stoffen zonder norm (ug/l).")

# --- TAB 5: PFAS ---
    with tab5:
        st.header("Effectbeoordeling PFAS")
        df_pfas_ref = load_pfas_ref(PFAS_FILE_PATH)
        
        if not df_pfas_ref.empty:
            # 1. Voorbereiding & Merging (Vectorized)
            # We maken een tijdelijke clean kolom in df_filtered voor de merge
            df_filtered['Stof_Clean'] = df_filtered['Stof'].str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True).str.strip()
            df_pfas_ref['Stofnaam'] = df_pfas_ref['Stofnaam'].str.lower().str.strip()
            
            # Inner join om alleen PFAS stoffen over te houden
            df_pfas_calc = pd.merge(df_filtered, df_pfas_ref, left_on='Stof_Clean', right_on='Stofnaam', how='inner')
            
            if not df_pfas_calc.empty:
                # 2. Globale Berekeningen (één keer voor alle grafieken)
                # Detectie limiet logica: als '<' in symbool, dan waarde = 0 voor berekening
                detected = ~df_pfas_calc['Limietsymbool'].astype(str).str.contains('<', na=False)
                
                df_pfas_calc['RPF_calc'] = np.where(detected, df_pfas_calc['RPF'], 0)
                df_pfas_calc['RBF_calc'] = np.where(detected, df_pfas_calc['RBF'], 0)
                
                # Waarde omrekenen van ug/l naar ng/l (* 1000)
                df_pfas_calc['Waarde_ng'] = df_pfas_calc['Waarde'] * 1000
                
                # PEQ en Bioaccumulatie berekenen
                df_pfas_calc['PEQ_Waarde'] = df_pfas_calc['Waarde_ng'] * df_pfas_calc['RPF_calc']
                df_pfas_calc['Bioacc_Waarde'] = df_pfas_calc['PEQ_Waarde'] * df_pfas_calc['RBF_calc']

                # --- DEEL A: Originele Grafieken (Per Meetpunt) ---
                mp_opts = sorted(df_pfas_calc['Meetpunt'].unique())
                sel_mp_pfas = st.selectbox("Selecteer meetpunt", mp_opts)
                
                # Filter voor de staafgrafieken
                df_plot = df_pfas_calc[df_pfas_calc['Meetpunt'] == sel_mp_pfas].copy()
                
                col_bar1, col_bar2 = st.columns(2)
                
                with col_bar1:
                    fig_rpf = px.bar(df_plot.sort_values('Datum'), x='Datum', y='PEQ_Waarde', color='Stof', title='Relatieve Toxiciteit (PEQ)')
                    fig_rpf.add_hline(y=4.4, line_dash="dash", line_color="red", annotation_text="Norm: 4.4")
                    st.plotly_chart(fig_rpf, use_container_width=True)
                
                with col_bar2:
                    fig_rbf = px.bar(df_plot.sort_values('Datum'), x='Datum', y='Bioacc_Waarde', color='Stof', title='Bioaccumulatie')
                    fig_rbf.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Drempel: 0.3")
                    st.plotly_chart(fig_rbf, use_container_width=True)

                # --- DEEL B: NIEUWE SPIDER CHART (Seizoenspatroon) ---
                st.markdown("---")
                st.subheader("🕸️ Seizoenspatroon: PFAS Toxiciteit (PEQ)")
                st.info("Deze grafiek toont de gemiddelde totale PEQ-waarde per maand. Hierbij wordt enkel gekeken naar relatieve toxiciteit (RPF), niet naar bioaccumulatie.")

                # 1. Selectie voor Spider Chart
                # Gebruik sel_mp_pfas (van hierboven) als slimme default
                default_spider = [sel_mp_pfas] if sel_mp_pfas in mp_opts else mp_opts[:1]
                
                selected_meetpunten_spider = st.multiselect(
                    "📍 Selecteer meetpunt(en) voor PEQ seizoensanalyse:",
                    options=mp_opts,
                    default=default_spider,
                    key="pfas_spider_multiselect"
                )

                if selected_meetpunten_spider:
                    # 2. Filteren (gebruik de reeds berekende df_pfas_calc)
                    df_spider_pfas = df_pfas_calc[df_pfas_calc['Meetpunt'].isin(selected_meetpunten_spider)].copy()

                    if not df_spider_pfas.empty:
                        # 3. Aggregeren
                        # Stap A: Sommeer PEQ per Datum + Meetpunt (Totaal PEQ van alle stoffen op één dag)
                        # observed=True is belangrijk voor performance bij categorische data (Meetpunt)
                        df_daily_sum = df_spider_pfas.groupby(['Datum', 'Meetpunt'], observed=True)['PEQ_Waarde'].sum().reset_index(name='Dag_Totaal_PEQ')

                        # Stap B: Tijdvariabelen toevoegen (Vectorized)
                        df_daily_sum['Jaar'] = df_daily_sum['Datum'].dt.year
                        df_daily_sum['MaandNr'] = df_daily_sum['Datum'].dt.month
                        
                        # Maak label voor de grafiek
                        df_daily_sum['Analyse_Groep'] = df_daily_sum['Meetpunt'].astype(str) + ' (' + df_daily_sum['Jaar'].astype(str) + ')'

                        # Stap C: Gemiddelde per Maand berekenen
                        df_monthly_avg = df_daily_sum.groupby(['Analyse_Groep', 'MaandNr'])['Dag_Totaal_PEQ'].mean().reset_index(name='Gemiddelde_PEQ')

                        # 4. Gaten vullen (Zorg dat maand 1 t/m 12 bestaat voor elke lijn)
                        unieke_groepen_pfas = df_monthly_avg['Analyse_Groep'].unique()
                        
                        if len(unieke_groepen_pfas) > 0:
                            full_index_pfas = pd.MultiIndex.from_product(
                                [unieke_groepen_pfas, range(1, 13)], 
                                names=['Analyse_Groep', 'MaandNr']
                            ).to_frame(index=False)

                            df_radar_pfas = pd.merge(full_index_pfas, df_monthly_avg, on=['Analyse_Groep', 'MaandNr'], how='left').fillna(0)

                            # 5. Maandnamen mappen
                            maand_namen = {
                                1: 'Jan', 2: 'Feb', 3: 'Mrt', 4: 'Apr', 5: 'Mei', 6: 'Jun',
                                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Okt', 11: 'Nov', 12: 'Dec'
                            }
                            df_radar_pfas['MaandNaam'] = df_radar_pfas['MaandNr'].map(maand_namen)

                            # 6. Plotten
                            fig_spider_pfas = px.line_polar(
                                df_radar_pfas, 
                                r='Gemiddelde_PEQ', 
                                theta='MaandNaam', 
                                color='Analyse_Groep', 
                                line_close=True,
                                markers=True,
                                title="Gemiddelde PEQ Toxiciteit (ng/l) per maand",
                                category_orders={"MaandNaam": ["Jan", "Feb", "Mrt", "Apr", "Mei", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]}
                            )

                            fig_spider_pfas.update_traces(fill='toself', opacity=0.3)
                            fig_spider_pfas.update_layout(
                                polar=dict(
                                    radialaxis=dict(visible=True, range=[0, df_radar_pfas['Gemiddelde_PEQ'].max() * 1.1])
                                ),
                                legend_title_text='Meetpunt (Jaar)'
                            )

                            st.plotly_chart(fig_spider_pfas, use_container_width=True)
                        else:
                            st.info("Geen data over om te plotten na aggregatie.")
                    else:
                        st.info("Geen PFAS data gevonden voor de geselecteerde meetpunten.")
                else:
                    st.info("Selecteer ten minste één meetpunt om de spider chart te genereren.")

                st.divider()
                
            else:
                st.warning("Geen PFAS matches gevonden in de huidige dataset (check spelling stoffen).")

# --- TAB 6: Herkomst en verdeling stoffen ---
    with tab6:
        st.header("🏭 Herkomst en verdeling van stoffen")
        
        # ---------------------------------------------------------
        # DATA VOORBEREIDING
        # ---------------------------------------------------------
        st.info("Hieronder worden de gegevens getoond op basis van de **jaren geselecteerd in de zijbalk**.")

        # We gebruiken df_filtered (al gefilterd op jaar in sidebar)
        df_herkomst = df_filtered.dropna(subset=['Stof']).copy()
        
        # Vectorized check voor detecties (sneller dan string contains in loop)
        # We nemen aan dat load_data 'Limietsymbool' al heeft opgeschoond (geen NaN)
        mask_detectie = ~df_herkomst['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_detecties = df_herkomst[mask_detectie].copy()

        # ---------------------------------------------------------
        # A. TOTAAL OVERZICHT
        # ---------------------------------------------------------
        st.subheader("Totaalbeeld verdeling (Geselecteerde periode)")
        st.markdown("### 🟢 uitgevoerde metingen vs. 🟠 aangetroffen")
        
        col_total_all, col_total_det = st.columns(2)

        with col_total_all:
            st.markdown("**Totaal: alle metingen**")
            # Value_counts is zeer snel op categorical data
            dist_total = df_herkomst['Stofgroep'].value_counts().reset_index()
            dist_total.columns = ['Stofgroep', 'Aantal metingen']
            
            fig_pie_total = px.pie(
                dist_total, values='Aantal metingen', names='Stofgroep',
                title='Aantal metingen (incl. <RG)', hole=0.4
            )
            st.plotly_chart(fig_pie_total, use_container_width=True)

            st.markdown("*Top 25 meest gemeten (totaal)*")
            # Limit voor display performance
            top_stoffen = df_herkomst['Stof'].value_counts().head(25).reset_index()
            top_stoffen.columns = ['Stof', 'Aantal']
            st.dataframe(top_stoffen, use_container_width=True)

        with col_total_det:
            st.markdown("**Filter: aangetroffen stoffen (boven rapportagegrens)**")
            if not df_detecties.empty:
                dist_det = df_detecties['Stofgroep'].value_counts().reset_index()
                dist_det.columns = ['Stofgroep', 'Aantal x aangetroffen']
                
                fig_pie_det = px.pie(
                    dist_det, values='Aantal x aangetroffen', names='Stofgroep',
                    title='Aantal keer aangetroffen (>RG)', hole=0.4
                )
                st.plotly_chart(fig_pie_det, use_container_width=True)

                st.markdown("*Top 25 meest aangetroffen (> RG)*")
                top_stoffen_det = df_detecties['Stof'].value_counts().head(25).reset_index()
                top_stoffen_det.columns = ['Stof', 'Aantal']
                st.dataframe(top_stoffen_det, use_container_width=True)
            else:
                st.warning("Geen waarden boven rapportagegrens gevonden in totale dataset.")

        st.divider()
        
        # ---------------------------------------------------------
        # B. TRENDANALYSE STOFGROEPEN (ALLE JAREN)
        # ---------------------------------------------------------
        st.subheader("📈 Trendverdeling Stofgroepen (Alle jaren)")
        st.markdown("De grafiek toont data uit de **volledige dataset**, ongeacht de jaar-filter in de zijbalk.")

        col_trend_sel1, col_trend_sel2 = st.columns([1, 3])
        
        with col_trend_sel1:
            alleen_detecties_trend = st.checkbox("Toon alleen aangetroffen stoffen (>RG)", value=False, key="tab6_trend_detectie")
        
        with col_trend_sel2:
            # Gebruik categories voor snelle lookup
            all_meetpunten_trend = sorted(df_main['Meetpunt'].unique())
            selected_meetpunten_trend = st.multiselect(
                "📍 Selecteer meetpunt(en)",
                options=all_meetpunten_trend,
                default=all_meetpunten_trend,
                key="tab6_trend_meetpunt"
            )

        if selected_meetpunten_trend:
            # Filter op df_main (alles)
            mask_mp = df_main['Meetpunt'].isin(selected_meetpunten_trend)
            
            if alleen_detecties_trend:
                # Combineer masks direct voor snelheid
                mask_det_main = ~df_main['Limietsymbool'].astype(str).str.contains('<', na=False)
                df_trend = df_main[mask_mp & mask_det_main].copy()
            else:
                df_trend = df_main[mask_mp].copy()

            if not df_trend.empty:
                df_trend['Jaar'] = df_trend['Datum'].dt.year
                
                # Groupby op categories is heel snel
                df_grouped = df_trend.groupby(['Jaar', 'Stofgroep'], observed=True).size().reset_index(name='Aantal')
                
                # Bereken percentages vectorized
                totaal_per_jaar = df_grouped.groupby('Jaar')['Aantal'].transform('sum')
                df_grouped['Percentage'] = (df_grouped['Aantal'] / totaal_per_jaar) * 100
                
                # Drop rows waar percentage 0 of NaN is (kan gebeuren door observed=True bij lege cats)
                df_grouped = df_grouped[df_grouped['Aantal'] > 0]

                if not df_grouped.empty:
                    fig_trend = px.bar(
                        df_grouped, x='Jaar', y='Percentage', color='Stofgroep',
                        title='Procentuele Verdeling (100% Gestapeld)',
                        barmode='stack' 
                    )
                    fig_trend.update_layout(yaxis=dict(range=[0, 100], ticksuffix="%"))
                    fig_trend.update_xaxes(type='category') # Zorgt dat jaren als labels worden gezien
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Geen data na groepering.")
            else:
                st.info("Geen data gevonden voor deze selectie.")
        else:
            st.warning("Selecteer ten minste één meetpunt.")

        st.divider()

        # ---------------------------------------------------------
        # C. PER MEETPUNT
        # ---------------------------------------------------------
        st.subheader("Overzicht per specifiek meetpunt")

        meetpunten_list = sorted(df_herkomst['Meetpunt'].unique())
        if meetpunten_list:
            selected_mp_herkomst = st.selectbox("Selecteer meetpunt:", meetpunten_list)

            # Filteren
            df_mp_all = df_herkomst[df_herkomst['Meetpunt'] == selected_mp_herkomst]
            df_mp_det = df_detecties[df_detecties['Meetpunt'] == selected_mp_herkomst]

            if not df_mp_all.empty:
                col_mp_1, col_mp_2 = st.columns(2)

                with col_mp_1:
                    st.markdown(f"**{selected_mp_herkomst}: Alle metingen**")
                    counts_all = df_mp_all['Stofgroep'].value_counts().reset_index()
                    counts_all.columns = ['Stofgroep', 'Aantal']
                    st.plotly_chart(px.pie(counts_all, values='Aantal', names='Stofgroep'), use_container_width=True)

                with col_mp_2:
                    st.markdown(f"**{selected_mp_herkomst}: > Rapportagegrens**")
                    if not df_mp_det.empty:
                        counts_det = df_mp_det['Stofgroep'].value_counts().reset_index()
                        counts_det.columns = ['Stofgroep', 'Aantal']
                        st.plotly_chart(px.pie(counts_det, values='Aantal', names='Stofgroep'), use_container_width=True)
                    else:
                        st.info("Geen detecties op dit meetpunt.")

                st.divider()
                st.markdown(f"### 📋 Detailoverzicht per stofgroep: {selected_mp_herkomst}")

                # 1. Filteroptie: Alles of alleen detecties
                show_only_detected = st.checkbox(
                    "Toon in tabel alleen waarden boven rapportagegrens (>RG)", 
                    value=False,
                    key="tab6_table_checkbox"
                )
                
                # Bepaal welke dataset we gebruiken
                df_table_source = df_mp_det if show_only_detected else df_mp_all

                if not df_table_source.empty:
                    # 2. Beschikbare stofgroepen bepalen (dynamisch op basis van filter)
                    # We gebruiken dropna() en unique() om alleen relevante groepen te tonen
                    beschikbare_groepen = sorted(list(df_table_source['Stofgroep'].dropna().unique()))
                    
                    if beschikbare_groepen:
                        # 3. Dropdown menu
                        selected_stofgroep_table = st.selectbox(
                            "Selecteer een stofgroep om de details te bekijken:", 
                            options=beschikbare_groepen,
                            key="tab6_table_selectbox"
                        )
                        
                        # 4. Data filteren op gekozen groep
                        df_table_filtered = df_table_source[df_table_source['Stofgroep'] == selected_stofgroep_table].copy()
                        
                        # 5. Tabel samenstellen (Aggregatie per stof voor overzichtelijkheid)
                        table_agg = df_table_filtered.groupby('Stof', observed=True).agg(
                            Aantal_Metingen=('Waarde', 'count'),
                            Gemiddelde=('Waarde', 'mean'),
                            Maximum=('Waarde', 'max'),
                            Laatste_Datum=('Datum', 'max'),
                            Eenheid=('Eenheid', 'first')
                        ).reset_index().sort_values('Maximum', ascending=False)

                        # Opmaak verfraaien
                        table_agg['Laatste_Datum'] = table_agg['Laatste_Datum'].dt.strftime('%Y-%m-%d')
                        table_agg['Gemiddelde'] = table_agg['Gemiddelde'].map('{:,.4f}'.format)
                        table_agg['Maximum'] = table_agg['Maximum'].map('{:,.4f}'.format)
                        
                        # 6. Weergeven
                        st.dataframe(table_agg, use_container_width=True)
                    else:
                        st.info("Geen stofgroepen gevonden in de huidige selectie.")
                else:
                    st.warning("Geen metingen gevonden voor de huidige filterinstellingen.")
                
                # ---------------------------------------------------------
            # D. STOFFEN ZONDER GROEP (ONBEKEND)
            # ---------------------------------------------------------
            st.divider()
            st.subheader("⚠️ Stoffen zonder toegewezen stofgroep")
            
            # We kijken naar df_herkomst (dat is df_filtered binnen tab 6 context)
            # In de load_data functie krijgen niet-gematchte stoffen de label 'Onbekend'
            df_onbekend = df_herkomst[df_herkomst['Stofgroep'] == 'Onbekend'].copy()
            
            if not df_onbekend.empty:
                st.warning(f"Er zijn **{df_onbekend['Stof'].nunique()}** unieke stoffen aangetroffen die niet in de configuratielijst staan.")
                st.info("Onderstaande tabel toont stoffen die de categorie 'Onbekend' hebben gekregen.")
                
                # We tonen de unieke stoffen, hoe vaak ze voorkomen en de meest recente datum
                onbekend_summary = df_onbekend.groupby('Stof', observed=True).agg(
                    Aantal_Metingen=('Waarde', 'count'),
                    Laatste_Datum=('Datum', 'max'),
                    Voorbeeld_Waarde=('Waarde', 'first'),
                    Eenheid=('Eenheid', 'first')
                ).sort_values('Aantal_Metingen', ascending=False).reset_index()
                
                # Formatteer de datum voor netheid
                onbekend_summary['Laatste_Datum'] = onbekend_summary['Laatste_Datum'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(onbekend_summary, use_container_width=True)
            else:
                st.success("✅ Alle aangetroffen stoffen zijn succesvol ingedeeld in een stofgroep.")
                
                if not source_df.empty:
                    # observed=True zorgt dat we alleen relevante groepen zien
                    beschikbare_groepen = source_df['Stofgroep'].unique()
                    # Filter lege categorieen eruit
                    beschikbare_groepen = [g for g in beschikbare_groepen if g in source_df['Stofgroep'].values]
                    
                    if beschikbare_groepen:
                        selected_groep = st.selectbox("Kies stofgroep:", sorted(beschikbare_groepen))

                        df_detail = source_df[source_df['Stofgroep'] == selected_groep]

                        # Aggregatie
                        detail_agg = df_detail.groupby('Stof', observed=True).agg(
                            Aantal=('Waarde', 'count'),
                            Gemiddelde=('Waarde', 'mean'),
                            Max=('Waarde', 'max'),
                            Eenheid=('Eenheid', 'first')
                        ).sort_values('Aantal', ascending=False).reset_index()
                        
                        detail_agg['Gemiddelde'] = detail_agg['Gemiddelde'].map('{:,.4f}'.format)
                        detail_agg['Max'] = detail_agg['Max'].map('{:,.4f}'.format)

                        st.dataframe(detail_agg, use_container_width=True)
                    else:
                        st.info("Geen stofgroepen beschikbaar in deze dataset.")
                else:
                    st.warning("Geen data beschikbaar voor tabel.")

    # --- TAB 7: Ruimtelijke analyse ---
    with tab7:
        st.header("🔍 Ruimtelijke analyse")
        
        # We werken hier verder met een kopie van df_filtered (al gefilterd op jaren sidebar)
        # Optimalisatie: Kopieer alleen relevante kolommen
        cols_needed = ['Datum', 'Meetpunt', 'Stof', 'Stofgroep', 'Waarde', 'Eenheid']
        df_space = df_filtered[cols_needed].copy()

        # Datum bereik bepalen
        if not df_space.empty:
            min_d, max_d = df_space['Datum'].min().date(), df_space['Datum'].max().date()
        else:
            min_d, max_d = datetime(2022,1,1).date(), datetime(2025,12,31).date()

        # Filters UI
        with st.container():
            c_zomer, c_start, c_end = st.columns([1.5, 2, 2])
            
            c_zomer.write("")
            c_zomer.write("")
            zomerhalfjaar = c_zomer.checkbox("Alleen Zomerhalfjaar (apr-sep)", value=False)
            
            start_date = c_start.date_input("Startdatum", value=min_d, min_value=min_d, max_value=max_d)
            end_date = c_end.date_input("Einddatum", value=max_d, min_value=min_d, max_value=max_d)
            
            c_loc, c_grp, c_stof = st.columns([2, 2, 2])
            
            # Opties ophalen (gebruik categories voor snelheid)
            loc_opts = sorted(df_space['Meetpunt'].unique())
            sel_loc = c_loc.multiselect("Meetpunt", loc_opts, default=loc_opts)
            
            grp_opts = sorted(df_space['Stofgroep'].unique())
            sel_grp = c_grp.multiselect("Stofgroep", grp_opts, default=[grp_opts[0]] if grp_opts else None)
            
            # Contextuele filter voor stof
            if sel_grp:
                # Filter de opties snel
                mask_grp_opt = df_space['Stofgroep'].isin(sel_grp)
                stof_opts = sorted(df_space.loc[mask_grp_opt, 'Stof'].unique())
            else:
                stof_opts = sorted(df_space['Stof'].unique())
            
            sel_stof = c_stof.multiselect("Stof", stof_opts, default=stof_opts[:1] if stof_opts else [])

        # --- DATA FILTERING LOGICA (VECTORIZED) ---
        
        # 1. Datum Filter (Pandas is optimized for datetime slicing)
        mask_date = (df_space['Datum'].dt.date >= start_date) & (df_space['Datum'].dt.date <= end_date)
        
        # 2. Zomer Filter
        if zomerhalfjaar:
            # dt.month is vectorized en erg snel
            mask_zomer = (df_space['Datum'].dt.month >= 4) & (df_space['Datum'].dt.month <= 9)
            mask_date = mask_date & mask_zomer
            
        # 3. Categorische filters
        # Check op None om warning te voorkomen bij lege selectie
        mask_loc = df_space['Meetpunt'].isin(sel_loc) if sel_loc else pd.Series(False, index=df_space.index)
        mask_grp = df_space['Stofgroep'].isin(sel_grp) if sel_grp else pd.Series(True, index=df_space.index)
        mask_stof = df_space['Stof'].isin(sel_stof) if sel_stof else pd.Series(False, index=df_space.index)

        # Pas alle filters in één keer toe
        dff_final = df_space[mask_date & mask_loc & mask_grp & mask_stof].copy()

        if dff_final.empty:
            st.warning("Geen data beschikbaar voor de geselecteerde criteria.")
        else:
            # --- PLOTS ---
            
            # A. Tijdlijn & Kaart
            col_kaart, col_tijd = st.columns(2)
            
            # Aggregaties voor plots (observed=True voor speed)
            time_agg = dff_final.groupby(['Datum', 'Meetpunt'], observed=True)['Waarde'].mean().reset_index()
            tijd_fig = px.line(time_agg, x="Datum", y="Waarde", color="Meetpunt", markers=True, title="Verloop in de tijd")
            col_tijd.plotly_chart(tijd_fig, use_container_width=True)

            # Kaart data
            loc_agg = dff_final.groupby("Meetpunt", observed=True)["Waarde"].mean().reset_index()
            
            # Efficiënte coördinaten lookup (zonder grote merge op de hele dataset)
            # We halen unieke coords uit df_main (origineel)
            coords_ref = df_main[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().dropna()
            
            # Merge alleen de geaggregeerde tabel
            loc_map = pd.merge(loc_agg, coords_ref, on='Meetpunt', how='inner')
            
            if not loc_map.empty:
                kaart_fig = px.scatter_mapbox(
                    loc_map, lat="Latitude", lon="Longitude", color="Waarde",
                    size=[15]*len(loc_map), hover_name="Meetpunt",
                    color_continuous_scale="YlOrRd", zoom=7.5, mapbox_style="open-street-map",
                    title="Gemiddelde waarde per locatie"
                )
                col_kaart.plotly_chart(kaart_fig, use_container_width=True)
            else:
                col_kaart.info("Geen coördinaten beschikbaar voor deze meetpunten.")

            # B. Boxplot & Strip
            col_box, col_strip = st.columns(2)
            
            box_fig = px.box(dff_final, x="Stof", y="Waarde", color="Meetpunt", title="Verdeling per stof")
            col_box.plotly_chart(box_fig, use_container_width=True)
            
            strip_fig = px.strip(dff_final, x="Waarde", y="Stof", color="Meetpunt", title="Individuele meetwaarden")
            col_strip.plotly_chart(strip_fig, use_container_width=True)

            # C. Heatmap & Bar
            col_heat, col_sub = st.columns(2)
            
            # Pivot voor heatmap (groupby + pivot is vaak schoner dan pivot_table op grote sets)
            heat_data = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().unstack()
            heat_fig = px.imshow(heat_data, aspect="auto", color_continuous_scale="YlGnBu", title="Heatmap Gemiddelden")
            col_heat.plotly_chart(heat_fig, use_container_width=True)
            
            sub_agg = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().reset_index()
            sub_fig = px.bar(sub_agg, x="Stof", y="Waarde", color="Meetpunt", barmode="group", title="Gemiddelde per meetpunt")
            col_sub.plotly_chart(sub_fig, use_container_width=True)

            # D. Fold Change Analyse
            st.markdown("---")
            st.subheader("Log2 Fold Change Analyse")
            
            # Opties beperken tot meetpunten die in de huidige gefilterde set zitten
            beschikbare_mp_fc = sorted(dff_final['Meetpunt'].unique())
            
            if len(beschikbare_mp_fc) > 1:
                ref_mp = st.selectbox("Selecteer referentie meetpunt", beschikbare_mp_fc)
                
                # Bereken gemiddelden
                means = dff_final.groupby(["Stof", "Meetpunt"], observed=True)["Waarde"].mean().reset_index()
                
                # Split in Ref en Rest
                ref_data = means[means['Meetpunt'] == ref_mp][['Stof', 'Waarde']].rename(columns={'Waarde': 'Ref_Waarde'})
                
                # Merge terug (Inner join: we kunnen alleen vergelijken als stof op beide plekken gemeten is)
                fc_data = pd.merge(means, ref_data, on='Stof', how='inner')
                
                # Vectorized berekening
                # Voorkom delen door nul
                fc_data = fc_data[fc_data['Ref_Waarde'] > 0]
                fc_data['Log2FC'] = np.log2(fc_data['Waarde'] / fc_data['Ref_Waarde'])
                
                # Filter referentie zelf eruit voor plot
                fc_plot = fc_data[fc_data['Meetpunt'] != ref_mp]
                
                if not fc_plot.empty:
                    fig_fc = px.scatter(
                        fc_plot, x="Log2FC", y="Stof", color="Meetpunt",
                        title=f"Fold Change t.o.v. {ref_mp}",
                        hover_data={'Waarde':':.2f', 'Ref_Waarde':':.2f'}
                    )
                    fig_fc.add_vline(x=0, line_dash="dash", line_color="black")
                    fig_fc.add_vline(x=1, line_dash="dot", line_color="gray")
                    fig_fc.add_vline(x=-1, line_dash="dot", line_color="gray")
                    st.plotly_chart(fig_fc, use_container_width=True)
                else:
                    st.info("Geen overlappende stoffen gevonden om te vergelijken.")
            else:
                st.info("Selecteer minimaal 2 meetpunten voor Fold Change analyse.")

            #st.plotly_chart(fold_change_fig, use_container_width=True)
            
        # Toon de gefilterde data
        #st.write("Gefilterde data:", df_space)       

if __name__ == "__main__":
    main()