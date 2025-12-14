# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- CONSTANTEN ---
DATA_FILE_PATH = 'IJG Chemie.csv'
NORMEN_FILE_PATH = 'KRW stoffen koppeltabel.csv'
PFAS_FILE_PATH = 'PFAS PEQ koppeltabel.csv'

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
        'guanylureum', 'avobenzone', 'octocrilene', 'paroxetine', 'fluoxetine', 'fenofibrinezuur',
        'gabapentine'
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
    # 1. Laden Meetdata
    try:
        df = pd.read_csv(DATA_FILE_PATH, delimiter=';', low_memory=False, encoding='latin-1')
    except FileNotFoundError:
        st.error(f"Bestand niet gevonden op pad: {DATA_FILE_PATH}.")
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
    
    # 1. Standaardiseer de tekstkolommen voor filtering
    df['Hoedanigheid_Omschr_lower'] = df['Hoedanigheid_Omschr'].astype(str).str.strip().str.lower()
    df['Eenheid_Omschr_lower'] = df['Eenheid_Omschr'].astype(str).str.strip().str.lower()
    
    # 2. Definieer de condities
    mask_nvt = df['Stof'] == 'NVT' # Basismarkering voor NVT
    
    # NIEUWE CONDITIE VOOR GADOLINIUM: Stof is 'gadolinium' EN Eenheid is 'dimensieloos'
    mask_gadolinium_antropogeen = (
        (df['Stof'].str.lower() == 'gadolinium') & 
        (df['Eenheid_Omschr_lower'] == 'dimensieloos')
    )
    
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
        # 9. Gadolinium (Antropogeen/Opgelost) - De gewenste uiteindelijke naam
        mask_gadolinium_antropogeen
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
def load_pfas_ref() -> pd.DataFrame:
    """
    Laadt de PFAS referentietabel en voert typeconversie uit.
    Gebruikt de constante PFAS_FILE_PATH.
    """
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
    """
    Deze functie zorgt dat de filters op elke pagina hetzelfde zijn
    en retourneert de gefilterde dataframe.
    """
    st.sidebar.header("📅 Filter op jaren")
    
    # Jaren ophalen
    if not df_main.empty and 'Datum' in df_main.columns:
        beschikbare_jaren = sorted(df_main['Datum'].dt.year.dropna().unique(), reverse=True)
    else:
        beschikbare_jaren = []
    
    # Multiselectbox
    geselecteerde_jaren = st.sidebar.multiselect(
        "Selecteer gewenste jaren:",
        options=beschikbare_jaren,
        default=beschikbare_jaren
    )

    # Filteren
    if geselecteerde_jaren and not df_main.empty:
        df_filtered = df_main[df_main['Datum'].dt.year.isin(geselecteerde_jaren)].copy()
    else:
        df_filtered = df_main.copy()
        
    st.sidebar.markdown("---")
    st.sidebar.info("Navigeer via het menu hierboven naar de verschillende analyses.")
    
    return df_filtered

@st.cache_data
def calculate_trends_optimized(df_in: pd.DataFrame, lt_optie: str, norm_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Geoptimaliseerde trendberekening.
    
    Args:
        df_in: De gefilterde meetdata
        lt_optie: String optie "Gebruik gemeten waarde" of "Sluit uit van berekening"
        norm_lookup_df: Een DataFrame met unieke 'Stof' en 'JG_MKN' kolommen (afkomstig uit df_main)
    """
    if df_in.empty:
        return pd.DataFrame()

    # 1. Voorbereiden data (Copy om warnings te voorkomen)
    df_calc = df_in.copy()
    
    # 2. Omgaan met < waarden
    if lt_optie == "Sluit uit van berekening":
        mask_lt = df_calc['Limietsymbool'].astype(str).str.contains('<', na=False)
        df_calc.loc[mask_lt, 'Waarde'] = np.nan

    # 3. Jaargemiddelden berekenen (Vectorized)
    df_calc['Jaar'] = df_calc['Datum'].dt.year
    # Eerst groeperen op Meetpunt/Stof/Jaar
    df_yearly = df_calc.groupby(['Meetpunt', 'Stof', 'Jaar'], observed=True)['Waarde'].mean().reset_index()
    
    # 4. Normen Map maken (Snelheidswinst!)
    # We maken een dictionary: {'stofnaam': 0.15, ...}
    # Dit maakt het opzoeken instant in plaats van dat we in de loop moeten filteren.
    norm_map = norm_lookup_df.set_index('Stof')['JG_MKN'].to_dict()

    trend_results = []

    # 5. Itereren over groepen (Dit is nu veel sneller omdat we geen zware operaties in de loop doen)
    # We filteren groepen met < 2 waarden direct weg in de groupby iteratie als optimalisatie
    grouped = df_yearly.dropna(subset=['Waarde']).groupby(['Meetpunt', 'Stof'], observed=True)

    for (meetpunt, stof), group in grouped:
        if len(group) < 2:
            continue
            
        # Sorteer op jaar (belangrijk voor polyfit)
        group = group.sort_values('Jaar')
        x = group['Jaar'].values
        y = group['Waarde'].values

        # Lineaire regressie (Numpy polyfit is snel)
        slope, intercept = np.polyfit(x, y, 1)

        # Alleen stijgende trends verwerken
        if slope > 0:
            laatste_jaargemiddelde = y[-1]
            
            # Snel de norm ophalen uit de dictionary
            jg_norm = norm_map.get(stof, np.nan)
            
            # Logicacheck: Is de norm al overschreden?
            if pd.notna(jg_norm) and laatste_jaargemiddelde >= jg_norm:
                # Urgentie 0 (Reeds overschreden), we willen deze WEL tonen in de tabel
                # De originele logica sloeg deze over ("continue"), maar in de tekst erboven
                # staat "Tijd tot norm 0 jaar = Kritisch Urgent". 
                # Als je ze wilt uitsluiten zoals in je originele code, uncomment dan de volgende regel:
                # continue 
                tijd_tot_norm = 0.0
            elif pd.notna(jg_norm) and slope > 0:
                tijd_tot_norm = (jg_norm - laatste_jaargemiddelde) / slope
            else:
                tijd_tot_norm = np.inf # Geen norm beschikbaar

            trend_results.append({
                'Meetpunt': meetpunt,
                'Stof': stof,
                'Startwaarde': y[0],
                'Eindwaarde': laatste_jaargemiddelde,
                'Trendscore': slope,
                'Tijd_tot_JG_normoverschrijding': tijd_tot_norm,
                'Aantal_jaren': len(group)
            })

    return pd.DataFrame(trend_results)