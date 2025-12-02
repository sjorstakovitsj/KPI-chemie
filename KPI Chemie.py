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
    })

    df['hoedanigheid'] = df['hoedanigheid'].astype(str).str.strip().str.lower()
    df['Stof'] = df['Stof'].astype(str).str.strip()
    df['Limietsymbool'] = df['Limietsymbool'].astype(str).replace('nan', '').fillna('')

    def specificeer_stofnaam(row):
        stof = row['Stof']
        hoedanigheid = row['hoedanigheid']
        if 'nf' in hoedanigheid or 'filtratie' in hoedanigheid or 'opgeloste' in hoedanigheid:
            return f"{stof} (opgelost)"
        elif 'nvt' in hoedanigheid or 'niet van toepassing' in hoedanigheid or hoedanigheid == 'nan':
             return f"{stof} (totaal)"
        else:
            return f"{stof} (totaal)"

    df['Stof'] = df.apply(specificeer_stofnaam, axis=1)
    df['Stof'] = df['Stof'].str.lower()
    df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%d', errors='coerce')
    df['Waarde'] = pd.to_numeric(df['Waarde'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    df = df[df['Waarde'] != 999999999999]
    df = df.dropna(subset=['Waarde', 'Datum', 'Meetpunt', 'Stof']).copy()

    # 2. KRW Normen Inlezen en Koppelen
    try:
        df_normen = pd.read_csv(NORMEN_FILE_PATH, delimiter=',', low_memory=False, encoding='latin-1')
        df_normen = df_normen.rename(columns={
            'Stofnaam': 'Stof',
            'Norm': 'NormType',
            'Waarde': 'NormWaarde'
        })

        df_normen['Stof'] = df_normen['Stof'].astype(str).str.strip()

        def match_norm_stofnaam(row):
            stof = row['Stof']
            omschrijving = str(row['NormType']).lower()
            if 'opgelost' in omschrijving:
                return f"{stof} (opgelost)"
            elif 'totaal' in omschrijving:
                return f"{stof} (totaal)"
            return stof

        df_normen['Stof'] = df_normen.apply(match_norm_stofnaam, axis=1)
        df_normen['Stof'] = df_normen['Stof'].str.lower()

        def map_norm_type(norm_omschrijving):
            norm_omschrijving = str(norm_omschrijving)
            if 'JG-MKN' in norm_omschrijving or 'Jaargemiddelde' in norm_omschrijving:
                return 'JG_MKN'
            if 'MAC-MKN' in norm_omschrijving or 'Maximaal' in norm_omschrijving:
                return 'MAC_MKN'
            return None

        df_normen['NormCode'] = df_normen['NormType'].apply(map_norm_type)
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

        df_normen_pivot['Stof'] = df_normen_pivot['Stof'].astype(str)

        df = pd.merge(
            df,
            df_normen_pivot[['Stof', 'JG_MKN', 'MAC_MKN']],
            on='Stof',
            how='left'
        )

    except FileNotFoundError:
        st.error(f"Koppeltabel KRW-normen niet gevonden op pad: {NORMEN_FILE_PATH}.")
        df['JG_MKN'] = None
        df['MAC_MKN'] = None

    # 3. Signaleringswaarden Berekenen
    df['KRW_Norm'] = df['JG_MKN']
    df['Signaleringswaarde'] = pd.NA

    uitgesloten_elementen = [
        'aluminium', 'ammonium', 'antimoon', 'arseen', 'arsenaat', 'arseniet', 'barium', 'beryllium', 'boor',
        'cadmium', 'cerium', 'cesium', 'chloride', 'chroom','calcium', 'cobalt', 'kobalt', 'koper', 'kwik',
        'lood', 'magnesium', 'mangaan', 'molybdeen', 'natrium', 'nikkel', 'dysprosium', 'erbium', 'europium',
        'kalium', 'seleen', 'silicium', 'strontium', 'thallium', 'tin', 'gadolinium', 'gallium', 'hafnium',
        'titanium', 'uranium', 'vanadium', 'ijzer', 'zilver', 'zink', 'holmium', 'indium', 'koolstof organisch',
        'lanthaan', 'lithium', 'lutetium', 'neodymium', 'niobium', 'nitraat', 'nitriet', 'platina', 'praseodymium',
        'rubidium', 'samarium', 'seleniet', 'selenaat', 'siliciumdioxide', 'sulfaat', 'tantalium', 'tellurium',
        'terbium', 'thallium', 'thorium', 'thulium', 'wolfraam', 'ytterbium', 'yttrium', 'zirkonium',
        'titaan', 'scandium', 'chlorofyl-a'
    ]

    base_stofnaam = df['Stof'].str.replace(r' \(totaal\)| \(opgelost\)', '', regex=True).str.strip()
    is_metaal_of_element = base_stofnaam.isin(uitgesloten_elementen)

    masker = (
        df['JG_MKN'].isna() &
        (df['Eenheid'].str.lower() == 'ug/l') &
        (~is_metaal_of_element)
    )

    df.loc[masker, 'Signaleringswaarde'] = 0.1

    st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Voeg de stofgroep kolom toe
    df['Stofgroep'] = df['Stof'].apply(bepaal_stofgroep)

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


def create_gauge(percentage: float, title_text: str = "Metingen onder Norm") -> go.Figure:

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title_text, 'font': {'size': 14}},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "green"},
                 'steps' : [
                     {'range': [0, 80], 'color': "lightgray"},
                     {'range': [80, 100], 'color': "green"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def bepaal_stofgroep(stofnaam: str) -> str:

    s = stofnaam.lower()

    pfas = ['perfluor-2-propoxypropaanzuur', 'perfluor-1-octaansulfonaat (lineair)', 'perfluoroctaanzuur', 'perfluorbutaanzuur', 'perfluorpentaansulfonzuur', 'perfluorhexaanzuur', 'pfhpa', 'perfluornonaansulfonzuur', 'perfluordecaanzuur', 'genx', 'perfluor', 'adona', 'N-ethyl-perfluoroctaan sulfonamidoazijnzuur', 'N-methyl-perfluoroctaan sulfonamidoazijnzuur',
                     'N-methylperfluorbutaansulfonamide', 'perfluor-3-methoxypropaanzuur', 'perfluor-3,6-dioxaheptaanzuur', 'perfluor-4-methoxybutaanzuur', 'perfluor(2-ethoxyethaan)sulfonzuur', 'perfluorbutaansulfonamide', 'perfluorbutaansulfonzuur', 'perfluordecaansulfonzuur', '4,8-dioxa-3H-perfluornonaanzuur', '4:2 fluortelomeersulfonzuur', '6:2 fluortelomeersulfonzuur',
                     'perfluordodecaanzuur', 'perfluorheptaansulfonzuur', 'perfluorheptaanzuur', 'perfluorhexaansulfonamide', 'perfluorhexaansulfonzuur', 'perfluornonaanzuur', 'perfluoroctaansulfonamide', '8:2 fluortelomeersulfonzuur', '9-chloorhexadecaanfluor-3-oxanon-1-sulfonzuur',
                     'perfluorpentaanzuur', 'perfluortetradecaanzuur', 'perfluortridecaanzuur', 'perfluorundecaanzuur', 'Som hexadecafluor-2-deceenzuur-isomerenÂ ', 'som vertakte perfluorhexaansulfonzuur-isomeren',
                     'som vertakte perfluoroctaansulfonzuur-isomeren', '10:2 fluortelomeersulfonzuur', '11-chlooreicosafluor-3-oxaundecaan-1-sulfonzuur', 'som hexadecafluor-2-deceenzuur-isomerenâ']
    if any(k in s for k in pfas):
        return 'PFAS'

    metalen = ['aluminium', 'antimoon', 'arseen', 'barium', 'beryllium', 'boor', 'cadmium', 'calcium', 'cerium', 'cesium', 'chroom', 'cobalt', 'kobalt', 'dysprosium', 'erbium', 'europium', 'kalium', 'koper', 'kwik',
               'lood', 'gadolinium', 'gallium', 'hafnium', 'magnesium', 'mangaan', 'molybdeen', 'natrium', 'nikkel', 'seleen', 'strontium', 'thallium', 'tin', 'titanium', 'uranium', 'vanadium', 'ijzer', 'zilver', 'zink',
               'holmium', 'indium','lanthaan', 'lithium', 'lutetium', 'neodymium', 'niobium','platina', 'praseodymium', 'rubidium', 'samarium','tantalium', 'tellurium','terbium', 'thallium', 'thorium', 'thulium', 'wolfraam',
               'ytterbium','yttrium', 'zirkonium', 'titaan', 'scandium', 'arsenaat', 'arseniet', 'selenaat', 'seleniet', 'chroom (zeswaardig)']
    if any(k in s for k in metalen):
        return 'Metalen & elementen'

    nutrienten = ['fluoride', 'Biochemisch zuurstofverbruik met allylthioureum', 'chlorofyl-a','siliciumdioxide', 'sulfaat','koolstof organisch','stikstof', 'nitraat', 'nitriet', 'ammonium', 'fosfor', 'fosfaat', 'fosfor totaal', 'chloride', 'zuurstof', 'silicium', 'zwevende stof', 'hardheid', 'temperatuur',
                  'zuurgraad', 'geleidbaarheid', 'Gloeirest', 'Onopgeloste stoffen', 'stikstof totaal', 'waterstofcarbonaat', 'zuurstof', 'cyanide', 'gloeirest', 'onopgeloste stoffen']
    if any(k in s for k in nutrienten):
        return 'Nutriënten & algemeen'

    paks_pcbs_pbdes = ['naftaleen', 'antraceen', 'fenantreen', 'fluorantheen', 'benzo(a)', 'benzo(ghi)', 'benzo(k)', 'chryseen', 'pyreen','dibenzo(a,h)antraceen', 'indeno(1,2,3-cd)pyreen', 'som benzo(b)fluorantheen en benzo(j)fluorantheen','som PCB28 en PCB31',
                       "2,2',3,3',4,4',5,5',6,6'-decabroomdiphenylether", "2,2',3,4,4'-pentabroomdifenylether", "2,2',3,4,4',5,5'-heptachloorbifenyl", "2,2',3,4,4',5'-hexabroomdifenylether", "2,2',3,4,4',5'-hexachloorbifenyl", "2,2',4,4'-tetrabroomdifenylether",
                       "2,2',4,4',5-pentabroomdifenylether", "2,2',4,4',5,5'-hexabroomdifenylether", "2,2',4,4',5,5'-hexachloorbifenyl", "2,2',4,4',5,6'-hexabroomdifenylether", "2,2',4,4',6-pentabroomdifenylether", "2,2',4,5,5'-pentachloorbifenyl", "2,2',4,5'-tetrabroomdifenylether",
                       "2,2',5,5'-tetrachloorbifenyl", "2,3',4,4',5-pentachloorbifenyl", "2,4,4'-tribroomdifenylether", 'som pcb28 en pcb31']
    if any(k in s for k in paks_pcbs_pbdes):
        return 'PAKs/PCBs/PBDEs'

    pesticiden = ['glyfosaat', 'ampa', 'metolachloor', 'imidacloprid', 'mcpa', 'mecoprop', 'terbutylazine', 'abamectine', 'aclonifen', 'alachloor', 'aldrin', 'endosulfan', 'hexachloorcyclohexaan', 'atrazine', 'bentazon', 'bifenox',
                  'chloorfenvinfos', 'chloortoluron', 'chloridazon' 'heptachloorepoxide', 'cumafos', 'cypermethrin', 'desethylatrazine', 'diazinon', 'dichloorvos', 'dicofol', 'dieldrin', 'dimethenamid-P', 'dimethoaat', 'dinoseb', 'dinoterb',
                  'diuron', 'dodine', 'endrin', 'ethylazinfos', 'ethylchloorpyrifos', 'fenamifos', 'fenoxycarb', 'heptachloor', 'heptenofos', 'hexachloorbenzeen', 'hexachloorbutadieen', 'irgarol', 'isodrin', 'isoproturon', 'linuron',
                  'malathion', 'methabenzthiazuron', 'metazachloor', 'methyl-metsulfuron', 'methylazinfos', 'methylpirimifos', 'metolachloor', 'mevinfos', 'monolinuron', 'pirimicarb', 'propazine', 'propiconazol (som cis- en trans-)',
                  'pyrazofos', 'pyridaben', 'pyriproxyfen', 'quinoxyfen', 'simazine', 'teflubenzuron', 'terbutrin', 'terbutylazine', 'thiacloprid', 'tolclofos-methyl', 'trans-heptachloorepoxide', 'triazofos', '2-methyl-4-chloorfenoxyazijnzuur',
                  '2-methyl-4-chloorfenoxyboterzuur', '2,4-dichloorfenoxyazijnzuur', '2,4-dichloorfenoxyboterzuur', '2,4-dichloorfenoxypropionzuur', '2,4,5-trichloorfenoxyazijnzuur', '2,4,5-trichloorfenoxypropionzuur', "2,4'-dichloordifenyltrichloorethaan",
                  "4,4'-dichloordifenyldichloorethaan", "4,4'-dichloordifenyldichlooretheen", "4,4'-dichloordifenyltrichloorethaan", '4,6-dinitro-o-cresol', 'chloridazon', 'dimethenamid-p', 'metabenzthiazuron']
    if any(k in s for k in pesticiden):
        return 'Bestrijdingsmiddelen'

    pharma = ['diclofenac', 'carbamazepine', 'metformine', 'tramadol', 'paracetamol', 'gadobutrol', 'gadopentetaat anion', 'gadoteraat anion', 'gadoteridol']
    if any(k in s for k in pharma):
        return 'Geneesmiddelen'

    btex = ['benzeen', 'tolueen', 'etylbenzeen', 'xyleen', 'styreen', 'chloorbenzeen', 'chlooretheen (vinylchloride)', '1,2-dichloorethaan', '1,3-dichloorpropeen', 'cumeen', 'cyclohexaan', 'dibroomchloormethaan', 'dibroommethaan', 'dichloorbroommethaan', 'dichloormethaan',
            'dicyclopentadieen', 'diisopropylether', 'dimethoxymethaan', 'dimethyldisulfide', 'epichloorhydrine', 'ethylbenzeen', 'hexachloorethaan', 'methyl-tertiair-butylether', 'som 1,3- en 1,4-xyleen', 'tertiair-butylbenzeen', '2-ethyltolueen',
            'tetrachlooretheen (per)', 'tetrachloormethaan (tetra)', 'trans-1,2-dichlooretheen', 'trans-1,3-dichloorpropeen', 'tribroommethaan', 'trichlooretheen (tri)', 'trichloormethaan (chloroform)', '1,2-xyleen', '2-chloortolueen',
            '1-propylbenzeen', '1,1-dichloorethaan', '1,1,1-trichloorethaan', '1,1,2-trichloor-1,2,2-trifluorethaan', '1,1,2-trichloorethaan', '1,1,2,2-tetrachloorethaan', '1,2-dichloorbenzeen', '1,2-dichloorethaan', '1,2-dichloorpropaan', '1,2,3-trichloorbenzeen',
            '1,2,3-trichloorpropaan', '1,2,3-trimethylbenzeen', '1,2,4-trichloorbenzeen', '1,2,4-trimethylbenzeen', '1,3-dichloorbenzeen', '1,3-dichloorpropaan','1,3,5-trichloorbenzeen', '1,3,5-trimethylbenzeen', '1,4-dichloorbenzeen', '2,2,5,5,-tetramethyl-tetrahydrofuran',
            '3-chloorpropeen', '3-chloortolueen', '3-ethyltolueen', '4-ethyltolueen', '1,1-dichlooretheen', 'cis-1,2-dichlooretheen', 'tetrachloorethaan']
    if any(k in s for k in btex):
        return 'Vluchtige organische stoffen'

    industrie = ['bisfenol-A', 'bisfenol-a', 'pentachloorbenzeen', 'pentachloorfenol', 'som 2,4- en 2,5-dichloorfenol', 'som 4-nonylfenol-isomeren (vertakt)', '2-chloorfenol', '2,3-dichloorfenol', '2,3,4-trichloorfenol', '2,3,4,5-tetrachloorfenol', '2,3,5-trichloorfenol', '2,3,5,6-tetrachloorfenol',
                 '2,3,6-trichloorfenol', '2,4-dinitrofenol', '2,4,5-trichloorfenol', '2,4,6-trichloorfenol', '2,6-dichloorfenol', '3-chloorfenol', '3,4-dichloorfenol', '3,4,5-trichloorfenol', '3,5-dichloorfenol', '4-chloorfenol', '4-tertiair-octylfenol', '2,3,4,6-tetrachloorfenol',
                 'di-ethyleentriaminepentaazijnzuur (dtpa)', 'methylmethacrylaat', 'nitrilotriazijnzuur (nta)', 'ethyleendiaminetetraethaanzuur (edta)']
    if any(k in s for k in industrie):
        return 'Industriestoffen & overigen'

    return 'Onbekend'


def main():

    st.set_page_config(layout="wide", page_title="Waterkwaliteit KPI Dashboard")

    st.title("💧 Chemisch Waterkwaliteit KPI Dashboard")
    if 'last_update' not in st.session_state:
         st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(f"**Laatste update:** {st.session_state.last_update}")

    df_main = load_data(DATA_FILE_PATH)

    if df_main.empty:
        st.error("🚨 Kritieke Fout: De geladen en opgeschoonde data is leeg. Controleer de invoer CSV of de opschoningsstappen in de code.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Toestand & Trend",
        "✅ KRW Normtoetsing",
        "📊 Overzicht Toestand & Kaart",
        "⚠️ Risicoanalyse (Signaleringswaarden)",
        "Effectbeoordeling PFAS",
        "Herkomst en verdeling van stoffen",
        "Ruimtelijke analyse"
    ])

    # --- TAB 1: Toestand & Trend Ontwikkeling ---
    with tab1:
        st.header("📈 Toestand en Trend Ontwikkeling")
        st.markdown("Elke geselecteerde stof wordt in een **aparte subplot** weergegeven om de juiste eenheden en schalen te garanderen. Elk punt is een individuele meting.")

        st.write("### 🔍 Selectie filters")
        stof_filter_type = st.radio(
            "Welk type stoffen wil je selecteren?",
            options=["Alle stoffen", "KRW-stoffen", "Niet-genormeerde stoffen"],
            index=0,
            horizontal=True
        )

        if "KRW-stoffen" in stof_filter_type:
            beschikbare_stoffen = df_main.dropna(subset=['JG_MKN', 'MAC_MKN'], how='all')['Stof'].unique()
        elif "Niet-genormeerde stoffen" in stof_filter_type:
            stoffen_met_norm = df_main.dropna(subset=['JG_MKN', 'MAC_MKN'], how='all')['Stof'].unique()
            beschikbare_stoffen = df_main[~df_main['Stof'].isin(stoffen_met_norm)]['Stof'].unique()
        else:
            beschikbare_stoffen = df_main['Stof'].unique()

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            sorted_meetpunten = sorted(df_main['Meetpunt'].unique())
            default_meetpunten = st.session_state.get("tab1_meetpunten_select", sorted_meetpunten[:1])
            selected_meetpunten = st.multiselect(
                "Selecteer Meetpunt(en)",
                options=sorted_meetpunten,
                default=default_meetpunten,
                key="tab1_meetpunten_select"
            )
        with col2:
            sorted_stoffen = sorted(beschikbare_stoffen)

            current_selection = st.session_state.get("tab1_stoffen_select", [])
            valid_default = [s for s in current_selection if s in sorted_stoffen]

            if not valid_default and len(sorted_stoffen) > 0:
                valid_default = [sorted_stoffen[0]]

            selected_stoffen = st.multiselect(
                "Selecteer Stof(fen)",
                options=sorted_stoffen,
                default=valid_default,
                key="tab1_stoffen_select"
            )

        if selected_meetpunten and selected_stoffen:
            df_trend = df_main[
                (df_main['Meetpunt'].isin(selected_meetpunten)) &
                (df_main['Stof'].isin(selected_stoffen))
            ].copy()

            if not df_trend.empty:

                df_trend['Meting Type'] = df_trend['Limietsymbool'].apply(
                    lambda x: '< Onder rapportagegrens' if '<' in str(x) else 'Gemeten waarde'
                )
                unique_stoffen = sorted(df_trend['Stof'].unique())

                stof_info = df_trend.groupby('Stof').agg({
                    'Eenheid': 'first',
                    'JG_MKN': lambda x: x.dropna().median() if x.dropna().size > 0 else None,
                    'MAC_MKN': lambda x: x.dropna().median() if x.dropna().size > 0 else None,
                }).to_dict('index')

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
                    title='Geselecteerde Metingen over Tijd',
                    hover_data={'Datum': True, 'Waarde': ':.4f', 'Eenheid': True, 'Meetpunt': True},
                    category_orders={"Stof": unique_stoffen},
                    height=350 * len(unique_stoffen)
                )

                fig.update_xaxes(matches='x')
                fig.update_yaxes(matches=None)

                for i, stof_naam in enumerate(unique_stoffen):
                    row_index = len(unique_stoffen) - i

                    if stof_naam in stof_info:
                        info = stof_info[stof_naam]
                        eenheid = info.get('Eenheid', '')

                        jg_norm = info.get('JG_MKN')
                        mac_norm = info.get('MAC_MKN')

                        max_waarde_data = df_trend[df_trend['Stof'] == stof_naam]['Waarde'].max()
                        current_max = max_waarde_data if pd.notna(max_waarde_data) else 0

                        vergelijkings_waarden = [current_max]
                        if jg_norm is not None and not pd.isna(jg_norm):
                            vergelijkings_waarden.append(jg_norm)
                        if mac_norm is not None and not pd.isna(mac_norm):
                            vergelijkings_waarden.append(mac_norm)
                        target_top = max(vergelijkings_waarden) * 1.15

                        fig.update_yaxes(
                            title_text=f"Waarde ({eenheid})",
                            range=[0, target_top],
                            row=row_index,
                            col=1,
                            showticklabels=True
                        )

                        if jg_norm is not None and not pd.isna(jg_norm):
                            fig.add_hline(
                                y=jg_norm,
                                line_dash="dash",
                                line_color="darkorange",
                                line_width=2,
                                row=row_index,
                                col=1,
                                annotation_text=f"JG: {jg_norm:.4f}",
                                annotation_position="top right",
                                annotation_font_size=10,
                                annotation_font_color="darkorange"
                            )

                        if mac_norm is not None and not pd.isna(mac_norm):
                            fig.add_hline(
                                y=mac_norm,
                                line_dash="dot",
                                line_color="red",
                                line_width=2,
                                row=row_index,
                                col=1,
                                annotation_text=f"MAC: {mac_norm:.2f}",
                                annotation_position="top left",
                                annotation_font_size=10,
                                annotation_font_color="red"
                            )

                fig.update_yaxes(showticklabels=True)
                fig.update_layout(title_x=0.5)
                fig.update_layout(margin=dict(l=80, r=80))

                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("Geen data gevonden voor de geselecteerde combinatie.")
        else:
            st.info("Selecteer minstens één meetpunt en één stof om de trend te zien.")

    # --- TAB 2: Normtoetsing met KRW (JG-MKN en MAC-MKN) ---
    with tab2:
        st.header("✅ KRW Normtoetsing per Meetpunt")
        st.markdown("Analyseer het percentage metingen boven de Kader Richtlijn Water normen (JG-MKN en MAC-MKN) per meetpunt.")

        col_filter_1, col_filter_2 = st.columns(2)

        with col_filter_1:
            all_stoffen_norm = sorted(df_main['Stof'].unique())
            stoffen_met_norm = df_main.dropna(subset=['JG_MKN', 'MAC_MKN'])['Stof'].unique()
            default_stoffen_norm = sorted(stoffen_met_norm)[:3]

            selected_stoffen_norm = st.multiselect(
                "🔎 **Selecteer Stof(fen)**",
                options=all_stoffen_norm,
                default=default_stoffen_norm
            )

        with col_filter_2:
            all_meetpunten_norm = sorted(df_main['Meetpunt'].unique())
            default_meetpunten_norm = all_meetpunten_norm[:3] if all_meetpunten_norm else []
            selected_meetpunten_norm = st.multiselect(
                "📍 **Selecteer Meetpunt(en)**",
                options=all_meetpunten_norm,
                default=default_meetpunten_norm
            )

        if not selected_stoffen_norm or not selected_meetpunten_norm:
            st.warning("Selecteer minstens één stof en één meetpunt om de normtoetsing te zien.")
        else:
            # JG-MKN Toetsing
            st.subheader("Jaargemiddelde toetsing (JG-MKN)")
            st.info("De JG-MKN toetsing berekent het jaargemiddelde per jaar, per meetpunt, en vergelijkt dit met de norm. De staafdiagram toont het percentage jaren waarin het jaargemiddelde de norm overschrijdt. **NB.** dit is geen officiële normtoetsing.")

            df_jg = df_main.dropna(subset=['JG_MKN']).copy()
            df_jg = df_jg[
                (df_jg['Stof'].isin(selected_stoffen_norm)) &
                (df_jg['Meetpunt'].isin(selected_meetpunten_norm))
            ]

            if not df_jg.empty:
                df_jg['Jaar'] = df_jg['Datum'].dt.year
                df_gemiddelde = df_jg.groupby(['Jaar', 'Meetpunt', 'Stof'])['Waarde'].mean().reset_index()

                df_gemiddelde = pd.merge(
                    df_gemiddelde,
                    df_jg[['Stof', 'JG_MKN']].drop_duplicates(),
                    on='Stof',
                    how='left'
                )

                df_gemiddelde['Overschrijding'] = df_gemiddelde['Waarde'] > df_gemiddelde['JG_MKN']

                jg_overschrijding_pct = df_gemiddelde.groupby(['Meetpunt', 'Stof'])['Overschrijding'].mean().reset_index()
                jg_overschrijding_pct['Overschrijding (%)'] = (jg_overschrijding_pct['Overschrijding'] * 100).round(2)

                jg_overschrijding_pct['PlotLabel'] = jg_overschrijding_pct['Meetpunt'] + ' (' + jg_overschrijding_pct['Stof'] + ')'

                fig_jg = px.bar(
                    jg_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
                    x='PlotLabel',
                    y='Overschrijding (%)',
                    title='Percentage boven JG-MKN norm per meetpunt',
                    color='Meetpunt',
                    labels={'PlotLabel': 'Meetpunt (Stof)'},
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig_jg, use_container_width=True)
            else:
                st.warning("Geen JG-MKN normen gedefinieerd of data gevonden voor de geselecteerde stof(fen)/meetpunt(en).")

            st.markdown("---")

            # MAC-MKN Toetsing
            st.subheader("Maximale aanvaardbare concentratie toetsing (MAC-MKN)")
            st.info("De MAC-MKN toetsing telt het percentage individuele metingen per meetpunt dat de norm overschrijdt. **NB.** dit is geen officiële normtoetsing.")

            df_mac = df_main.dropna(subset=['MAC_MKN']).copy()
            df_mac = df_mac[
                (df_mac['Stof'].isin(selected_stoffen_norm)) &
                (df_mac['Meetpunt'].isin(selected_meetpunten_norm))
            ]

            if not df_mac.empty:
                df_mac['Overschrijding'] = df_mac['Waarde'] > df_mac['MAC_MKN']

                mac_overschrijding_pct = df_mac.groupby(['Meetpunt', 'Stof'])['Overschrijding'].mean().reset_index()
                mac_overschrijding_pct['Overschrijding (%)'] = (mac_overschrijding_pct['Overschrijding'] * 100).round(2)

                mac_overschrijding_pct['PlotLabel'] = mac_overschrijding_pct['Meetpunt'] + ' (' + mac_overschrijding_pct['Stof'] + ')'

                fig_mac = px.bar(
                    mac_overschrijding_pct.sort_values(['Stof', 'Meetpunt']),
                    x='PlotLabel', y='Overschrijding (%)',
                    title='Percentage individuele metingen boven MAC-MKN norm per meetpunt',
                    color='Meetpunt',
                    labels={'PlotLabel': 'Meetpunt (Stof)'},
                    color_continuous_scale=px.colors.sequential.Reds
                )
                st.plotly_chart(fig_mac, use_container_width=True)
            else:
                st.warning("Geen MAC-MKN normen gedefinieerd of data gevonden voor de geselecteerde stof(fen)/meetpunt(en).")

    # --- TAB 3: In één Oogopslag Toestand & Kaart ---
    with tab3:
        st.header("📊 Normoverschrijdingen KPI")

        st.subheader("Totaalbeeld (alle meetpunten)")
        col_kpi_info, col_kpi_gauges, col_map = st.columns([1, 2, 2])

        df_jg_overall = df_main.dropna(subset=['JG_MKN']).copy()
        pct_jg_total = 0
        if not df_jg_overall.empty:
            aantal_jg_totaal = len(df_jg_overall)
            aantal_jg_voldoet = len(df_jg_overall[df_jg_overall['Waarde'] <= df_jg_overall['JG_MKN']])
            pct_jg_total = round((aantal_jg_voldoet / aantal_jg_totaal * 100), 1) if aantal_jg_totaal > 0 else 0

        df_mac_overall = df_main.dropna(subset=['MAC_MKN']).copy()
        pct_mac_total = 0
        if not df_mac_overall.empty:
            aantal_mac_totaal = len(df_mac_overall)
            aantal_mac_voldoet = len(df_mac_overall[df_mac_overall['Waarde'] <= df_mac_overall['MAC_MKN']])
            pct_mac_total = round((aantal_mac_voldoet / aantal_mac_totaal * 100), 1) if aantal_mac_totaal > 0 else 0

        with col_kpi_info:
            st.markdown("### ℹ️ Info")
            st.metric(label="Unieke meetpunten", value=df_main['Meetpunt'].nunique())
            st.markdown("---")
            st.metric(label="Metingen met JG-norm", value=len(df_jg_overall))
            st.metric(label="Metingen met MAC-norm", value=len(df_mac_overall))

        with col_kpi_gauges:
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.plotly_chart(create_gauge(pct_jg_total, "Totaal: voldoet JG (%)"), use_container_width=True)
            with sub_col2:
                st.plotly_chart(create_gauge(pct_mac_total, "Totaal: voldoet MAC (%)"), use_container_width=True)

        with col_map:
            st.markdown("### 📍 Meetpunten Kaart")
            df_map = df_main[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates()

            if not df_map[['Latitude', 'Longitude']].isnull().all().all():
                df_map = df_map.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}).dropna(subset=['lat', 'lon']).copy()

                df_map['Grootte'] = 20

                center_lat = df_map['lat'].mean()
                center_lon = df_map['lon'].mean()

                fig_map = px.scatter_mapbox(
                    df_map,
                    lat='lat',
                    lon='lon',
                    hover_name='Meetpunt',
                    size='Grootte',
                    size_max=20,
                    zoom=8,
                    color_discrete_sequence=['red'],
                    mapbox_style="open-street-map"
                )

                fig_map.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox_center={"lat": center_lat, "lon": center_lon}
                )

                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info("Geen geldige coördinaten gevonden in de data om de kaart te tonen.")

        st.divider()

        st.subheader("📍 KPI per specifiek meetpunt")

        unieke_meetpunten = sorted(df_main['Meetpunt'].unique())

        with st.expander(f"Toon detailmeters voor alle {len(unieke_meetpunten)} meetpunten", expanded=True):

            for meetpunt in unieke_meetpunten:
                st.markdown(f"**Meetpunt: {meetpunt}**")

                df_mp = df_main[df_main['Meetpunt'] == meetpunt]

                # JG voor dit punt
                df_jg_mp = df_mp.dropna(subset=['JG_MKN'])
                pct_jg_mp = None

                if not df_jg_mp.empty:
                    aantal_jg = len(df_jg_mp)
                    voldoet_jg = len(df_jg_mp[df_jg_mp['Waarde'] <= df_jg_mp['JG_MKN']])
                    pct_jg_mp = round((voldoet_jg / aantal_jg * 100), 1)

                # MAC voor dit punt
                df_mac_mp = df_mp.dropna(subset=['MAC_MKN'])
                pct_mac_mp = None

                if not df_mac_mp.empty:
                    aantal_mac = len(df_mac_mp)
                    voldoet_mac = len(df_mac_mp[df_mac_mp['Waarde'] <= df_mac_mp['MAC_MKN']])
                    pct_mac_mp = round((voldoet_mac / aantal_mac * 100), 1)

                jg_display = f"{pct_jg_mp}%" if pct_jg_mp is not None else "N.V.T."
                mac_display = f"{pct_mac_mp}%" if pct_mac_mp is not None else "N.V.T."

                st.markdown(f"### 📍 **{meetpunt}** | Voldoet JG: **{jg_display}** | Voldoet MAC: **{mac_display}**")

                col_mp_1, col_mp_2 = st.columns(2)

                with col_mp_1:
                    if pct_jg_mp is not None:
                        st.plotly_chart(create_gauge(pct_jg_mp, f"JG-MKN: {meetpunt}"), use_container_width=True, key=f"gauge_jg_{meetpunt}")
                    else:
                        st.info(f"Geen JG-normen of data voor {meetpunt}")

                with col_mp_2:
                    if pct_mac_mp is not None:
                        st.plotly_chart(create_gauge(pct_mac_mp, f"MAC-MKN: {meetpunt}"), use_container_width=True, key=f"gauge_mac_{meetpunt}")
                    else:
                        st.info(f"Geen MAC-normen of data voor {meetpunt}")

                st.markdown("---")

        st.subheader("⚠️ Meest recente overschrijdingen")

        df_overschrijdingen = df_main.copy()

        df_overschrijdingen['Overschrijding_JG'] = df_overschrijdingen.apply(
            lambda row: row['Waarde'] > row['JG_MKN'] if pd.notna(row['JG_MKN']) else False, axis=1
        )
        df_overschrijdingen['Overschrijding_MAC'] = df_overschrijdingen.apply(
            lambda row: row['Waarde'] > row['MAC_MKN'] if pd.notna(row['MAC_MKN']) else False, axis=1
        )

        df_violations = df_overschrijdingen[
            (df_overschrijdingen['Overschrijding_JG']) | (df_overschrijdingen['Overschrijding_MAC'])
        ].sort_values('Datum', ascending=False)

        if not df_violations.empty:
            def type_overschrijding(row):
                types = []
                if row['Overschrijding_JG']: types.append("JG")
                if row['Overschrijding_MAC']: types.append("MAC")
                return "+".join(types)

            df_violations['Type'] = df_violations.apply(type_overschrijding, axis=1)

            st.dataframe(
                df_violations[['Datum', 'Meetpunt', 'Stof', 'Waarde', 'Eenheid', 'JG_MKN', 'MAC_MKN', 'Type']].head(15),
                use_container_width=True
            )
        else:
            st.success("Geen overschrijdingen gevonden in de huidige dataset!")

    # --- TAB 4: Risicoanalyse Stoffen zonder Normen ---
    with tab4:
        st.header("⚠️ Risicoanalyse (Signaleringswaarden)")
        st.info("Deze analyse toont stoffen waarvan **geen MKN normen** bekend zijn. Voor deze stoffen wordt getoetst aan een generieke signaleringswaarde van **0.1 ug/l**. Er wordt enkel gekeken naar **aangetroffen waarden** (boven de rapportagegrens).")

        # 1. Selecteer stoffen met een signaleringswaarde
        df_risico = df_main.dropna(subset=['Signaleringswaarde']).copy()

        # 2. FILTER: Verwijder metingen onder de rapportagegrens (<)
        # We kijken of het limietsymbool een '<' bevat. De tilde (~) draait de selectie om (behoud wat GEEN < heeft).
        df_risico = df_risico[~df_risico['Limietsymbool'].astype(str).str.contains('<', na=False)]

        if not df_risico.empty:

            df_risico['Boven_Signalering'] = df_risico['Waarde'] > df_risico['Signaleringswaarde']

            risico_summary = df_risico.groupby('Stof').agg(
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

            st.subheader("Actuele Risico-overschrijdingen")

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

    # --- TAB 5: PFAS effectbeoordeling ---
    with tab5:
        st.header("Effectbeoordeling PFAS (relatieve toxiciteit & bioaccumulatie)")
        st.info("De berekende waarden zijn op basis van lower bound concentraties.")

        df_pfas_ref = load_pfas_ref(PFAS_FILE_PATH)

        if df_pfas_ref.empty:
            st.warning("Kan de PFAS koppeltabel niet laden.")
        else:

            df_pfas_ref['Stofnaam'] = df_pfas_ref['Stofnaam'].str.lower().str.strip()

            df_main['Stof_Match'] = df_main['Stof'].str.replace(r' \(totaal\)', '', regex=True)
            df_main['Stof_Match'] = df_main['Stof_Match'].str.replace(r' \(opgelost\)', '', regex=True).str.strip()

            df_pfas_calc = pd.merge(
                df_main,
                df_pfas_ref,
                left_on='Stof_Match',
                right_on='Stofnaam',
                how='inner'
            )

            if df_pfas_calc.empty:
                st.info("Geen PFAS stoffen uit de koppeltabel gevonden in de huidige meetdata.")
                st.markdown("""
                **Mogelijke oorzaken:**
                1. De namen komen niet exact overeen (bijv. 'PFOS' vs 'Perfluoroctaansulfonaat').
                2. Er zitten typefouten in de CSV.
                            """)
            else:
                meetpunten = df_pfas_calc['Meetpunt'].unique()
                selected_meetpunt = st.selectbox("Selecteer meetpunt voor PFAS PEQ analyse", meetpunten)

                df_plot = df_pfas_calc[df_pfas_calc['Meetpunt'] == selected_meetpunt].copy()

                if df_plot.empty:
                    st.warning("Geen data voor dit meetpunt.")
                else:
                    mask_detected = df_plot['Limietsymbool'].astype(str).str.strip().eq('')

                    df_plot['RPF_calc'] = 0.0
                    df_plot['RBF_calc'] = 0.0

                    df_plot.loc[mask_detected, 'RPF_calc'] = df_plot['RPF']
                    df_plot.loc[mask_detected, 'RBF_calc'] = df_plot['RBF']

                    df_plot['Waarde_ng'] = df_plot['Waarde'] * 1000

                    # Berekeningen
                    df_plot['PEQ_Waarde'] = df_plot['Waarde_ng'] * df_plot['RPF_calc']
                    df_plot['Bioacc_Waarde'] = df_plot['PEQ_Waarde'] * df_plot['RBF_calc']

                    # GRAFIEK 1: RPF (PEQ)
                    st.subheader("Relatieve toxiciteit (PEQ in ng/l)")
                    st.markdown("Relatieve toxiciteit **Waarde (ng/l) × RPF** per individuele stof. De totale staafhoogte is de som PEQ.")
                    st.info("Deze toetsing is voor de beoordeling van geschiktheid van oppervlaktewater voor de bereiding van drinkwater. De som PEQ risicogrenswaarde is 4.4 ng/PEQ L.")

                    df_plot = df_plot.sort_values('Datum')

                    fig_rpf = px.bar(
                        df_plot,
                        x='Datum',
                        y='PEQ_Waarde',
                        color='Stof',
                        title=f'Trend relatieve toxiciteit {selected_meetpunt}',
                        labels={'PEQ_Waarde': 'PEQ Waarde (ng/l)', 'Datum': 'Datum'},
                        hover_data={'Waarde_ng': ':.2f', 'RPF': True, 'PEQ_Waarde': ':.2f'}
                    )
                    fig_rpf.update_traces(hovertemplate='<b>%{x}</b><br>Stof: %{legendgroup}<br>Conc: %{customdata[0]:.2f} ng/l<br>RPF: %{customdata[1]}<br>PEQ: %{y:.2f}')

                    fig_rpf.add_hline(
                        y=4.4,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text="Risicogrens: 4.4",
                        annotation_position="top left",
                        annotation_font_color="black"
                    )

                    st.plotly_chart(fig_rpf, use_container_width=True)

                    st.write("**Som van PEQ-waarden (ng/l) per datum:**")
                    df_sum_rpf = df_plot.groupby('Datum')['PEQ_Waarde'].sum().reset_index()
                    st.dataframe(df_sum_rpf.style.format({"PEQ_Waarde": "{:.2f}"}), use_container_width=True)

                    st.divider()

                    # GRAFIEK 2: RBF (Bioaccumulatie)
                    st.subheader("Relatieve toxiciteit en bioaccumulatie")
                    st.markdown("Relatieve toxiciteit en bioaccumulatie  **Waarde (ng/l) × RPF × RBF** per individuele PFAS stof. De totale staafhoogte is de som PEQ.")
                    st.info("Deze toetsing is voor de beoordeling van geschiktheid van oppervlaktewater voor visconsumptie. De som PEQ risicogrenswaarde is 0.3 ng/PEQ L.")

                    fig_rbf = px.bar(
                        df_plot,
                        x='Datum',
                        y='Bioacc_Waarde',
                        color='Stof',
                        title=f'Trend relatieve toxiciteit x bioaccumulatie (Waarde ng/l × RPF × RBF) - {selected_meetpunt}',
                        labels={'Bioacc_Waarde': 'Bioaccumulatie Score', 'Datum': 'Datum'},
                        hover_data={'Waarde_ng': ':.2f', 'RPF': True, 'RBF': True, 'Bioacc_Waarde': ':.2f'}
                    )
                    fig_rbf.update_traces(hovertemplate='<b>%{x}</b><br>Stof: %{legendgroup}<br>Conc: %{customdata[0]:.2f} ng/l<br>RBF: %{customdata[2]}<br>Score: %{y:.2f}')

                    fig_rbf.add_hline(
                        y=0.3,
                        line_dash="dash",
                        line_color="red",
                        line_width=2,
                        annotation_text="Risicogrens: 0.3",
                        annotation_position="top left",
                        annotation_font_color="black"
                    )

                    st.plotly_chart(fig_rbf, use_container_width=True)

                    st.write("**Som van Bioaccumulatie-scores per datum:**")
                    df_sum_rbf = df_plot.groupby('Datum')['Bioacc_Waarde'].sum().reset_index()
                    st.dataframe(df_sum_rbf.style.format({"Bioacc_Waarde": "{:.2f}"}), use_container_width=True)

# --- TAB 6: Herkomst en verdeling stoffen ---
    with tab6:
        st.header("🏭 Herkomst en verdeling van stoffen")
        st.info("De indeling in stofgroepen is gebaseerd op trefwoord-herkenning. Hieronder worden twee zaken getoond: alle uitgevoerde metingen (inclusief < rapportagegrens-waarden) en alleen de daadwerkelijk aangetoonde stoffen (> rapportagegrens).")

        df_herkomst = df_main.copy()
        df_herkomst = df_herkomst.dropna(subset=['Stof'])
        
        # Voeg stofgroep toe
        #df_herkomst['Stofgroep'] = df_herkomst['Stof'].apply(bepaal_stofgroep)

        # Maak een subset voor alleen waarden boven de rapportagegrens
        # We filteren rijen weg waar een '<' in het Limietsymbool staat
        df_detecties = df_herkomst[~df_herkomst['Limietsymbool'].astype(str).str.contains('<', na=False)].copy()

        # --- A. TOTAAL OVERZICHT ---
        st.subheader("Totaalbeeld (alle locaties)")
        
        st.markdown("### 🟠 Alle uitgevoerde metingen vs. 🟢 Boven rapportagegrens")
        
        col_total_all, col_total_det = st.columns(2)

        # 1. Linkerkolom: Alles
        with col_total_all:
            st.markdown("**Totaal: alle metingen**")
            dist_total = df_herkomst['Stofgroep'].value_counts().reset_index()
            dist_total.columns = ['Stofgroep', 'Aantal metingen']
            
            fig_pie_total = px.pie(
                dist_total,
                values='Aantal metingen',
                names='Stofgroep',
                title='Aantal metingen (incl. <RG)',
                hole=0.4
            )
            st.plotly_chart(fig_pie_total, use_container_width=True)

            st.markdown("*Top 25 meest gemeten (totaal)*")
            top_stoffen = df_herkomst['Stof'].value_counts().head(25).reset_index()
            top_stoffen.columns = ['Stof', 'Aantal']
            st.dataframe(top_stoffen, use_container_width=True)

        # 2. Rechterkolom: Alleen detecties
        with col_total_det:
            st.markdown("**Filter: aangetroffen stoffen (boven rapportagegrens)**")
            if not df_detecties.empty:
                dist_det = df_detecties['Stofgroep'].value_counts().reset_index()
                dist_det.columns = ['Stofgroep', 'Aantal x aangetroffen']
                
                fig_pie_det = px.pie(
                    dist_det,
                    values='Aantal x aangetroffen',
                    names='Stofgroep',
                    title='Aantal keer aangetroffen (>RG)',
                    hole=0.4
                )
                st.plotly_chart(fig_pie_det, use_container_width=True)

                st.markdown("*Top 25 meest aangetroffen (> RG)*")
                top_stoffen_det = df_detecties['Stof'].value_counts().head(25).reset_index()
                top_stoffen_det.columns = ['Stof', 'Aantal']
                st.dataframe(top_stoffen_det, use_container_width=True)
            else:
                st.warning("Geen waarden boven rapportagegrens gevonden in totale dataset.")

        st.divider()

        # --- B. PER MEETPUNT ---
        st.subheader("Overzicht per specifiek meetpunt")

        meetpunten_list = sorted(df_herkomst['Meetpunt'].unique())
        selected_mp_herkomst = st.selectbox("Selecteer een meetpunt voor detailanalyse:", meetpunten_list)

        # Dataframes filteren voor meetpunt
        df_mp_all = df_herkomst[df_herkomst['Meetpunt'] == selected_mp_herkomst]
        df_mp_det = df_detecties[df_detecties['Meetpunt'] == selected_mp_herkomst]

        if not df_mp_all.empty:
            col_mp_1, col_mp_2 = st.columns(2)

            # Grafiek Alle Metingen
            with col_mp_1:
                st.markdown(f"**{selected_mp_herkomst}: Alle metingen**")
                dist_mp_all = df_mp_all['Stofgroep'].value_counts().reset_index()
                dist_mp_all.columns = ['Stofgroep', 'Aantal']
                
                fig_pie_mp_all = px.pie(
                    dist_mp_all,
                    values='Aantal',
                    names='Stofgroep',
                    title=f'Aantal metingen'
                )
                st.plotly_chart(fig_pie_mp_all, use_container_width=True)

            # Grafiek Alleen Detecties
            with col_mp_2:
                st.markdown(f"**{selected_mp_herkomst}: > Rapportagegrens**")
                if not df_mp_det.empty:
                    dist_mp_det = df_mp_det['Stofgroep'].value_counts().reset_index()
                    dist_mp_det.columns = ['Stofgroep', 'Aantal']
                    
                    fig_pie_mp_det = px.pie(
                        dist_mp_det,
                        values='Aantal',
                        names='Stofgroep',
                        title=f'Aantal keer aangetroffen'
                    )
                    st.plotly_chart(fig_pie_mp_det, use_container_width=True)
                else:
                    st.info(f"Op meetpunt {selected_mp_herkomst} zijn geen stoffen boven de rapportagegrens aangetroffen.")

            # Detailtabel
            st.markdown(f"**Details per stofgroep voor {selected_mp_herkomst}**")
            
            # Toggle voor tabel weergave
            toon_alleen_detecties = st.checkbox("Toon in tabel alleen stoffen > Rapportagegrens", value=True)
            
            source_df = df_mp_det if toon_alleen_detecties else df_mp_all
            
            if not source_df.empty:
                beschikbare_groepen = source_df['Stofgroep'].unique()
                selected_groep = st.selectbox("Welke stofgroep wil je in detail zien?", sorted(beschikbare_groepen))

                df_detail_groep = source_df[source_df['Stofgroep'] == selected_groep]

                detail_agg = df_detail_groep.groupby('Stof').agg(
                    Aantal_Metingen=('Waarde', 'count'),
                    Gemiddelde_Waarde=('Waarde', 'mean'),
                    Max_Waarde=('Waarde', 'max'),
                    Eenheid=('Eenheid', 'first')
                ).sort_values('Aantal_Metingen', ascending=False).reset_index()
                
                # Format getallen
                detail_agg['Gemiddelde_Waarde'] = detail_agg['Gemiddelde_Waarde'].map('{:,.4f}'.format)
                detail_agg['Max_Waarde'] = detail_agg['Max_Waarde'].map('{:,.4f}'.format)

                st.dataframe(detail_agg, use_container_width=True)
            else:
                st.warning("Geen data beschikbaar voor de huidige selectie/filter.")

        else:
            st.warning("Geen data gevonden voor dit meetpunt.")

        st.divider()

        # --- C. OVERZICHT ONBEKENDE STOFFEN ---
        st.subheader("❓ Overzicht categorie 'onbekend'")
        st.info("Onderstaande tabel toont de unieke stoffen die niet automatisch herkend werden door de categoriseringsfunctie.")

        # We kijken hier naar de unieke stoffen uit de HELE dataset (zodat je weet wat je mist in je filters)
        df_onbekend = df_herkomst[df_herkomst['Stofgroep'] == 'Onbekend']

        if not df_onbekend.empty:
            df_onbekend_unique = df_onbekend.groupby('Stof').agg(
                Aantal_Metingen=('Waarde', 'count'),
                Eenheid=('Eenheid', 'first'),
                Voorbeeld_Waarde=('Waarde', 'mean')
            ).reset_index().sort_values('Stof')

            df_onbekend_unique['Voorbeeld_Waarde'] = df_onbekend_unique['Voorbeeld_Waarde'].round(2)

            st.dataframe(df_onbekend_unique, use_container_width=True)
        else:
            st.success("Alle gemeten stoffen zijn succesvol ingedeeld in een categorie.")
            
# --- TAB 7: Ruimtelijke analyse ---
    with tab7:
        st.header("🔍 Ruimtelijke analyse")
        st.markdown("Selecteer filters om data te analyseren.")

        # Maak een kopie van df_main om bewerkingen op uit te voeren
        df_space = df_main.copy()

        # Datumbereik bepalen voor de date pickers
        min_date = df_space['Datum'].min().date() if not df_space['Datum'].isnull().all() else datetime.date(2023, 1, 1)
        max_date = df_space['Datum'].max().date() if not df_space['Datum'].isnull().all() else datetime.date(2025, 12, 31)

        # Gegevens voor dropdowns (opties)
        loc_opties_all = sorted(df_space['Meetpunt'].unique())
        stof_opties = sorted(df_space['Stof'].unique())
        fold_change_opties = sorted(df_space['Meetpunt'].unique())
        default_stoffen = df_space['Stof'].value_counts().head(5).index.tolist() if not df_space.empty else []
    
        # Haal unieke jaren op uit de data
        if not df_space.empty and 'Datum' in df_space.columns:
            jaar_opties = sorted(df_space['Datum'].dt.year.unique())
        else:
            jaar_opties = []

        # --- Rijk 1: Filters (Nu over 2 rijen verdeeld voor duidelijkheid) ---
        with st.container():
            # Rij 1: Tijdinstellingen (Jaar, Zomer, Datum bereik)
            col_jaar, col_zomer, col_start, col_end = st.columns([2, 1.5, 2, 2])
            
            # A. Jaar Selectie
            jaren_selected = col_jaar.multiselect(
                "Selecteer Jaar/Jaren",
                options=jaar_opties,
                default=jaar_opties, # Standaard alles geselecteerd
                #key=f"jaar_{'KPI'}"
            )

            # B. Zomerhalfjaar Schakelaar
            col_zomer.write("") 
            col_zomer.write("") 
            zomerhalfjaar = col_zomer.checkbox(
                "Alleen Zomerhalfjaar\n(1 apr - 30 sep)", 
                value=False,
                #key=f"zomer_{dashboard_titel}"
            )

            # C. Datum Pickers (als extra verfijning)
            start_date = col_start.date_input(
                "Startdatum (verfijning)", 
                value=min_date, 
                min_value=min_date, 
                max_value=max_date,
                #key=f"start_{dashboard_titel}"
            )

            end_date = col_end.date_input(
                "Einddatum (verfijning)", 
                value=max_date, 
                min_value=min_date, 
                max_value=max_date,
                #key=f"end_{dashboard_titel}"
            )

            # Rij 2: Locatie en stof instellingen
            col_loc, col_stof = st.columns([3, 3])

            #D. Meetpunt
            meetpunt_options = loc_opties_all 
                
            locaties_selected = col_loc.multiselect(
                "Meetpunt",
                options=meetpunt_options,
                default=meetpunt_options,
                #key=f"locatie_{dashboard_titel}"
            )

            # E. Stof
            stoffen_selected = col_stof.multiselect(
                "Stof",
                options=stof_opties,
                default=default_stoffen,
                #key=f"stof_{dashboard_titel}"
            )
        
        # ======================================================
        # DATA FILTEREN
        # ======================================================
            
        dff = df_space

        # NIEUW: 0. Filter op Jaar
        if jaren_selected:
            dff = dff[dff['Datum'].dt.year.isin(jaren_selected)].copy()

        # NIEUW: 0b. Filter op Zomerhalfjaar (maand 4 t/m 9)
        if zomerhalfjaar:
            # Maand 4 is april, Maand 9 is september
            dff = dff[(dff['Datum'].dt.month >= 4) & (dff['Datum'].dt.month <= 9)].copy()

        # Datumconversie
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)

        # 1. Filter op Datum (Verfijning)
        if start_date_dt and end_date_dt:
            dff = dff[
                (dff['Datum'] >= start_date_dt) & 
                (dff['Datum'] <= end_date_dt)
            ].copy()
            
        # 3. Filter op Meetpunt
        if locaties_selected:
            dff = dff[dff['Meetpunt'].isin(locaties_selected)].copy()

        # --- Geen data gevonden fallback ---
        if dff.empty or locaties_selected is None or not locaties_selected:
            st.warning("Geen data beschikbaar voor de geselecteerde criteria.")
            return # Stop de rendering voor dit tabblad
            
        # Filter data voor grafieken die afhankelijk zijn van stoffen_selected
        dff_stof = dff[dff['Stof'].isin(stoffen_selected)].copy() if stoffen_selected else pd.DataFrame(columns=dff.columns)
        
        # ======================================================
        # PLOT GENERATIE
        # ======================================================

        # --- Tijdlijn Grafiek ---
        if dff_stof.empty:
            time_df = pd.DataFrame(columns=['Datum', 'Meetpunt', 'Waarde'])
        else:
            time_df = dff_stof.groupby(['Datum', 'Meetpunt'])['Waarde'].mean().reset_index()
        
        tijd_fig = px.line(
            time_df, x="Datum", y="Waarde", color="Meetpunt",
            title="Gemiddelde waarde (van geselecteerde stoffen) per locatie in de tijd", markers=True)
        
        # --- Kaart Grafiek ---
        # zodat de kaart alleen het gemiddelde toont van de geselecteerde Stofen.
        
        # Maak een dataframe met de coördinaten
        coords = df_main[['Meetpunt', 'Latitude', 'Longitude']].drop_duplicates().rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            
        if dff_stof.empty:
            # Voorkom errors als er nog geen Stof is gekozen
            loc_df = pd.DataFrame(columns=["Meetpunt", "Waarde", "lat", "lon"])
        else:
            loc_df = dff_stof.groupby("Meetpunt")["Waarde"].mean().reset_index().merge(coords, on="Meetpunt", how="left")
        
        # Bepaal het midden van de kaart op basis van de beschikbare punten
        center_lat = loc_df['lat'].mean() if not loc_df.empty else 52.4 
        center_lon = loc_df['lon'].mean() if not loc_df.empty else 5.6

        if not loc_df.empty:
            kaart_fig = px.scatter_mapbox(
                loc_df, 
                lat="lat", 
                lon="lon", 
                color="Waarde", 
                color_continuous_scale="YlOrRd",
                size=[12] * len(loc_df) if not loc_df.empty else [], # Voorkomt fout bij lege data
                hover_name="Meetpunt",
                hover_data={
                    "lat": False, 
                    "lon": False, 
                    "Waarde": ":.2f"
                },
                center={"lat": center_lat, "lon": center_lon},
                mapbox_style="open-street-map",
                zoom=7.5, 
                title="Gemiddelde waarde per locatie (van geselecteerde stoffen)"
            )
        
        # Rijk 2: Kaart en Tijdlijn
        col_kaart, col_tijdlijn = st.columns(2)
        col_kaart.plotly_chart(kaart_fig, use_container_width=True)
        col_tijdlijn.plotly_chart(tijd_fig, use_container_width=True)

    # --- Plotten die afhankelijk zijn van stofselectie ---
        if dff_stof.empty:
            st.warning("Selecteer stoffen om de resterende grafieken te zien.")
        else:
            # Rijk 3: Boxplot en Strip Plot
            col_box, col_strip = st.columns(2)
        
        # --- Boxplot Grafiek ---
        boxplot_fig = px.box(
            dff_stof, 
            x="Stof",
            y="Waarde",
            color="Meetpunt",
            points="all", 
            hover_data=["Datum", "Waarde", "Eenheid"],
            title='Verdeling per stof(fen) en locatie'
        )
        boxplot_fig.update_layout(xaxis_title='Stof', yaxis_title='Waarde')
        boxplot_fig.update_traces(quartilemethod="exclusive")
        col_box.plotly_chart(boxplot_fig, use_container_width=True)

        # --- Strip/Dumbbell Plot Grafiek ---
        dumbbell_fig = px.strip(
            dff_stof, 
            x="Waarde", 
            y="Stof", 
            color="Meetpunt",
            stripmode="group",
            hover_data=["Datum", "Meetpunt", "Waarde"],
            title='Individuele waarde(n) per stof(fen) en locatie'
        )
        dumbbell_fig.update_layout(xaxis_title='Individuele Meetwaarde', yaxis_title='Stof')
        dumbbell_fig.update_traces(orientation='h')
        col_strip.plotly_chart(dumbbell_fig, use_container_width=True)
        
        # Rijk 4: Heatmap en Subset/Stofgroep
        col_heat, col_subset = st.columns(2)
        
        # --- Heatmap Grafiek ---
        heat_df = (
            dff_stof.groupby(["Stof", "Meetpunt"])["Waarde"]
            .mean().reset_index()
            .pivot(index="Stof", columns="Meetpunt", values="Waarde")
        )
        heat_fig = px.imshow(heat_df, aspect="auto", color_continuous_scale="YlGnBu", text_auto=".3f",
                            title="Gemiddelde waarden per Stof en Meetpunt")
        col_heat.plotly_chart(heat_fig, use_container_width=True)
        
        # --- Subsetanalyse Grafiek ---
        subset_df = dff_stof.groupby(["Stof", "Meetpunt"])["Waarde"].mean().reset_index()
        
        subset_fig = px.bar(
            subset_df, 
            x="Stof", 
            y="Waarde", 
            color="Meetpunt",  # Kleur geeft nu het Waterlichaam aan
            barmode="group",  # Zet de staven naast elkaar in plaats van op elkaar
            title="Gemiddelde waarde per meetpunt per stof(fen)", 
            text_auto=".2f"
        )
        col_subset.plotly_chart(subset_fig, use_container_width=True)
        
        # Rijk 5: Fold Change Analyse
        st.markdown("---")
        st.subheader("Log2 Fold Change Analyse")
        
        fold_change_ref_selected = st.selectbox(
            "Selecteer referentielocatie voor Log2Fold Change Analyse",
            options=fold_change_opties,
            index=0 if fold_change_opties else None, # Selecteer de eerste als default
            #key=f"fc_ref_{dashboard_title}" # Unieke key voor elk tabblad
        )
        
        if not fold_change_ref_selected:
            fold_change_fig = px.scatter(title="Selecteer een referentielocatie voor fc plot")
            st.plotly_chart(fold_change_fig, use_container_width=True)
        else:
            # 1. Bereken gemiddelde waarde per Stof en Meetpunt
            fold_change_df = dff_stof.groupby(["Stof", "Meetpunt"])["Waarde"].mean().reset_index()
            
            # 2. Haal de gemiddelden van de Referentie Locatie op
            ref_df = fold_change_df[fold_change_df["Meetpunt"] == fold_change_ref_selected].rename(columns={'Waarde': 'Ref_Waarde'})
            ref_df = ref_df[["Stof", "Ref_Waarde"]]
            
            # 3. Merge met de volledige data
            stof_df = fold_change_df.merge(ref_df, on="Stof", how="left")
            
            # 4. Bereken Log2 Fold Change
            stof_df['Log2_Fold_Change'] = np.log2(stof_df['Waarde'] / stof_df['Ref_Waarde'])

            # 5. Filter de referentie locatie uit de plot
            plot_df = stof_df[stof_df["Meetpunt"] != fold_change_ref_selected]
            
            # 6. Creëer de Fold Change Plot (Scatter plot)
            fold_change_fig = px.scatter(
                plot_df, 
                x="Log2_Fold_Change", 
                y="Stof", 
                color="Meetpunt",
                hover_data={"Log2_Fold_Change": ':.2f', "Waarde": ':.2f', "Ref_Waarde": ':.2f'},
                title=f"Log2 Fold Change (gemiddeld) t.o.v. {fold_change_ref_selected}"
            )
            
            # 7. Voeg verticale lijnen toe
            fold_change_fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="red", annotation_text="Geen Verschil", annotation_position="top left")
            fold_change_fig.add_vline(x=1, line_width=0.5, line_dash="dash", line_color="gray")
            fold_change_fig.add_vline(x=-1, line_width=0.5, line_dash="dash", line_color="gray")

            fold_change_fig.update_layout(
                xaxis_title='log2(waarde/waarde referentie)',
                yaxis_title='Stof'
            )

            st.plotly_chart(fold_change_fig, use_container_width=True)
            
        # Toon de gefilterde data
        #st.write("Gefilterde data:", df_space)       

if __name__ == "__main__":
    main()