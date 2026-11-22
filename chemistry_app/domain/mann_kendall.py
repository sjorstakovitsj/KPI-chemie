from __future__ import annotations
from math import erfc, sqrt
import numpy as np
import pandas as pd

def _mk_component(waarden):
    """Bereken S en tie-gecorrigeerde variantie voor een Mann-Kendall-reeks."""
    y = np.asarray(waarden, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2:
        return 0.0, 0.0, 0
    s = 0.0
    for i in range(n - 1):
        s += np.sign(y[i + 1:] - y[i]).sum()
    _, aantallen = np.unique(y, return_counts=True)
    ties = aantallen[aantallen > 1]
    tie_correctie = np.sum(ties * (ties - 1) * (2 * ties + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_correctie) / 18.0
    return float(s), float(var_s), int(n * (n - 1) / 2)

def _mk_resultaat(s, var_s, paren, n, methode, alpha=0.05):
    """Zet Mann-Kendall-componenten om naar tau, p-waarde en trendlabel."""
    if n < 8 or paren <= 0 or var_s <= 0:
        return {'Methode': methode, 'N': int(n), 'Tau': np.nan,
                'p-waarde': np.nan, 'Trend': 'Onvoldoende gegevens',
                'Significant': False}
    z = (s - 1) / np.sqrt(var_s) if s > 0 else (
        (s + 1) / np.sqrt(var_s) if s < 0 else 0.0
    )
    p_waarde = float(erfc(abs(z) / sqrt(2.0)))
    tau = float(s / paren)
    significant = p_waarde < alpha
    if not significant:
        trend = 'Geen significante trend'
    elif tau > 0:
        trend = 'Significant stijgend'
    elif tau < 0:
        trend = 'Significant dalend'
    else:
        trend = 'Geen significante trend'
    return {'Methode': methode, 'N': int(n), 'Tau': tau,
            'p-waarde': p_waarde, 'Trend': trend,
            'Significant': significant}

def bereken_mann_kendall(groep, seasonal):
    """Bereken MK uitsluitend met metingen boven de rapportagegrens."""
    benodigde_kolommen = ['Datum', 'Waarde', 'Limietsymbool']
    data = groep[benodigde_kolommen].copy()
    data['Datum'] = pd.to_datetime(data['Datum'], errors='coerce')
    data['Waarde'] = pd.to_numeric(data['Waarde'], errors='coerce')

    onder_rg = data['Limietsymbool'].astype(str).str.contains('<', na=False)
    aantal_onder_rg = int(onder_rg.sum())
    data = data.loc[~onder_rg].dropna(subset=['Datum', 'Waarde']).sort_values('Datum')
    aantal_boven_rg = int(len(data))

    if not seasonal:
        s, var_s, paren = _mk_component(data['Waarde'].to_numpy())
        resultaat = _mk_resultaat(
            s, var_s, paren, aantal_boven_rg, 'Mann-Kendall'
        )
        resultaat['Aantal onder RG'] = aantal_onder_rg
        resultaat['Aantal boven RG'] = aantal_boven_rg
        return resultaat

    # Gemiddelde per jaar-maand voorkomt pseudoreplicatie binnen één maand.
    data['Jaar'] = data['Datum'].dt.year
    data['Maand'] = data['Datum'].dt.month
    maandwaarden = (data.groupby(['Jaar', 'Maand'], observed=True)['Waarde']
                    .mean().reset_index().sort_values(['Maand', 'Jaar']))
    totaal_s = totaal_var = 0.0
    totaal_paren = bruikbare_maanden = 0
    for _, maandgroep in maandwaarden.groupby('Maand', observed=True):
        maandgroep = maandgroep.sort_values('Jaar')
        s, var_s, paren = _mk_component(maandgroep['Waarde'].to_numpy())
        if len(maandgroep) >= 2 and paren > 0:
            bruikbare_maanden += 1
            totaal_s += s
            totaal_var += var_s
            totaal_paren += paren
    resultaat = _mk_resultaat(
        totaal_s, totaal_var, totaal_paren, aantal_boven_rg,
        'Seasonal Mann-Kendall'
    )
    if bruikbare_maanden < 2:
        resultaat.update({'Tau': np.nan, 'p-waarde': np.nan,
                           'Trend': 'Onvoldoende seizoensgegevens',
                           'Significant': False})
    resultaat['Aantal onder RG'] = aantal_onder_rg
    resultaat['Aantal boven RG'] = aantal_boven_rg
    return resultaat
