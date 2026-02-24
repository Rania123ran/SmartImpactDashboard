"""
data_generator.py
Generates 12 months of simulated KPI data.
"""
import pandas as pd
import numpy as np

MONTHS = [
    "Jan 2024", "Fév 2024", "Mar 2024", "Avr 2024",
    "Mai 2024", "Juin 2024", "Juil 2024", "Aoû 2024",
    "Sep 2024", "Oct 2024", "Nov 2024", "Déc 2024",
]

def generate_monthly_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Base values
    ca_base      = 230_000
    marge_base   = 31_000
    energie_base = 53_000
    co2_base     = 22.0
    abs_base     = 5.8
    sat_base     = 78.0
    prod_base    = 82.0

    records = []
    for i, m in enumerate(MONTHS):
        # Seasonal + noise
        season = np.sin(i / 12 * 2 * np.pi) * 0.05
        records.append({
            "mois_label":      m,
            "mois_idx":        i,
            "chiffre_affaires": round(ca_base * (1 + season + rng.normal(0.02, 0.03)), 0),
            "marge":            round(marge_base * (1 + season + rng.normal(0.01, 0.04)), 0),
            # Spike in energy in April & November
            "energie":  int(energie_base * (1.19 if i == 3 else 1.12 if i == 10 else 1 + rng.normal(0, 0.04))),
            "co2":      round(co2_base * (1.06 if i == 3 else 1.04 if i == 10 else 1 + rng.normal(0, 0.03)), 1),
            "absenteisme": round(abs_base + rng.normal(0, 0.4) + (0.5 if i in [3, 10] else 0), 1),
            "satisfaction": round(min(100, max(50, sat_base + rng.normal(0, 3) + (-5 if i == 3 else 0))), 0),
            "productivite": round(min(100, max(60, prod_base + rng.normal(0, 2))), 1),
        })

    return pd.DataFrame(records)