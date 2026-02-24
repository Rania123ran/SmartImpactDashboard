"""
anomaly_detector.py
ML-based anomaly detection using:
  - Isolation Forest  (multivariate, unsupervised)
  - Z-score           (per-KPI, statistical)
Replaces all hard-coded thresholds.
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ── KPI columns used for ML ───────────────────────────────────────────────────
ML_FEATURES = ["chiffre_affaires", "marge", "energie", "co2", "absenteisme", "satisfaction", "productivite"]

# ── Human-readable labels ─────────────────────────────────────────────────────
KPI_LABELS = {
    "chiffre_affaires": "Chiffre d'Affaires",
    "marge":            "Marge",
    "energie":          "Énergie",
    "co2":              "CO₂",
    "absenteisme":      "Absentéisme",
    "satisfaction":     "Satisfaction",
    "productivite":     "Productivité",
}

# Direction: "up_bad" = spike is bad, "down_bad" = drop is bad
KPI_DIRECTION = {
    "chiffre_affaires": "down_bad",
    "marge":            "down_bad",
    "energie":          "up_bad",
    "co2":              "up_bad",
    "absenteisme":      "up_bad",
    "satisfaction":     "down_bad",
    "productivite":     "down_bad",
}

# ── Severity from z-score magnitude ──────────────────────────────────────────
def _zscore_level(z: float) -> str:
    az = abs(z)
    if az >= 2.5:  return "critique"
    if az >= 1.8:  return "élevé"
    if az >= 1.2:  return "modéré"
    return "normal"


def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of z-scores for each KPI column."""
    result = df[ML_FEATURES].copy()
    for col in ML_FEATURES:
        mu  = df[col].mean()
        std = df[col].std(ddof=1)
        result[col] = (df[col] - mu) / std if std > 0 else 0.0
    return result


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.1) -> np.ndarray:
    """
    Returns array of -1 (anomaly) or 1 (normal) for each row.
    Falls back to z-score if sklearn unavailable.
    """
    if not SKLEARN_OK or len(df) < 6:
        # Fallback: flag rows where any z > 2
        zs = compute_zscores(df)
        flags = (zs.abs() > 2.0).any(axis=1).map({True: -1, False: 1})
        return flags.values

    X = df[ML_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        max_samples="auto",
    )
    return iso.fit_predict(X_scaled)


# ── Main detection function ───────────────────────────────────────────────────
def detect_anomalies(current, previous, df: pd.DataFrame) -> List[Dict]:
    """
    Combines:
      1. Isolation Forest global anomaly flag
      2. Per-KPI Z-score for root cause identification
    """
    anomalies = []

    # ── Z-scores over all history ─────────────────────────────────────────────
    zscores_df = compute_zscores(df)
    current_idx = df[df["mois_label"] == current["mois_label"]].index[0]
    current_z   = zscores_df.iloc[current_idx]

    # ── Isolation Forest flag ─────────────────────────────────────────────────
    if_labels = run_isolation_forest(df)
    is_global_anomaly = (if_labels[current_idx] == -1)

    # ── Per-KPI analysis ──────────────────────────────────────────────────────
    for kpi in ML_FEATURES:
        z     = float(current_z[kpi])
        level = _zscore_level(z)
        if level == "normal":
            continue

        direction = KPI_DIRECTION.get(kpi, "down_bad")
        # Only flag if the direction is "bad"
        is_bad = (direction == "up_bad" and z > 0) or (direction == "down_bad" and z < 0)
        if not is_bad:
            continue

        # Compute % delta vs previous
        prev_val = previous[kpi]
        curr_val = current[kpi]
        if prev_val != 0:
            delta_pct = (curr_val - prev_val) / abs(prev_val) * 100
        else:
            delta_pct = 0.0

        # Boost level if also flagged by Isolation Forest
        if is_global_anomaly and level == "modéré":
            level = "élevé"
        elif is_global_anomaly and level == "élevé":
            level = "critique"

        label = KPI_LABELS.get(kpi, kpi)
        sign  = "+" if delta_pct > 0 else ""
        anomalies.append({
            "kpi":       kpi,
            "level":     level,
            "delta":     delta_pct,
            "zscore":    round(z, 2),
            "method":    "Isolation Forest + Z-score" if is_global_anomaly else "Z-score",
            "message":   f"{label} : {sign}{delta_pct:.1f}% (z={z:+.2f})",
            "global_anomaly": is_global_anomaly,
        })

    # Sort: critique → élevé → modéré
    order = {"critique": 0, "élevé": 1, "modéré": 2}
    anomalies.sort(key=lambda x: order.get(x["level"], 3))
    return anomalies


def get_all_anomaly_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run anomaly detection on every row — used for the Alert History log.
    Returns a flat DataFrame of all detected anomalies across all months.
    """
    if_labels  = run_isolation_forest(df)
    zscores_df = compute_zscores(df)
    records    = []

    for idx in range(1, len(df)):  # skip first row (no previous)
        current  = df.iloc[idx]
        previous = df.iloc[idx - 1]
        is_global = (if_labels[idx] == -1)

        for kpi in ML_FEATURES:
            z     = float(zscores_df.iloc[idx][kpi])
            level = _zscore_level(z)
            if level == "normal":
                continue

            direction = KPI_DIRECTION.get(kpi, "down_bad")
            is_bad    = (direction == "up_bad" and z > 0) or (direction == "down_bad" and z < 0)
            if not is_bad:
                continue

            prev_val  = previous[kpi]
            curr_val  = current[kpi]
            delta_pct = (curr_val - prev_val) / abs(prev_val) * 100 if prev_val != 0 else 0.0

            if is_global and level == "modéré":  level = "élevé"
            elif is_global and level == "élevé": level = "critique"

            records.append({
                "Mois":           current["mois_label"],
                "KPI":            KPI_LABELS.get(kpi, kpi),
                "Niveau":         level.capitalize(),
                "Variation":      f"{'+' if delta_pct > 0 else ''}{delta_pct:.1f}%",
                "Z-Score":        f"{z:+.2f}",
                "Méthode":        "IF + Z-score" if is_global else "Z-score",
                "Anomalie globale": "✅" if is_global else "—",
                "_level_order":   {"critique": 0, "élevé": 1, "modéré": 2}.get(level, 3),
            })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.sort_values(["_level_order", "Mois"]).drop(columns=["_level_order"])
    return result.reset_index(drop=True)


# ── Priorities & Recommendations (unchanged API) ──────────────────────────────
PRIORITY_MAP = {
    "energie":          {"title": "Réduire la consommation énergétique",
                         "desc_template": "L'énergie a augmenté de {delta:.0f}% (z-score: {z:+.2f})."},
    "co2":              {"title": "Réduire les émissions CO₂",
                         "desc_template": "Les émissions CO₂ ont augmenté de {delta:.0f}% (z={z:+.2f})."},
    "marge":            {"title": "Améliorer la rentabilité",
                         "desc_template": "La marge a chuté de {delta:.0f}% (z={z:+.2f})."},
    "satisfaction":     {"title": "Améliorer la satisfaction client",
                         "desc_template": "Satisfaction client en baisse de {delta:.0f} pts (z={z:+.2f})."},
    "absenteisme":      {"title": "Réduire l'absentéisme",
                         "desc_template": "Absentéisme en hausse de {delta:.1f}% (z={z:+.2f})."},
    "chiffre_affaires": {"title": "Relancer le chiffre d'affaires",
                         "desc_template": "CA en baisse de {delta:.0f}% (z={z:+.2f})."},
    "productivite":     {"title": "Améliorer la productivité",
                         "desc_template": "Productivité en baisse de {delta:.0f}% (z={z:+.2f})."},
}

RECO_MAP = {
    "energie":          {"title": "Audit énergétique",
                         "description": "Vérifier et optimiser les machines énergivores. Programmer les équipements hors heures de pointe."},
    "co2":              {"title": "Plan de réduction carbone",
                         "description": "Identifier les sources d'émissions et mettre en place un plan d'action mensuel."},
    "marge":            {"title": "Revue des coûts opérationnels",
                         "description": "Analyser les postes de dépenses et négocier avec les fournisseurs clés."},
    "satisfaction":     {"title": "Enquête satisfaction client",
                         "description": "Lancer une enquête rapide pour identifier les points de friction."},
    "absenteisme":      {"title": "Programme bien-être RH",
                         "description": "Organiser des entretiens individuels et renforcer les actions de prévention."},
    "chiffre_affaires": {"title": "Revue pipeline commercial",
                         "description": "Analyser le pipeline des ventes et accélérer les actions prioritaires."},
    "productivite":     {"title": "Optimisation des processus",
                         "description": "Identifier les goulots d'étranglement et revoir les workflows opérationnels."},
}


def get_priorities(anomalies: List[Dict]) -> List[Dict]:
    priorities = []
    for a in anomalies:
        if a["kpi"] in PRIORITY_MAP:
            tpl = PRIORITY_MAP[a["kpi"]]
            priorities.append({
                "title":       tpl["title"],
                "description": tpl["desc_template"].format(delta=abs(a["delta"]), z=a["zscore"]),
                "level":       a["level"],
                "kpi":         a["kpi"],
                "zscore":      a["zscore"],
                "method":      a.get("method", "Z-score"),
            })
    return priorities[:5]


def get_recommendations(anomalies: List[Dict]) -> List[Dict]:
    seen, recos = set(), []
    for a in anomalies:
        if a["kpi"] in RECO_MAP and a["kpi"] not in seen:
            recos.append(RECO_MAP[a["kpi"]])
            seen.add(a["kpi"])
    return recos[:4]