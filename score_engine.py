"""
score_engine.py
Computes the global composite score (0–100) and sub-scores.
"""
import numpy as np


# ── Weights for global score ──────────────────────────────────────────────────
WEIGHTS = {
    "finance":     0.30,
    "energie":     0.20,
    "co2":         0.15,
    "rh":          0.15,
    "satisfaction": 0.20,
}

# ── Thresholds (good → bad) per KPI ──────────────────────────────────────────
THRESHOLDS = {
    "ca_growth":         (5, 0, -5),      # pct: excellent, ok, bad
    "marge_growth":      (3, 0, -5),
    "energie_growth":    (-5, 5, 15),     # lower is better
    "co2_growth":        (-5, 5, 12),
    "absenteisme_abs":   (4, 6, 9),       # absolute % value
    "satisfaction_abs":  (80, 70, 60),
    "productivite_abs":  (85, 75, 65),
}


def _normalize(value: float, good: float, bad: float) -> float:
    """Map a raw value to 0–100 score where good=100 and bad=0."""
    if good == bad:
        return 50.0
    score = (value - bad) / (good - bad) * 100
    return float(np.clip(score, 0, 100))


def compute_score(current, previous) -> dict:
    # Growth rates
    ca_growth    = (current["chiffre_affaires"] - previous["chiffre_affaires"]) / previous["chiffre_affaires"] * 100
    m_growth     = (current["marge"] - previous["marge"]) / previous["marge"] * 100
    e_growth     = (current["energie"] - previous["energie"]) / previous["energie"] * 100
    co2_growth   = (current["co2"] - previous["co2"]) / previous["co2"] * 100

    # Sub-scores
    finance_score = (
        _normalize(ca_growth, THRESHOLDS["ca_growth"][0], THRESHOLDS["ca_growth"][2]) * 0.6 +
        _normalize(m_growth,  THRESHOLDS["marge_growth"][0], THRESHOLDS["marge_growth"][2]) * 0.4
    )
    energie_score = _normalize(e_growth, THRESHOLDS["energie_growth"][0], THRESHOLDS["energie_growth"][2])
    co2_score     = _normalize(co2_growth, THRESHOLDS["co2_growth"][0], THRESHOLDS["co2_growth"][2])
    rh_score = (
        _normalize(current["absenteisme"], THRESHOLDS["absenteisme_abs"][0], THRESHOLDS["absenteisme_abs"][2]) * 0.5 +
        _normalize(current["productivite"], THRESHOLDS["productivite_abs"][2], THRESHOLDS["productivite_abs"][0]) * 0.5
    )
    sat_score = _normalize(current["satisfaction"], THRESHOLDS["satisfaction_abs"][0], THRESHOLDS["satisfaction_abs"][2])

    sub_scores = {
        "finance":     round(finance_score),
        "energie":     round(energie_score),
        "co2":         round(co2_score),
        "rh":          round(rh_score),
        "satisfaction": round(sat_score),
    }

    global_score = int(round(
        sub_scores["finance"]     * WEIGHTS["finance"]     +
        sub_scores["energie"]     * WEIGHTS["energie"]     +
        sub_scores["co2"]         * WEIGHTS["co2"]         +
        sub_scores["rh"]          * WEIGHTS["rh"]          +
        sub_scores["satisfaction"] * WEIGHTS["satisfaction"]
    ))
    global_score = max(0, min(100, global_score))

    # Sustainability index (energy + co2 weighted)
    sustainability_score = int(round(co2_score * 0.5 + energie_score * 0.3 + sat_score * 0.2))

    # Bonus logic
    bonus_points = 0
    bonus_reason = "Aucun bonus ce mois"
    if ca_growth > 3 and m_growth > 2:
        bonus_points = 5
        bonus_reason = "CA + Marge en hausse ✅"
    elif co2_growth < 0 and e_growth < 0:
        bonus_points = 8
        bonus_reason = "Énergie & CO₂ réduits 🌱"
    elif current["satisfaction"] >= 82:
        bonus_points = 4
        bonus_reason = "Satisfaction excellente 😊"

    return {
        "global_score":       global_score,
        "sub_scores":         sub_scores,
        "sustainability_score": sustainability_score,
        "bonus_points":       bonus_points,
        "bonus_reason":       bonus_reason,
    }


def generate_report(current, previous, score_data, priorities, recommendations, month: str) -> str:
    lines = [
        "=" * 60,
        f"  SMART IMPACT DASHBOARD — Rapport {month}",
        "=" * 60,
        "",
        f"Score Global : {score_data['global_score']}/100",
        f"Sustainability Index : {score_data['sustainability_score']}/100",
        f"Bonus : +{score_data['bonus_points']} pts — {score_data['bonus_reason']}",
        "",
        "── KPIs Clés ─────────────────────────────────────────",
        f"  Chiffre d'affaires : {current['chiffre_affaires']:,.0f} €",
        f"  Marge              : {current['marge']:,.0f} €",
        f"  Énergie            : {current['energie']:,} kWh",
        f"  CO₂                : {current['co2']:.1f} T",
        f"  Absentéisme        : {current['absenteisme']:.1f}%",
        f"  Satisfaction       : {current['satisfaction']:.0f}/100",
        f"  Productivité       : {current['productivite']:.1f}%",
        "",
        "── Scores par Dimension ──────────────────────────────",
    ]
    for k, v in score_data["sub_scores"].items():
        lines.append(f"  {k.capitalize():15s} : {v}/100")
    lines += ["", "── Priorités ─────────────────────────────────────────"]
    for i, p in enumerate(priorities, 1):
        lines.append(f"  {i}. [{p['level'].upper()}] {p['title']}")
        lines.append(f"     {p['description']}")
    lines += ["", "── Recommandations ───────────────────────────────────"]
    for r in recommendations:
        lines.append(f"  • {r['title']}")
        lines.append(f"    {r['description']}")
    lines += ["", "=" * 60, "  Généré par Smart Impact Dashboard", "=" * 60]
    return "\n".join(lines)