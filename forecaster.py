"""
forecaster.py
ARIMA-based forecasting for KPI time series.
Falls back to linear regression if statsmodels is unavailable.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False


KPI_COLS = [
    "chiffre_affaires", "marge", "energie",
    "co2", "absenteisme", "satisfaction", "productivite"
]

# ARIMA orders per KPI (p, d, q) — tuned for monthly business data
ARIMA_ORDERS = {
    "chiffre_affaires": (1, 1, 1),
    "marge":            (1, 1, 0),
    "energie":          (2, 1, 1),
    "co2":              (1, 1, 1),
    "absenteisme":      (1, 0, 1),
    "satisfaction":     (1, 1, 0),
    "productivite":     (1, 0, 1),
}


def _arima_forecast(series: np.ndarray, order: Tuple, n_periods: int) -> np.ndarray:
    """Fit ARIMA and return n_periods future values."""
    try:
        model  = ARIMA(series, order=order)
        fitted = model.fit()
        fc     = fitted.forecast(steps=n_periods)
        return np.array(fc)
    except Exception:
        return _linear_forecast(series, n_periods)


def _linear_forecast(series: np.ndarray, n_periods: int) -> np.ndarray:
    """Simple linear regression fallback."""
    x = np.arange(len(series))
    m, b = np.polyfit(x, series, 1)
    future_x = np.arange(len(series), len(series) + n_periods)
    return m * future_x + b


def forecast_all_kpis(df: pd.DataFrame, n_periods: int = 3) -> Dict[str, Dict]:
    """
    Forecast each KPI for n_periods months ahead.
    Returns dict: { kpi_col: { "forecast": [...], "method": "ARIMA"|"Linear" } }
    """
    results = {}
    for kpi in KPI_COLS:
        series = df[kpi].values.astype(float)
        order  = ARIMA_ORDERS.get(kpi, (1, 1, 1))

        if STATSMODELS_OK and len(series) >= 8:
            forecast = _arima_forecast(series, order, n_periods)
            method   = f"ARIMA{order}"
        else:
            forecast = _linear_forecast(series, n_periods)
            method   = "Régression linéaire"

        results[kpi] = {
            "forecast": forecast.tolist(),
            "method":   method,
        }
    return results


def get_forecast_months(df: pd.DataFrame, n_periods: int = 3) -> List[str]:
    """Generate future month labels."""
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    MONTH_FR = {
        1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr",
        5: "Mai", 6: "Juin", 7: "Juil", 8: "Aoû",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc"
    }

    last_label = df["mois_label"].iloc[-1]  # e.g. "Déc 2024"
    # Parse
    parts = last_label.split(" ")
    month_name, year = parts[0], int(parts[1])
    FR_TO_NUM = {v: k for k, v in MONTH_FR.items()}
    month_num = FR_TO_NUM.get(month_name, 12)
    dt = datetime(year, month_num, 1)

    labels = []
    for i in range(1, n_periods + 1):
        try:
            future = dt + relativedelta(months=i)
        except Exception:
            # fallback: just add ~30 days * i
            import datetime as dt_mod
            future = dt + dt_mod.timedelta(days=30 * i)
        labels.append(f"{MONTH_FR[future.month]} {future.year}")
    return labels


def forecast_global_score(df: pd.DataFrame, score_fn, n_periods: int = 3) -> List[Dict]:
    """
    Forecast the global score for n_periods ahead using individual KPI forecasts.
    Returns list of { month, score, lower, upper }
    """
    kpi_forecasts = forecast_all_kpis(df, n_periods)
    future_months = get_forecast_months(df, n_periods)

    last_row  = df.iloc[-1].copy()
    prev_row  = df.iloc[-2].copy()
    results   = []

    for i in range(n_periods):
        # Build synthetic "future" row
        future_row = last_row.copy()
        for kpi in ["chiffre_affaires", "marge", "energie", "co2",
                    "absenteisme", "satisfaction", "productivite"]:
            future_row[kpi] = kpi_forecasts[kpi]["forecast"][i]

        score_data = score_fn(future_row, prev_row if i == 0 else last_row)
        sc = score_data["global_score"]

        # Simple confidence interval: ±5 * sqrt(i+1)
        uncertainty = 5 * np.sqrt(i + 1)
        results.append({
            "month": future_months[i],
            "score": sc,
            "lower": max(0,   sc - uncertainty),
            "upper": min(100, sc + uncertainty),
        })
        prev_row = last_row
        last_row = future_row

    return results