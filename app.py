"""
app.py  —  Smart Impact Dashboard v3
Light theme · single-viewport · no scrolling on main view
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import data_generator as dg
import anomaly_detector as ad
import score_engine as se
import forecaster as fc

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Impact Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }

/* ── App shell ── */
.stApp {
    background: #f4f6fb;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #1a1d2e;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e5e9f2 !important;
    padding: 0 20px !important;
    gap: 2px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #8b92a9 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 12px 18px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #5b4fcf !important;
    border-bottom: 2px solid #5b4fcf !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding: 16px 20px !important;
}

/* ── KPI cards ── */
.kcard {
    background: #ffffff;
    border: 1px solid #eaedf5;
    border-radius: 14px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.kcard::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #5b4fcf, #818cf8);
    opacity: 0;
    transition: opacity 0.2s;
}
.kcard:hover::after { opacity: 1; }
.kcard.bad  { border-left: 3px solid #ef4444; }
.kcard.warn { border-left: 3px solid #f97316; }
.kcard.good { border-left: 3px solid #22c55e; }

.klabel {
    font-size: 10px; font-weight: 700; color: #8b92a9;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 4px;
    display: flex; align-items: center; gap: 5px;
}
.kval {
    font-size: 22px; font-weight: 800; color: #1a1d2e;
    line-height: 1.1; margin-bottom: 4px;
}
.kdelta-pos  { color: #16a34a; font-size: 11px; font-weight: 700; }
.kdelta-neg  { color: #dc2626; font-size: 11px; font-weight: 700; }
.kdelta-warn { color: #ea580c; font-size: 11px; font-weight: 700; }
.kbadge {
    display: inline-block;
    font-size: 9px; font-weight: 700;
    padding: 1px 6px; border-radius: 20px;
    background: #fef2f2; color: #dc2626; border: 1px solid #fecaca;
    vertical-align: middle; margin-left: 3px;
}
.kbadge.warn { background:#fff7ed; color:#c2410c; border-color:#fed7aa; }
.kbadge.mod  { background:#fefce8; color:#854d0e; border-color:#fef08a; }

/* ── Score hero ── */
.score-hero {
    background: linear-gradient(135deg, #5b4fcf 0%, #818cf8 100%);
    border-radius: 18px;
    padding: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(91,79,207,0.25);
}
.score-num {
    font-size: 56px; font-weight: 800; line-height: 1;
    color: white;
}
.score-label { font-size: 11px; font-weight: 600; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Side panel cards ── */
.scard {
    background: #ffffff;
    border: 1px solid #eaedf5;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.scard-title {
    font-size: 10px; font-weight: 700; color: #8b92a9;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-bottom: 10px;
}

/* ── Priority / alert rows ── */
.prow {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 9px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid #eaedf5;
    background: #fafbfd;
}
.prow.crit { border-left: 3px solid #ef4444; background: #fff8f8; }
.prow.high { border-left: 3px solid #f97316; background: #fffaf6; }
.prow.mod  { border-left: 3px solid #eab308; background: #fefdf0; }
.prow-title { font-size: 12px; font-weight: 700; color: #1a1d2e; margin-bottom: 2px; }
.prow-desc  { font-size: 11px; color: #6b7280; margin: 0; }
.prow-meta  { font-size: 10px; color: #9ca3af; margin-top: 3px; }

/* ── Alert history table ── */
div[data-testid="stDataFrame"] { border-radius: 12px !important; }

/* ── Selectbox light ── */
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #dde2ee !important;
    color: #1a1d2e !important;
    border-radius: 10px !important;
    font-size: 13px !important;
}

/* ── Download button ── */
.stDownloadButton button {
    background: linear-gradient(135deg, #5b4fcf, #818cf8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 13px !important; padding: 10px 16px !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(91,79,207,0.25) !important;
}

/* ── Multiselect ── */
div[data-baseweb="tag"] { background: #ede9fe !important; color: #5b4fcf !important; }

/* ── Info box ── */
.ibox {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 10px; padding: 10px 14px;
    font-size: 12px; color: #1e40af; margin-bottom: 12px;
}

/* ── Section header ── */
.sec-title {
    font-size: 13px; font-weight: 700; color: #374151;
    margin-bottom: 10px; padding-bottom: 6px;
    border-bottom: 1px solid #eaedf5;
}

hr { border-color: #eaedf5 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():       return dg.generate_monthly_data()
@st.cache_data
def get_history(_df):  return ad.get_all_anomaly_rows(_df)
@st.cache_data
def get_forecasts(_df):return fc.forecast_all_kpis(_df, n_periods=3)
@st.cache_data
def get_sc_fc(_df):    return fc.forecast_global_score(_df, se.compute_score, n_periods=3)
@st.cache_data
def get_fut_m(_df):    return fc.get_forecast_months(_df, n_periods=3)

df             = load_data()
alert_history  = get_history(df)
all_forecasts  = get_forecasts(df)
score_fc       = get_sc_fc(df)
future_months  = get_fut_m(df)
months         = df["mois_label"].tolist()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#ffffff;border-bottom:1px solid #eaedf5;
            padding:12px 24px;display:flex;align-items:center;
            justify-content:space-between;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="width:36px;height:36px;border-radius:10px;
                background:linear-gradient(135deg,#5b4fcf,#818cf8);
                display:flex;align-items:center;justify-content:center;font-size:18px;">🎯</div>
    <div>
      <span style="font-size:16px;font-weight:800;color:#1a1d2e;">Smart Impact Dashboard</span>
      <span style="font-size:11px;background:#ede9fe;color:#5b4fcf;border-radius:20px;
                   padding:2px 9px;margin-left:8px;font-weight:700;">ML · v3</span>
    </div>
  </div>
  <span style="font-size:12px;color:#9ca3af;">Isolation Forest · ARIMA · Z-score</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Vue Principale",
    "📈  Prévisions",
    "🤖  Analyse ML",
    "🔔  Alertes",
])

# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────
def pct_delta(new, old):
    return (new - old) / abs(old) * 100 if old else 0

def delta_html(val, unit="", inverse=False):
    pos = val >= 0
    if inverse: pos = not pos
    cls  = "kdelta-pos" if pos else "kdelta-neg"
    arr  = "▲" if val >= 0 else "▼"
    sign = "+" if val > 0 else ""
    return f'<span class="{cls}">{arr} {sign}{val:.1f}{unit}</span>'

def ml_badge_html(anomalies, kpi):
    for a in anomalies:
        if a["kpi"] == kpi:
            lvl = a["level"]
            cls = "" if lvl == "critique" else "warn" if lvl == "élevé" else "mod"
            return f'<span class="kbadge {cls}">ML·{lvl}</span>'
    return ""

def card_class(anomalies, kpi, inverse=False):
    for a in anomalies:
        if a["kpi"] == kpi:
            return "bad" if a["level"] == "critique" else "warn"
    return ""

PLOT_CFG = {"displayModeBar": False, "staticPlot": False}
PLOT_BG  = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

def light_axis():
    return dict(
        xaxis=dict(showgrid=False, tickfont=dict(color="#9ca3af", size=9), tickangle=-30),
        yaxis=dict(gridcolor="#f0f0f4", tickfont=dict(color="#9ca3af", size=9)),
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAIN VIEW  (everything fits in one viewport)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # Month picker in a slim top bar
    hc1, hc2, hc3 = st.columns([5, 2, 1])
    with hc2:
        sel_month = st.selectbox("", months, index=len(months)-1,
                                 label_visibility="collapsed", key="m1")

    curr_idx = df[df["mois_label"] == sel_month].index[0]
    prev_idx = max(0, curr_idx - 1)
    current  = df.iloc[curr_idx]
    previous = df.iloc[prev_idx]

    score_data  = se.compute_score(current, previous)
    anomalies   = ad.detect_anomalies(current, previous, df)
    priorities  = ad.get_priorities(anomalies)
    recos       = ad.get_recommendations(anomalies)
    gscore      = score_data["global_score"]

    # ── Layout: left(3) | centre(5) | right(4) ────────────────────────────────
    col_l, col_m, col_r = st.columns([3, 5, 4], gap="medium")

    # ── LEFT: Score + sub-scores + sustainability ─────────────────────────────
    with col_l:
        sc_color = "#16a34a" if gscore >= 75 else "#ea580c" if gscore >= 55 else "#dc2626"
        sc_label = "Bonne performance" if gscore >= 75 else "En baisse" if gscore >= 55 else "Critique"
        sc_icon  = "✅" if gscore >= 75 else "⚠️" if gscore >= 55 else "🚨"

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gscore,
            domain={"x":[0,1],"y":[0,1]},
            number={"font":{"size":46,"family":"Plus Jakarta Sans","color":"white"}},
            gauge={
                "axis":{"range":[0,100],"visible":False},
                "bar":{"color":"rgba(255,255,255,0.9)","thickness":0.18},
                "bgcolor":"rgba(255,255,255,0.1)",
                "borderwidth":0,
                "steps":[
                    {"range":[0,40],  "color":"rgba(255,255,255,0.15)"},
                    {"range":[40,65], "color":"rgba(255,255,255,0.15)"},
                    {"range":[65,80], "color":"rgba(255,255,255,0.15)"},
                    {"range":[80,100],"color":"rgba(255,255,255,0.15)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"thickness":0.85,"value":gscore},
            }
        ))
        fig_g.update_layout(**PLOT_BG, height=150, margin=dict(l=10,r=10,t=10,b=0))

        st.markdown(f"""
        <div class="score-hero">
          <div class="score-label">Score Global · {sel_month}</div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_g, use_container_width=True, config=PLOT_CFG)
        st.markdown(f"""
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:-8px;">
            <span style="background:rgba(255,255,255,0.2);border-radius:8px;
                         padding:4px 10px;font-size:12px;font-weight:700;">
              {sc_icon} {sc_label}
            </span>
            <span style="font-size:12px;opacity:0.8;">{len(anomalies)} alerte(s) ML</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Sub-scores mini bars
        st.markdown('<div class="scard" style="margin-top:10px;">', unsafe_allow_html=True)
        st.markdown('<div class="scard-title">Dimensions</div>', unsafe_allow_html=True)
        dim_icons = {"finance":"💰","energie":"⚡","co2":"🌿","rh":"👥","satisfaction":"😊"}
        dim_colors= {"finance":"#5b4fcf","energie":"#f97316","co2":"#16a34a","rh":"#0ea5e9","satisfaction":"#ec4899"}
        for k, v in score_data["sub_scores"].items():
            c = dim_colors.get(k,"#5b4fcf")
            st.markdown(f"""
            <div style="margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:11px;font-weight:600;color:#374151;">{dim_icons.get(k,'')} {k.capitalize()}</span>
                <span style="font-size:11px;font-weight:800;color:{c};">{v}</span>
              </div>
              <div style="background:#f0f2f8;border-radius:4px;height:5px;">
                <div style="background:{c};width:{v}%;height:5px;border-radius:4px;
                             transition:width 0.4s;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Sustainability + Bonus
        sus = score_data["sustainability_score"]
        st.markdown(f"""
        <div class="scard">
          <div class="scard-title">🌱 Sustainability</div>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <span style="font-size:26px;font-weight:800;color:#16a34a;">{sus}<span style="font-size:13px;color:#9ca3af;">/100</span></span>
            <div style="text-align:right;">
              <div style="font-size:11px;color:#9ca3af;">Bonus</div>
              <div style="font-size:15px;font-weight:800;color:#5b4fcf;">+{score_data['bonus_points']} pts</div>
            </div>
          </div>
          <div style="font-size:11px;color:#9ca3af;margin-top:4px;">{score_data['bonus_reason']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── CENTRE: KPI grid (2×3) + sparkline chart ──────────────────────────────
    with col_m:
        # KPI grid: 3 columns × 2 rows
        KPI_DEF = [
            ("chiffre_affaires","💰","CA",          "€",    False),
            ("marge",           "📊","Marge",        "€",    False),
            ("energie",         "⚡","Énergie",      " kWh", True),
            ("co2",             "🌿","CO₂",          " T",   True),
            ("absenteisme",     "👥","Absentéisme",  "%",    True),
            ("satisfaction",    "😊","Satisfaction", "/100", False),
        ]
        g1, g2, g3 = st.columns(3, gap="small")
        gcols = [g1, g2, g3]

        for i, (kpi, icon, lbl, unit, inv) in enumerate(KPI_DEF):
            cv, pv = float(current[kpi]), float(previous[kpi])
            if unit == "€":
                d = pct_delta(cv, pv)
                val_str = f"{cv:,.0f}{unit}"
                d_html  = delta_html(d, "%", inverse=inv)
            elif unit == "%":
                d = cv - pv
                val_str = f"{cv:.1f}{unit}"
                d_html  = delta_html(d, " pp", inverse=inv)
            elif unit == "/100":
                d = cv - pv
                val_str = f"{cv:.0f}{unit}"
                d_html  = delta_html(d, " pts", inverse=inv)
            else:
                d = pct_delta(cv, pv)
                val_str = f"{cv:,.0f}{unit}"
                d_html  = delta_html(d, "%", inverse=inv)

            cls = card_class(anomalies, kpi)
            badge = ml_badge_html(anomalies, kpi)

            with gcols[i % 3]:
                st.markdown(f"""
                <div class="kcard {cls}">
                  <div class="klabel">{icon} {lbl}{badge}</div>
                  <div class="kval">{val_str}</div>
                  {d_html}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Mini trend chart — all KPIs normalised 0–100 on same axis
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="scard-title">Tendances historiques (normalisées)</div>', unsafe_allow_html=True)

        TREND_KPIS = {
            "chiffre_affaires": ("#5b4fcf","CA"),
            "energie":          ("#f97316","Énergie"),
            "satisfaction":     ("#ec4899","Satisf."),
            "co2":              ("#16a34a","CO₂"),
        }
        fig_t = go.Figure()
        for kpi, (color, label) in TREND_KPIS.items():
            vals = df[kpi].values.astype(float)
            norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9) * 100
            fig_t.add_trace(go.Scatter(
                x=df["mois_label"], y=norm,
                mode="lines", name=label,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{label}</b>: %{{customdata:.1f}}<extra></extra>",
                customdata=vals,
            ))
        # Mark selected month
        sel_x = [sel_month, sel_month]
        fig_t.add_shape(type="line", x0=sel_month, x1=sel_month, y0=0, y1=100,
                        line=dict(color="#5b4fcf", width=1.5, dash="dot"))

        fig_t.update_layout(
            **PLOT_BG,
            height=170, margin=dict(l=0,r=0,t=4,b=0),
            **light_axis(),
            legend=dict(orientation="h", y=1.15, font=dict(size=10,color="#6b7280"),
                        bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True, config=PLOT_CFG)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: Priorities + Recommendations + Export ───────────────────────────
    with col_r:
        # Score forecast mini
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="scard-title">📈 Prévision Score (3 mois)</div>', unsafe_allow_html=True)

        hist_scores = []
        for i in range(len(df)):
            r = df.iloc[i]; p = df.iloc[max(0,i-1)]
            hist_scores.append(se.compute_score(r,p)["global_score"])

        fc_x = [df["mois_label"].iloc[-1]] + [s["month"] for s in score_fc]
        fc_y = [hist_scores[-1]] + [s["score"] for s in score_fc]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df["mois_label"].tolist()[-8:], y=hist_scores[-8:],
            mode="lines+markers", line=dict(color="#5b4fcf",width=2.5),
            marker=dict(size=5,color="#5b4fcf"), showlegend=False,
        ))
        fig_fc.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines+markers", line=dict(color="#ef4444",width=2,dash="dot"),
            marker=dict(size=7,symbol="diamond",color="#ef4444"), showlegend=False,
        ))
        # CI band
        fc_upper = [hist_scores[-1]] + [s["upper"] for s in score_fc]
        fc_lower = [hist_scores[-1]] + [s["lower"] for s in score_fc]
        fig_fc.add_trace(go.Scatter(
            x=fc_x+fc_x[::-1], y=fc_upper+fc_lower[::-1],
            fill="toself", fillcolor="rgba(239,68,68,0.07)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
        ))
        fig_fc.add_hline(y=55, line_dash="dash", line_color="rgba(239,68,68,0.35)")
        fig_fc.update_layout(
            **PLOT_BG, height=145, margin=dict(l=0,r=0,t=4,b=0),
            **light_axis(),
        )
        st.plotly_chart(fig_fc, use_container_width=True, config=PLOT_CFG)
        proj = score_fc[-1]["score"] if score_fc else gscore
        trend_txt = "📉 tendance baisse" if proj < gscore else "📈 tendance hausse"
        st.markdown(f'<p style="font-size:11px;color:#9ca3af;text-align:center;margin-top:-4px;">Prév. M+3 : <b style="color:{"#dc2626" if proj<55 else "#16a34a"};">{proj:.0f}/100</b> · {trend_txt}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Priorities
        if priorities:
            st.markdown('<div class="sec-title">🔥 Priorités ML</div>', unsafe_allow_html=True)
            for i, p in enumerate(priorities[:3]):
                lvl = p["level"]
                css = "crit" if lvl=="critique" else "high" if lvl=="élevé" else "mod"
                icon = "🚨" if lvl=="critique" else "⚠️" if lvl=="élevé" else "💡"
                st.markdown(f"""
                <div class="prow {css}">
                  <span style="font-size:14px;flex-shrink:0;">{icon}</span>
                  <div>
                    <div class="prow-title">{p['title']}</div>
                    <div class="prow-desc">{p['description']}</div>
                    <div class="prow-meta">{p.get('method','Z-score')} · z={p.get('zscore','—')}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
                        padding:16px;text-align:center;">
              <div style="font-size:22px;">✅</div>
              <div style="font-size:13px;font-weight:700;color:#15803d;margin-top:4px;">Aucune anomalie détectée</div>
              <div style="font-size:11px;color:#86efac;">Tous les KPIs sont normaux</div>
            </div>
            """, unsafe_allow_html=True)

        # Recommendations
        if recos:
            st.markdown('<div class="sec-title" style="margin-top:10px;">💡 Recommandations</div>', unsafe_allow_html=True)
            for r in recos[:2]:
                st.markdown(f"""
                <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:10px;
                            padding:9px 12px;margin-bottom:6px;">
                  <div style="font-size:12px;font-weight:700;color:#7c3aed;">🔧 {r['title']}</div>
                  <div style="font-size:11px;color:#6b7280;margin-top:3px;">{r['description']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Export
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        report = se.generate_report(current, previous, score_data, priorities, recos, sel_month)
        st.download_button("📄 Exporter le rapport", data=report,
                           file_name=f"rapport_{sel_month.replace(' ','_')}.txt",
                           mime="text/plain", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="ibox">📡 Prévisions ARIMA (statsmodels) avec intervalles de confiance · fallback régression linéaire.</div>', unsafe_allow_html=True)

    # Score forecast full
    hist_scores2 = []
    for i in range(len(df)):
        r=df.iloc[i]; p=df.iloc[max(0,i-1)]
        hist_scores2.append(se.compute_score(r,p)["global_score"])

    fc_x2 = [df["mois_label"].iloc[-1]] + [s["month"] for s in score_fc]
    fc_y2 = [hist_scores2[-1]] + [s["score"] for s in score_fc]
    fc_u2 = [hist_scores2[-1]] + [s["upper"] for s in score_fc]
    fc_l2 = [hist_scores2[-1]] + [s["lower"] for s in score_fc]

    fig_sf = go.Figure()
    fig_sf.add_trace(go.Scatter(x=df["mois_label"].tolist(), y=hist_scores2,
        mode="lines+markers", name="Historique",
        line=dict(color="#5b4fcf",width=2.5), marker=dict(size=6,color="#5b4fcf"),
        fill="tozeroy", fillcolor="rgba(91,79,207,0.05)"))
    fig_sf.add_trace(go.Scatter(x=fc_x2+fc_x2[::-1], y=fc_u2+fc_l2[::-1],
        fill="toself", fillcolor="rgba(239,68,68,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
    fig_sf.add_trace(go.Scatter(x=fc_x2, y=fc_y2,
        mode="lines+markers", name="Prévision ARIMA",
        line=dict(color="#ef4444",width=2.5,dash="dot"),
        marker=dict(size=9,symbol="diamond",color="#ef4444")))
    fig_sf.add_hline(y=55, line_dash="dash", line_color="rgba(239,68,68,0.4)",
                     annotation_text="Seuil critique", annotation_font_color="#ef4444",annotation_font_size=11)
    fig_sf.update_layout(**PLOT_BG, height=280, margin=dict(l=0,r=0,t=10,b=0),
                         **light_axis(),
                         legend=dict(font=dict(size=11,color="#6b7280"),bgcolor="rgba(0,0,0,0)"))
    st.markdown('<div class="scard">', unsafe_allow_html=True)
    st.markdown('<div class="scard-title">🎯 Prévision Score Global</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_sf, use_container_width=True, config=PLOT_CFG)
    st.markdown('</div>', unsafe_allow_html=True)

    # Per-KPI forecasts in 3 columns
    KPI_FC = [
        ("energie","⚡ Énergie (kWh)","#f97316"),
        ("co2","🌿 CO₂ (T)","#16a34a"),
        ("chiffre_affaires","💰 CA (€)","#5b4fcf"),
        ("marge","📊 Marge (€)","#0ea5e9"),
        ("satisfaction","😊 Satisfaction","#ec4899"),
        ("absenteisme","👥 Absentéisme (%)","#eab308"),
    ]
    c1,c2,c3 = st.columns(3, gap="small")
    fcols = [c1,c2,c3]
    for i,(kpi,lbl,color) in enumerate(KPI_FC):
        hist_v = df[kpi].tolist()
        fc_v   = all_forecasts[kpi]["forecast"]
        method = all_forecasts[kpi]["method"]
        x_h    = df["mois_label"].tolist()
        x_fc2  = [x_h[-1]] + future_months
        y_fc2  = [hist_v[-1]] + fc_v

        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(x=x_h, y=hist_v, mode="lines",
            line=dict(color=color,width=2), showlegend=False))
        fig_k.add_trace(go.Scatter(x=x_fc2, y=y_fc2, mode="lines+markers",
            line=dict(color="#ef4444",width=2,dash="dot"),
            marker=dict(size=7,symbol="diamond",color="#ef4444"), showlegend=False))
        fig_k.update_layout(**PLOT_BG, height=160,
            title=dict(text=f"{lbl} <span style='font-size:10px;color:#9ca3af;'>· {method}</span>",
                       font=dict(size=12,color="#374151")),
            margin=dict(l=0,r=0,t=36,b=0), **light_axis())
        with fcols[i%3]:
            st.markdown('<div class="scard">', unsafe_allow_html=True)
            st.plotly_chart(fig_k, use_container_width=True, config=PLOT_CFG)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="ibox">🤖 <b>Isolation Forest</b> détecte les mois globalement anormaux · <b>Z-score</b> identifie le KPI responsable.</div>', unsafe_allow_html=True)

    if_labels  = ad.run_isolation_forest(df)
    zscores_df = ad.compute_zscores(df)

    ml1, ml2 = st.columns([3,5], gap="medium")

    with ml1:
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="scard-title">🗓️ Mois flaggés — Isolation Forest</div>', unsafe_allow_html=True)
        for i, m in enumerate(months):
            is_a = (if_labels[i] == -1)
            bg   = "#fef2f2" if is_a else "#f0fdf4"
            bd   = "#fecaca" if is_a else "#bbf7d0"
            c    = "#dc2626" if is_a else "#16a34a"
            ic   = "🚨" if is_a else "✅"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        background:{bg};border:1px solid {bd};border-radius:8px;
                        padding:5px 10px;margin-bottom:4px;">
              <span style="font-size:12px;font-weight:600;color:{c};">{ic} {m}</span>
              <span style="font-size:10px;color:{c};background:{'#fee2e2' if is_a else '#dcfce7'};
                           border-radius:4px;padding:1px 6px;">
                {'Anomalie' if is_a else 'Normal'}
              </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ml2:
        # Z-score heatmap
        st.markdown('<div class="scard">', unsafe_allow_html=True)
        st.markdown('<div class="scard-title">🌡️ Heatmap Z-scores</div>', unsafe_allow_html=True)
        kpi_short = {"chiffre_affaires":"CA","marge":"Marge","energie":"Énergie",
                     "co2":"CO₂","absenteisme":"Absent.","satisfaction":"Satisf.","productivite":"Produc."}
        z_mat = zscores_df[ad.ML_FEATURES].T
        z_mat.index   = [kpi_short[k] for k in ad.ML_FEATURES]
        z_mat.columns = df["mois_label"].tolist()

        fig_h = go.Figure(go.Heatmap(
            z=z_mat.values, x=z_mat.columns.tolist(), y=z_mat.index.tolist(),
            colorscale=[[0,"#3730a3"],[0.35,"#818cf8"],[0.5,"#f8fafc"],
                        [0.65,"#fb923c"],[1,"#dc2626"]],
            zmid=0,
            colorbar=dict(tickfont=dict(color="#6b7280",size=10),thickness=10),
            text=z_mat.round(1).values, texttemplate="%{text}",
            textfont=dict(size=9),
            hovertemplate="<b>%{y}</b> | %{x}<br>Z: %{z:.2f}<extra></extra>",
        ))
        fig_h.update_layout(**PLOT_BG, height=240, margin=dict(l=0,r=0,t=4,b=0),
            xaxis=dict(showgrid=False, tickfont=dict(color="#9ca3af",size=9), tickangle=-35),
            yaxis=dict(tickfont=dict(color="#374151",size=10)))
        st.plotly_chart(fig_h, use_container_width=True, config=PLOT_CFG)
        st.markdown('<p style="font-size:11px;color:#9ca3af;text-align:center;margin-top:-4px;">Rouge = trop haut · Bleu = trop bas · Blanc = normal</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Z-score detail for a month
        sel_ml = st.selectbox("Détail pour :", months, index=len(months)-1, key="ml_m")
        sel_z  = zscores_df.iloc[df[df["mois_label"]==sel_ml].index[0]]

        zcols = st.columns(7)
        for i, kpi in enumerate(ad.ML_FEATURES):
            z    = float(sel_z[kpi])
            lvl  = ad._zscore_level(z)
            c    = "#dc2626" if lvl=="critique" else "#ea580c" if lvl=="élevé" else "#ca8a04" if lvl=="modéré" else "#16a34a"
            bg   = "#fef2f2" if lvl=="critique" else "#fff7ed" if lvl=="élevé" else "#fefce8" if lvl=="modéré" else "#f0fdf4"
            with zcols[i]:
                st.markdown(f"""
                <div style="background:{bg};border-radius:10px;padding:10px 6px;text-align:center;">
                  <div style="font-size:9px;font-weight:700;color:#6b7280;margin-bottom:4px;">{kpi_short[kpi]}</div>
                  <div style="font-size:18px;font-weight:800;color:{c};">{z:+.1f}</div>
                  <div style="font-size:9px;color:{c};">{lvl}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALERT HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if alert_history.empty:
        st.info("Aucune anomalie détectée sur la période.")
    else:
        # Summary counters
        n_c = (alert_history["Niveau"]=="Critique").sum()
        n_e = (alert_history["Niveau"]=="Élevé").sum()
        n_m = (alert_history["Niveau"]=="Modéré").sum()
        n_g = (alert_history["Anomalie globale"]=="✅").sum()

        s1,s2,s3,s4 = st.columns(4, gap="small")
        for col,(val,lbl,bg,c,bd) in zip([s1,s2,s3,s4],[
            (n_c,"Critiques","#fef2f2","#dc2626","#fecaca"),
            (n_e,"Élevés",  "#fff7ed","#ea580c","#fed7aa"),
            (n_m,"Modérés", "#fefce8","#ca8a04","#fef08a"),
            (n_g,"IF Global","#faf5ff","#7c3aed","#e9d5ff"),
        ]):
            with col:
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bd};border-radius:14px;
                            padding:16px;text-align:center;">
                  <div style="font-size:32px;font-weight:800;color:{c};">{val}</div>
                  <div style="font-size:11px;font-weight:600;color:{c};margin-top:2px;">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Filters + chart side by side
        fa, fb = st.columns([3, 4], gap="medium")
        with fa:
            f_lv = st.multiselect("Niveau", ["Critique","Élevé","Modéré"],
                                   default=["Critique","Élevé"], key="fv")
            f_kp = st.multiselect("KPI", sorted(alert_history["KPI"].unique()), default=[], key="fk")
            filtered = alert_history.copy()
            if f_lv: filtered = filtered[filtered["Niveau"].isin(f_lv)]
            if f_kp: filtered = filtered[filtered["KPI"].isin(f_kp)]
            st.markdown(f'<p style="font-size:12px;color:#9ca3af;">{len(filtered)} alerte(s)</p>', unsafe_allow_html=True)
            st.download_button("📥 Export CSV", data=filtered.to_csv(index=False).encode(),
                               file_name="alertes.csv", mime="text/csv", use_container_width=True)

        with fb:
            freq = filtered.groupby(["Mois","Niveau"]).size().reset_index(name="n")
            cmap = {"Critique":"#ef4444","Élevé":"#f97316","Modéré":"#eab308"}
            fig_b = go.Figure()
            for lvl in ["Critique","Élevé","Modéré"]:
                sub = freq[freq["Niveau"]==lvl]
                if sub.empty: continue
                fig_b.add_trace(go.Bar(x=sub["Mois"],y=sub["n"],name=lvl,
                    marker_color=cmap[lvl], marker_line_width=0))
            fig_b.update_layout(**PLOT_BG, barmode="stack", height=180,
                margin=dict(l=0,r=0,t=10,b=0), **light_axis(),
                legend=dict(font=dict(size=10,color="#6b7280"),bgcolor="rgba(0,0,0,0)"))
            st.markdown('<div class="scard">', unsafe_allow_html=True)
            st.plotly_chart(fig_b, use_container_width=True, config=PLOT_CFG)
            st.markdown('</div>', unsafe_allow_html=True)

        # Table
        def style_niveau(val):
            m = {"Critique":"background:#fef2f2;color:#dc2626;font-weight:700",
                 "Élevé":"background:#fff7ed;color:#ea580c;font-weight:700",
                 "Modéré":"background:#fefce8;color:#ca8a04;font-weight:700"}
            return m.get(val,"")
        styled = filtered.style.applymap(style_niveau, subset=["Niveau"])
        st.dataframe(styled, use_container_width=True, hide_index=True, height=320)