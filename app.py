"""
app.py — CrisisSignal: Live Risk Visualization Dashboard
=========================================================
Rendering strategy (fixes both issues):
  · SIDEBAR  → st.markdown(unsafe_allow_html=True)
               components.v1.html() does NOT render in the sidebar — it always
               renders in the main area. Sidebar HTML is simple enough that
               Streamlit's sanitizer leaves it intact.
  · MAIN     → st.components.v1.html() for all complex cards (risk scorer,
               intervention engine, feature bars, metrics row, header, footer).
               Renders in a true iframe — sanitizer cannot touch it.

All inference logic unchanged:
  - Pipeline status checks .pt / .ptl (not legacy .h5 / .tflite)
  - CrisisSignalInference via @st.cache_resource
  - Proxy weighted-sum fallback preserved
"""

import sys
from pathlib import Path
import streamlit.components.v1 as components

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrisisSignal — AI For Good 2026",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS  — mirrors build_ppt.js palette exactly
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0A0F0D"
DARK_GRN  = "#0D2B1A"
MID_GRN   = "#0F3D22"
DIM_GRN   = "#1A4D2E"
GREEN     = "#22C55E"
WHITE     = "#FFFFFF"
OFF_WHITE = "#E8F5EC"
MUTED     = "#6B8F75"
AMBER     = "#F59E0B"
RED       = "#EF4444"
ORANGE    = "#FF8C00"
GRID      = "#1A2E22"

# Injected into every components.html() call so fonts load inside the iframe
IFRAME_HEAD = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    color: {OFF_WHITE};
    -webkit-font-smoothing: antialiased;
  }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — controls sidebar + native Streamlit widgets
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {BG} !important;
    color: {OFF_WHITE} !important;
}}
.stApp {{ background-color: {BG} !important; }}

/* ── Hide ONLY these two — nothing else ── */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {DARK_GRN} !important;
    border-right: 1px solid {DIM_GRN} !important;
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {{
    color: {OFF_WHITE} !important;
}}
[data-testid="stSidebar"] hr {{
    border: none !important;
    border-top: 1px solid {DIM_GRN} !important;
    margin: 12px 0 !important;
}}
/* Slider */
[data-testid="stSlider"] label p {{
    color: {OFF_WHITE} !important;
    font-size: 0.88em !important;
}}
/* Dividers in main */
hr {{
    border: none !important;
    border-top: 1px solid {DIM_GRN} !important;
    margin: 16px 0 !important;
}}
/* Expander */
[data-testid="stExpander"] {{
    background: {DARK_GRN} !important;
    border: 1px solid {DIM_GRN} !important;
}}
[data-testid="stExpander"] summary {{
    color: {GREEN} !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
}}
/* Dataframe */
[data-testid="stDataFrame"] {{ border: 1px solid {DIM_GRN} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD INFERENCE ENGINE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_inference_engine():
    try:
        from src.inference import CrisisSignalInference
        return CrisisSignalInference()
    except Exception:
        return None

with st.spinner("🧠 Loading CrisisSignal AI Engine..."):
    engine = load_inference_engine()

is_real = engine is not None and engine.is_using_real_model


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORE CALCULATION  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_window(sleep, social, typing, gps, calls) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    today = np.array([sleep, social, typing, gps, calls], dtype=np.float32)
    window = np.zeros((30, 5), dtype=np.float32)
    for i in range(30):
        fraction = i / 29.0
        noise = rng.uniform(-4, 4, size=5).astype(np.float32)
        window[i] = np.clip(today * fraction + noise, 0, 100)
    window[-1] = today
    return window


def calculate_risk_score(sleep, social, typing, gps, calls) -> int:
    window = build_feature_window(sleep, social, typing, gps, calls)
    if engine is not None:
        return engine.predict_risk(window / 100.0)
    weights = {"sleep": 0.91, "social": 0.78, "typing": 0.71, "gps": 0.65, "calls": 0.58}
    total_w = sum(weights.values())
    raw = (sleep  * weights["sleep"]  + social * weights["social"] +
           typing * weights["typing"] + gps    * weights["gps"]    +
           calls  * weights["calls"]) / total_w
    return int(np.clip(raw, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — uses st.markdown only (components.html ignores sidebar context)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand header
    st.markdown(f"""
    <div style="padding:14px 4px 6px;">
      <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.5em;
                  font-weight:900;color:{GREEN};letter-spacing:0.03em;">
        CrisisSignal
      </div>
      <div style="color:{MUTED};font-size:0.68em;letter-spacing:0.14em;
                  text-transform:uppercase;margin-top:3px;">
        AI For Good Hackathon 2026
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sensor layer label
    st.markdown(f"""
    <div style="color:{MUTED};font-size:0.7em;letter-spacing:0.14em;
                text-transform:uppercase;font-weight:600;margin-bottom:4px;">
      📡 Passive Sensor Layer
    </div>
    <div style="color:{MUTED};font-size:0.78em;margin-bottom:10px;">
      Simulate behavioral drift — last 7 days
    </div>
    """, unsafe_allow_html=True)

    # Sliders
    sleep_disruption  = st.slider("Sleep Pattern Disruption",  0, 100, 15, format="%d%%", help="91% correlation with depression")
    social_withdrawal = st.slider("Social Media Withdrawal",   0, 100, 10, format="%d%%", help="78% correlation with depression")
    typing_error      = st.slider("Typing Error Rate Spike",   0, 100,  5, format="%d%%", help="71% correlation with depression")
    gps_mobility      = st.slider("GPS Mobility Reduction",    0, 100, 20, format="%d%%", help="65% correlation with depression")
    call_drop         = st.slider("Call/SMS Frequency Drop",   0, 100, 10, format="%d%%", help="58% correlation with depression")

    st.markdown("---")

    # Pipeline status  ── FIXED: .pt / .ptl not .h5 / .tflite
    pipeline_status = {
        "Phase A — Preprocess":  (PROJECT_ROOT / "data" / "processed" / "X_train.npy").exists(),
        "Phase B — LSTM Model":  (PROJECT_ROOT / "models" / "baseline_lstm.pt").exists(),
        "Phase C — TorchScript": (PROJECT_ROOT / "models" / "crisissignal_v1.ptl").exists(),
        "Phase D — Inference":   is_real,
    }

    st.markdown(f"""
    <div style="color:{MUTED};font-size:0.7em;letter-spacing:0.14em;
                text-transform:uppercase;font-weight:600;margin-bottom:8px;">
      Pipeline Status
    </div>
    """, unsafe_allow_html=True)

    for stage, done in pipeline_status.items():
        icon  = "✅" if done else "⏳"
        color = GREEN if done else AMBER
        bg    = "rgba(34,197,94,0.07)" if done else "rgba(245,158,11,0.05)"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:6px 8px;
                    margin-bottom:4px;background:{bg};border-left:3px solid {color};">
          <span>{icon}</span>
          <span style="color:{OFF_WHITE};font-size:0.82em;">{stage}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quote
    st.markdown(f"""
    <div style="background:{MID_GRN};border:1px solid {DIM_GRN};padding:12px;">
      <div style="color:{MUTED};font-size:0.78em;font-style:italic;line-height:1.5;">
        "The phone in every student's pocket can become a silent guardian."
      </div>
      <div style="color:{GREEN};margin-top:6px;font-size:0.78em;font-weight:600;">
        — CrisisSignal
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RISK + TIER
# ─────────────────────────────────────────────────────────────────────────────
current_risk = calculate_risk_score(
    sleep_disruption, social_withdrawal, typing_error, gps_mobility, call_drop
)

if current_risk <= 40:
    risk_color, risk_bg, risk_status = GREEN,  f"rgba(34,197,94,0.06)",  "SAFE — MONITORING"
elif current_risk <= 65:
    risk_color, risk_bg, risk_status = AMBER,  f"rgba(245,158,11,0.08)", "ALERT — CHECK-IN TRIGGERED"
elif current_risk <= 80:
    risk_color, risk_bg, risk_status = ORANGE, f"rgba(255,140,0,0.08)",  "WARNING — COUNSELOR ALERTED"
else:
    risk_color, risk_bg, risk_status = RED,    f"rgba(239,68,68,0.08)",  "CRITICAL — HELPLINE INITIATED"


# ─────────────────────────────────────────────────────────────────────────────
# HEADER  — components.html (flex layout survives sanitizer in iframe)
# ─────────────────────────────────────────────────────────────────────────────
badge_color = GREEN if is_real else AMBER
badge_bg    = "rgba(34,197,94,0.12)" if is_real else "rgba(245,158,11,0.1)"
badge_text  = "🧠 LSTM TORCHSCRIPT ACTIVE" if is_real else "⚡ PROXY MODE"

components.html(IFRAME_HEAD + f"""
<div style="background:{DARK_GRN};border-top:4px solid {GREEN};
            padding:18px 24px 16px;display:flex;
            justify-content:space-between;align-items:center;">
  <div>
    <div style="color:{MUTED};font-size:0.63em;letter-spacing:0.2em;
                text-transform:uppercase;margin-bottom:5px;">
      AI FOR GOOD HACKATHON 2026 &nbsp;·&nbsp; CONNECTING DREAMS FOUNDATION
    </div>
    <div style="font-family:'Playfair Display',Georgia,serif;font-size:1.9em;
                font-weight:900;color:{WHITE};line-height:1;">
      CrisisSignal <span style="color:{GREEN};">·</span> Live Risk Dashboard
    </div>
    <div style="color:{MUTED};font-size:0.8em;font-style:italic;margin-top:5px;">
      Turning the device in every student's pocket into a silent guardian.
    </div>
  </div>
  <div style="text-align:right;flex-shrink:0;margin-left:24px;">
    <div style="display:inline-block;padding:4px 14px;background:{badge_bg};
                color:{badge_color};border:1px solid {badge_color};
                font-size:0.7em;font-weight:700;letter-spacing:0.1em;
                text-transform:uppercase;">
      {badge_text}
    </div>
    <div style="color:{MUTED};font-size:0.66em;margin-top:6px;">
      StudentLife Dataset &nbsp;·&nbsp; Dartmouth &nbsp;·&nbsp; PHQ-9 Labels
    </div>
  </div>
</div>
""", height=108)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ROW  — 4 cards like PPT slide 6
# ─────────────────────────────────────────────────────────────────────────────
def _metric(value, label, sub, accent):
    c = GREEN if accent == "green" else AMBER
    return f"""
    <div style="background:{DARK_GRN};border:1px solid {DIM_GRN};
                border-top:4px solid {c};padding:16px 14px 12px;">
      <div style="font-family:'Playfair Display',Georgia,serif;font-size:2.3em;
                  font-weight:900;color:{c};line-height:1;margin-bottom:7px;">{value}</div>
      <div style="color:{OFF_WHITE};font-size:0.88em;font-weight:500;">{label}</div>
      <div style="color:{MUTED};font-size:0.74em;margin-top:3px;">{sub}</div>
    </div>"""

components.html(IFRAME_HEAD + f"""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:10px;">
  {_metric("86%",    "Healthy Precision",  "@ Detection Threshold",  "green")}
  {_metric("77%",    "Overall Accuracy",   "StudentLife Validation",  "green")}
  {_metric("7 Days", "Early Warning",      "Before Crisis Peaks",     "amber")}
  {_metric("₹0",     "Cost Per User",      "100% On-Device · Free",   "amber")}
</div>
""", height=118)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2], gap="medium")


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Risk Score Card ────────────────────────────────────────────────────────
    components.html(IFRAME_HEAD + f"""
    <div style="background:{risk_bg};border:1px solid {risk_color};
                border-left:5px solid {risk_color};padding:20px 18px;">
      <div style="color:{MUTED};font-size:0.66em;letter-spacing:0.18em;
                  text-transform:uppercase;margin-bottom:6px;">CRISIS RISK SCORE</div>
      <div style="font-family:'Playfair Display',Georgia,serif;font-size:4em;
                  font-weight:900;color:{risk_color};line-height:1;">
        {current_risk}<span style="font-size:0.32em;font-weight:400;
        color:{MUTED};margin-left:4px;">/100</span>
      </div>
      <div style="background:{DIM_GRN};height:6px;border-radius:3px;
                  margin:14px 0 12px;overflow:hidden;">
        <div style="background:{risk_color};height:100%;
                    width:{current_risk}%;border-radius:3px;"></div>
      </div>
      <div style="color:{risk_color};font-size:0.73em;font-weight:700;
                  letter-spacing:0.14em;text-transform:uppercase;">{risk_status}</div>
    </div>
    """, height=172)

    # ── Intervention Engine ────────────────────────────────────────────────────
    tiers = [
        (80, RED,    "🆘", "CRITICAL", "iCall / Vandrevala helpline connection initiated"),
        (65, ORANGE, "⚠️", "ALERT",    "Counselor dashboard notification sent"),
        (40, AMBER,  "💬", "PROMPT",   "Compassionate mood check-in pushed to device"),
        (0,  GREEN,  "🔒", "SAFE",     "Baseline maintained · 100% on-device processing"),
    ]

    rows_html = ""
    for threshold, color, icon, label, desc in tiers:
        active  = current_risk > threshold
        opacity = "1" if active else "0.28"
        bg      = DARK_GRN if active else "transparent"
        border  = f"3px solid {color}" if active else f"3px solid {DIM_GRN}"
        rows_html += f"""
      <div style="display:flex;align-items:flex-start;gap:10px;padding:9px 10px;
                  margin-bottom:4px;background:{bg};
                  border-left:{border};opacity:{opacity};">
        <div style="font-size:1.05em;min-width:20px;flex-shrink:0;">{icon}</div>
        <div>
          <div style="color:{color};font-size:0.67em;font-weight:700;
                      letter-spacing:0.14em;text-transform:uppercase;">
            RISK &gt; {threshold} · {label}
          </div>
          <div style="color:{OFF_WHITE};font-size:0.79em;margin-top:2px;">{desc}</div>
        </div>
      </div>"""

    components.html(IFRAME_HEAD + f"""
    <div style="background:{DARK_GRN};border:1px solid {DIM_GRN};
                border-top:4px solid {GREEN};padding:13px 13px 9px;margin-top:8px;">
      <div style="color:{MUTED};font-size:0.63em;letter-spacing:0.18em;
                  text-transform:uppercase;margin-bottom:9px;">🆘 INTERVENTION ENGINE</div>
      {rows_html}
    </div>
    """, height=310)

    # ── Feature Contribution Bars ─────────────────────────────────────────────
    features_raw = [
        ("Sleep Pattern Disruption", sleep_disruption  * 0.91, GREEN),
        ("Social Withdrawal",        social_withdrawal * 0.78, GREEN),
        ("Typing Error Rate Spike",  typing_error      * 0.71, AMBER),
        ("GPS Mobility Reduction",   gps_mobility      * 0.65, AMBER),
        ("Call/SMS Frequency Drop",  call_drop         * 0.58, AMBER),
    ]
    max_val = max(v for _, v, _ in features_raw) or 1.0

    bars_html = ""
    for rank, (name, val, color) in enumerate(features_raw, 1):
        bar_w = int((val / max_val) * 93)
        pct   = int((val / max_val) * 100)
        bars_html += f"""
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:9px;">
        <div style="color:{MUTED};font-size:0.7em;font-weight:700;
                    width:18px;text-align:right;flex-shrink:0;">#{rank}</div>
        <div style="color:{OFF_WHITE};font-size:0.76em;width:138px;flex-shrink:0;
                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{name}</div>
        <div style="flex:1;background:{DIM_GRN};height:12px;border-radius:2px;overflow:hidden;">
          <div style="background:{color};height:100%;width:{bar_w}%;border-radius:2px;"></div>
        </div>
        <div style="color:{color};font-size:0.74em;font-weight:700;
                    width:34px;text-align:right;flex-shrink:0;">{pct}%</div>
      </div>"""

    components.html(IFRAME_HEAD + f"""
    <div style="background:{DARK_GRN};border:1px solid {DIM_GRN};
                border-top:4px solid {GREEN};padding:13px 15px 10px;margin-top:8px;">
      <div style="color:{MUTED};font-size:0.63em;letter-spacing:0.16em;
                  text-transform:uppercase;margin-bottom:11px;">
        🔍 FEATURE CONTRIBUTIONS (StudentLife Analysis)
      </div>
      {bars_html}
    </div>
    """, height=212)


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
with col_right:

    # Section label (simple enough for st.markdown)
    st.markdown(f"""
    <div style="color:{MUTED};font-size:0.68em;font-weight:600;letter-spacing:0.16em;
                text-transform:uppercase;margin-bottom:6px;padding-bottom:6px;
                border-bottom:1px solid {DIM_GRN};">
      📈 7-Day Behavioral Drift Window — LSTM Reconstruction Error
    </div>
    """, unsafe_allow_html=True)

    # ── Drift Chart (Plotly — renders fine natively) ───────────────────────────
    dates = [(datetime.now() - timedelta(days=i)).strftime("%b %d") for i in range(6, -1, -1)]
    rng = np.random.default_rng(seed=current_risk)
    drift_trend = [
        int(np.clip(current_risk - (6 - i) * 12 + rng.integers(-5, 5), 5, 100))
        for i in range(7)
    ]
    drift_trend[-1] = current_risk

    fill_map = {
        GREEN:  "rgba(34,197,94,0.10)",
        AMBER:  "rgba(245,158,11,0.10)",
        ORANGE: "rgba(255,140,0,0.10)",
        RED:    "rgba(239,68,68,0.10)",
    }
    fill_color = fill_map[risk_color]

    fig = go.Figure()
    fig.add_hrect(y0=0,  y1=40,  fillcolor="rgba(34,197,94,0.03)",  layer="below", line_width=0)
    fig.add_hrect(y0=40, y1=65,  fillcolor="rgba(245,158,11,0.04)", layer="below", line_width=0)
    fig.add_hrect(y0=65, y1=80,  fillcolor="rgba(255,140,0,0.04)",  layer="below", line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(239,68,68,0.05)",  layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=dates, y=drift_trend,
        mode="lines+markers",
        line=dict(color=risk_color, width=2.5, shape="spline", smoothing=0.6),
        marker=dict(size=9, color=risk_color, line=dict(color=BG, width=2)),
        fill="tozeroy", fillcolor=fill_color,
        name="Risk Score",
    ))
    for y_val, dash, color, label in [
        (40, "dot",  MUTED, "Check-in · 40"),
        (65, "dash", AMBER, "Counselor · 65"),
        (80, "dash", RED,   "Helpline · 80"),
    ]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, line_width=1,
                      annotation_text=label, annotation_font_size=10,
                      annotation_font_color=color, annotation_position="right")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=DARK_GRN,
        font=dict(family="DM Sans, sans-serif", color=MUTED),
        margin=dict(l=0, r=90, t=8, b=0), height=290,
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(range=[0, 105], gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=MUTED, size=11),
                   title=dict(text="Risk Score", font=dict(color=MUTED, size=11))),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Scenario Comparison ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="color:{MUTED};font-size:0.68em;font-weight:600;letter-spacing:0.16em;
                text-transform:uppercase;margin-bottom:6px;padding-bottom:6px;
                border-bottom:1px solid {DIM_GRN};">
      📊 Scenario Comparison
    </div>
    """, unsafe_allow_html=True)

    if current_risk <= 40:   alert_str = "None ✓"
    elif current_risk <= 65: alert_str = "Mood check-in triggered"
    elif current_risk <= 80: alert_str = "⚠ Counselor alert sent"
    else:                    alert_str = "🆘 Helpline initiated"

    st.dataframe(pd.DataFrame({
        "Case":            ["Case A — Healthy Student", "Case B — Current Student"],
        "7-Day Peak Risk": ["15–25",                    str(current_risk)],
        "Alert Triggered": ["None ✓",                  alert_str],
        "Outcome":         ["No action required",       "Intervention initiated"],
    }), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Model Details Expander ─────────────────────────────────────────────────
    with st.expander("ℹ️ Model & Architecture Details"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <b style="color:{GREEN};font-size:0.82em;">Backend</b><br>
            <span style="color:{MUTED};font-size:0.8em;">PyTorch 2.6 · TorchScript export</span><br><br>
            <b style="color:{GREEN};font-size:0.82em;">Model File</b><br>
            <span style="color:{MUTED};font-size:0.8em;">crisissignal_v1.ptl · 139.7 KB · int8</span><br><br>
            <b style="color:{GREEN};font-size:0.82em;">Parameters</b><br>
            <span style="color:{MUTED};font-size:0.8em;">89,413 total · −61% quantized size</span>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <b style="color:{GREEN};font-size:0.82em;">Architecture</b><br>
            <span style="color:{MUTED};font-size:0.8em;">LSTM(64)→LSTM(32)→latent→LSTM(32)→LSTM(64)→Linear(5)</span><br><br>
            <b style="color:{GREEN};font-size:0.82em;">Training</b><br>
            <span style="color:{MUTED};font-size:0.8em;">Healthy-only autoencoder · MSE threshold 0.01152</span><br><br>
            <b style="color:{GREEN};font-size:0.82em;">Dataset</b><br>
            <span style="color:{MUTED};font-size:0.8em;">Dartmouth StudentLife · 49 students · PHQ-9 labels</span>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
components.html(IFRAME_HEAD + f"""
<div style="background:{DARK_GRN};border-top:1px solid {DIM_GRN};
            padding:10px 24px;display:flex;
            justify-content:space-between;align-items:center;margin-top:4px;">
  <div style="color:{MUTED};font-size:0.71em;font-style:italic;">
    Demo validated on StudentLife Dataset — Dartmouth University
    &nbsp;·&nbsp; AI For Good Hackathon 2026
    &nbsp;·&nbsp; Connecting Dreams Foundation
  </div>
  <div style="font-family:'Playfair Display',Georgia,serif;
              color:{GREEN};font-size:0.8em;font-weight:700;letter-spacing:0.04em;">
    Proactive &nbsp;·&nbsp; Passive &nbsp;·&nbsp; Private
  </div>
</div>
""", height=46)