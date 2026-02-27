"""
app.py — CrisisSignal: Live Risk Visualization Dashboard
=========================================================
Streamlit demo for the CrisisSignal AI For Good Hackathon 2026.

Inference Priority:
  1. Real TFLite model (crisissignal_v1.tflite) if Phase A–C are complete
  2. Proxy calculation using published feature weights (Slide 6) as fallback

Features:
  - 7-day behavioral drift chart
  - Risk scorer with intervention engine
  - SHAP-style feature contribution panel
  - Pipeline status indicator
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrisisSignal Demo",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0e1117; }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 18px 22px;
        border-radius: 12px;
        border-left: 4px solid #00ff88;
        margin-bottom: 12px;
    }
    .metric-box.warning  { border-left-color: #ffa500; }
    .metric-box.danger   { border-left-color: #ff4b4b; }
    .metric-box.critical { border-left-color: #ff0000; background: linear-gradient(135deg, #2d0000 0%, #1a0000 100%); }
    .model-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75em;
        font-weight: 600;
    }
    .badge-real   { background: #003322; color: #00ff88; border: 1px solid #00ff88; }
    .badge-proxy  { background: #332200; color: #ffa500; border: 1px solid #ffa500; }
    </style>
""", unsafe_allow_html=True)

# ── Load Inference Engine ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CrisisSignal AI Engine...")
def load_inference_engine():
    """Load inference engine (cached so it only loads once per session)."""
    try:
        from src.inference import CrisisSignalInference
        engine = CrisisSignalInference()
        return engine
    except Exception:
        return None

engine = load_inference_engine()

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("📱 CrisisSignal: Live Risk Visualization")
    st.markdown("#### Turning the device in every student's pocket into a silent guardian.")

with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if engine and engine.is_using_real_model:
        st.markdown('<span class="model-badge badge-real">🧠 LSTM TFLite Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="model-badge badge-proxy">⚡ Proxy Mode</span>', unsafe_allow_html=True)
        st.caption("Run Phases A–C to enable real LSTM inference.")

st.divider()

# ── Sidebar: Passive Sensor Simulator ────────────────────────────────────────
st.sidebar.header("📡 Passive Sensor Layer (Last 7 Days)")
st.sidebar.markdown(
    "Adjust sliders to simulate behavioral drift. "
    "These represent the top predictive features from the StudentLife dataset analysis."
)

sleep_disruption  = st.sidebar.slider("Sleep Pattern Disruption (Anomaly %)",  0, 100, 15, help="Feature weight: 91%")
social_withdrawal = st.sidebar.slider("Social Media Withdrawal (Anomaly %)",   0, 100, 10, help="Feature weight: 78%")
typing_error      = st.sidebar.slider("Typing Error Rate Spike (Anomaly %)",   0, 100,  5, help="Feature weight: 71%")
gps_mobility      = st.sidebar.slider("GPS Mobility Reduction (Anomaly %)",    0, 100, 20, help="Feature weight: 65%")
call_drop         = st.sidebar.slider("Call/SMS Frequency Drop (Anomaly %)",   0, 100, 10, help="Feature weight: 58%")

st.sidebar.divider()
st.sidebar.markdown("**Pipeline Status**")
pipeline_status = {
    "Phase A — Preprocess": (PROJECT_ROOT / "data" / "processed" / "X_train.npy").exists(),
    "Phase B — LSTM Model": (PROJECT_ROOT / "models" / "baseline_lstm.h5").exists(),
    "Phase C — TFLite":     (PROJECT_ROOT / "models" / "crisissignal_v1.tflite").exists(),
    "Phase D — Inference":  engine is not None and engine.is_using_real_model,
}
for stage, done in pipeline_status.items():
    icon = "✅" if done else "⏳"
    st.sidebar.markdown(f"{icon} {stage}")

# ── Risk Score Calculation ────────────────────────────────────────────────────
def build_feature_window(
    sleep: int, social: int, typing: int, gps: int, calls: int
) -> np.ndarray:
    """
    Build a synthetic 30-day feature window from today's slider values.
    The last row = current values. Earlier rows show a gradual drift.

    Returns:
        ndarray of shape (30, 5) — raw values in [0, 100] range
    """
    rng = np.random.default_rng(seed=42)
    today = np.array([sleep, social, typing, gps, calls], dtype=np.float32)
    window = np.zeros((30, 5), dtype=np.float32)
    for i in range(30):
        # Linearly interpolate from near-baseline (day 0) to current (day 29)
        fraction = i / 29.0
        noise = rng.uniform(-4, 4, size=5).astype(np.float32)
        window[i] = np.clip(today * fraction + noise, 0, 100)
    window[-1] = today  # Last day = exact current slider values
    return window


def calculate_risk_score(sleep: int, social: int, typing: int, gps: int, calls: int) -> int:
    """Calculate risk score using TFLite engine or weighted proxy fallback."""
    window = build_feature_window(sleep, social, typing, gps, calls)

    if engine is not None:
        # Normalize to [0, 1] for the engine (raw values are 0–100)
        window_normalized = window / 100.0
        return engine.predict_risk(window_normalized)
    else:
        # Direct proxy: weighted sum from pitch deck Slide 6
        weights = {"sleep": 0.91, "social": 0.78, "typing": 0.71, "gps": 0.65, "calls": 0.58}
        total_weight = sum(weights.values())
        raw = (
            sleep  * weights["sleep"]  +
            social * weights["social"] +
            typing * weights["typing"] +
            gps    * weights["gps"]    +
            calls  * weights["calls"]
        ) / total_weight
        return int(np.clip(raw, 0, 100))


current_risk = calculate_risk_score(
    sleep_disruption, social_withdrawal, typing_error, gps_mobility, call_drop
)

# ── Determine status tier ─────────────────────────────────────────────────────
if current_risk <= 40:
    status_text  = "System: Normal. No alert triggered. ✓"
    box_class    = "metric-box"
    delta_color  = "off"
elif current_risk <= 65:
    status_text  = "Risk > 40: Mood check-in triggered."
    box_class    = "metric-box warning"
    delta_color  = "inverse"
elif current_risk <= 80:
    status_text  = "Risk > 65: Counselor Alert sent automatically. ⚠"
    box_class    = "metric-box danger"
    delta_color  = "inverse"
else:
    status_text  = "Risk > 80: iCall / Vandrevala helpline connection initiated. 🆘"
    box_class    = "metric-box critical"
    delta_color  = "inverse"

# ── Main Layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    # ── Risk Scorer ───────────────────────────────────────────────────────────
    st.subheader("🎯 Crisis Risk Scorer")
    st.metric(
        label="Daily Risk Score (0–100)",
        value=current_risk,
        delta=f"{current_risk - 15:+d} from Baseline",
        delta_color=delta_color,
    )
    st.markdown(f"**Status:** {status_text}")

    st.divider()

    # ── Intervention Engine ───────────────────────────────────────────────────
    st.subheader("🆘 Intervention Engine")
    st.progress(current_risk / 100)

    if current_risk > 80:
        st.error("🚨 CRITICAL: Federated iCall API Pinged. Immediate Intervention Required.")
    elif current_risk > 65:
        st.warning("⚠️ ALERT: Dashboard notification sent to institutional counselor.")
    elif current_risk > 40:
        st.info("💬 GENTLE PROMPT: Automated compassionate mood check-in pushed to device.")
    else:
        st.success("🔒 Baseline maintained. 100% On-Device Processing active.")

    st.divider()

    # ── SHAP-style Feature Contribution Panel ─────────────────────────────────
    st.subheader("🔍 Feature Contributions")
    st.caption("Relative weight each behavioral signal contributes to your risk score.")

    feature_weights = {
        "Sleep Pattern Disruption":  sleep_disruption  * 0.91,
        "Social Media Withdrawal":   social_withdrawal * 0.78,
        "Typing Error Rate Spike":   typing_error      * 0.71,
        "GPS Mobility Reduction":    gps_mobility      * 0.65,
        "Call/SMS Frequency Drop":   call_drop         * 0.58,
    }
    total_contribution = sum(feature_weights.values()) or 1.0

    labels = list(feature_weights.keys())
    values = list(feature_weights.values())
    colors = [
        f"rgba(255, 75, 75, {0.4 + 0.6 * (v / max(values))})" if max(values) > 0
        else "rgba(255,75,75,0.5)"
        for v in values
    ]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig_shap.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
        xaxis_title="Weighted Contribution",
        showlegend=False,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

with col2:
    # ── 7-Day Behavioral Drift Chart ──────────────────────────────────────────
    st.subheader("📈 7-Day Behavioral Drift Window")

    dates = [
        (datetime.now() - timedelta(days=i)).strftime("%b %d")
        for i in range(6, -1, -1)
    ]

    rng = np.random.default_rng(seed=current_risk)
    drift_trend = [
        int(np.clip(current_risk - (6 - i) * 12 + rng.integers(-5, 5), 5, 100))
        for i in range(7)
    ]
    drift_trend[-1] = current_risk  # Current day always matches score

    line_color = "#00ff88" if current_risk < 40 else ("#ffa500" if current_risk < 65 else "#ff4b4b")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=drift_trend,
        mode="lines+markers",
        name="Anomaly Score",
        line=dict(color=line_color, width=3),
        marker=dict(size=8, color=line_color),
        fill="tozeroy",
        fillcolor=line_color.replace(")", ", 0.1)").replace("rgb", "rgba").replace("#00ff88", "rgba(0,255,136,0.1)").replace("#ffa500", "rgba(255,165,0,0.1)").replace("#ff4b4b", "rgba(255,75,75,0.1)"),
    ))

    fig.add_hline(y=40, line_dash="dot",  line_color="#888888", annotation_text="Check-in Threshold (40)")
    fig.add_hline(y=65, line_dash="dash", line_color="orange",  annotation_text="Counselor Alert Threshold (65)")
    fig.add_hline(y=80, line_dash="dash", line_color="red",     annotation_text="Helpline Threshold (80)")

    fig.update_layout(
        title="Deviation from Personal Baseline (LSTM Reconstruction Error)",
        xaxis_title="Days",
        yaxis_title="Aggregated Risk Score",
        yaxis_range=[0, 100],
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scenario Comparison Table ─────────────────────────────────────────────
    st.subheader("📊 Scenario Comparison")
    scenario_data = {
        "Case":           ["Case A — Healthy Student",      "Case B — At-Risk Student (Your Current)"],
        "7-Day Peak Risk": ["15–25",                        str(current_risk)],
        "Alert Triggered": ["None ✓",                       status_text],
        "Outcome":         ["No action required",           "Intervention initiated"],
    }
    st.dataframe(
        pd.DataFrame(scenario_data),
        use_container_width=True,
        hide_index=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "*(Demo validated on StudentLife Dataset — Dartmouth University | "
    "AI For Good Hackathon 2026 | Connecting Dreams Foundation)*"
)