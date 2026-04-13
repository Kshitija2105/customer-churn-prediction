import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e8eaf0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #818cf8;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    color: #f0f2ff;
    line-height: 1.1;
    margin: 0 0 0.8rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4f5668;
    margin: 2.2rem 0 1rem;
    padding-left: 2px;
    border-left: 2px solid #6366f1;
    padding-left: 10px;
}

/* ── Card wrapper ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(8px);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #9ca3b0 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 4px !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Selectbox arrow + dropdown */
[data-baseweb="select"] svg { fill: #6b7280 !important; }
[data-baseweb="popover"] { background: #141929 !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; }
[data-baseweb="menu"] li { color: #d1d5db !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important; }
[data-baseweb="menu"] li:hover { background: rgba(99,102,241,0.15) !important; }

/* Number input buttons */
div[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #9ca3b0 !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 2rem;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    color: #fff !important;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
    margin-top: 1rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.5) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Result box ── */
.result-box {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    animation: fadeSlideIn 0.4s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-box.churn {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
}
.result-box.safe {
    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.28);
}
.result-icon { font-size: 2.8rem; line-height: 1; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.25rem;
}
.result-box.churn .result-label { color: #f87171; }
.result-box.safe  .result-label { color: #4ade80; }
.result-meta {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: #6b7280;
    margin: 0;
}

/* ── Gauge / probability bar ── */
.gauge-wrap {
    margin-top: 1.8rem;
}
.gauge-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
}
.gauge-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    color: #6b7280;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.gauge-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.gauge-track {
    height: 10px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
    margin-bottom: 0.4rem;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.23,1,.32,1);
}
.gauge-ticks {
    display: flex;
    justify-content: space-between;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    color: #374151;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.risk-low    { background: rgba(34,197,94,0.15);  color: #4ade80; }
.risk-medium { background: rgba(234,179,8,0.15);  color: #facc15; }
.risk-high   { background: rgba(239,68,68,0.15);  color: #f87171; }

/* ── Metric cards ── */
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.6rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.metric-card .mc-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
}
.metric-card .mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #c7d2fe;
}
.metric-card .mc-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: #4b5563;
    margin-top: 2px;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2.5rem 0;
}

/* ── Expander ── */
details summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: #6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model Files ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load("churn_model.pkl")
    scaler  = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# ─── Hero Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-tag">ML · Predictive Analytics</div>
  <h1 class="hero-title">Customer <span>Churn Intelligence</span></h1>
  <p class="hero-sub">Predict churn probability using a trained gradient-boosted model</p>
</div>
""", unsafe_allow_html=True)

# ─── Layout: Two columns ─────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        credit_score     = st.number_input("Credit Score", 300, 900, 650, step=10)
        balance          = st.number_input("Account Balance (₹)", 0.0, 250000.0, 50000.0, step=500.0, format="%.2f")
    with c2:
        estimated_salary = st.number_input("Estimated Salary (₹)", 0.0, 200000.0, 60000.0, step=500.0, format="%.2f")
        num_products     = st.number_input("No. of Products", 1, 4, 2)

    st.markdown('<div class="section-label">Engagement & Tenure</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        tenure    = st.number_input("Tenure (Years)", 0, 10, 5)
        has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x else "No")
    with c4:
        is_active = st.selectbox("Active Member", [1, 0], format_func=lambda x: "Yes" if x else "No")
        age       = st.number_input("Age", 18, 100, 35)

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    with c6:
        gender = st.selectbox("Gender", ["Female", "Male"])

    predict_clicked = st.button("⚡  Run Churn Analysis", use_container_width=True)

# ─── Build Input ─────────────────────────────────────────────────────────────
input_dict = {
    "CreditScore":       credit_score,
    "Age":               age,
    "Tenure":            tenure,
    "Balance":           balance,
    "NumOfProducts":     num_products,
    "HasCrCard":         has_cr_card,
    "IsActiveMember":    is_active,
    "EstimatedSalary":   estimated_salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain":   1 if geography == "Spain" else 0,
    "Gender_Male":       1 if gender == "Male" else 0,
}
input_df = pd.DataFrame([input_dict])[features]

# ─── Right Panel ─────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.02);
            border: 1px dashed rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            color: #374151;
            font-family: 'DM Sans', sans-serif;
            font-size: 0.9rem;
        ">
            <div style="font-size:2.4rem; margin-bottom:0.8rem;">🔍</div>
            Fill in the customer details on the left<br>and click <strong style="color:#6366f1">Run Churn Analysis</strong>
        </div>
        """, unsafe_allow_html=True)

    else:
        input_scaled = scaler.transform(input_df)
        prob         = model.predict_proba(input_scaled)[0][1]
        pct          = prob * 100
        is_churn     = prob > 0.3

        # Risk tier
        if pct < 30:
            risk_cls, risk_lbl = "risk-low",    "Low Risk"
        elif pct < 60:
            risk_cls, risk_lbl = "risk-medium", "Medium Risk"
        else:
            risk_cls, risk_lbl = "risk-high",   "High Risk"

        # Gauge colour
        if pct < 30:
            gauge_color = "linear-gradient(90deg, #22c55e, #4ade80)"
        elif pct < 60:
            gauge_color = "linear-gradient(90deg, #eab308, #facc15)"
        else:
            gauge_color = "linear-gradient(90deg, #ef4444, #f87171)"

        # Result card
        box_cls    = "churn" if is_churn else "safe"
        icon       = "⚠️" if is_churn else "✅"
        verdict    = "Likely to Churn" if is_churn else "Retention Expected"
        sub_msg    = "Immediate intervention recommended." if is_churn else "Customer engagement is healthy."

        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-label">{verdict}</div>
                <div class="result-meta">{sub_msg}</div>
                <span class="risk-badge {risk_cls}">{risk_lbl}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-header">
                <span class="gauge-title">Churn Probability</span>
                <span class="gauge-value" style="color:{'#f87171' if pct>=60 else ('#facc15' if pct>=30 else '#4ade80')}">{pct:.1f}%</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{pct:.1f}%; background:{gauge_color};"></div>
            </div>
            <div class="gauge-ticks">
                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics row
        retention_pct = 100 - pct
        confidence    = abs(prob - 0.5) * 200  # 0–100 scale
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="mc-label">Retention Score</div>
                <div class="mc-value">{retention_pct:.1f}%</div>
                <div class="mc-sub">probability of staying</div>
            </div>
            <div class="metric-card">
                <div class="mc-label">Model Confidence</div>
                <div class="mc-value">{confidence:.0f}%</div>
                <div class="mc-sub">decision margin</div>
            </div>
            <div class="metric-card">
                <div class="mc-label">Decision Threshold</div>
                <div class="mc-value">0.30</div>
                <div class="mc-sub">adjusted cutoff</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="section-label">Recommended Actions</div>', unsafe_allow_html=True)
        if is_churn:
            actions = [
                ("📞", "Proactive outreach",         "Schedule a retention call within 48 hours."),
                ("🎁", "Personalised offer",          "Offer loyalty rewards or a fee waiver."),
                ("💳", "Product upsell",              "Suggest relevant products to increase stickiness."),
            ]
        else:
            actions = [
                ("🌟", "Loyalty programme",           "Enrol in a premium rewards tier."),
                ("📈", "Cross-sell opportunity",      "Present investment or savings products."),
                ("📋", "Periodic check-in",           "Schedule quarterly satisfaction review."),
            ]

        for emoji, title, detail in actions:
            st.markdown(f"""
            <div style="display:flex; gap:0.9rem; align-items:flex-start; margin-bottom:0.9rem;">
                <div style="
                    background:rgba(99,102,241,0.12);
                    border-radius:10px;
                    width:38px; height:38px;
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.1rem; flex-shrink:0;
                ">{emoji}</div>
                <div>
                    <div style="font-family:'Syne',sans-serif; font-size:0.88rem; font-weight:600; color:#c7d2fe; margin-bottom:2px;">{title}</div>
                    <div style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#6b7280;">{detail}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Raw data expander
        with st.expander("🔬 View raw input vector"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )