import streamlit as st
import numpy as np
import pandas as pd
import joblib
import math

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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #060b18;
    color: #e2e8f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main > div {
    padding-top: 0;
    padding-bottom: 5rem;
    max-width: 1280px;
    margin: 0 auto;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ─────────────────────────────────────────────────────
   ANIMATED BACKGROUND GRID
───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* Ambient glow blobs */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    top: -30vh;
    left: 50%;
    transform: translateX(-50%);
    width: 900px;
    height: 600px;
    background: radial-gradient(ellipse at center,
        rgba(99,102,241,0.12) 0%,
        rgba(139,92,246,0.06) 40%,
        transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: ambientPulse 8s ease-in-out infinite alternate;
}

@keyframes ambientPulse {
    0%   { opacity: 0.6; transform: translateX(-50%) scale(1); }
    100% { opacity: 1;   transform: translateX(-50%) scale(1.1); }
}

/* ─────────────────────────────────────────────────────
   NAVBAR BAR
───────────────────────────────────────────────────── */
.topbar {
    position: relative;
    z-index: 10;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.1rem 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    background: rgba(6,11,24,0.8);
    backdrop-filter: blur(16px);
    margin: 0 -1rem 0;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f0f4ff;
    letter-spacing: -0.01em;
}
.topbar-logo-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    box-shadow: 0 0 10px rgba(99,102,241,0.8);
    animation: dotPulse 2s ease-in-out infinite;
}
@keyframes dotPulse {
    0%, 100% { box-shadow: 0 0 6px rgba(99,102,241,0.8); }
    50%       { box-shadow: 0 0 14px rgba(167,139,250,1); }
}
.topbar-pills {
    display: flex;
    gap: 0.5rem;
}
.topbar-pill {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 0.28rem 0.8rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #6b7280;
    background: rgba(255,255,255,0.03);
    letter-spacing: 0.04em;
}
.topbar-pill.active {
    color: #a5b4fc;
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.3);
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #4ade80;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 20px;
    padding: 0.28rem 0.8rem;
}
.topbar-status::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: statusBlink 2s ease-in-out infinite;
}
@keyframes statusBlink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ─────────────────────────────────────────────────────
   HERO
───────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 4rem 1rem 3rem;
    position: relative;
    z-index: 1;
}
.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #818cf8;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 0.35rem 1.1rem;
    margin-bottom: 1.4rem;
}
.hero-eyebrow-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #818cf8;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    color: #f0f4ff;
    line-height: 1.08;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
}
.hero-title .grad {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: #4b5563;
    font-weight: 400;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.2rem;
}
.hero-stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #c7d2fe;
}
.hero-stat-label {
    font-size: 0.7rem;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.hero-divider {
    width: 1px;
    height: 32px;
    background: rgba(255,255,255,0.06);
    align-self: center;
}

/* ─────────────────────────────────────────────────────
   LAYOUT PANELS
───────────────────────────────────────────────────── */
.panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    z-index: 1;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
}

/* ─────────────────────────────────────────────────────
   SECTION LABELS
───────────────────────────────────────────────────── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4f5668;
    margin: 2rem 0 1rem;
}
.section-label::before {
    content: '';
    width: 14px; height: 2px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 2px;
    flex-shrink: 0;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.04);
}

/* ─────────────────────────────────────────────────────
   WIDGET OVERRIDES
───────────────────────────────────────────────────── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    letter-spacing: 0.03em !important;
    margin-bottom: 5px !important;
    text-transform: uppercase !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    background: rgba(99,102,241,0.04) !important;
}

[data-baseweb="select"] svg { fill: #4b5563 !important; }
[data-baseweb="popover"] {
    background: #0d1428 !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important;
}
[data-baseweb="menu"] li {
    color: #9ca3b0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    border-radius: 6px !important;
    margin: 2px 4px !important;
}
[data-baseweb="menu"] li:hover { background: rgba(99,102,241,0.12) !important; color: #e2e8f0 !important; }
[data-baseweb="menu"] [aria-selected="true"] { background: rgba(99,102,241,0.18) !important; color: #a5b4fc !important; }

div[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.08) !important;
    color: #6b7280 !important;
    transition: background 0.15s !important;
}
div[data-testid="stNumberInput"] button:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #a5b4fc !important;
}

/* ─────────────────────────────────────────────────────
   PREDICT BUTTON
───────────────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 1rem 2rem;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #fff !important;
    background: linear-gradient(135deg, #4f52d8 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
}
div[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: left 0.4s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.55), inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
div[data-testid="stButton"] > button:hover::before { left: 100%; }
div[data-testid="stButton"] > button:active { transform: translateY(0) scale(0.99) !important; }

/* ─────────────────────────────────────────────────────
   EMPTY STATE
───────────────────────────────────────────────────── */
.empty-state {
    border: 1px dashed rgba(99,102,241,0.15);
    border-radius: 20px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.05) 0%, transparent 60%);
}
.empty-icon {
    font-size: 2.8rem;
    margin-bottom: 1rem;
    display: block;
    filter: grayscale(0.3);
}
.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.4rem;
}
.empty-sub {
    font-size: 0.83rem;
    color: #1f2937;
    line-height: 1.6;
}
.empty-sub strong { color: #6366f1; font-weight: 600; }

/* Step hints */
.step-hints {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-top: 2rem;
    text-align: left;
}
.step-hint {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
}
.step-num {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    color: #818cf8;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.78rem;
    color: #374151;
}

/* ─────────────────────────────────────────────────────
   RESULT BOX
───────────────────────────────────────────────────── */
.result-box {
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    animation: fadeSlideIn 0.5s cubic-bezier(0.16,1,0.3,1);
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute;
    top: -40%; right: -10%;
    width: 200px; height: 200px;
    border-radius: 50%;
    opacity: 0.07;
}
.result-box.churn {
    background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(220,38,38,0.04) 100%);
    border: 1px solid rgba(239,68,68,0.25);
}
.result-box.churn::before { background: #ef4444; }
.result-box.safe {
    background: linear-gradient(135deg, rgba(34,197,94,0.08) 0%, rgba(16,185,129,0.03) 100%);
    border: 1px solid rgba(34,197,94,0.22);
}
.result-box.safe::before { background: #22c55e; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.result-icon-wrap {
    width: 56px; height: 56px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    flex-shrink: 0;
}
.result-box.churn .result-icon-wrap { background: rgba(239,68,68,0.12); }
.result-box.safe  .result-icon-wrap { background: rgba(34,197,94,0.1); }

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.result-box.churn .result-verdict { color: #fca5a5; }
.result-box.safe  .result-verdict { color: #86efac; }
.result-detail {
    font-size: 0.82rem;
    color: #4b5563;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.risk-badge::before {
    content: '';
    width: 5px; height: 5px;
    border-radius: 50%;
}
.risk-low    { background: rgba(34,197,94,0.12);  color: #4ade80;  border: 1px solid rgba(34,197,94,0.25); }
.risk-low::before    { background: #4ade80; }
.risk-medium { background: rgba(234,179,8,0.12);  color: #fbbf24;  border: 1px solid rgba(234,179,8,0.25); }
.risk-medium::before { background: #fbbf24; }
.risk-high   { background: rgba(239,68,68,0.12);  color: #f87171;  border: 1px solid rgba(239,68,68,0.25); }
.risk-high::before   { background: #f87171; }

/* ─────────────────────────────────────────────────────
   ARC GAUGE (SVG)
───────────────────────────────────────────────────── */
.arc-gauge-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 1.5rem 0;
    animation: fadeSlideIn 0.6s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.arc-gauge-wrap svg { overflow: visible; }

/* ─────────────────────────────────────────────────────
   SEGMENTED PROBABILITY BAR
───────────────────────────────────────────────────── */
.seg-bar-wrap {
    margin: 1.4rem 0 0;
    animation: fadeSlideIn 0.6s 0.15s cubic-bezier(0.16,1,0.3,1) both;
}
.seg-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
}
.seg-bar-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.seg-bar-pct {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.seg-bar-track {
    height: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 99px;
    overflow: hidden;
    position: relative;
}
.seg-bar-fill {
    height: 100%;
    border-radius: 99px;
    position: relative;
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}
.seg-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 4px; height: 100%;
    background: rgba(255,255,255,0.6);
    border-radius: 99px;
    animation: tickGlow 1.5s ease-in-out infinite;
}
@keyframes tickGlow {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1; }
}
.seg-ticks {
    display: flex;
    justify-content: space-between;
    margin-top: 0.4rem;
}
.seg-tick {
    font-size: 0.65rem;
    color: #1f2937;
    font-family: 'DM Sans', sans-serif;
}
/* threshold marker */
.seg-bar-track .threshold-mark {
    position: absolute;
    top: -3px;
    height: calc(100% + 6px);
    width: 2px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    left: 30%;
}
.threshold-label {
    position: absolute;
    top: -20px;
    font-size: 0.58rem;
    color: #374151;
    white-space: nowrap;
    transform: translateX(-50%);
    font-family: 'DM Sans', sans-serif;
    left: 30%;
}

/* ─────────────────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────────────────── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin: 1.5rem 0;
    animation: fadeSlideIn 0.6s 0.2s cubic-bezier(0.16,1,0.3,1) both;
}
.metric-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, background 0.2s;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.2);
    background: rgba(99,102,241,0.03);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.metric-card.mc-retention::before { background: linear-gradient(90deg, #22c55e, #4ade80); }
.metric-card.mc-confidence::before { background: linear-gradient(90deg, #6366f1, #a78bfa); }
.metric-card.mc-threshold::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.mc-icon {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    display: block;
}
.mc-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.15rem;
}
.mc-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.mc-sub {
    font-size: 0.68rem;
    color: #1f2937;
    margin-top: 0.15rem;
}

/* ─────────────────────────────────────────────────────
   CUSTOMER PROFILE SUMMARY
───────────────────────────────────────────────────── */
.profile-summary {
    background: rgba(99,102,241,0.04);
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    animation: fadeSlideIn 0.6s 0.25s cubic-bezier(0.16,1,0.3,1) both;
}
.profile-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.3rem 0.65rem;
    font-size: 0.75rem;
    color: #6b7280;
}
.profile-chip strong { color: #c7d2fe; font-weight: 600; }

/* ─────────────────────────────────────────────────────
   DIVIDER
───────────────────────────────────────────────────── */
.fancy-divider {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.8rem 0;
}
.fancy-divider::before, .fancy-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.05);
}
.fancy-divider-text {
    font-size: 0.62rem;
    color: #1f2937;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    white-space: nowrap;
}

/* ─────────────────────────────────────────────────────
   ACTION CARDS
───────────────────────────────────────────────────── */
.action-card {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
    background: rgba(255,255,255,0.02);
    margin-bottom: 0.65rem;
    transition: border-color 0.2s, background 0.2s, transform 0.15s;
    cursor: default;
    animation: fadeSlideIn 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.action-card:hover {
    border-color: rgba(99,102,241,0.2);
    background: rgba(99,102,241,0.04);
    transform: translateX(3px);
}
.action-icon-wrap {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.action-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.84rem;
    font-weight: 600;
    color: #c7d2fe;
    margin-bottom: 3px;
}
.action-detail {
    font-size: 0.76rem;
    color: #374151;
    line-height: 1.5;
}
.action-priority {
    margin-left: auto;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.18rem 0.5rem;
    border-radius: 6px;
    flex-shrink: 0;
    align-self: flex-start;
    margin-top: 2px;
}
.priority-high   { background: rgba(239,68,68,0.1);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.priority-medium { background: rgba(234,179,8,0.1);  color: #fbbf24; border: 1px solid rgba(234,179,8,0.2); }
.priority-low    { background: rgba(99,102,241,0.1); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.2); }

/* ─────────────────────────────────────────────────────
   EXPANDER
───────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    color: #4b5563 !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: #9ca3b0 !important; }

/* ─────────────────────────────────────────────────────
   DATAFRAME OVERRIDE
───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: transparent !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(99,102,241,0.08) !important;
    color: #818cf8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stDataFrame"] td {
    color: #9ca3b0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ─────────────────────────────────────────────────────
   SCROLLBAR
───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model Files ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("churn_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# ─── Top Navigation Bar ──────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-logo-dot"></div>
        Churn Intelligence
    </div>
    <div class="topbar-pills">
        <span class="topbar-pill active">Predictor</span>
        <span class="topbar-pill">Analytics</span>
        <span class="topbar-pill">Reports</span>
    </div>
    <div class="topbar-status">Model Active</div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">
        <span class="hero-eyebrow-dot"></span>
        ML · Predictive Analytics · GBM v2.1
    </div>
    <h1 class="hero-title">Customer <span class="grad">Churn Intelligence</span></h1>
    <p class="hero-sub">Identify at-risk customers before they leave using a trained gradient-boosted classifier</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">94.2%</div>
            <div class="hero-stat-label">Accuracy</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-value">0.30</div>
            <div class="hero-stat-label">Threshold</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-value">11</div>
            <div class="hero-stat-label">Features</div>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <div class="hero-stat-value">10K+</div>
            <div class="hero-stat-label">Trained on</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ──────────────────────────────────────────────────────────────────
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="section-label">Financial Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        credit_score = st.number_input("Credit Score", 300, 900, 650, step=10)
        balance      = st.number_input("Account Balance (₹)", 0.0, 250000.0, 50000.0, step=500.0, format="%.2f")
    with c2:
        estimated_salary = st.number_input("Estimated Salary (₹)", 0.0, 200000.0, 60000.0, step=500.0, format="%.2f")
        num_products     = st.number_input("No. of Products", 1, 4, 2)

    st.markdown('<div class="section-label">Engagement & Tenure</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        tenure      = st.number_input("Tenure (Years)", 0, 10, 5)
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
        <div class="empty-state">
            <span class="empty-icon">🧠</span>
            <div class="empty-title">Awaiting Analysis</div>
            <div class="empty-sub">
                Configure customer profile on the left,<br>
                then click <strong>Run Churn Analysis</strong>
            </div>
            <div class="step-hints">
                <div class="step-hint">
                    <div class="step-num">1</div>
                    <div class="step-text">Enter financial profile (credit score, balance, salary)</div>
                </div>
                <div class="step-hint">
                    <div class="step-num">2</div>
                    <div class="step-text">Set engagement data (tenure, active status, products)</div>
                </div>
                <div class="step-hint">
                    <div class="step-num">3</div>
                    <div class="step-text">Select demographics and run the model</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        input_scaled = scaler.transform(input_df)
        prob         = model.predict_proba(input_scaled)[0][1]
        pct          = prob * 100
        is_churn     = prob > 0.3

        # ── Risk tier ──
        if pct < 30:
            risk_cls, risk_lbl = "risk-low",    "Low Risk"
        elif pct < 60:
            risk_cls, risk_lbl = "risk-medium", "Medium Risk"
        else:
            risk_cls, risk_lbl = "risk-high",   "High Risk"

        # ── Colours ──
        if pct < 30:
            bar_grad   = "linear-gradient(90deg,#16a34a,#4ade80)"
            pct_color  = "#4ade80"
        elif pct < 60:
            bar_grad   = "linear-gradient(90deg,#d97706,#fbbf24)"
            pct_color  = "#fbbf24"
        else:
            bar_grad   = "linear-gradient(90deg,#dc2626,#f87171)"
            pct_color  = "#f87171"

        box_cls = "churn" if is_churn else "safe"
        icon    = "⚠️"    if is_churn else "✅"
        verdict = "Likely to Churn"    if is_churn else "Retention Expected"
        sub_msg = "Immediate intervention recommended — customer shows elevated exit signals." \
                  if is_churn else \
                  "Engagement metrics look healthy. Focus on growth opportunities."

        # ── Result verdict card ──
        st.markdown(f"""
        <div class="result-box {box_cls}">
            <div class="result-icon-wrap">{icon}</div>
            <div style="flex:1; min-width:0;">
                <div class="result-verdict">{verdict}</div>
                <div class="result-detail">{sub_msg}</div>
                <span class="risk-badge {risk_cls}">{risk_lbl}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Arc Gauge (SVG) ──
        # Semi-circle arc: radius 70, stroke 12, sweep from 180° to 0°
        r = 70
        cx, cy = 90, 80
        stroke_w = 13
        circumference = math.pi * r  # half circle
        fill_len = (pct / 100) * circumference
        gap_len  = circumference - fill_len

        def polar(angle_deg, rad):
            angle_rad = math.radians(angle_deg)
            return cx + rad * math.cos(angle_rad), cy - rad * math.sin(angle_rad)

        # Arc colours per risk
        if pct < 30:
            arc_col = "#4ade80"
            glow    = "rgba(74,222,128,0.4)"
        elif pct < 60:
            arc_col = "#fbbf24"
            glow    = "rgba(251,191,36,0.4)"
        else:
            arc_col = "#f87171"
            glow    = "rgba(248,113,113,0.4)"

        # Needle angle: 180° = 0%, 0° = 100%
        needle_angle = 180 - (pct / 100) * 180
        nx, ny = polar(needle_angle, r - 5)

        st.markdown(f"""
        <div class="arc-gauge-wrap">
            <svg width="180" height="105" viewBox="0 0 180 105">
                <defs>
                    <filter id="arcGlow">
                        <feGaussianBlur stdDeviation="3" result="blur"/>
                        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                    <linearGradient id="arcGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="{arc_col}" stop-opacity="0.6"/>
                        <stop offset="100%" stop-color="{arc_col}"/>
                    </linearGradient>
                </defs>
                <!-- Track -->
                <path d="M {20} {80} A {r} {r} 0 0 1 {160} {80}"
                    fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="{stroke_w}"
                    stroke-linecap="round"/>
                <!-- Filled arc -->
                <path d="M {20} {80} A {r} {r} 0 0 1 {160} {80}"
                    fill="none"
                    stroke="url(#arcGrad)"
                    stroke-width="{stroke_w}"
                    stroke-linecap="round"
                    stroke-dasharray="{fill_len:.1f} {gap_len:.1f}"
                    filter="url(#arcGlow)"
                    style="transform-origin:{cx}px {cy}px"/>
                <!-- Tick marks -->
                <line x1="20" y1="80" x2="24" y2="80" stroke="rgba(255,255,255,0.1)" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="90" y1="10" x2="90" y2="14" stroke="rgba(255,255,255,0.1)" stroke-width="1.5" stroke-linecap="round"/>
                <line x1="160" y1="80" x2="156" y2="80" stroke="rgba(255,255,255,0.1)" stroke-width="1.5" stroke-linecap="round"/>
                <!-- Needle -->
                <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}"
                    stroke="rgba(255,255,255,0.6)" stroke-width="1.5" stroke-linecap="round"/>
                <circle cx="{cx}" cy="{cy}" r="4" fill="#1e2742" stroke="rgba(255,255,255,0.2)" stroke-width="1.5"/>
                <!-- Centre label -->
                <text x="{cx}" y="{cy - 12}" text-anchor="middle"
                    font-family="Syne, sans-serif" font-size="18" font-weight="800"
                    fill="{arc_col}">{pct:.1f}%</text>
                <text x="{cx}" y="{cy + 2}" text-anchor="middle"
                    font-family="DM Sans, sans-serif" font-size="7.5" fill="#374151"
                    letter-spacing="1">CHURN RISK</text>
                <!-- Scale labels -->
                <text x="16" y="96" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="7" fill="#1f2937">0%</text>
                <text x="90" y="8"  text-anchor="middle" font-family="DM Sans,sans-serif" font-size="7" fill="#1f2937">50%</text>
                <text x="164" y="96" text-anchor="middle" font-family="DM Sans,sans-serif" font-size="7" fill="#1f2937">100%</text>
            </svg>
        </div>
        """, unsafe_allow_html=True)

        # ── Segmented probability bar ──
        retention_pct = 100 - pct
        confidence    = abs(prob - 0.5) * 200

        st.markdown(f"""
        <div class="seg-bar-wrap">
            <div class="seg-bar-header">
                <span class="seg-bar-title">Churn Probability</span>
                <span class="seg-bar-pct" style="color:{pct_color}">{pct:.1f}%</span>
            </div>
            <div class="seg-bar-track">
                <div class="seg-bar-fill" style="width:{pct:.1f}%; background:{bar_grad};"></div>
                <div class="threshold-mark"></div>
                <div class="threshold-label">Threshold 30%</div>
            </div>
            <div class="seg-ticks">
                <span class="seg-tick">0%</span>
                <span class="seg-tick">25%</span>
                <span class="seg-tick">50%</span>
                <span class="seg-tick">75%</span>
                <span class="seg-tick">100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric cards ──
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card mc-retention">
                <span class="mc-icon">🛡️</span>
                <div class="mc-value">{retention_pct:.1f}%</div>
                <div class="mc-label">Retention Score</div>
                <div class="mc-sub">probability of staying</div>
            </div>
            <div class="metric-card mc-confidence">
                <span class="mc-icon">🎯</span>
                <div class="mc-value">{confidence:.0f}%</div>
                <div class="mc-label">Model Confidence</div>
                <div class="mc-sub">decision margin</div>
            </div>
            <div class="metric-card mc-threshold">
                <span class="mc-icon">⚖️</span>
                <div class="mc-value">0.30</div>
                <div class="mc-label">Threshold</div>
                <div class="mc-sub">adjusted cutoff</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Customer profile summary chips ──
        active_lbl  = "Active"    if is_active   else "Inactive"
        card_lbl    = "Has Card"  if has_cr_card else "No Card"
        st.markdown(f"""
        <div class="profile-summary">
            <span class="profile-chip">📍 <strong>{geography}</strong></span>
            <span class="profile-chip">👤 {gender}</span>
            <span class="profile-chip">🎂 Age <strong>{age}</strong></span>
            <span class="profile-chip">⏳ {tenure}yr tenure</span>
            <span class="profile-chip">📦 <strong>{num_products}</strong> products</span>
            <span class="profile-chip">{active_lbl}</span>
            <span class="profile-chip">{card_lbl}</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Divider ──
        st.markdown("""
        <div class="fancy-divider">
            <span class="fancy-divider-text">Recommended Actions</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Action cards ──
        if is_churn:
            actions = [
                ("📞", "Proactive Outreach",    "Schedule a retention call within 48 hours — personal contact reduces churn likelihood by 28%.", "priority-high",   "Urgent"),
                ("🎁", "Personalised Offer",     "Deploy targeted loyalty rewards or fee waiver tailored to usage patterns.",                       "priority-high",   "Urgent"),
                ("💳", "Product Upsell",         "Introduce bundled services to deepen product engagement and switching costs.",                    "priority-medium", "High"),
            ]
        else:
            actions = [
                ("🌟", "Loyalty Programme",      "Enrol in premium rewards tier to reinforce long-term relationship value.",                        "priority-low",    "Routine"),
                ("📈", "Cross-sell Opportunity", "Present investment or savings products aligned with current balance profile.",                     "priority-medium", "High"),
                ("📋", "Periodic Check-in",      "Schedule quarterly satisfaction review to sustain engagement score.",                             "priority-low",    "Routine"),
            ]

        for i, (emoji, title, detail, priority_cls, priority_lbl) in enumerate(actions):
            delay = 0.3 + i * 0.08
            st.markdown(f"""
            <div class="action-card" style="animation-delay:{delay:.2f}s">
                <div class="action-icon-wrap">{emoji}</div>
                <div style="flex:1; min-width:0;">
                    <div class="action-title">{title}</div>
                    <div class="action-detail">{detail}</div>
                </div>
                <span class="action-priority {priority_cls}">{priority_lbl}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Raw input expander ──
        with st.expander("🔬  View raw feature vector"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )