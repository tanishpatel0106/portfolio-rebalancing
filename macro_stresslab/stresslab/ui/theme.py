# stresslab/ui/theme.py
from __future__ import annotations
import streamlit as st

from stresslab.ui import styles

DARK_CSS = """
<style>
/* App background */
.main {
    background-color: #0E1117;
}
.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* KPI cards */
.kpi-card {
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    background: #111827;
    border: 1px solid #1F2937;
    color: #F9FAFB;
    font-size: 0.95rem;
}
.kpi-label {
    font-size: 0.80rem;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.kpi-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #E5E7EB;
}
.kpi-sub {
    font-size: 0.80rem;
    color: #6B7280;
    margin-top: 0.15rem;
}

/* Section headers */
.section-header {
    margin-top: 0.6rem;
    margin-bottom: 0.6rem;
}
.section-header h3 {
    color: #F9FAFB;
    font-weight: 600;
    margin-bottom: 0.1rem;
}
.section-header p {
    color: #6B7280;
    font-size: 0.85rem;
    margin-top: 0.15rem;
}

/* Banner header */
.app-banner {
    padding: 1rem 1.25rem;
    border-radius: 0.9rem;
    background: linear-gradient(90deg, #111827, #020617);
    border: 1px solid #1F2937;
    margin-bottom: 1rem;
}
.app-banner h2 {
    color: #F9FAFB;
    margin-bottom: 0.1rem;
}
.app-banner p {
    color: #9CA3AF;
    font-size: 0.85rem;
    margin-top: 0.15rem;
}
.app-banner .meta {
    text-align: right;
    color: #9CA3AF;
    font-size: 0.80rem;
}
.app-banner .meta span {
    color: #E5E7EB;
}

/* Make Streamlit widgets feel tighter on dark */
div[data-testid="stMetric"] {
    background: transparent;
}
</style>
"""

def apply_dark_theme() -> None:
    """Injects legacy global CSS for the StressLab UI system."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def apply_theme() -> None:
    """
    Apply the primary StressLab theme.

    We keep a thin wrapper so app.py can call a stable function name.
    """
    styles.apply_global_styles()
