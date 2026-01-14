"""
stresslab/ui/styles.py
======================
Centralized UI styling + reusable UI primitives for Streamlit.

Goals:
- Single source of truth for CSS tokens (colors, spacing, shadows)
- Reusable components: header banner, section header, KPI cards, info panels
- Consistent "dark terminal" aesthetic, but more polished than ad-hoc CSS blocks
- Streamlit-safe: inject CSS once per session (idempotent)
- Minimal coupling to app logic

Usage:
    from ui.styles import apply_global_styles, page_header, section_header, kpi_row, kpi_card

    apply_global_styles()
    page_header("StressLab", "Research-grade, interactive stress analytics")
    section_header("Overview", "Key metrics for the selected cohort")
    kpi_row([...])

Notes:
- We intentionally avoid Tailwind. Pure CSS + HTML injected via st.markdown.
- Streamlit components have internal DOM; we style "softly" with stable selectors.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import streamlit as st


# =========================
# THEME TOKENS
# =========================

@dataclass(frozen=True)
class Theme:
    # Base
    bg: str = "#0B1220"         # main background
    panel: str = "#0F172A"      # card/panel bg
    panel2: str = "#0B1020"     # alt panel bg
    border: str = "#1F2A44"     # card border
    border_soft: str = "#18243D"

    # Text
    text: str = "#E5E7EB"
    text_dim: str = "#9CA3AF"
    text_muted: str = "#6B7280"

    # Accents
    accent: str = "#60A5FA"     # blue
    accent2: str = "#22C55E"    # green
    warn: str = "#FBBF24"       # amber
    danger: str = "#FB7185"     # pinkish red
    purple: str = "#A78BFA"

    # Effects
    shadow: str = "0 12px 30px rgba(0,0,0,0.35)"
    shadow_soft: str = "0 8px 22px rgba(0,0,0,0.22)"
    radius: str = "16px"
    radius_sm: str = "12px"

    # Typography
    font_mono: str = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace"
    font_sans: str = "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'"

    # Spacing
    pad: str = "14px"
    pad_sm: str = "10px"


DEFAULT_THEME = Theme()


# =========================
# GLOBAL CSS
# =========================

def _global_css(theme: Theme) -> str:
    """
    Returns a single CSS string to inject into Streamlit.
    Keep it stable and avoid overly-specific selectors.
    """
    # Streamlit uses various class names; we use data-testid + generic tags.
    return f"""
<style>
/* ---------- Page + container ---------- */
html, body, [data-testid="stAppViewContainer"] {{
  background: {theme.bg} !important;
  color: {theme.text};
  font-family: {theme.font_sans};
}}

[data-testid="stHeader"] {{
  background: transparent !important;
}}

[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, {theme.panel2}, {theme.bg}) !important;
  border-right: 1px solid {theme.border_soft};
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2.2rem;
  max-width: 1400px;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
  background: {theme.panel};
  color: {theme.text};
  border: 1px solid {theme.border};
  border-radius: 12px;
  padding: 0.55rem 0.8rem;
  box-shadow: {theme.shadow_soft};
  transition: transform 120ms ease, border-color 120ms ease;
}}
.stButton > button:hover {{
  transform: translateY(-1px);
  border-color: {theme.accent};
}}
.stButton > button:active {{
  transform: translateY(0px);
}}

/* ---------- Inputs ---------- */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stMultiSelect"] div {{
  background: {theme.panel} !important;
  color: {theme.text} !important;
  border: 1px solid {theme.border} !important;
  border-radius: 12px !important;
}}

/* ---------- Tabs ---------- */
[data-testid="stTabs"] button {{
  color: {theme.text_dim} !important;
  font-weight: 600;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
  color: {theme.text} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
  gap: 6px;
}}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] {{
  border: 1px solid {theme.border_soft};
  border-radius: {theme.radius};
  overflow: hidden;
}}

/* ---------- Expanders ---------- */
[data-testid="stExpander"] {{
  border: 1px solid {theme.border_soft};
  border-radius: {theme.radius};
  background: {theme.panel2};
}}

/* ---------- Custom components ---------- */
.sl-card {{
  background: {theme.panel};
  border: 1px solid {theme.border};
  border-radius: {theme.radius};
  padding: {theme.pad};
  box-shadow: {theme.shadow_soft};
}}

.sl-card--subtle {{
  background: {theme.panel2};
  border: 1px solid {theme.border_soft};
  box-shadow: none;
}}

.sl-row {{
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 12px;
}}

.sl-kpi {{
  background: radial-gradient(1200px 220px at 20% 0%, rgba(96,165,250,0.16), rgba(0,0,0,0)) , {theme.panel};
  border: 1px solid {theme.border};
  border-radius: {theme.radius};
  padding: 14px 14px;
  box-shadow: {theme.shadow_soft};
  min-height: 86px;
}}

.sl-kpi .label {{
  font-size: 0.78rem;
  color: {theme.text_dim};
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 6px;
}}

.sl-kpi .value {{
  font-size: 1.25rem;
  font-weight: 750;
  color: {theme.text};
  line-height: 1.1;
}}

.sl-kpi .sub {{
  font-size: 0.82rem;
  color: {theme.text_muted};
  margin-top: 6px;
}}

.sl-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  color: {theme.text_dim};
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid {theme.border_soft};
  background: rgba(255,255,255,0.03);
}}

.sl-header {{
  padding: 16px 18px;
  border-radius: {theme.radius};
  border: 1px solid {theme.border};
  background:
    linear-gradient(90deg, rgba(15,23,42,1) 0%, rgba(2,6,23,1) 100%);
  box-shadow: {theme.shadow_soft};
  margin-bottom: 12px;
}}

.sl-header .title {{
  color: {theme.text};
  font-size: 1.55rem;
  font-weight: 800;
  margin: 0;
}}

.sl-header .subtitle {{
  margin-top: 6px;
  color: {theme.text_dim};
  font-size: 0.92rem;
}}

.sl-section {{
  margin-top: 8px;
  margin-bottom: 8px;
}}

.sl-section .h {{
  color: {theme.text};
  font-weight: 800;
  margin: 0;
  font-size: 1.05rem;
}}

.sl-section .p {{
  color: {theme.text_muted};
  margin: 4px 0 0 0;
  font-size: 0.88rem;
}}

</style>
"""


def apply_global_styles(theme: Theme = DEFAULT_THEME) -> None:
    """
    Inject global CSS once. Safe across Streamlit reruns.
    """
    # Streamlit reruns; ensure we only inject once per session.
    if st.session_state.get("_sl_styles_applied", False):
        return
    st.markdown(_global_css(theme), unsafe_allow_html=True)
    st.session_state["_sl_styles_applied"] = True


# =========================
# SMALL HTML HELPERS
# =========================

def _esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


def _maybe(text: Optional[str], cls: str) -> str:
    if text is None or str(text).strip() == "":
        return ""
    return f"<div class='{cls}'>{_esc(text)}</div>"


# =========================
# COMPONENTS
# =========================

def page_header(
    title: str,
    subtitle: Optional[str] = None,
    right_top: Optional[str] = None,
    right_bottom: Optional[str] = None,
    pills: Optional[Sequence[str]] = None,
) -> None:
    """
    Render a top banner header.

    pills: list of tiny labels shown under the title (e.g., "v1.0", "Local Mode").
    """
    pills_html = ""
    if pills:
        pills_html = " ".join([f"<span class='sl-pill'>{_esc(p)}</span>" for p in pills])

    rt = _maybe(right_top, "sub")
    rb = _maybe(right_bottom, "sub")

    right_block = ""
    if right_top or right_bottom:
        right_block = f"""
        <div style="text-align:right;">
          <div class="sub" style="color:#9CA3AF; font-size:0.80rem;">{_esc(right_top or "")}</div>
          <div class="sub" style="color:#9CA3AF; font-size:0.80rem;">{_esc(right_bottom or "")}</div>
        </div>
        """

    html_block = f"""
    <div class="sl-header">
      <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:14px;">
        <div style="min-width:0;">
          <div class="title">{_esc(title)}</div>
          {_maybe(subtitle, "subtitle")}
          <div style="margin-top:10px;">{pills_html}</div>
        </div>
        {right_block}
      </div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Render a section header with optional subtitle.
    """
    html_block = f"""
    <div class="sl-section">
      <div class="h">{_esc(title)}</div>
      {_maybe(subtitle, "p")}
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def info_card(title: str, body: str, *, subtle: bool = True) -> None:
    """
    Render an informational panel (for configuration snapshots, notes, etc.).
    """
    cls = "sl-card sl-card--subtle" if subtle else "sl-card"
    html_block = f"""
    <div class="{cls}">
      <div style="font-weight:800; color:#E5E7EB; margin-bottom:6px;">{_esc(title)}</div>
      <div style="color:#9CA3AF; font-size:0.92rem; line-height:1.45;">{_esc(body)}</div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str,
    sub: Optional[str] = None,
    *,
    col_span: int = 3,
) -> str:
    """
    Build a KPI card HTML string.
    - col_span: 12-grid span (3 => 4 cards per row)
    """
    col_span = max(1, min(12, int(col_span)))
    return f"""
    <div style="grid-column: span {col_span};">
      <div class="sl-kpi">
        <div class="label">{_esc(label)}</div>
        <div class="value">{value}</div>
        {_maybe(sub, "sub")}
      </div>
    </div>
    """


def kpi_row(cards_html: Iterable[str]) -> None:
    """
    Render a row of KPI cards (expects `kpi_card()` outputs).
    """
    html_block = "<div class='sl-row'>" + "\n".join(cards_html) + "</div>"
    st.markdown(html_block, unsafe_allow_html=True)


def fmt_money(x: float, decimals: int = 0, prefix: str = "$") -> str:
    try:
        return f"{prefix}{x:,.{decimals}f}"
    except Exception:
        return f"{prefix}{_esc(x)}"


def fmt_pct(x: float, decimals: int = 2, signed: bool = False) -> str:
    try:
        s = f"{x:.{decimals}f}%"
        if signed and not s.startswith("-"):
            s = "+" + s
        return s
    except Exception:
        return _esc(x)


def fmt_float(x: float, decimals: int = 3, signed: bool = False) -> str:
    try:
        s = f"{x:.{decimals}f}"
        if signed and not s.startswith("-"):
            s = "+" + s
        return s
    except Exception:
        return _esc(x)


def value_span(text: str, color: Optional[str] = None, bold: bool = True) -> str:
    """
    Return a small HTML span for KPI values (allows accent coloring).
    """
    style = ""
    if color:
        style += f"color:{color};"
    if bold:
        style += "font-weight:800;"
    return f"<span style='{style}'>{_esc(text)}</span>"
