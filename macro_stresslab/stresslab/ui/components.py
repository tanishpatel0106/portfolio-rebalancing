# stresslab/ui/components.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict
import streamlit as st

import plotly.graph_objects as go

from stresslab.ui import styles

Formatter = Callable[[Any], str]

@dataclass(frozen=True)
class KPI:
    label: str
    value: Any
    sub: Optional[str] = None
    fmt: Optional[Formatter] = None

def section_header(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
          <h3>{title}</h3>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def app_banner(title: str, subtitle: str, meta: Dict[str, str] | None = None) -> None:
    meta_html = ""
    if meta:
        rows = "".join([f"<div>{k}: <span>{v}</span></div>" for k, v in meta.items()])
        meta_html = f"<div class='meta'>{rows}</div>"
    st.markdown(
        f"""
        <div class="app-banner">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
            <div>
              <h2>{title}</h2>
              <p>{subtitle}</p>
            </div>
            {meta_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(col, kpi: KPI) -> None:
    with col:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-label'>{kpi.label}</div>", unsafe_allow_html=True)
        v = kpi.fmt(kpi.value) if kpi.fmt else str(kpi.value)
        st.markdown(f"<div class='kpi-value'>{v}</div>", unsafe_allow_html=True)
        if kpi.sub:
            st.markdown(f"<div class='kpi-sub'>{kpi.sub}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def kpi_row(kpis: list[KPI], widths: list[float] | None = None) -> None:
    if widths is None:
        cols = st.columns(len(kpis))
    else:
        cols = st.columns(widths)
    for c, k in zip(cols, kpis):
        kpi_card(c, k)

def plotly_defaults(fig: go.Figure, height: int = 500, title: str | None = None) -> go.Figure:
    """
    Standardize plotly layout to match the StressLab aesthetic.
    (We keep template=plotly_dark like your code, but enforce consistent margins/legend/fonts.)
    """
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=40, b=20),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title=None, showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title=None, showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def render_app_header(title: str, subtitle: str, right_meta: Dict[str, str] | None = None) -> None:
    pills = None
    right_top = None
    right_bottom = None
    if right_meta:
        items = list(right_meta.items())
        if items:
            right_top = f"{items[0][0]}: {items[0][1]}"
        if len(items) > 1:
            right_bottom = f"{items[1][0]}: {items[1][1]}"
        if len(items) > 2:
            pills = [f"{k}: {v}" for k, v in items[2:]]
    styles.page_header(title=title, subtitle=subtitle, right_top=right_top, right_bottom=right_bottom, pills=pills)


def render_section_header(title: str, subtitle: str | None = None) -> None:
    styles.section_header(title=title, subtitle=subtitle)


def render_footer(left: str, right: str) -> None:
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; color:#6B7280; font-size:0.8rem; padding-top:1rem;">
            <span>{left}</span>
            <span>{right}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def toast_success(message: str) -> None:
    st.toast(message, icon="✅")


def toast_warning(message: str) -> None:
    st.toast(message, icon="⚠️")


def toast_error(message: str) -> None:
    st.toast(message, icon="❌")


def ui_divider(location: str = "main") -> None:
    if location == "sidebar":
        st.sidebar.markdown("<hr style='border-color:#1F2937; opacity:0.6;'>", unsafe_allow_html=True)
    else:
        st.markdown("<hr style='border-color:#1F2937; opacity:0.6;'>", unsafe_allow_html=True)
