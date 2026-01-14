from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.store.portfolios import list_portfolios, load_portfolio
from src.services.compute import build_portfolio_state
from src.ui.components import kpi_cards, warning_box
from src.ui.layout import sidebar_header
from src.ui.charts import allocation_pie, bar_chart, line_chart, drawdown_chart


st.set_page_config(page_title="Dashboard", layout="wide")
sidebar_header()

st.title("Portfolio Dashboard")

portfolio_names = list_portfolios()
if not portfolio_names:
    st.info("No saved portfolios. Create one in Portfolio Builder.")
    st.stop()

selected = st.sidebar.selectbox("Portfolio", options=portfolio_names)
end_date = st.sidebar.date_input("End Date", dt.date.today())
start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=365))

holdings = load_portfolio(selected)
state, missing, meta = build_portfolio_state(holdings, start_date, end_date)

if missing:
    warning_box(f"Missing price data for: {', '.join(missing)}")

kpi_cards(
    [
        {"label": "NAV", "value": f"${state.nav:,.0f}"},
        {
            "label": "Daily Return",
            "value": f"{state.performance_df['portfolio_return'].iloc[-1]:.2%}" if not state.performance_df.empty else "-",
        },
        {
            "label": "Vol (20D)",
            "value": f"{state.performance_df['vol_20d'].iloc[-1]:.2%}" if not state.performance_df.empty else "-",
        },
        {
            "label": "Max Drawdown",
            "value": f"{state.performance_df['drawdown'].min():.2%}" if not state.performance_df.empty else "-",
        },
        {"label": "Top-5 Concentration", "value": f"{state.concentration.get('top5', 0):.2%}"},
        {"label": "Effective N", "value": f"{state.concentration.get('effective_n', 0):.2f}"},
    ]
)

st.subheader("Holdings Overview")
st.dataframe(state.holdings_view, use_container_width=True)

col1, col2 = st.columns(2)
if not state.sector_weights.empty:
    col1.plotly_chart(allocation_pie(state.sector_weights, "sector", "sector_weight", "Sector Allocation"), use_container_width=True)

if not state.holdings_view.empty:
    top_positions = state.holdings_view.sort_values("weight", ascending=False).head(10)
    col2.plotly_chart(bar_chart(top_positions, "ticker", "weight", "Top Positions"), use_container_width=True)

if not state.performance_df.empty:
    st.subheader("Performance")
    col3, col4 = st.columns(2)
    col3.plotly_chart(line_chart(state.performance_df, "date", "cum_return", "Cumulative Return"), use_container_width=True)
    col4.plotly_chart(drawdown_chart(state.performance_df), use_container_width=True)

st.caption(f"Prices source: {meta['latest_source']}, last update: {meta['latest_timestamp']}")
