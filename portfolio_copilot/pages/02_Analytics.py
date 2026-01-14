from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.services.analytics_service import get_rebalance_history
from src.services.compute import build_portfolio_state
from src.store.audit import load_audit_runs
from src.store.portfolios import list_portfolios, load_portfolio
from src.ui.charts import drawdown_chart, line_chart, returns_histogram, bar_chart
from src.ui.layout import sidebar_header
from src.ui.components import warning_box


st.set_page_config(page_title="Analytics", layout="wide")
sidebar_header()

st.title("Analytics & Audit Logs")

portfolio_names = list_portfolios()
if not portfolio_names:
    st.info("No saved portfolios available.")
    st.stop()

selected = st.sidebar.selectbox("Portfolio", options=portfolio_names)
end_date = st.sidebar.date_input("End Date", dt.date.today())
start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=365))

holdings = load_portfolio(selected)
state, missing, _ = build_portfolio_state(holdings, start_date, end_date)

if missing:
    warning_box(f"Missing price data for: {', '.join(missing)}")

st.subheader("Performance Analytics")
if state.performance_df.empty:
    st.info("Not enough history to compute performance.")
else:
    col1, col2 = st.columns(2)
    col1.plotly_chart(line_chart(state.performance_df, "date", "cum_return", "Cumulative Return"), use_container_width=True)
    col2.plotly_chart(drawdown_chart(state.performance_df), use_container_width=True)
    st.plotly_chart(returns_histogram(state.performance_df["portfolio_return"]), use_container_width=True)
    st.download_button(
        "Download Performance CSV",
        data=state.performance_df.to_csv(index=False),
        file_name="portfolio_performance.csv",
    )

st.subheader("Rebalance Run History")
history = get_rebalance_history()
if history.empty:
    st.info("No audit runs saved yet.")
else:
    st.dataframe(history, use_container_width=True)

runs = load_audit_runs()
if runs:
    trade_rows = []
    for run in runs:
        for trade in run.get("trade_blotter", []):
            trade_rows.append(
                {
                    "timestamp": run.get("timestamp"),
                    "ticker": trade.get("ticker"),
                    "side": trade.get("side"),
                    "notional": trade.get("notional"),
                }
            )
    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty:
        top_traded = trades_df.groupby("ticker")["notional"].sum().abs().sort_values(ascending=False).head(10).reset_index()
        st.plotly_chart(bar_chart(top_traded, "ticker", "notional", "Top Traded Tickers"), use_container_width=True)

st.subheader("Audit Run Details")
if runs:
    st.json(runs[-1])
