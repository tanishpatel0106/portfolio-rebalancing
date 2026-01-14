from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.data.market_data_provider import YFinanceMarketDataProvider
from src.services.compute import build_portfolio_state
from src.services.rebalance_service import build_rebalance, build_trade_blotter
from src.services.strategy_service import generate_signals
from src.store.audit import save_audit_run
from src.store.portfolios import list_portfolios, load_portfolio
from src.store.settings import load_settings
from src.ui.charts import allocation_pie, bar_chart
from src.ui.components import kpi_cards, warning_box
from src.ui.layout import sidebar_header


st.set_page_config(page_title="Rebalance", layout="wide")
sidebar_header()

st.title("Rebalance Optimizer")

portfolio_names = list_portfolios()
if not portfolio_names:
    st.info("No saved portfolios.")
    st.stop()

selected = st.sidebar.selectbox("Portfolio", options=portfolio_names)
end_date = st.sidebar.date_input("End Date", dt.date.today())
start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=365))

settings = load_settings()
constraints = settings.get("constraints", {})
strategy_defaults = settings.get("strategy", {})

strategy_tilt = st.sidebar.slider("Strategy Tilt Strength", min_value=0.0, max_value=5.0, value=1.0)

holdings = load_portfolio(selected)
state, missing, meta = build_portfolio_state(holdings, start_date, end_date, constraints.get("sector_targets", {}))

if missing:
    warning_box(f"Missing price data for: {', '.join(missing)}")

provider = YFinanceMarketDataProvider()

signal_summary = []
for ticker in holdings["ticker"].dropna().unique():
    ohlcv = provider.get_ohlcv(ticker, start_date, end_date)
    if ohlcv.empty:
        continue
    price_df = ohlcv[["date", "close"]].copy()
    signals = generate_signals(
        price_df,
        "MA Crossover",
        {
            "ma_fast": int(strategy_defaults.get("ma_fast", 20)),
            "ma_slow": int(strategy_defaults.get("ma_slow", 50)),
            "rsi_window": int(strategy_defaults.get("rsi_window", 14)),
            "rsi_lower": int(strategy_defaults.get("rsi_lower", 30)),
            "rsi_upper": int(strategy_defaults.get("rsi_upper", 70)),
        },
    )
    if signals.empty:
        continue
    latest_signal = signals["signal"].iloc[-1]
    signal_summary.append(
        {
            "ticker": ticker,
            "signal": latest_signal,
        }
    )

signal_df = pd.DataFrame(signal_summary)

st.subheader("Current Portfolio")
st.dataframe(state.holdings_view, use_container_width=True)

result = build_rebalance(state.holdings_view, constraints.get("sector_targets", {}), signal_df, constraints, strategy_tilt)
if result["status"] != "optimal":
    warning_box("Optimization failed. Check constraints.")
    st.stop()

optimized_weights = result["weights"]
delta_weights = result.get("dw", pd.Series(dtype=float))
trade_blotter = build_trade_blotter(
    state.holdings_view,
    optimized_weights,
    state.latest_prices,
    constraints.get("min_trade_notional", 250.0),
)

if not trade_blotter.empty:
    trade_blotter["reason_code"] = "DRIFT_REBALANCE"
    if not signal_df.empty:
        trade_blotter.loc[trade_blotter["ticker"].isin(signal_df[signal_df["signal"] > 0]["ticker"]), "reason_code"] = "STRATEGY_TILT"
    if (trade_blotter["notional"].abs() > state.nav * constraints.get("max_weight_per_asset", 0.25)).any():
        trade_blotter.loc[trade_blotter["notional"].abs() > state.nav * constraints.get("max_weight_per_asset", 0.25), "reason_code"] = "CAP_ENFORCEMENT"
    turnover_used = float(delta_weights.abs().sum()) if not delta_weights.empty else 0.0
    if turnover_used >= constraints.get("turnover_max", 0.2) * 0.95:
        trade_blotter["reason_code"] = trade_blotter["reason_code"].where(
            trade_blotter["reason_code"].isin(["CAP_ENFORCEMENT", "STRATEGY_TILT"]), "TURNOVER_LIMITED"
        )

kpi_cards(
    [
        {"label": "Turnover Used", "value": f"{trade_blotter['notional'].abs().sum() / state.nav:.2%}" if state.nav else "-"},
        {"label": "Strategy Tilt", "value": f"{strategy_tilt:.2f}"},
    ]
)

col1, col2 = st.columns(2)
if not state.sector_weights.empty:
    col1.plotly_chart(allocation_pie(state.sector_weights, "sector", "sector_weight", "Current Sector Allocation"), use_container_width=True)

post_alloc = state.holdings_view.copy()
post_alloc["weight"] = post_alloc["ticker"].map(optimized_weights)
post_sector = post_alloc.groupby("sector")["weight"].sum().reset_index()
col2.plotly_chart(allocation_pie(post_sector, "sector", "weight", "Post-Rebalance Sector Allocation"), use_container_width=True)

st.subheader("Trade Blotter")
st.dataframe(trade_blotter, use_container_width=True)

if st.button("Approve & Save Run", type="primary"):
    post_sector_targets = constraints.get("sector_targets", {})
    post_sector["target"] = post_sector["sector"].map(post_sector_targets).fillna(0.0)
    post_sector["drift"] = post_sector["weight"] - post_sector["target"]
    audit_payload = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "portfolio_name": selected,
        "constraints": constraints,
        "metrics": {
            "turnover": float(trade_blotter["notional"].abs().sum() / state.nav) if state.nav else 0.0,
            "drift_pre": float(state.drift["drift"].abs().sum()) if not state.drift.empty else 0.0,
            "drift_post": float(post_sector["drift"].abs().sum()) if not post_sector.empty else 0.0,
            "cost_proxy": float(trade_blotter["notional"].abs().sum()) * 0.0005,
        },
        "trade_blotter": trade_blotter.to_dict(orient="records"),
        "signal_summary": signal_df.to_dict(orient="records"),
        "price_snapshot": state.latest_prices.to_dict(orient="records"),
        "price_metadata": meta,
    }
    save_audit_run(audit_payload)
    st.success("Audit run saved.")

st.download_button("Download Trade Blotter", data=trade_blotter.to_csv(index=False), file_name="trade_blotter.csv")
