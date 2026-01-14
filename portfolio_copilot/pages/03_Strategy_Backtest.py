from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from src.config.defaults import DEFAULT_STRATEGY_UNIVERSE
from src.data.market_data_provider import YFinanceMarketDataProvider
from src.services.strategy_service import generate_signals, run_backtest
from src.store.settings import load_settings
from src.ui.charts import ohlcv_chart, line_chart, drawdown_chart
from src.ui.components import kpi_cards, warning_box
from src.ui.layout import sidebar_header


st.set_page_config(page_title="Strategy Backtest", layout="wide")
sidebar_header()

st.title("Strategy Signals & Backtesting")

settings = load_settings()
strategy_defaults = settings.get("strategy", {})
backtest_defaults = settings.get("backtest", {})

provider = YFinanceMarketDataProvider()

universe = st.sidebar.text_input("Strategy Universe (comma-separated)", value=", ".join(DEFAULT_STRATEGY_UNIVERSE))
selected_ticker = st.sidebar.selectbox("Primary Ticker", options=[t.strip().upper() for t in universe.split(",") if t.strip()])
strategy = st.sidebar.selectbox("Strategy", options=["MA Crossover", "RSI Mean Reversion"])

ma_fast = st.sidebar.number_input("MA Fast", min_value=5, max_value=200, value=int(strategy_defaults.get("ma_fast", 20)))
ma_slow = st.sidebar.number_input("MA Slow", min_value=10, max_value=400, value=int(strategy_defaults.get("ma_slow", 50)))

rsi_window = st.sidebar.number_input("RSI Window", min_value=5, max_value=50, value=int(strategy_defaults.get("rsi_window", 14)))
rsi_lower = st.sidebar.slider("RSI Lower", min_value=10, max_value=40, value=int(strategy_defaults.get("rsi_lower", 30)))
rsi_upper = st.sidebar.slider("RSI Upper", min_value=60, max_value=90, value=int(strategy_defaults.get("rsi_upper", 70)))

transaction_cost = st.sidebar.slider("Transaction Cost (bps)", min_value=0.0, max_value=50.0, value=float(backtest_defaults.get("transaction_cost_bps", 5.0)))

end_date = st.sidebar.date_input("End Date", dt.date.today())
start_date = st.sidebar.date_input("Start Date", end_date - dt.timedelta(days=365))

if not selected_ticker:
    st.info("Provide at least one ticker in the universe.")
    st.stop()

ohlcv = provider.get_ohlcv(selected_ticker, start_date, end_date)
if ohlcv.empty:
    warning_box("Not enough data for the selected ticker.")
    st.stop()

price_df = ohlcv[["date", "close", "open", "high", "low", "volume"]].copy()
params = {
    "ma_fast": ma_fast,
    "ma_slow": ma_slow,
    "rsi_window": rsi_window,
    "rsi_lower": rsi_lower,
    "rsi_upper": rsi_upper,
}

signals = generate_signals(price_df, strategy, params)

st.subheader("OHLCV with Signals")
chart_df = ohlcv.copy()
chart_df["date"] = pd.to_datetime(chart_df["date"])
chart_fig = ohlcv_chart(chart_df, signals.set_index(pd.to_datetime(chart_df["date"])))
st.plotly_chart(chart_fig, use_container_width=True)

st.subheader("Latest Signal")
latest_signal = signals["signal"].iloc[-1] if not signals.empty else 0
latest_conf = signals["confidence"].iloc[-1] if "confidence" in signals.columns else 0
signal_label = "BUY" if latest_signal > 0 else "SELL" if latest_signal < 0 else "HOLD"
kpi_cards(
    [
        {"label": "Signal", "value": signal_label},
        {"label": "Confidence", "value": f"{latest_conf:.2f}"},
    ]
)

st.subheader("Suggested Trade (Staging)")
max_trade = st.number_input("Max Trade Notional", min_value=100.0, value=1000.0)
min_trade = st.number_input("Min Trade Notional", min_value=50.0, value=250.0)
notional = max_trade if signal_label != "HOLD" else 0.0
if notional < min_trade:
    st.info("Signal does not meet min trade notional.")
else:
    st.write(
        {
            "ticker": selected_ticker,
            "signal": signal_label,
            "suggested_notional": notional,
        }
    )

st.subheader("Backtest")
execution_price = backtest_defaults.get("execution_price", "close")
backtest = run_backtest(price_df, signals, execution_price, transaction_cost)
if backtest["equity"].empty:
    warning_box("Not enough data for backtest (check indicator windows).")
else:
    equity = backtest["equity"]
    col1, col2 = st.columns(2)
    col1.plotly_chart(line_chart(equity.reset_index(), "date", "equity_curve", "Equity Curve"), use_container_width=True)
    col2.plotly_chart(drawdown_chart(equity.reset_index()), use_container_width=True)
    st.dataframe(backtest["trades"], use_container_width=True)
    st.download_button("Download Trade Log", data=backtest["trades"].to_csv(index=False), file_name="trade_log.csv")
    st.json(backtest["metrics"])
