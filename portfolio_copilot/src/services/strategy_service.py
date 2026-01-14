from __future__ import annotations

import pandas as pd

from ..strategy.backtest import backtest_signals
from ..strategy.metrics import compute_backtest_metrics
from ..strategy.signals import moving_average_crossover, rsi_signal


def generate_signals(price_df: pd.DataFrame, strategy: str, params: dict) -> pd.DataFrame:
    close = price_df["close"]
    if strategy == "MA Crossover":
        return moving_average_crossover(close, params["ma_fast"], params["ma_slow"])
    if strategy == "RSI Mean Reversion":
        return rsi_signal(close, params["rsi_window"], params["rsi_lower"], params["rsi_upper"])
    return pd.DataFrame()


def run_backtest(price_df: pd.DataFrame, signals: pd.DataFrame, execution_price: str, cost_bps: float) -> dict:
    merged = price_df.copy()
    merged = merged.set_index("date")
    if signals.empty:
        return {"equity": pd.DataFrame(), "trades": pd.DataFrame(), "metrics": {}}
    merged = merged.join(signals)
    equity, trades = backtest_signals(merged, execution_price=execution_price, transaction_cost_bps=cost_bps)
    metrics = compute_backtest_metrics(equity["strategy_return"], trades)
    return {"equity": equity, "trades": trades, "metrics": metrics}
