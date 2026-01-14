from __future__ import annotations

import pandas as pd


def backtest_signals(
    price_df: pd.DataFrame,
    signal_col: str = "signal",
    execution_price: str = "close",
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if price_df.empty or signal_col not in price_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = price_df.copy().dropna(subset=[execution_price])
    df["position"] = df[signal_col].shift(1).fillna(0)
    df["return"] = df[execution_price].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["return"]

    trades = df["position"].diff().fillna(0).abs()
    cost = trades * (transaction_cost_bps / 10000)
    df["strategy_return"] -= cost

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["drawdown"] = df["equity_curve"] / df["equity_curve"].cummax() - 1

    trade_log = _build_trade_log(df, execution_price)
    return df, trade_log


def _build_trade_log(df: pd.DataFrame, execution_price: str) -> pd.DataFrame:
    trades = []
    current = None
    for idx, row in df.iterrows():
        pos = row["position"]
        if current is None and pos != 0:
            current = {
                "entry_date": idx,
                "entry_price": row[execution_price],
                "side": "LONG" if pos > 0 else "SHORT",
            }
        elif current is not None and pos == 0:
            exit_price = row[execution_price]
            pnl = (exit_price - current["entry_price"]) / current["entry_price"]
            if current["side"] == "SHORT":
                pnl = -pnl
            holding_period = (idx - current["entry_date"]).days if hasattr(idx, "days") else 0
            trades.append(
                {
                    "entry_date": current["entry_date"],
                    "exit_date": idx,
                    "side": current["side"],
                    "pnl": pnl,
                    "holding_period": holding_period,
                    "notional": abs(exit_price),
                }
            )
            current = None
    return pd.DataFrame(trades)
