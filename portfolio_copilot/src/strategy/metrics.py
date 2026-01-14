from __future__ import annotations

import numpy as np
import pandas as pd


def compute_backtest_metrics(returns: pd.Series, trades: pd.DataFrame) -> dict:
    if returns.empty:
        return {}
    cumulative = (1 + returns).cumprod()
    years = max(len(returns) / 252, 1e-6)
    cagr = cumulative.iloc[-1] ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0
    drawdown = (cumulative / cumulative.cummax()) - 1
    max_dd = drawdown.min()
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
    avg_trade = float(trades["pnl"].mean()) if not trades.empty else 0.0
    turnover = float(trades["notional"].abs().sum()) if not trades.empty else 0.0
    return {
        "cagr": float(cagr),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade,
        "turnover": turnover,
    }
