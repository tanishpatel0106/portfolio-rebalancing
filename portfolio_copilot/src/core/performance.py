from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .drawdown import compute_drawdown


def compute_portfolio_performance(
    history: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    if history.empty or weights.empty:
        return pd.DataFrame(columns=["date", "portfolio_return", "cum_return", "drawdown", "vol_20d", "sharpe_20d"]), {}

    price_pivot = history.pivot(index="date", columns="ticker", values="close").sort_index()
    price_pivot = price_pivot.ffill().dropna()
    aligned_weights = weights.reindex(price_pivot.columns).fillna(0.0)

    returns = price_pivot.pct_change().dropna()
    portfolio_returns = returns.mul(aligned_weights, axis=1).sum(axis=1)
    cum_return = (1 + portfolio_returns).cumprod() - 1
    drawdown = compute_drawdown(cum_return)
    vol_20d = portfolio_returns.rolling(20).std() * np.sqrt(252)
    sharpe_20d = (portfolio_returns.rolling(20).mean() / portfolio_returns.rolling(20).std()) * np.sqrt(252)

    performance_df = pd.DataFrame(
        {
            "date": portfolio_returns.index,
            "portfolio_return": portfolio_returns.values,
            "cum_return": cum_return.values,
            "drawdown": drawdown.values,
            "vol_20d": vol_20d.values,
            "sharpe_20d": sharpe_20d.values,
        }
    )

    stats_dict = {
        "mean": float(portfolio_returns.mean()),
        "std": float(portfolio_returns.std()),
        "skew": float(stats.skew(portfolio_returns)),
        "kurtosis": float(stats.kurtosis(portfolio_returns)),
    }
    return performance_df, stats_dict
