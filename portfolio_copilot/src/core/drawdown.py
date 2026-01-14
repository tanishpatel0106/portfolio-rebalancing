from __future__ import annotations

import pandas as pd


def compute_drawdown(cum_return: pd.Series) -> pd.Series:
    if cum_return.empty:
        return pd.Series(dtype=float)
    cum_curve = (1 + cum_return)
    running_max = cum_curve.cummax()
    drawdown = (cum_curve / running_max) - 1
    return drawdown
