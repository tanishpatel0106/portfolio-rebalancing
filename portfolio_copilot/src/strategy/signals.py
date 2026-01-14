from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average_crossover(close: pd.Series, fast: int, slow: int) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame(columns=["signal", "fast_ma", "slow_ma", "confidence"])
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal = pd.Series(signal, index=close.index).where(~fast_ma.isna() & ~slow_ma.isna(), 0)
    confidence = (fast_ma - slow_ma).abs() / close
    return pd.DataFrame({"signal": signal, "fast_ma": fast_ma, "slow_ma": slow_ma, "confidence": confidence})


def rsi_signal(close: pd.Series, window: int, lower: float, upper: float) -> pd.DataFrame:
    if close.empty:
        return pd.DataFrame(columns=["signal", "rsi", "confidence"])
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    signal = pd.Series(0, index=close.index)
    signal = signal.where(~rsi.isna(), 0)
    signal = signal.mask(rsi < lower, 1)
    signal = signal.mask(rsi > upper, -1)
    confidence = (rsi - (lower + upper) / 2).abs() / 50
    return pd.DataFrame({"signal": signal, "rsi": rsi, "confidence": confidence})
