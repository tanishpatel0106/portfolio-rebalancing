from __future__ import annotations

import pandas as pd


def compute_sector_weights(holdings: pd.DataFrame) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=["sector", "weight"])
    sector_weights = holdings.groupby("sector", dropna=False)["weight"].sum().reset_index()
    sector_weights.rename(columns={"weight": "sector_weight"}, inplace=True)
    return sector_weights.sort_values("sector_weight", ascending=False)


def compute_concentration(holdings: pd.DataFrame) -> dict:
    if holdings.empty:
        return {"top5": 0.0, "hhi": 0.0, "effective_n": 0.0}
    weights = holdings["weight"].clip(lower=0).values
    top5 = float(pd.Series(weights).nlargest(5).sum())
    hhi = float((weights**2).sum())
    effective_n = float(1.0 / hhi) if hhi > 0 else 0.0
    return {"top5": top5, "hhi": hhi, "effective_n": effective_n}
