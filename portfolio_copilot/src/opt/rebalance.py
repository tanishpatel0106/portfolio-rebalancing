from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd


def optimize_rebalance(
    weights: pd.Series,
    sector_map: pd.Series,
    sector_targets: dict,
    signal_vector: pd.Series | None,
    constraints: dict,
    strategy_tilt: float,
) -> dict:
    tickers = list(weights.index)
    n_assets = len(tickers)
    w = weights.values
    dw = cp.Variable(n_assets)
    w_new = w + dw

    max_weight = constraints.get("max_weight_per_asset", 0.25)
    turnover_max = constraints.get("turnover_max", 0.2)
    cash_min = constraints.get("cash_min", 0.02)

    sector_targets_vec = _build_sector_target_vector(tickers, sector_map, sector_targets)
    sector_matrix, sector_names = _build_sector_matrix(tickers, sector_map)

    sector_weights = sector_matrix @ w_new
    sector_drift = sector_weights - sector_targets_vec

    l1_penalty = constraints.get("l1_penalty", 1.0)
    l2_penalty = constraints.get("l2_penalty", 1.0)

    signal_term = 0
    if signal_vector is not None and not signal_vector.empty and strategy_tilt > 0:
        signal_array = signal_vector.reindex(tickers).fillna(0).values
        signal_term = -strategy_tilt * signal_array @ w_new

    objective = cp.Minimize(cp.sum_squares(sector_drift) + l1_penalty * cp.norm1(dw) + l2_penalty * cp.sum_squares(dw) + signal_term)

    constraints_list = [
        w_new >= 0,
        cp.sum(w_new) == 1,
        w_new <= max_weight,
        cp.norm1(dw) <= turnover_max,
    ]

    if cash_min > 0 and "CASH" in tickers:
        cash_idx = tickers.index("CASH")
        constraints_list.append(w_new[cash_idx] >= cash_min)

    problem = cp.Problem(objective, constraints_list)
    problem.solve(solver=cp.ECOS)

    if w_new.value is None:
        return {"status": "failed", "weights": weights, "dw": pd.Series(dtype=float), "sector_names": sector_names}

    optimized = pd.Series(w_new.value, index=tickers)
    delta = pd.Series(dw.value, index=tickers)
    return {"status": "optimal", "weights": optimized, "dw": delta, "sector_names": sector_names}


def _build_sector_matrix(tickers: list[str], sector_map: pd.Series) -> tuple[np.ndarray, list[str]]:
    sectors = sorted(sector_map.fillna("UNKNOWN").unique())
    matrix = np.zeros((len(sectors), len(tickers)))
    for i, sector in enumerate(sectors):
        mask = sector_map.reindex(tickers).fillna("UNKNOWN") == sector
        matrix[i, :] = mask.astype(float)
    return matrix, sectors


def _build_sector_target_vector(tickers: list[str], sector_map: pd.Series, sector_targets: dict) -> np.ndarray:
    sectors = sorted(sector_map.fillna("UNKNOWN").unique())
    targets = []
    for sector in sectors:
        targets.append(sector_targets.get(sector, 0.0))
    return np.array(targets)
