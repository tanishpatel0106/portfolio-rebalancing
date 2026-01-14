from __future__ import annotations

import pandas as pd

from ..opt.rebalance import optimize_rebalance


def build_rebalance(
    holdings_view: pd.DataFrame,
    sector_targets: dict,
    signal_summary: pd.DataFrame,
    constraints: dict,
    strategy_tilt: float,
) -> dict:
    weights = holdings_view.set_index("ticker")["weight"]
    sector_map = holdings_view.set_index("ticker")["sector"]
    signal_vector = None
    if not signal_summary.empty and "signal" in signal_summary.columns:
        signal_vector = signal_summary.set_index("ticker")["signal"]

    result = optimize_rebalance(weights, sector_map, sector_targets, signal_vector, constraints, strategy_tilt)
    return result


def build_trade_blotter(
    holdings_view: pd.DataFrame,
    optimized_weights: pd.Series,
    latest_prices: pd.DataFrame,
    min_trade_notional: float,
) -> pd.DataFrame:
    price_map = latest_prices.set_index("ticker")["price"] if not latest_prices.empty else pd.Series(dtype=float)
    current_weight = holdings_view.set_index("ticker")["weight"]
    nav = float(holdings_view["market_value"].sum())
    rows = []
    for ticker, new_weight in optimized_weights.items():
        price = price_map.get(ticker, 0.0)
        current = current_weight.get(ticker, 0.0)
        delta_weight = new_weight - current
        notional = delta_weight * nav
        if abs(notional) < min_trade_notional:
            continue
        shares = notional / price if price else 0.0
        side = "BUY" if notional > 0 else "SELL"
        rows.append({
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "notional": notional,
        })
    return pd.DataFrame(rows)
