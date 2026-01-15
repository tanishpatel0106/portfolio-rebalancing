"""
stresslab/data/demo.py
======================
Deterministic demo portfolio + dataset for StressLab.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEMO_ASSETS = [
    ("AAPL", "Apple", "equity", "USD", "Technology", "US"),
    ("MSFT", "Microsoft", "equity", "USD", "Technology", "US"),
    ("NVDA", "NVIDIA", "equity", "USD", "Technology", "US"),
    ("AMZN", "Amazon", "equity", "USD", "Consumer Discretionary", "US"),
    ("GOOGL", "Alphabet", "equity", "USD", "Communication", "US"),
    ("META", "Meta", "equity", "USD", "Communication", "US"),
    ("JPM", "JPMorgan", "equity", "USD", "Financials", "US"),
    ("BAC", "Bank of America", "equity", "USD", "Financials", "US"),
    ("XOM", "Exxon", "equity", "USD", "Energy", "US"),
    ("CVX", "Chevron", "equity", "USD", "Energy", "US"),
    ("UNH", "UnitedHealth", "equity", "USD", "Healthcare", "US"),
    ("JNJ", "Johnson & Johnson", "equity", "USD", "Healthcare", "US"),
    ("PG", "Procter & Gamble", "equity", "USD", "Consumer Staples", "US"),
    ("KO", "Coca-Cola", "equity", "USD", "Consumer Staples", "US"),
    ("TSLA", "Tesla", "equity", "USD", "Consumer Discretionary", "US"),
    ("HD", "Home Depot", "equity", "USD", "Industrials", "US"),
    ("CAT", "Caterpillar", "equity", "USD", "Industrials", "US"),
    ("DUK", "Duke Energy", "equity", "USD", "Utilities", "US"),
    ("NEE", "NextEra", "equity", "USD", "Utilities", "US"),
    ("TMO", "Thermo Fisher", "equity", "USD", "Healthcare", "US"),
    ("AVGO", "Broadcom", "equity", "USD", "Technology", "US"),
    ("COST", "Costco", "equity", "USD", "Consumer Staples", "US"),
    ("V", "Visa", "equity", "USD", "Financials", "US"),
    ("NFLX", "Netflix", "equity", "USD", "Communication", "US"),
    ("BA", "Boeing", "equity", "USD", "Industrials", "US"),
]


def demo_holdings() -> pd.DataFrame:
    n = len(DEMO_ASSETS)
    base_weight = 1.0 / n
    rows = []
    for asset_id, name, asset_type, currency, sector, region in DEMO_ASSETS:
        rows.append(
            {
                "asset_id": asset_id,
                "asset_name": name,
                "asset_type": asset_type,
                "currency": currency,
                "quantity": 100,
                "price": 100.0,
                "notional": 10000.0,
                "weight": base_weight,
                "sector": sector,
                "region": region,
            }
        )
    return pd.DataFrame(rows)


def demo_prices(days: int = 756, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.utcnow().date() - timedelta(days=int(days * 1.4))
    dates = pd.bdate_range(start=start, periods=days)

    prices = {}
    for asset_id, *_ in DEMO_ASSETS:
        drift = rng.normal(0.0003, 0.0001)
        vol = rng.uniform(0.012, 0.028)
        shocks = rng.normal(drift, vol, size=len(dates))
        series = 100 * np.exp(np.cumsum(shocks))
        prices[asset_id] = series
    return pd.DataFrame(prices, index=dates)


def demo_metadata(prices: pd.DataFrame) -> Dict[str, str]:
    return {
        "source": "demo",
        "rows": str(len(prices)),
        "columns": ",".join(prices.columns),
    }
