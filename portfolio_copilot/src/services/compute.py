from __future__ import annotations

import datetime as dt

import pandas as pd

from ..core.portfolio_state import PortfolioState
from ..data.market_data_provider import YFinanceMarketDataProvider


def build_portfolio_state(
    holdings: pd.DataFrame,
    start: dt.date,
    end: dt.date,
    sector_targets: dict | None = None,
) -> tuple[PortfolioState, list[str], dict]:
    provider = YFinanceMarketDataProvider()
    tickers = holdings["ticker"].dropna().unique().tolist()
    latest = provider.get_latest_prices(tickers)

    missing = [t for t in tickers if t not in latest["ticker"].unique()]

    history = provider.get_close_history(tickers, start, end)
    meta = {
        "latest_timestamp": latest["timestamp"].iloc[0] if not latest.empty else None,
        "latest_source": "yfinance",
        "history_start": start.isoformat(),
        "history_end": end.isoformat(),
    }

    state = PortfolioState(holdings, latest, history, sector_targets)
    return state, missing, meta
