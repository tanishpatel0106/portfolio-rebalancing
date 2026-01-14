from __future__ import annotations

DEFAULT_SETTINGS = {
    "constraints": {
        "turnover_max": 0.2,
        "max_weight_per_asset": 0.25,
        "cash_min": 0.02,
        "min_trade_notional": 250.0,
        "sector_bounds": {},
        "sector_targets": {},
    },
    "backtest": {
        "transaction_cost_bps": 5.0,
        "execution_price": "close",
        "slippage_enabled": False,
    },
    "strategy": {
        "ma_fast": 20,
        "ma_slow": 50,
        "rsi_window": 14,
        "rsi_lower": 30,
        "rsi_upper": 70,
    },
}

DEFAULT_PORTFOLIO = {
    "name": "Sample Portfolio",
    "holdings": [
        {
            "ticker": "AAPL",
            "quantity": 20.0,
            "sector": "Technology",
            "asset_class": "Equity",
            "currency": "USD",
        },
        {
            "ticker": "MSFT",
            "quantity": 15.0,
            "sector": "Technology",
            "asset_class": "Equity",
            "currency": "USD",
        },
        {
            "ticker": "JPM",
            "quantity": 25.0,
            "sector": "Financials",
            "asset_class": "Equity",
            "currency": "USD",
        },
    ],
}

DEFAULT_STRATEGY_UNIVERSE = ["AAPL", "MSFT", "SPY", "QQQ"]
