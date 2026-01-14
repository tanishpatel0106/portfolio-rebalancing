from __future__ import annotations

HOLDINGS_COLUMNS = ["ticker", "quantity", "sector", "asset_class", "currency"]
REQUIRED_HOLDINGS_COLUMNS = ["ticker", "quantity", "sector"]

PORTFOLIO_SCHEMA_HINT = {
    "ticker": "string",
    "quantity": "float",
    "sector": "string (recommended)",
    "asset_class": "optional",
    "currency": "optional",
}
