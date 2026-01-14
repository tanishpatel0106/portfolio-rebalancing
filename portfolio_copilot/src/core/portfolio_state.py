from __future__ import annotations

import pandas as pd

from .exposures import compute_concentration, compute_sector_weights
from .performance import compute_portfolio_performance


class PortfolioState:
    def __init__(
        self,
        holdings: pd.DataFrame,
        latest_prices: pd.DataFrame,
        history: pd.DataFrame | None = None,
        sector_targets: dict | None = None,
    ) -> None:
        self.holdings = holdings.copy()
        self.latest_prices = latest_prices.copy()
        self.history = history if history is not None else pd.DataFrame()
        self.sector_targets = sector_targets or {}

        self.holdings_view = self._build_holdings_view()
        self.nav = float(self.holdings_view["market_value"].sum()) if not self.holdings_view.empty else 0.0
        self.weights = (
            self.holdings_view.set_index("ticker")["weight"] if not self.holdings_view.empty else pd.Series(dtype=float)
        )
        self.sector_weights = compute_sector_weights(self.holdings_view)
        self.concentration = compute_concentration(self.holdings_view)
        self.performance_df, self.return_stats = compute_portfolio_performance(self.history, self.weights)
        self.drift = self._compute_sector_drift()

    def _build_holdings_view(self) -> pd.DataFrame:
        if self.holdings.empty:
            return pd.DataFrame()
        prices = self.latest_prices.set_index("ticker")["price"] if not self.latest_prices.empty else pd.Series()
        df = self.holdings.copy()
        df["ticker"] = df["ticker"].str.upper().str.strip()
        df["price"] = df["ticker"].map(prices).fillna(0.0)
        df["market_value"] = df["quantity"].astype(float) * df["price"].astype(float)
        total_value = df["market_value"].sum()
        df["weight"] = df["market_value"] / total_value if total_value > 0 else 0.0
        df["sector"] = df["sector"].fillna("UNKNOWN")
        return df

    def _compute_sector_drift(self) -> pd.DataFrame:
        if self.sector_weights.empty:
            return pd.DataFrame(columns=["sector", "sector_weight", "target", "drift"])
        sector_df = self.sector_weights.copy()
        sector_df["target"] = sector_df["sector"].map(self.sector_targets).fillna(0.0)
        sector_df["drift"] = sector_df["sector_weight"] - sector_df["target"]
        return sector_df

    def snapshot_summary(self) -> dict:
        return {
            "nav": self.nav,
            "top5": self.concentration.get("top5", 0.0),
            "hhi": self.concentration.get("hhi", 0.0),
            "effective_n": self.concentration.get("effective_n", 0.0),
        }

    # TODO: add factor risk model + tracking error computations for V1.
