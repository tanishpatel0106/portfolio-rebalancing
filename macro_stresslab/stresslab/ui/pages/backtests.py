"""
Backtests (optional module).
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from stresslab.analytics import stress as stx
from stresslab.app import services, store
from stresslab.ui import styles
from stresslab.ui.components import plotly_defaults


def _build_portfolio_spec(holdings: list[dict]) -> stx.PortfolioSpec:
    weights = {row["asset_id"]: float(row.get("weight") or 0.0) for row in holdings}
    return stx.PortfolioSpec(weights=weights, normalize_weights=True, notional_mode=False)


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Backtests", "Strategy or portfolio backtest configuration")

    if st.session_state.get("market_data") is None:
        st.warning("Load market data in Market Data.")
        return

    if not st.session_state.get("active_portfolio_id"):
        st.warning("Select an active portfolio.")
        return

    holdings = store.get_holdings(config.sqlite_path, st.session_state["active_portfolio_id"])
    portfolio_spec = _build_portfolio_spec(holdings)

    transaction_cost = st.slider("Transaction cost (bps)", min_value=0, max_value=100, value=5)
    rebalance = st.selectbox("Rebalance frequency", options=["Monthly", "Quarterly", "Annually"], index=0)

    if st.button("Run Backtest"):
        data = st.session_state["market_data"].copy()
        risk_cfg = stx.RiskConfig(alpha=0.05, method="historical")
        payload = services.build_risk_payload(
            data,
            portfolio_spec,
            risk=risk_cfg,
            starting_capital=10_000.0,
            input_kind="prices",
            return_method="simple",
        )
        st.session_state["backtest_payload"] = payload

    payload = st.session_state.get("backtest_payload")
    if not payload:
        st.info("Run a backtest to see results.")
        return

    eq = payload["equity_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
    fig.update_layout(title="Equity Curve (Buy & Hold)")
    st.plotly_chart(plotly_defaults(fig, height=380), use_container_width=True)

    st.caption(f"Transaction cost: {transaction_cost} bps | Rebalance: {rebalance}")
