"""
Risk analytics dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stresslab.analytics import stress as stx
from stresslab.app import services, store
from stresslab.ui import styles
from stresslab.ui.components import plotly_defaults, kpi_row, KPI


def _build_portfolio_spec(holdings: list[dict]) -> stx.PortfolioSpec:
    weights = {row["asset_id"]: float(row.get("weight") or 0.0) for row in holdings}
    return stx.PortfolioSpec(weights=weights, normalize_weights=True, notional_mode=False)


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Risk Analytics", "Baseline risk metrics and attribution")

    if st.session_state.get("market_data") is None:
        st.warning("Load market data in Market Data.")
        return

    if not st.session_state.get("active_portfolio_id"):
        st.warning("Select an active portfolio.")
        return

    holdings = store.get_holdings(config.sqlite_path, st.session_state["active_portfolio_id"])
    portfolio_spec = _build_portfolio_spec(holdings)

    col_cfg, col_action = st.columns([1.2, 1])
    with col_cfg:
        lookback = st.number_input("Lookback (days)", min_value=60, max_value=3000, value=756)
        alpha = st.slider("Confidence level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        method = st.selectbox("VaR method", options=["historical", "parametric_normal", "cornish_fisher"])

    with col_action:
        if st.button("Compute Baseline Risk"):
            data = st.session_state["market_data"].copy()
            if lookback and len(data) > lookback:
                data = data.iloc[-int(lookback):]
            risk_cfg = stx.RiskConfig(alpha=1 - float(alpha), method=method)
            payload = services.build_risk_payload(
                data,
                portfolio_spec,
                risk=risk_cfg,
                starting_capital=10_000.0,
                input_kind="prices",
                return_method="simple",
            )
            st.session_state["risk_payload"] = payload

    payload = st.session_state.get("risk_payload")
    if not payload:
        st.info("Run baseline risk to populate the dashboard.")
        return

    metrics = payload["metrics"]
    kpis = [
        KPI(label="Ann. Return", value=f"{metrics['Annualized Return']:.2%}"),
        KPI(label="Ann. Vol", value=f"{metrics['Annualized Volatility']:.2%}"),
        KPI(label="VaR", value=f"{metrics['VaR (alpha)']:.2%}"),
        KPI(label="CVaR", value=f"{metrics['ES (alpha)']:.2%}"),
        KPI(label="Sharpe", value=f"{metrics['Sharpe Ratio']:.2f}"),
        KPI(label="Max Drawdown", value=f"{metrics['Max Drawdown ($)']:.2%}"),
    ]
    kpi_row(kpis)

    tabs = st.tabs(["Portfolio Overview", "Risk & VaR", "Attribution", "Distributions"])

    with tabs[0]:
        eq = payload["equity_curve"]
        dd = payload["drawdown"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
        fig.update_layout(title="Equity Curve")
        st.plotly_chart(plotly_defaults(fig, height=380), use_container_width=True)

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown", fill="tozeroy"))
        fig_dd.update_layout(title="Drawdown")
        st.plotly_chart(plotly_defaults(fig_dd, height=300), use_container_width=True)

    with tabs[1]:
        rolling = payload["rolling_var"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling["VaR"], name="VaR"))
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling["ES"], name="ES"))
        fig.update_layout(title="Rolling VaR/ES")
        st.plotly_chart(plotly_defaults(fig, height=360), use_container_width=True)

        corr = payload["returns_df"].corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu", title="Correlation Heatmap")
        st.plotly_chart(plotly_defaults(fig_corr, height=420), use_container_width=True)

    with tabs[2]:
        contrib = payload["risk_contributions"]
        assets = payload["assets"]
        df = pd.DataFrame(
            {
                "asset": assets,
                "component": contrib["component"],
                "pct": contrib["pct"],
            }
        ).sort_values("pct", ascending=False)
        fig = px.bar(df, x="asset", y="pct", title="Contribution to Volatility")
        st.plotly_chart(plotly_defaults(fig, height=360), use_container_width=True)

        var_contrib = payload["var_contributions"]
        df_var = pd.DataFrame(
            {
                "asset": assets,
                "component_var": var_contrib["component_var"],
                "pct_var": var_contrib["pct_var"],
            }
        )
        fig_var = px.bar(df_var, x="asset", y="component_var", title="VaR Contributions")
        st.plotly_chart(plotly_defaults(fig_var, height=360), use_container_width=True)

    with tabs[3]:
        pr = payload["portfolio_series"]
        fig = px.histogram(pr, nbins=50, title="Portfolio Return Distribution")
        st.plotly_chart(plotly_defaults(fig, height=360), use_container_width=True)
