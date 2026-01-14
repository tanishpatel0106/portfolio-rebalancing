"""
Run stress tests and save runs.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from stresslab.analytics import stress as stx
from stresslab.app import services, store
from stresslab.ui import styles
from stresslab.ui.components import plotly_defaults, toast_success, toast_error, kpi_row, KPI


def _build_portfolio_spec(holdings: list[dict]) -> stx.PortfolioSpec:
    weights = {row["asset_id"]: float(row.get("weight") or 0.0) for row in holdings}
    return stx.PortfolioSpec(weights=weights, normalize_weights=True, notional_mode=False)


def _scenario_from_payload(payload: dict) -> stx.Scenario:
    kind = payload.get("kind", "asset_shock")
    return stx.Scenario(
        name=payload.get("name", "Scenario"),
        kind=kind,
        shocks=payload.get("shocks"),
        factor_shocks=payload.get("factor_shocks"),
        corr_target=payload.get("corr_target"),
        corr_blend=payload.get("corr_blend"),
    )


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Run Stress", "Configure and execute scenario analytics")

    if st.session_state.get("market_data") is None:
        st.warning("Load market data before running stress tests.")
        return

    if not st.session_state.get("active_portfolio_id"):
        st.warning("Select an active portfolio first.")
        return

    holdings = store.get_holdings(config.sqlite_path, st.session_state["active_portfolio_id"])
    portfolio_spec = _build_portfolio_spec(holdings)

    col_config, col_run = st.columns([1.2, 1])
    with col_config:
        st.markdown("#### Run Configuration")
        lookback = st.number_input("Lookback (days)", min_value=60, max_value=3000, value=756)
        frequency = st.selectbox("Frequency", options=["D", "W", "M"], index=0)
        alpha = st.slider("Confidence level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
        var_method = st.selectbox("VaR method", options=["historical", "parametric_normal", "cornish_fisher"])
        ewma_lambda = st.slider("EWMA lambda", min_value=0.80, max_value=0.99, value=0.94, step=0.01)
        risk_free = st.number_input("Risk-free rate", value=0.0, step=0.01)

    with col_run:
        st.markdown("#### Scenario")
        scenarios = store.list_scenarios(config.sqlite_path)
        scenario_map = {"None": None}
        for sc in scenarios:
            scenario_map[f"{sc['name']} ({sc['scenario_id']})"] = sc
        scenario_label = st.selectbox("Scenario", options=list(scenario_map.keys()))

        run_name = st.text_input("Run name", value=f"Stress Run {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

        if st.button("🚀 Run Stress Test"):
            try:
                data = st.session_state["market_data"].copy()
                if lookback and len(data) > lookback:
                    data = data.iloc[-int(lookback):]

                risk_cfg = stx.RiskConfig(
                    alpha=1 - float(alpha),
                    method=var_method,
                    annualization_factor=252,
                    ewma_lambda=float(ewma_lambda),
                    cov_shrink="lw_diag",
                )

                risk_payload = services.build_risk_payload(
                    data,
                    portfolio_spec,
                    risk=risk_cfg,
                    starting_capital=10_000.0,
                    input_kind="prices",
                    return_method="simple",
                )

                scenario_payload = None
                selected = scenario_map.get(scenario_label)
                scenarios_payload = []
                if selected:
                    payload = selected["payload"]
                    payload["name"] = selected["name"]
                    scenario_obj = _scenario_from_payload(payload)
                    scenarios_payload.append(scenario_obj)
                    scenario_payload = services.build_scenario_payload(
                        data,
                        portfolio_spec,
                        scenarios=scenarios_payload,
                        input_kind="prices",
                        return_method="simple",
                        factor_returns=st.session_state.get("factor_returns"),
                        betas=st.session_state.get("factor_betas"),
                        starting_capital=10_000.0,
                    )

                results = {
                    "risk": risk_payload,
                    "scenario": scenario_payload,
                }
                st.session_state["run_results"] = results

                run_id = store.create_run(
                    config.sqlite_path,
                    name=run_name,
                    portfolio_id=st.session_state["active_portfolio_id"],
                    scenario_id=selected["scenario_id"] if selected else None,
                    config={
                        "lookback": lookback,
                        "frequency": frequency,
                        "alpha": alpha,
                        "var_method": var_method,
                        "ewma_lambda": ewma_lambda,
                        "risk_free": risk_free,
                    },
                    outputs={
                        "metrics": risk_payload["metrics"],
                        "scenario_table": scenario_payload["scenario_table"].to_dict()
                        if scenario_payload is not None
                        else None,
                    },
                    artifacts={},
                )
                st.session_state["last_run_id"] = run_id
                toast_success("Run completed and saved.")
            except Exception as exc:
                toast_error(f"Run failed: {exc}")

    if st.session_state.get("run_results") is None:
        return

    st.divider()
    styles.section_header("Run Output", "Baseline and scenario metrics")
    results = st.session_state["run_results"]
    metrics = results["risk"]["metrics"]

    kpis = [
        KPI(label="Ann. Vol", value=f"{metrics['Annualized Volatility']:.2%}"),
        KPI(label="VaR", value=f"{metrics['VaR (alpha)']:.2%}"),
        KPI(label="CVaR", value=f"{metrics['ES (alpha)']:.2%}"),
        KPI(label="Max Drawdown", value=f"{metrics['Max Drawdown ($)']:.2%}"),
    ]
    kpi_row(kpis)

    tabs = st.tabs(["Scenario Table", "Waterfall P&L", "Risk Contributions"])

    with tabs[0]:
        scenario_payload = results.get("scenario")
        if scenario_payload is None:
            st.info("No scenario selected for this run.")
        else:
            st.dataframe(scenario_payload["scenario_table"], use_container_width=True)

    with tabs[1]:
        scenario_payload = results.get("scenario")
        if scenario_payload is None:
            st.info("Scenario required for waterfall.")
        else:
            table = scenario_payload["scenario_table"]
            pnl_rows = table[table["ImpactType"].str.contains("PnL", na=False)]
            if pnl_rows.empty:
                st.info("No PnL impacts available.")
            else:
                fig = go.Figure(
                    go.Waterfall(
                        x=pnl_rows["Scenario"],
                        y=pnl_rows["ImpactValue"],
                        measure=["relative"] * len(pnl_rows),
                    )
                )
                st.plotly_chart(plotly_defaults(fig, height=400), use_container_width=True)

    with tabs[2]:
        contrib = results["risk"]["risk_contributions"]
        assets = results["risk"]["assets"]
        df = pd.DataFrame(
            {
                "asset": assets,
                "component": contrib["component"],
                "pct": contrib["pct"],
            }
        )
        fig = px.bar(df, x="asset", y="pct", title="Percent Contribution to Vol")
        st.plotly_chart(plotly_defaults(fig, height=400), use_container_width=True)
