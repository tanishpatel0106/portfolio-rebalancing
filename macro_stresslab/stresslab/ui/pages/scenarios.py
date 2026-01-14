"""
Scenario builder and library.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from stresslab.app import store
from stresslab.ui import styles
from stresslab.ui.components import toast_success, toast_error


def _asset_shock_editor(assets: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"asset": assets, "shock_pct": 0.0})
    return st.data_editor(
        base,
        use_container_width=True,
        num_rows="dynamic",
        key="asset_shock_editor",
    )


def _factor_shock_editor(factors: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"factor": factors, "shock_pct": 0.0})
    return st.data_editor(
        base,
        use_container_width=True,
        num_rows="dynamic",
        key="factor_shock_editor",
    )


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Scenario Builder", "Create deterministic or factor-aware stress scenarios")

    assets = []
    if st.session_state.get("active_portfolio_id"):
        holdings = store.get_holdings(config.sqlite_path, st.session_state["active_portfolio_id"])
        assets = [h["asset_id"] for h in holdings]

    scenario_kind = st.selectbox(
        "Scenario type",
        options=["asset_shock", "factor_beta", "corr_stress"],
        index=0,
    )
    name = st.text_input("Scenario name", value="Risk-Off Shock")
    description = st.text_area("Description", value="Equity selloff, rate rally")
    tags = st.text_input("Tags (comma separated)", value="macro,risk-off")

    payload: dict[str, object] = {"kind": scenario_kind}

    if scenario_kind == "asset_shock":
        st.caption("Apply direct price shocks to assets (percent, e.g. -0.05 = -5%).")
        shocks_df = _asset_shock_editor(assets)
        shocks = {row["asset"]: float(row["shock_pct"]) for _, row in shocks_df.iterrows() if row["asset"]}
        payload["shocks"] = shocks
    elif scenario_kind == "factor_beta":
        st.caption("Apply shocks to factors and map via beta exposure matrix.")
        betas = st.session_state.get("factor_betas")
        factor_cols = list(betas.columns) if isinstance(betas, pd.DataFrame) else []
        shocks_df = _factor_shock_editor(factor_cols)
        factor_shocks = {row["factor"]: float(row["shock_pct"]) for _, row in shocks_df.iterrows() if row["factor"]}
        payload["factor_shocks"] = factor_shocks
    else:
        st.caption("Stress correlations toward a target level.")
        target_corr = st.slider("Target average correlation", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
        blend = st.slider("Blend strength", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        payload["corr_target"] = target_corr
        payload["corr_blend"] = blend

    if st.button("💾 Save Scenario"):
        try:
            scenario_id = store.create_scenario(
                config.sqlite_path,
                name=name,
                description=description,
                tags=tags,
                payload=payload,
            )
            st.session_state["active_scenario_id"] = scenario_id
            toast_success("Scenario saved.")
        except Exception as exc:
            toast_error(f"Failed to save scenario: {exc}")

    st.divider()
    styles.section_header("Scenario Library", "Reuse and manage saved scenarios")

    scenarios = store.list_scenarios(config.sqlite_path)
    if not scenarios:
        st.info("No scenarios saved yet.")
        return

    scenario_labels = [f"{s['name']} ({s['scenario_id']})" for s in scenarios]
    selected_label = st.selectbox("Select scenario", scenario_labels)
    selected = scenarios[scenario_labels.index(selected_label)]

    st.json(selected["payload"])

    if st.button("Set as active scenario"):
        st.session_state["active_scenario_id"] = selected["scenario_id"]
        toast_success("Active scenario set.")
