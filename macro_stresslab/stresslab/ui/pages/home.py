"""
Home page for StressLab.
"""
from __future__ import annotations

import streamlit as st

from stresslab.app import store
from stresslab.ui import styles


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Overview", "Latest activity and quick start checklist")

    runs = store.list_runs(config.sqlite_path)
    portfolios = store.list_portfolios(config.sqlite_path)
    scenarios = store.list_scenarios(config.sqlite_path)

    latest_run = runs[0] if runs else None
    latest_run_name = latest_run["name"] if latest_run else "—"
    latest_run_ts = latest_run["created_at"] if latest_run else "—"

    kpi_cards = [
        styles.kpi_card("Portfolios", str(len(portfolios)), col_span=3),
        styles.kpi_card("Scenarios", str(len(scenarios)), col_span=3),
        styles.kpi_card("Runs", str(len(runs)), col_span=3),
        styles.kpi_card("Latest Run", latest_run_name, sub=latest_run_ts, col_span=3),
    ]
    styles.kpi_row(kpi_cards)

    st.divider()

    cols = st.columns([1.3, 1])
    with cols[0]:
        styles.section_header("Quick Start", "End-to-end steps")
        st.markdown(
            """
            1. Upload or build a portfolio in **Portfolios**.
            2. Load market data (CSV or yfinance) in **Market Data**.
            3. Create a scenario in **Scenarios**.
            4. Configure and run stress tests in **Run Stress**.
            5. Review baseline analytics in **Risk Analytics** and export in **Reports**.
            """
        )

    with cols[1]:
        styles.section_header("System Status", "Data and session state")
        dataset_loaded = "Yes" if st.session_state.get("market_data") is not None else "No"
        portfolio_loaded = "Yes" if st.session_state.get("active_portfolio_id") else "No"
        scenario_loaded = "Yes" if st.session_state.get("active_scenario_id") else "No"
        styles.info_card(
            "Current Context",
            f\"Dataset loaded: {dataset_loaded}\\nActive portfolio: {portfolio_loaded}\\nActive scenario: {scenario_loaded}\",
            subtle=False,
        )

    if latest_run:
        st.divider()
        styles.section_header("Last Run Snapshot", "Saved configuration and results")
        st.json(
            {
                "run_id": latest_run["run_id"],
                "portfolio_id": latest_run["portfolio_id"],
                "scenario_id": latest_run["scenario_id"],
                "created_at": latest_run["created_at"],
                "metrics": latest_run["outputs"].get("metrics", {}),
            }
        )
