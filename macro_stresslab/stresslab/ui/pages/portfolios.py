"""
Portfolio builder and manager.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from stresslab.app import store
from stresslab.ui import styles
from stresslab.ui.components import toast_success, toast_error
from stresslab.utils.validation import validate_holdings, ValidationError


DEFAULT_TEMPLATE = pd.DataFrame(
    [
        {
            "asset_id": "AAPL",
            "asset_name": "Apple",
            "asset_type": "equity",
            "currency": "USD",
            "quantity": 10,
            "price": 190.0,
            "notional": None,
            "weight": None,
            "sector": "Technology",
            "region": "US",
        },
        {
            "asset_id": "TLT",
            "asset_name": "iShares 20+ Year Treasury",
            "asset_type": "rates",
            "currency": "USD",
            "quantity": 15,
            "price": 93.0,
            "notional": None,
            "weight": None,
            "sector": "Rates",
            "region": "US",
        },
    ]
)


def _load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Portfolio Builder", "Upload holdings or edit in-app")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Upload Holdings CSV")
        uploaded = st.file_uploader("Holdings CSV", type=["csv"], key="portfolio_upload")
        if uploaded is not None:
            try:
                raw_df = _load_csv(uploaded)
                st.session_state["portfolio_cache"] = raw_df
                toast_success("Holdings CSV loaded. Review and save below.")
            except Exception as exc:
                toast_error(f"Failed to load CSV: {exc}")

        st.markdown("#### Edit Holdings")
        df = st.session_state.get("portfolio_cache")
        if df is None:
            df = DEFAULT_TEMPLATE.copy()
        editor = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="holdings_editor",
        )
        st.session_state["portfolio_cache"] = editor

    with col_right:
        st.markdown("#### Portfolio Metadata")
        name = st.text_input("Portfolio name", value="Macro Multi-Asset")
        base_currency = st.text_input("Base currency", value="USD")
        description = st.text_area("Description", value="Core macro sleeve")
        save_as_new = st.toggle("Save as new version", value=True)

        if st.button("💾 Save Portfolio"):
            try:
                clean_df, warnings = validate_holdings(editor)
                holdings = clean_df.to_dict(orient="records")
                save_name = name
                if save_as_new:
                    save_name = f"{name} (v{len(store.list_portfolios(config.sqlite_path)) + 1})"
                portfolio_id = store.create_portfolio(
                    config.sqlite_path,
                    name=save_name,
                    base_currency=base_currency,
                    description=description,
                    holdings=holdings,
                )
                st.session_state["active_portfolio_id"] = portfolio_id
                toast_success(f"Saved portfolio {save_name}.")
                if warnings:
                    st.warning("; ".join(warnings))
            except ValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to save portfolio: {exc}")

    st.divider()

    styles.section_header("Saved Portfolios", "Review and update existing portfolios")
    portfolios = store.list_portfolios(config.sqlite_path)
    if not portfolios:
        st.info("No portfolios saved yet.")
        return

    portfolio_options = {f"{p['name']} ({p['portfolio_id']})": p for p in portfolios}
    selected_label = st.selectbox("Select portfolio", list(portfolio_options.keys()))
    selected = portfolio_options[selected_label]
    holdings = store.get_holdings(config.sqlite_path, selected["portfolio_id"])

    if holdings:
        holdings_df = pd.DataFrame(holdings).drop(columns=["id", "portfolio_id"])
        st.dataframe(holdings_df, use_container_width=True)

        if st.button("Update holdings with current editor"):
            try:
                clean_df, warnings = validate_holdings(editor)
                store.replace_holdings(
                    config.sqlite_path,
                    selected["portfolio_id"],
                    clean_df.to_dict(orient="records"),
                )
                toast_success("Holdings updated.")
                if warnings:
                    st.warning("; ".join(warnings))
            except ValidationError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Failed to update holdings: {exc}")
