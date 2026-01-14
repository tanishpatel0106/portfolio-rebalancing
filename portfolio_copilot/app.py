from __future__ import annotations

import streamlit as st

from src.ui.layout import set_page_config, sidebar_header


set_page_config()
sidebar_header()

st.title("Portfolio Rebalancing + Strategy Copilot")
st.markdown(
    """
Welcome to the MVP for an audit-ready portfolio rebalancing and strategy copilot.
Use the sidebar to build portfolios, run analytics, and approve rebalancing runs.
"""
)

st.success("Select a page from the sidebar to get started.")
