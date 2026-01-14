from __future__ import annotations

import streamlit as st


def set_page_config() -> None:
    st.set_page_config(page_title="Portfolio Rebalancing + Strategy Copilot", layout="wide")


def sidebar_header() -> None:
    st.sidebar.title("Portfolio Copilot")
    st.sidebar.caption("MVP with audit-ready rebalancing")
