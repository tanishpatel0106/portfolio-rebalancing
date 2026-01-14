from __future__ import annotations

import streamlit as st
import pandas as pd

from src.config.defaults import DEFAULT_PORTFOLIO
from src.config.schemas import HOLDINGS_COLUMNS
from src.data.validators import validate_holdings
from src.store.portfolios import delete_portfolio, list_portfolios, load_portfolio, save_portfolio
from src.ui.layout import sidebar_header
from src.ui.components import warning_box


st.set_page_config(page_title="Portfolio Builder", layout="wide")
sidebar_header()

st.title("Portfolio Builder")

portfolio_names = list_portfolios()
selected = st.sidebar.selectbox("Load Portfolio", options=["<new>"] + portfolio_names)
portfolio_name = st.sidebar.text_input("Portfolio Name", value="New Portfolio" if selected == "<new>" else selected)

if selected != "<new>":
    holdings_df = load_portfolio(selected)
else:
    holdings_df = pd.DataFrame(DEFAULT_PORTFOLIO["holdings"], columns=HOLDINGS_COLUMNS)

st.subheader("Holdings")
editable_df = st.data_editor(holdings_df, num_rows="dynamic", use_container_width=True)

valid, errors = validate_holdings(editable_df)
if errors:
    for err in errors:
        warning_box(err)

col1, col2, col3 = st.columns(3)
if col1.button("Save Portfolio", type="primary", disabled=not valid):
    save_portfolio(portfolio_name, editable_df)
    st.success("Portfolio saved.")

if col2.button("Load Sample"):
    editable_df = pd.DataFrame(DEFAULT_PORTFOLIO["holdings"], columns=HOLDINGS_COLUMNS)
    st.rerun()

if col3.button("Delete Portfolio", disabled=selected == "<new>"):
    delete_portfolio(selected)
    st.success("Portfolio deleted.")

st.caption("Holdings schema: ticker, quantity, sector (recommended), asset_class, currency.")
