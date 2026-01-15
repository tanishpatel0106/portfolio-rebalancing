"""
Market data configuration and validation.
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from stresslab.app import store
from stresslab.data import sources
from stresslab.ui import styles
from stresslab.ui.components import plotly_defaults, toast_success, toast_error
from stresslab.utils.validation import data_quality_summary


@st.cache_data(ttl=3600)
def _load_csv(file) -> pd.DataFrame:
    content = file.read()
    df = pd.read_csv(io.BytesIO(content))
    return df


@st.cache_data(ttl=3600)
def _load_yfinance(ticker: str, start: date, end: date, timeframe: str) -> pd.DataFrame:
    df = sources.get_prices(ticker, start, end, timeframe=timeframe)
    return df


def _extract_prices(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.sort_index()
    return df


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Market Data", "Load time series data with checks and caching")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("#### Source")
        source = st.radio("Data source", options=["CSV Upload", "yfinance"], horizontal=True)

        if source == "CSV Upload":
            with st.form("csv_load_form"):
                uploaded = st.file_uploader("Upload time series CSV", type=["csv"], key="market_upload")
                submit_csv = st.form_submit_button("Load CSV")
            if uploaded is not None and submit_csv:
                try:
                    raw = _load_csv(uploaded)
                    prices = _extract_prices(raw)
                    st.session_state["market_data"] = prices
                    st.session_state["market_meta"] = {
                        "source": "csv",
                        "rows": len(prices),
                        "columns": list(prices.columns),
                    }
                    toast_success("Market data loaded from CSV.")
                except Exception as exc:
                    toast_error(f"Failed to load CSV: {exc}")
        else:
            with st.form("yf_load_form"):
                ticker = st.text_input("Ticker", value="SPY")
                start = st.date_input("Start date", value=date(2018, 1, 1))
                end = st.date_input("End date", value=date.today())
                timeframe = st.selectbox("Frequency", options=["Daily", "Weekly", "Monthly", "1H"])
                submit_yf = st.form_submit_button("Fetch from yfinance")
            if submit_yf:
                try:
                    df = _load_yfinance(ticker, start, end, timeframe)
                    prices = df[["Close"]].rename(columns={"Close": ticker})
                    st.session_state["market_data"] = prices
                    st.session_state["market_meta"] = {
                        "source": "yfinance",
                        "ticker": ticker,
                        "rows": len(prices),
                        "columns": list(prices.columns),
                    }
                    toast_success("Market data loaded from yfinance.")
                except Exception as exc:
                    toast_error(f"yfinance error: {exc}")

        st.markdown("#### Factor Data")
        with st.form("factor_upload_form"):
            factor_upload = st.file_uploader("Upload factor returns CSV", type=["csv"], key="factor_upload")
            beta_upload = st.file_uploader("Upload factor betas CSV", type=["csv"], key="beta_upload")
            submit_factor = st.form_submit_button("Load factor data")
        if submit_factor:
            if factor_upload is not None:
                try:
                    factor_df = _load_csv(factor_upload)
                    factor_df = _extract_prices(factor_df)
                    st.session_state["factor_returns"] = factor_df
                    toast_success("Factor returns loaded.")
                except Exception as exc:
                    toast_error(f"Failed to load factor returns: {exc}")
            if beta_upload is not None:
                try:
                    betas = pd.read_csv(beta_upload, index_col=0)
                    st.session_state["factor_betas"] = betas
                    toast_success("Factor betas loaded.")
                except Exception as exc:
                    toast_error(f"Failed to load betas: {exc}")

    with col_right:
        st.markdown("#### Dataset Registry")
        if st.session_state.get("market_data") is not None:
            data = st.session_state["market_data"]
            name = st.text_input("Dataset name", value="Primary Market Data")
            if st.button("Register dataset"):
                try:
                    dataset_id = store.create_dataset(
                        config.sqlite_path,
                        name=name,
                        source=st.session_state["market_meta"].get("source", "unknown"),
                        start_date=str(data.index.min().date()),
                        end_date=str(data.index.max().date()),
                        frequency=pd.infer_freq(data.index) or "unknown",
                        metadata=st.session_state.get("market_meta", {}),
                    )
                    st.session_state["active_dataset_id"] = dataset_id
                    toast_success("Dataset registered.")
                except Exception as exc:
                    toast_error(f"Failed to register dataset: {exc}")

        datasets = store.list_datasets(config.sqlite_path)
        if datasets:
            st.dataframe(pd.DataFrame(datasets), use_container_width=True)

    if st.session_state.get("market_data") is None:
        st.info("Load market data to view quality checks.")
        return

    st.divider()
    styles.section_header("Data Quality", "Coverage, previews, and missingness diagnostics")
    data = st.session_state["market_data"]
    summary = data_quality_summary(data)
    st.dataframe(summary, use_container_width=True)

    st.markdown("#### Preview")
    st.dataframe(data.tail(10), use_container_width=True)

    missing = data.isna().astype(int)
    fig = px.imshow(missing.T, aspect="auto", color_continuous_scale="RdBu", title="Missingness Heatmap")
    st.plotly_chart(plotly_defaults(fig, height=400), use_container_width=True)
