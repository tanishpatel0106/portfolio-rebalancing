"""
stresslab/ui/Home.py
====================

Streamlit entry point (landing page) for StressLab.

Responsibilities:
- Allow user to load data (CSV upload)
- Define whether input is prices or returns
- Hold app-wide state in st.session_state:
    - raw_df
    - input_kind ("prices" or "returns")
    - portfolio_weights dict
    - starting_capital
    - return_method ("simple" or "log")
- Provide navigation hints

No analytics is computed here (keep it fast + stable).
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from stresslab.utils.logging import get_logger

log = get_logger(__name__)


APP_STATE_KEYS = [
    "raw_df",
    "input_kind",
    "return_method",
    "portfolio_weights",
    "starting_capital",
]


def _init_state() -> None:
    for k in APP_STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None

    if st.session_state["input_kind"] is None:
        st.session_state["input_kind"] = "prices"

    if st.session_state["return_method"] is None:
        st.session_state["return_method"] = "simple"

    if st.session_state["starting_capital"] is None:
        st.session_state["starting_capital"] = 10_000.0

    if st.session_state["portfolio_weights"] is None:
        st.session_state["portfolio_weights"] = {}  # type: ignore


def _read_csv(uploaded_file) -> pd.DataFrame:
    try:
        content = uploaded_file.read()
        # try utf-8; if fails, pandas will error and we show message
        s = io.BytesIO(content)
        df = pd.read_csv(s)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        raise ValueError("CSV was read successfully but is empty.")

    # Require a date column or index-like first column
    # Strategy:
    # 1) If first column looks like dates, use it as index.
    # 2) Else if there's a column named "date" (case-insensitive), use it.
    cols = list(df.columns)
    first_col = cols[0]

    date_col = None
    if first_col:
        try:
            tmp = pd.to_datetime(df[first_col], errors="raise")
            date_col = first_col
        except Exception:
            pass

    if date_col is None:
        for c in cols:
            if str(c).strip().lower() in {"date", "datetime", "time", "timestamp"}:
                date_col = c
                break

    if date_col is None:
        raise ValueError(
            "Could not infer a date column. "
            "Make sure your CSV has a date-like first column or a column named 'date'."
        )

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    # Keep only numeric columns as assets
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="all", axis=1)

    if out.shape[1] == 0:
        raise ValueError("After parsing, there are no numeric asset columns to use.")

    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")

    if out.empty:
        raise ValueError("All rows became empty after cleaning NaNs/infs.")

    return out


def _default_weights_from_df(df: pd.DataFrame) -> Dict[str, float]:
    assets = list(df.columns)
    n = len(assets)
    if n == 0:
        return {}
    w = 1.0 / n
    return {a: float(w) for a in assets}


def _weights_editor(weights: Dict[str, float], assets: Tuple[str, ...]) -> Dict[str, float]:
    st.subheader("Portfolio Weights")

    if not assets:
        st.info("Load data first to define portfolio assets.")
        return weights

    # Build an editable dataframe
    rows = []
    for a in assets:
        rows.append({"asset": a, "weight": float(weights.get(a, 0.0))})
    wdf = pd.DataFrame(rows)

    edited = st.data_editor(
        wdf,
        use_container_width=True,
        hide_index=True,
        column_config={
            "asset": st.column_config.TextColumn(disabled=True),
            "weight": st.column_config.NumberColumn(min_value=None, max_value=None, step=0.01),
        },
        key="weights_editor_df",
    )

    new_w = {str(r["asset"]): float(r["weight"]) for _, r in edited.iterrows()}
    return new_w


def main() -> None:
    st.set_page_config(page_title="StressLab", layout="wide")
    _init_state()

    st.title("StressLab")
    st.caption("Macro stress + risk analytics — data → portfolio → risk → stress → MC → regimes")

    with st.sidebar:
        st.header("Data & Global Settings")

        input_kind = st.selectbox(
            "Input type",
            options=["prices", "returns"],
            index=0 if st.session_state["input_kind"] == "prices" else 1,
            help="If your CSV has price levels, choose 'prices'. If it already has daily returns, choose 'returns'.",
        )
        st.session_state["input_kind"] = input_kind

        return_method = st.selectbox(
            "Return method (if input=prices)",
            options=["simple", "log"],
            index=0 if st.session_state["return_method"] == "simple" else 1,
            help="Only used when converting prices → returns.",
        )
        st.session_state["return_method"] = return_method

        starting_capital = st.number_input(
            "Starting capital",
            min_value=0.0,
            value=float(st.session_state["starting_capital"]),
            step=1000.0,
            help="Used for equity curve / drawdown scaling and pnl-mode normalization.",
        )
        st.session_state["starting_capital"] = float(starting_capital)

        st.divider()
        st.subheader("Load CSV")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

        if uploaded is not None:
            try:
                df = _read_csv(uploaded)
                st.session_state["raw_df"] = df

                # If weights empty OR mismatch, set defaults
                if not st.session_state["portfolio_weights"]:
                    st.session_state["portfolio_weights"] = _default_weights_from_df(df)

                else:
                    # ensure all assets exist
                    w = dict(st.session_state["portfolio_weights"])
                    for a in df.columns:
                        if a not in w:
                            w[a] = 0.0
                    # drop weights for missing assets
                    w = {a: w[a] for a in w.keys() if a in df.columns}
                    st.session_state["portfolio_weights"] = w

                st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} assets.")
            except Exception as e:
                st.error(str(e))

        if st.button("Reset app state", type="secondary"):
            for k in APP_STATE_KEYS:
                st.session_state[k] = None
            st.rerun()

    df: Optional[pd.DataFrame] = st.session_state.get("raw_df", None)

    if df is None:
        st.info("Upload a CSV from the sidebar to begin.")
        st.markdown(
            """
**CSV format expected**
- One date column (first column preferred, or named `date`)
- Remaining columns are assets (prices or returns)

Then go to **pages → 01_Risk** to compute risk metrics.
"""
        )
        return

    st.subheader("Data Preview")
    st.write(df.tail(20))

    st.divider()

    # Weights editor
    assets = tuple(df.columns.astype(str).tolist())
    new_w = _weights_editor(dict(st.session_state["portfolio_weights"]), assets)
    st.session_state["portfolio_weights"] = new_w

    # Diagnostics
    wsum = float(np.sum(list(new_w.values()))) if new_w else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Assets", len(assets))
    col2.metric("Weight Sum", f"{wsum:.6f}")
    col3.metric("Input Kind", st.session_state["input_kind"])

    st.success("Ready. Next: open **pages → 01_Risk**.")


if __name__ == "__main__":
    main()
