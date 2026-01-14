from __future__ import annotations

import streamlit as st


def kpi_cards(items: list[dict]) -> None:
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.metric(item["label"], item.get("value", "-"), item.get("delta"))


def warning_box(message: str) -> None:
    st.warning(message)


def info_box(message: str) -> None:
    st.info(message)
