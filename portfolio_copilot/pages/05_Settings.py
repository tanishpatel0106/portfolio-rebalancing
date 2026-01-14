from __future__ import annotations

import json

import streamlit as st

from src.config.defaults import DEFAULT_SETTINGS
from src.store.settings import load_settings, save_settings, reset_settings
from src.ui.layout import sidebar_header


st.set_page_config(page_title="Settings", layout="wide")
sidebar_header()

st.title("Settings")

settings = load_settings()

st.subheader("Settings JSON")
json_text = st.text_area("Edit settings", value=json.dumps(settings, indent=2), height=400)

col1, col2 = st.columns(2)
if col1.button("Save Settings", type="primary"):
    try:
        payload = json.loads(json_text)
        save_settings(payload)
        st.success("Settings saved.")
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")

if col2.button("Restore Defaults"):
    reset_settings()
    st.rerun()

st.caption("Settings include constraints, backtest defaults, and strategy defaults.")
