"""
Settings and diagnostics.
"""
from __future__ import annotations

import streamlit as st

from stresslab.ui import styles


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Settings", "Configuration, cache, and diagnostics")

    st.markdown("#### Paths")
    st.code(
        f"Data dir: {config.data_dir}\nSQLite: {config.sqlite_path}\nArtifacts: {config.artifacts_dir}",
        language="text",
    )

    st.markdown("#### Cache")
    if st.button("Reset cache"):
        st.session_state["cache_bust"] = int(st.session_state.get("cache_bust", 0)) + 1
        st.success("Cache bust applied. Reload data for fresh computations.")

    st.markdown("#### Session State")
    st.json({k: v for k, v in st.session_state.items() if not k.startswith("_toast")})
