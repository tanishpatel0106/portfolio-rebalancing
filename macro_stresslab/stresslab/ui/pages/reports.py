"""
Reports and run history.
"""
from __future__ import annotations

import csv
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from stresslab.app import store
from stresslab.ui import styles
from stresslab.ui.components import toast_success


def _build_pdf(run: dict) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, height - 40, "StressLab Report")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 70, f"Run ID: {run['run_id']}")
    c.drawString(40, height - 85, f"Portfolio: {run['portfolio_id']}")
    c.drawString(40, height - 100, f"Scenario: {run.get('scenario_id') or 'None'}")
    c.drawString(40, height - 115, f"Timestamp: {run['created_at']}")

    metrics = run.get("outputs", {}).get("metrics", {})
    y = height - 150
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Key Metrics")
    y -= 20
    c.setFont("Helvetica", 9)
    for key, val in metrics.items():
        c.drawString(40, y, f"{key}: {val}")
        y -= 14
        if y < 80:
            c.showPage()
            y = height - 50

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()


def render_page(ctx) -> None:
    config = ctx.config
    styles.section_header("Reports & History", "Review prior runs and export")

    runs = store.list_runs(config.sqlite_path)
    if not runs:
        st.info("No runs available yet.")
        return

    portfolio_filter = st.text_input("Filter by portfolio id", value="")
    scenario_filter = st.text_input("Filter by scenario id", value="")

    filtered = []
    for run in runs:
        if portfolio_filter and portfolio_filter not in run["portfolio_id"]:
            continue
        if scenario_filter and scenario_filter not in (run.get("scenario_id") or ""):
            continue
        filtered.append(run)

    df = pd.DataFrame(filtered)
    st.dataframe(df[["run_id", "name", "portfolio_id", "scenario_id", "created_at"]], use_container_width=True)

    selected_run_id = st.selectbox("Select run", df["run_id"].tolist())
    selected = store.get_run(config.sqlite_path, selected_run_id)
    if not selected:
        return

    st.json(selected["outputs"])

    st.markdown("#### Export")
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["metric", "value"])
    for key, val in selected.get("outputs", {}).get("metrics", {}).items():
        writer.writerow([key, val])

    st.download_button(
        "Download metrics CSV",
        data=csv_buffer.getvalue(),
        file_name=f"stresslab_metrics_{selected_run_id}.csv",
        mime="text/csv",
    )

    pdf_bytes = _build_pdf(selected)
    st.download_button(
        "Download PDF report",
        data=pdf_bytes,
        file_name=f"stresslab_report_{selected_run_id}.pdf",
        mime="application/pdf",
    )

    if st.button("Save report artifact"):
        artifacts_dir = Path(config.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = artifacts_dir / f"stresslab_report_{selected_run_id}.pdf"
        pdf_path.write_bytes(pdf_bytes)
        toast_success(f"Report saved to {pdf_path}")
