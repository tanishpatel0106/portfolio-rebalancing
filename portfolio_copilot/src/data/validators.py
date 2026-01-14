from __future__ import annotations

import pandas as pd

from ..config.schemas import REQUIRED_HOLDINGS_COLUMNS


def validate_holdings(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if df is None or df.empty:
        errors.append("Holdings table is empty.")
        return False, errors

    for col in REQUIRED_HOLDINGS_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}.")

    if "ticker" in df.columns:
        if df["ticker"].isna().any():
            errors.append("Ticker column contains missing values.")
        if (df["ticker"].astype(str).str.strip() == "").any():
            errors.append("Ticker column contains blank values.")

    if "quantity" in df.columns:
        try:
            qty = pd.to_numeric(df["quantity"], errors="coerce")
        except Exception:
            qty = pd.Series([None] * len(df))
        if qty.isna().any():
            errors.append("Quantity column has non-numeric values.")
        if (qty <= 0).any():
            errors.append("Quantity must be positive for all rows.")

    if "sector" in df.columns:
        if df["sector"].isna().any():
            errors.append("Sector column has missing values; please fill or use 'UNKNOWN'.")

    return len(errors) == 0, errors
