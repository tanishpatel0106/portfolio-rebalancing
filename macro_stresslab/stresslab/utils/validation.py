"""
stresslab/utils/validation.py
============================
Validation helpers for holdings, scenarios, and datasets.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


REQUIRED_HOLDING_COLUMNS = [
    "asset_id",
    "asset_name",
    "asset_type",
    "currency",
]

OPTIONAL_COLUMNS = ["sector", "region", "quantity", "price", "notional", "weight"]


class ValidationError(ValueError):
    """User-facing validation error."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def validate_holdings(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and normalize a holdings DataFrame.

    Returns (clean_df, warnings).
    """
    if df is None or df.empty:
        raise ValidationError("Holdings table is empty.")

    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_HOLDING_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    warnings: List[str] = []

    df = df[REQUIRED_HOLDING_COLUMNS + OPTIONAL_COLUMNS].copy()

    df["asset_id"] = df["asset_id"].astype(str).str.strip()
    df["asset_name"] = df["asset_name"].astype(str).str.strip()
    df["asset_type"] = df["asset_type"].astype(str).str.strip().str.lower()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()

    if df["asset_id"].eq("").any():
        raise ValidationError("asset_id cannot be blank.")

    if df["currency"].eq("").any():
        raise ValidationError("currency cannot be blank.")

    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    has_notional = df["notional"].notna().any()
    has_qty_price = df["quantity"].notna().any() and df["price"].notna().any()
    if not has_notional and not has_qty_price:
        raise ValidationError("Provide either notional or quantity + price columns.")

    if not has_notional:
        df["notional"] = df["quantity"].fillna(0.0) * df["price"].fillna(0.0)
        warnings.append("Computed notional from quantity * price.")

    if df["weight"].isna().all():
        total_notional = df["notional"].sum()
        if total_notional <= 0:
            raise ValidationError("Total notional must be > 0 to compute weights.")
        df["weight"] = df["notional"] / total_notional
        warnings.append("Computed weights from notional.")

    weight_sum = float(df["weight"].sum())
    if not np.isclose(weight_sum, 1.0, atol=1e-4):
        df["weight"] = df["weight"] / weight_sum
        warnings.append("Weights normalized to sum to 1.")

    duplicates = df.duplicated(subset=["asset_id", "currency"], keep=False)
    if duplicates.any():
        warnings.append("Duplicate asset_id entries detected; consolidating positions.")
        df = (
            df.groupby(["asset_id", "currency"], as_index=False)
            .agg(
                {
                    "asset_name": "first",
                    "asset_type": "first",
                    "quantity": "sum",
                    "price": "mean",
                    "notional": "sum",
                    "weight": "sum",
                    "sector": "first",
                    "region": "first",
                }
            )
            .copy()
        )
        df["weight"] = df["weight"] / df["weight"].sum()

    return df, warnings


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missingness and coverage summary."""
    if df is None or df.empty:
        raise ValidationError("Dataset is empty.")
    summary = pd.DataFrame(index=df.columns)
    summary["missing_pct"] = df.isna().mean() * 100.0
    summary["coverage_pct"] = 100.0 - summary["missing_pct"]
    summary["min"] = df.min(numeric_only=True)
    summary["max"] = df.max(numeric_only=True)
    return summary.sort_values("missing_pct", ascending=False)
