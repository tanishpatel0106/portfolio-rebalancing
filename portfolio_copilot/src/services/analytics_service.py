from __future__ import annotations

import pandas as pd

from ..store.audit import load_audit_runs


def get_rebalance_history() -> pd.DataFrame:
    runs = load_audit_runs()
    if not runs:
        return pd.DataFrame()
    rows = []
    for run in runs:
        rows.append(
            {
                "timestamp": run.get("timestamp"),
                "portfolio": run.get("portfolio_name"),
                "turnover": run.get("metrics", {}).get("turnover", 0.0),
                "drift_pre": run.get("metrics", {}).get("drift_pre", 0.0),
                "drift_post": run.get("metrics", {}).get("drift_post", 0.0),
                "cost_proxy": run.get("metrics", {}).get("cost_proxy", 0.0),
            }
        )
    return pd.DataFrame(rows)
