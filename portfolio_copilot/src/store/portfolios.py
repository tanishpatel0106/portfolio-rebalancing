from __future__ import annotations

import pandas as pd

from .persistence import PORTFOLIO_DIR, ensure_storage
from ..config.schemas import HOLDINGS_COLUMNS


def list_portfolios() -> list[str]:
    ensure_storage()
    return sorted([p.stem for p in PORTFOLIO_DIR.glob("*.csv")])


def load_portfolio(name: str) -> pd.DataFrame:
    ensure_storage()
    path = PORTFOLIO_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame(columns=HOLDINGS_COLUMNS)
    return pd.read_csv(path)


def save_portfolio(name: str, holdings: pd.DataFrame) -> None:
    ensure_storage()
    path = PORTFOLIO_DIR / f"{name}.csv"
    holdings.to_csv(path, index=False)


def delete_portfolio(name: str) -> None:
    ensure_storage()
    path = PORTFOLIO_DIR / f"{name}.csv"
    if path.exists():
        path.unlink()
