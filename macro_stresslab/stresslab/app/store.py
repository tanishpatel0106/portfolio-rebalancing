"""
stresslab/app/store.py
======================
SQLite persistence layer for StressLab.

Responsibilities:
- Initialize schema
- CRUD for portfolios, holdings, scenarios, datasets, runs
- Store JSON payloads for configs/results

This module is intentionally free of Streamlit imports.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class PortfolioRecord:
    portfolio_id: str
    name: str
    base_currency: str
    description: str
    created_at: str


@dataclass(frozen=True)
class ScenarioRecord:
    scenario_id: str
    name: str
    description: str
    tags: str
    payload: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    source: str
    start_date: str
    end_date: str
    frequency: str
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    portfolio_id: str
    scenario_id: Optional[str]
    config: Dict[str, Any]
    outputs: Dict[str, Any]
    artifacts: Dict[str, Any]
    created_at: str


@contextmanager
def _connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().strftime(ISO_FORMAT)


def init_db(db_path: str) -> None:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                base_currency TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                asset_name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                currency TEXT NOT NULL,
                quantity REAL,
                price REAL,
                notional REAL,
                weight REAL,
                sector TEXT,
                region TEXT,
                FOREIGN KEY (portfolio_id) REFERENCES portfolios (portfolio_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                scenario_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                frequency TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                portfolio_id TEXT NOT NULL,
                scenario_id TEXT,
                config TEXT NOT NULL,
                outputs TEXT NOT NULL,
                artifacts TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _from_json(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _uuid() -> str:
    return uuid.uuid4().hex[:12]


# ----------------------
# Portfolios & holdings
# ----------------------

def create_portfolio(
    db_path: str,
    *,
    name: str,
    base_currency: str,
    description: str,
    holdings: Iterable[Dict[str, Any]],
) -> str:
    portfolio_id = _uuid()
    created_at = _now()
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO portfolios (portfolio_id, name, base_currency, description, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (portfolio_id, name, base_currency, description, created_at),
        )
        for row in holdings:
            cur.execute(
                """
                INSERT INTO holdings (
                    portfolio_id, asset_id, asset_name, asset_type, currency,
                    quantity, price, notional, weight, sector, region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    row.get("asset_id"),
                    row.get("asset_name"),
                    row.get("asset_type"),
                    row.get("currency"),
                    row.get("quantity"),
                    row.get("price"),
                    row.get("notional"),
                    row.get("weight"),
                    row.get("sector"),
                    row.get("region"),
                ),
            )
        conn.commit()
    return portfolio_id


def replace_holdings(db_path: str, portfolio_id: str, holdings: Iterable[Dict[str, Any]]) -> None:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM holdings WHERE portfolio_id = ?", (portfolio_id,))
        for row in holdings:
            cur.execute(
                """
                INSERT INTO holdings (
                    portfolio_id, asset_id, asset_name, asset_type, currency,
                    quantity, price, notional, weight, sector, region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    portfolio_id,
                    row.get("asset_id"),
                    row.get("asset_name"),
                    row.get("asset_type"),
                    row.get("currency"),
                    row.get("quantity"),
                    row.get("price"),
                    row.get("notional"),
                    row.get("weight"),
                    row.get("sector"),
                    row.get("region"),
                ),
            )
        conn.commit()


def list_portfolios(db_path: str) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute("SELECT * FROM portfolios ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_portfolio(db_path: str, portfolio_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        row = cur.execute("SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,)).fetchone()
    return dict(row) if row else None


def get_holdings(db_path: str, portfolio_id: str) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT * FROM holdings WHERE portfolio_id = ? ORDER BY asset_id",
            (portfolio_id,),
        ).fetchall()
    return [dict(row) for row in rows]


# ----------------------
# Scenarios
# ----------------------

def create_scenario(
    db_path: str,
    *,
    name: str,
    description: str,
    tags: str,
    payload: Dict[str, Any],
) -> str:
    scenario_id = _uuid()
    created_at = _now()
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scenarios (scenario_id, name, description, tags, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scenario_id, name, description, tags, _to_json(payload), created_at),
        )
        conn.commit()
    return scenario_id


def list_scenarios(db_path: str) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM scenarios ORDER BY created_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["payload"] = _from_json(item.get("payload"))
        out.append(item)
    return out


def get_scenario(db_path: str, scenario_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["payload"] = _from_json(item.get("payload"))
    return item


# ----------------------
# Datasets
# ----------------------

def create_dataset(
    db_path: str,
    *,
    name: str,
    source: str,
    start_date: str,
    end_date: str,
    frequency: str,
    metadata: Dict[str, Any],
) -> str:
    dataset_id = _uuid()
    created_at = _now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO datasets (dataset_id, name, source, start_date, end_date, frequency, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, name, source, start_date, end_date, frequency, _to_json(metadata), created_at),
        )
        conn.commit()
    return dataset_id


def list_datasets(db_path: str) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["metadata"] = _from_json(item.get("metadata"))
        out.append(item)
    return out


def get_dataset(db_path: str, dataset_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["metadata"] = _from_json(item.get("metadata"))
    return item


# ----------------------
# Runs
# ----------------------

def create_run(
    db_path: str,
    *,
    name: str,
    portfolio_id: str,
    scenario_id: Optional[str],
    config: Dict[str, Any],
    outputs: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> str:
    run_id = _uuid()
    created_at = _now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, name, portfolio_id, scenario_id, config, outputs, artifacts, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                name,
                portfolio_id,
                scenario_id,
                _to_json(config),
                _to_json(outputs),
                _to_json(artifacts),
                created_at,
            ),
        )
        conn.commit()
    return run_id


def list_runs(db_path: str) -> List[Dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["config"] = _from_json(item.get("config"))
        item["outputs"] = _from_json(item.get("outputs"))
        item["artifacts"] = _from_json(item.get("artifacts"))
        out.append(item)
    return out


def get_run(db_path: str, run_id: str) -> Optional[Dict[str, Any]]:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["config"] = _from_json(item.get("config"))
    item["outputs"] = _from_json(item.get("outputs"))
    item["artifacts"] = _from_json(item.get("artifacts"))
    return item
