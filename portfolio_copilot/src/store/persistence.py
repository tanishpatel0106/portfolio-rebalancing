from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
LOCAL_DATA = BASE_DIR / "local_data"
PORTFOLIO_DIR = LOCAL_DATA / "portfolios"
AUDIT_DIR = LOCAL_DATA / "audit"
SETTINGS_FILE = LOCAL_DATA / "settings.json"


def ensure_storage() -> None:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))
