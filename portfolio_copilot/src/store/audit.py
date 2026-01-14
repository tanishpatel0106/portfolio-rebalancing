from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .persistence import AUDIT_DIR, ensure_storage


def save_audit_run(payload: dict) -> Path:
    ensure_storage()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = AUDIT_DIR / f"rebalance_{timestamp}.json"
    # TODO: extend audit payload with LLM explain JSON and chat what-if context for V1.
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_audit_runs() -> list[dict]:
    ensure_storage()
    runs = []
    for path in sorted(AUDIT_DIR.glob("rebalance_*.json")):
        runs.append(json.loads(path.read_text()))
    return runs
