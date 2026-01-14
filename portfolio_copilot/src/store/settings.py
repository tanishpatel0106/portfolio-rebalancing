from __future__ import annotations

from .persistence import SETTINGS_FILE, ensure_storage, read_json, write_json
from ..config.defaults import DEFAULT_SETTINGS


def load_settings() -> dict:
    ensure_storage()
    stored = read_json(SETTINGS_FILE)
    if not stored:
        write_json(SETTINGS_FILE, DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS
    return stored


def save_settings(settings: dict) -> None:
    ensure_storage()
    write_json(SETTINGS_FILE, settings)


def reset_settings() -> dict:
    ensure_storage()
    write_json(SETTINGS_FILE, DEFAULT_SETTINGS)
    return DEFAULT_SETTINGS
