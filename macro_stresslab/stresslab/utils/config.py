"""
stresslab/utils/config.py
=========================
App configuration layer.

Goals:
- Single source of truth for configuration
- Clear defaults for local dev
- Env-based overrides for deployment
- Explicit typing + validation (no silent misconfig)
- Future-proof (feature flags, secret requirements, DB paths)

Usage:
    from utils.config import load_config
    cfg = load_config()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _env_str(key: str, default: str) -> str:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip()


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _normalize_path(p: str) -> str:
    # Expand ~ and environment variables, then make absolute relative to cwd.
    path = Path(os.path.expandvars(os.path.expanduser(p)))
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return str(path)


@dataclass(frozen=True)
class AppConfig:
    # -----------------------------
    # Identity / environment
    # -----------------------------
    environment: str = "local"  # local | dev | staging | prod
    debug: bool = True
    build_tag: str = "dev"

    # -----------------------------
    # Storage / data
    # -----------------------------
    data_dir: str = field(default_factory=lambda: _normalize_path("./.stresslab_data"))
    # SQLite path (we'll use later for portfolios, scenario library, run history)
    sqlite_path: str = field(default_factory=lambda: _normalize_path("./.stresslab_data/stresslab.db"))
    # Artifacts dir for exports (PDF/CSV/Parquet) and run bundles
    artifacts_dir: str = field(default_factory=lambda: _normalize_path("./.stresslab_data/artifacts"))

    # -----------------------------
    # App behavior flags
    # -----------------------------
    # If True, we enforce required_secret_keys presence.
    require_secrets: bool = False
    required_secret_keys: List[str] = field(default_factory=list)

    # -----------------------------
    # External data providers (future)
    # -----------------------------
    # We keep these here to avoid scattering env reads throughout the app.
    fred_api_key: Optional[str] = None

    # -----------------------------
    # Performance / caching
    # -----------------------------
    # Allows us to centralize caching behavior later
    cache_ttl_seconds: int = 3600
    max_rows_preview: int = 5000

    # -----------------------------
    # Guardrails / limits (important)
    # -----------------------------
    max_upload_mb: int = 50
    max_portfolio_positions: int = 25000
    max_scenario_shocks: int = 5000

    # -----------------------------
    # UI preferences
    # -----------------------------
    default_theme: str = "dark"  # dark | light (we’ll implement dark now)
    show_experimental_panels: bool = False


def _infer_env_defaults(environment: str) -> dict:
    """
    Environment-driven defaults. We still allow explicit env overrides.
    """
    env = environment.lower().strip()
    if env in {"prod", "production"}:
        return {
            "debug": False,
            "require_secrets": True,
            "build_tag": _env_str("STRESSLAB_BUILD_TAG", "prod"),
        }
    if env in {"staging"}:
        return {
            "debug": False,
            "require_secrets": True,
            "build_tag": _env_str("STRESSLAB_BUILD_TAG", "staging"),
        }
    if env in {"dev"}:
        return {
            "debug": True,
            "require_secrets": False,
            "build_tag": _env_str("STRESSLAB_BUILD_TAG", "dev"),
        }
    # local
    return {
        "debug": True,
        "require_secrets": False,
        "build_tag": _env_str("STRESSLAB_BUILD_TAG", "local"),
    }


def load_config() -> AppConfig:
    """
    Load configuration from environment variables with safe defaults.

    Supported environment variables:
        STRESSLAB_ENV                 (default: local)
        STRESSLAB_DEBUG               (true/false)
        STRESSLAB_BUILD_TAG           (string)

        STRESSLAB_DATA_DIR            (path)
        STRESSLAB_SQLITE_PATH         (path)
        STRESSLAB_ARTIFACTS_DIR       (path)

        STRESSLAB_REQUIRE_SECRETS     (true/false)
        STRESSLAB_REQUIRED_SECRETS    (comma-separated list)
        FRED_API_KEY                  (string)

        STRESSLAB_CACHE_TTL_SECONDS   (int)
        STRESSLAB_MAX_ROWS_PREVIEW    (int)

        STRESSLAB_MAX_UPLOAD_MB       (int)
        STRESSLAB_MAX_POSITIONS       (int)
        STRESSLAB_MAX_SCENARIO_SHOCKS (int)

        STRESSLAB_THEME               (dark/light)
        STRESSLAB_EXPERIMENTAL        (true/false)
    """
    environment = _env_str("STRESSLAB_ENV", "local")
    env_defaults = _infer_env_defaults(environment)

    # Debug: explicit env overrides env defaults
    debug = _env_bool("STRESSLAB_DEBUG", env_defaults["debug"])

    build_tag = _env_str("STRESSLAB_BUILD_TAG", env_defaults["build_tag"])

    data_dir = _normalize_path(_env_str("STRESSLAB_DATA_DIR", "./.stresslab_data"))
    sqlite_path = _normalize_path(_env_str("STRESSLAB_SQLITE_PATH", f"{data_dir}/stresslab.db"))
    artifacts_dir = _normalize_path(_env_str("STRESSLAB_ARTIFACTS_DIR", f"{data_dir}/artifacts"))

    require_secrets = _env_bool("STRESSLAB_REQUIRE_SECRETS", env_defaults["require_secrets"])
    required_secret_keys_raw = _env_str("STRESSLAB_REQUIRED_SECRETS", "")
    required_secret_keys = [
        s.strip() for s in required_secret_keys_raw.split(",") if s.strip()
    ]

    # External providers
    fred_api_key = os.getenv("FRED_API_KEY")
    if fred_api_key is not None:
        fred_api_key = fred_api_key.strip() or None

    cache_ttl_seconds = _env_int("STRESSLAB_CACHE_TTL_SECONDS", 3600)
    max_rows_preview = _env_int("STRESSLAB_MAX_ROWS_PREVIEW", 5000)

    max_upload_mb = _env_int("STRESSLAB_MAX_UPLOAD_MB", 50)
    max_positions = _env_int("STRESSLAB_MAX_POSITIONS", 25000)
    max_scenario_shocks = _env_int("STRESSLAB_MAX_SCENARIO_SHOCKS", 5000)

    theme = _env_str("STRESSLAB_THEME", "dark").lower().strip()
    if theme not in {"dark", "light"}:
        theme = "dark"

    show_experimental = _env_bool("STRESSLAB_EXPERIMENTAL", False)

    # Ensure directories exist (don’t fail silently)
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    return AppConfig(
        environment=environment,
        debug=debug,
        build_tag=build_tag,
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        artifacts_dir=artifacts_dir,
        require_secrets=require_secrets,
        required_secret_keys=required_secret_keys,
        fred_api_key=fred_api_key,
        cache_ttl_seconds=cache_ttl_seconds,
        max_rows_preview=max_rows_preview,
        max_upload_mb=max_upload_mb,
        max_portfolio_positions=max_positions,
        max_scenario_shocks=max_scenario_shocks,
        default_theme=theme,
        show_experimental_panels=show_experimental,
    )
