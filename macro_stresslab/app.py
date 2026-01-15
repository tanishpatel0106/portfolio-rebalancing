"""
stresslab/app.py
================
Main entrypoint for Macro StressLab / StressLab.

This file is responsible for:
- Streamlit app configuration
- Global theme & layout bootstrapping
- Centralized navigation/routing
- Session state initialization
- Cross-page utilities (toasts, error boundary, debug panel)
- App-level dependency checks
- App-level logging integration

NOTE:
- Business logic must NOT live here. It belongs in core/ and services/.
- UI components must come from ui/components.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Local imports (must exist; we will implement these files next)
# ------------------------------------------------------------
# These are intentionally imported at top-level to fail-fast in dev
# and to ensure we keep a clean dependency graph.
from stresslab.ui.theme import apply_theme  # noqa: F401
from stresslab.ui.components import (
    render_app_header,
    render_footer,
    render_section_header,
    toast_success,
    toast_warning,
    toast_error,
    ui_divider,
)
from stresslab.utils.config import AppConfig, load_config
from stresslab.utils.logging import get_logger, log_exception
from stresslab.app import store
from stresslab.data import demo


# ============================================================
# 1) Streamlit page config (MUST be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="StressLab — Macro Stress Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2) Globals
# ============================================================
APP_ROOT = Path(__file__).resolve().parent  # macro_stresslab/
PAGES_DIR = APP_ROOT / "stresslab" / "ui" / "pages"
LOGGER = get_logger("stresslab.app")

DEFAULT_ROUTE = "Home"


# ============================================================
# 3) Routing primitives
# ============================================================
@dataclass(frozen=True)
class Route:
    key: str
    title: str
    page_module: str  # e.g. "ui.pages.00_Home"
    icon: str = "📄"
    description: str = ""


def _safe_import_page(module_path: str):
    """
    Import a page module safely.
    Requirements:
      - Module must define a function: `render_page(ctx: AppContext) -> None`
    """
    import importlib

    mod = importlib.import_module(module_path)
    if not hasattr(mod, "render_page"):
        raise AttributeError(
            f"Page module '{module_path}' must define a function render_page(ctx)."
        )
    return mod


# ============================================================
# 4) App context shared across pages
# ============================================================
@dataclass
class AppContext:
    config: AppConfig
    logger_name: str
    app_start_ts: float

    # For future expansion:
    # - db/session handles
    # - current user
    # - feature flags
    # - caching layer handles


def build_context(config: AppConfig) -> AppContext:
    return AppContext(
        config=config,
        logger_name="stresslab",
        app_start_ts=st.session_state.get("_app_start_ts", time.time()),
    )


# ============================================================
# 5) Dependency & environment checks (no corners)
# ============================================================
def check_environment(config: AppConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Return a dict of checks:
      { "Check Name": (passed, message) }

    This does NOT stop the app by default. It reports to a health panel.
    """
    checks: Dict[str, Tuple[bool, str]] = {}

    # Python version check
    py_ok = sys.version_info >= (3, 10)
    checks["Python >= 3.10"] = (
        py_ok,
        f"Detected {sys.version.split()[0]}",
    )

    # Writable data directory check (for SQLite, cached artifacts, exports)
    data_dir = Path(config.data_dir)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        checks["Data directory writable"] = (True, str(data_dir))
    except Exception as e:
        checks["Data directory writable"] = (False, f"{data_dir} — {repr(e)}")

    # Secrets check (optional; we won’t block dev)
    # Example: FRED / AlphaVantage / etc. later
    if config.require_secrets:
        missing = []
        for key in config.required_secret_keys:
            if not os.getenv(key):
                missing.append(key)
        if missing:
            checks["Required secrets present"] = (False, "Missing: " + ", ".join(missing))
        else:
            checks["Required secrets present"] = (True, "All present")
    else:
        checks["Required secrets present"] = (True, "Not required in this environment")

    # Pages directory check
    pages_ok = PAGES_DIR.exists()
    checks["Pages directory exists"] = (pages_ok, str(PAGES_DIR))

    return checks


def render_health_panel(checks: Dict[str, Tuple[bool, str]]) -> None:
    """
    Sidebar health/status panel. Visible when Debug mode enabled.
    """
    st.sidebar.markdown("### 🩺 Health")
    passed = sum(1 for v in checks.values() if v[0])
    total = len(checks)
    st.sidebar.caption(f"{passed}/{total} checks passed")

    for name, (ok, msg) in checks.items():
        if ok:
            st.sidebar.success(f"{name}: {msg}")
        else:
            st.sidebar.error(f"{name}: {msg}")


# ============================================================
# 6) Session state initialization (strict and explicit)
# ============================================================
def ensure_session_state_defaults(config: AppConfig) -> None:
    """
    One place to initialize ALL app-level session state keys.

    We do this so that:
    - page modules never "invent" keys implicitly
    - we can reason about state transitions
    - persistence integration later becomes trivial
    """
    defaults = {
        "_app_start_ts": time.time(),
        "_route": DEFAULT_ROUTE,
        "_debug": bool(config.debug),
        "_last_exception": None,
        "_toast_queue": [],  # list[tuple[level, message]]
        # Domain state placeholders
        "active_portfolio_id": None,
        "active_scenario_id": None,
        "active_dataset_id": None,
        "last_run_id": None,
        "market_data": None,
        "market_returns": None,
        "market_meta": None,
        "factor_returns": None,
        "factor_betas": None,
        "portfolio_cache": None,
        "run_results": None,
        "_demo_initialized": False,
        # Cache/compute controls
        "cache_bust": 0,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ensure_demo_seed(config: AppConfig) -> None:
    if st.session_state.get("_demo_initialized", False):
        return

    portfolios = store.list_portfolios(config.sqlite_path)
    datasets = store.list_datasets(config.sqlite_path)

    if not portfolios:
        holdings = demo.demo_holdings()
        portfolio_id = store.create_portfolio(
            config.sqlite_path,
            name="Demo Multi-Asset Portfolio",
            base_currency="USD",
            description="Auto-seeded demo portfolio with diversified equities.",
            holdings=holdings.to_dict(orient="records"),
        )
        st.session_state["active_portfolio_id"] = portfolio_id

    if not datasets:
        prices = demo.demo_prices()
        dataset_id = store.create_dataset(
            config.sqlite_path,
            name="Demo Market Dataset",
            source="demo",
            start_date=str(prices.index.min().date()),
            end_date=str(prices.index.max().date()),
            frequency=pd.infer_freq(prices.index) or "B",
            metadata=demo.demo_metadata(prices),
        )
        st.session_state["active_dataset_id"] = dataset_id
        st.session_state["market_data"] = prices
        st.session_state["market_meta"] = {"source": "demo", "rows": len(prices), "columns": list(prices.columns)}

    st.session_state["_demo_initialized"] = True


def enqueue_toast(level: str, message: str) -> None:
    """
    Queue toast messages to show after layout is established.
    """
    if "_toast_queue" not in st.session_state:
        st.session_state["_toast_queue"] = []
    st.session_state["_toast_queue"].append((level, message))


def flush_toasts() -> None:
    q = st.session_state.get("_toast_queue", [])
    if not q:
        return
    for level, msg in q:
        if level == "success":
            toast_success(msg)
        elif level == "warning":
            toast_warning(msg)
        else:
            toast_error(msg)
    st.session_state["_toast_queue"] = []


# ============================================================
# 7) Error boundary (don’t crash the whole app)
# ============================================================
def error_boundary(fn: Callable[[], None], where: str = "page") -> None:
    """
    Wrap any render function so we can:
    - show a friendly error block
    - store traceback in session state
    - log the exception
    """
    try:
        fn()
    except Exception as e:
        tb = traceback.format_exc()
        st.session_state["_last_exception"] = {
            "where": where,
            "error": repr(e),
            "traceback": tb,
            "ts": datetime.utcnow().isoformat() + "Z",
        }

        log_exception(LOGGER, e, extra={"where": where})
        st.error("Something went wrong while rendering this section.")
        with st.expander("Show technical details"):
            st.code(tb, language="text")

        enqueue_toast("error", f"Render error ({where}): {type(e).__name__}")


# ============================================================
# 8) Route table (single source of truth)
# ============================================================
def get_routes() -> Dict[str, Route]:
    """
    Central route table.
    Each module will be created inside ui/pages/ with a render_page(ctx) function.

    Naming convention:
      ui/pages/00_Home.py => module ui.pages.00_Home
    """
    return {
        "Home": Route(
            key="Home",
            title="Home",
            page_module="stresslab.ui.pages.home",
            icon="🏠",
            description="Overview, quick actions, recent runs",
        ),
        "Portfolios": Route(
            key="Portfolios",
            title="Portfolios",
            page_module="stresslab.ui.pages.portfolios",
            icon="💼",
            description="Upload & manage portfolios / holdings",
        ),
        "Scenarios": Route(
            key="Scenarios",
            title="Scenarios",
            page_module="stresslab.ui.pages.scenarios",
            icon="🧩",
            description="Build and save stress scenarios",
        ),
        "Market Data": Route(
            key="Market Data",
            title="Market Data",
            page_module="stresslab.ui.pages.market_data",
            icon="🗃️",
            description="Load, validate, and align datasets",
        ),
        "Run Stress": Route(
            key="Run Stress",
            title="Run Stress",
            page_module="stresslab.ui.pages.stress_run",
            icon="🧪",
            description="Execute scenarios and compute deltas",
        ),
        "Risk Analytics": Route(
            key="Risk Analytics",
            title="Risk Analytics",
            page_module="stresslab.ui.pages.risk_analytics",
            icon="📊",
            description="VaR, CVaR, exposures, sensitivities",
        ),
        "Backtests": Route(
            key="Backtests",
            title="Backtests",
            page_module="stresslab.ui.pages.backtests",
            icon="📈",
            description="Strategy backtests & trade logs",
        ),
        "Reports": Route(
            key="Reports",
            title="Reports",
            page_module="stresslab.ui.pages.reports",
            icon="🧾",
            description="Generate & export reports; history",
        ),
        "Settings": Route(
            key="Settings",
            title="Settings",
            page_module="stresslab.ui.pages.settings",
            icon="⚙️",
            description="App configuration & diagnostics",
        ),
    }


# ============================================================
# 9) Sidebar navigation + global controls
# ============================================================
def render_sidebar(routes: Dict[str, Route], config: AppConfig) -> str:
    """
    Render sidebar and return selected route key.
    """
    st.sidebar.markdown("## 🧪 StressLab")
    st.sidebar.caption("Macro stress testing & risk analytics (production-grade build).")
    ui_divider(location="sidebar")

    # Route selection
    route_keys = list(routes.keys())
    icons = {k: routes[k].icon for k in route_keys}

    # Keep current route stable across reruns
    current = st.session_state.get("_route", DEFAULT_ROUTE)
    if current not in route_keys:
        current = DEFAULT_ROUTE

    # Streamlit selectbox (with icon prefix)
    def _label(k: str) -> str:
        return f"{icons.get(k, '📄')} {k}"

    route_label_to_key = {_label(k): k for k in route_keys}
    default_label = _label(current)

    selected_label = st.sidebar.selectbox(
        "Navigate",
        options=list(route_label_to_key.keys()),
        index=list(route_label_to_key.keys()).index(default_label),
        key="_nav_selectbox",
    )
    selected_route = route_label_to_key[selected_label]
    st.session_state["_route"] = selected_route

    ui_divider(location="sidebar")

    # Global toggles
    st.sidebar.markdown("### Active Context")
    portfolios = store.list_portfolios(config.sqlite_path)
    portfolio_labels = ["—"] + [f"{p['name']} ({p['portfolio_id']})" for p in portfolios]
    portfolio_index = 0
    active_id = st.session_state.get("active_portfolio_id")
    if active_id:
        for idx, p in enumerate(portfolios, start=1):
            if p["portfolio_id"] == active_id:
                portfolio_index = idx
                break
    selected_portfolio = st.sidebar.selectbox(
        "Active portfolio",
        options=portfolio_labels,
        index=portfolio_index,
    )
    if selected_portfolio != "—":
        st.session_state["active_portfolio_id"] = selected_portfolio.split("(")[-1].strip(")")
    else:
        st.session_state["active_portfolio_id"] = None

    datasets = store.list_datasets(config.sqlite_path)
    dataset_labels = ["—"] + [f"{d['name']} ({d['dataset_id']})" for d in datasets]
    dataset_index = 0
    active_dataset = st.session_state.get("active_dataset_id")
    if active_dataset:
        for idx, d in enumerate(datasets, start=1):
            if d["dataset_id"] == active_dataset:
                dataset_index = idx
                break
    selected_dataset = st.sidebar.selectbox(
        "Active dataset",
        options=dataset_labels,
        index=dataset_index,
    )
    if selected_dataset != "—":
        st.session_state["active_dataset_id"] = selected_dataset.split("(")[-1].strip(")")
    else:
        st.session_state["active_dataset_id"] = None

    ui_divider(location="sidebar")

    st.sidebar.markdown("### Global")
    if st.sidebar.button("⚡ Run Stress Test"):
        st.session_state["_route"] = "Run Stress"
        st.rerun()
    debug = st.sidebar.toggle("Debug mode", value=st.session_state.get("_debug", False))
    st.session_state["_debug"] = bool(debug)

    # Cache bust button (forces any cached function that uses st.session_state['cache_bust'])
    if st.sidebar.button("🔄 Hard refresh (cache bust)"):
        st.session_state["cache_bust"] = int(st.session_state.get("cache_bust", 0)) + 1
        enqueue_toast("success", "Cache-bust applied. Recomputations will rerun.")
        st.rerun()

    # Quick state preview (only debug)
    if st.session_state.get("_debug", False):
        with st.sidebar.expander("Session state snapshot", expanded=False):
            # Filter out very large objects in future (for now show all keys)
            keys = sorted(st.session_state.keys())
            preview = {k: st.session_state[k] for k in keys if not k.startswith("_toast")}
            st.json(preview)

    return selected_route


# ============================================================
# 10) Main router
# ============================================================
def render_route(route_key: str, routes: Dict[str, Route], ctx: AppContext) -> None:
    """
    Render selected route.
    """
    if route_key not in routes:
        st.warning(f"Unknown route '{route_key}'. Redirecting to Home.")
        st.session_state["_route"] = DEFAULT_ROUTE
        route_key = DEFAULT_ROUTE

    route = routes[route_key]

    # Page header section
    render_section_header(
        title=f"{route.icon} {route.title}",
        subtitle=route.description,
    )

    # Attempt to import & render the page
    def _render_page():
        mod = _safe_import_page(route.page_module)
        mod.render_page(ctx)

    error_boundary(_render_page, where=f"route:{route.key}")


# ============================================================
# 11) App main
# ============================================================
def main() -> None:
    # Load config (env-based) — implemented in utils/config.py
    config = load_config()

    # Initialize persistence
    store.init_db(config.sqlite_path)

    # Theme (CSS) applied globally — implemented in ui/theme.py
    apply_theme()

    # Session defaults
    ensure_session_state_defaults(config)
    ensure_demo_seed(config)

    # Build context
    ctx = build_context(config)

    # Health checks
    checks = check_environment(config)

    # Top app header (global) — implemented in ui/components.py
    render_app_header(
        title="StressLab",
        subtitle="Macro Stress Testing • Risk Analytics • Backtesting • Reporting",
        right_meta={
            "Environment": config.environment,
            "Build": config.build_tag,
        },
    )

    # Sidebar navigation
    routes = get_routes()
    selected_route = render_sidebar(routes, config)

    # Debug health panel
    if st.session_state.get("_debug", False):
        render_health_panel(checks)

    # Main page render
    render_route(selected_route, routes, ctx)

    # Footer (global)
    render_footer(
        left=f"© {datetime.now().year} StressLab",
        right=f"UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    # Flush queued toasts at end of layout
    flush_toasts()


if __name__ == "__main__":
    main()
