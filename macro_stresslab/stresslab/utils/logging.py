"""
stresslab/utils/logging.py
==========================
Central logging utilities.

Design goals:
- One consistent logger config across the app
- Streamlit-friendly output (no duplicate handlers)
- Structured-ish formatting with timestamps + levels
- Helpers for safe exception logging (with stack traces)
- Works both locally and when deployed (no special infra required)

Usage:
    from utils.logging import get_logger, log_exception, init_logging

    init_logging(debug=cfg.debug)
    logger = get_logger(__name__)
    logger.info("Hello")

Notes:
- Streamlit reruns the script; we must be careful not to add handlers repeatedly.
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


_LOGGER_NAME_ROOT = "stresslab"


@dataclass(frozen=True)
class LoggingConfig:
    debug: bool = True
    level: int = logging.INFO
    to_stdout: bool = True
    include_process: bool = False
    include_module: bool = True


class _UTCFormatter(logging.Formatter):
    """
    A formatter that uses UTC timestamps and a clean, compact format.
    """

    def formatTime(self, record, datefmt=None):
        # ISO-ish UTC time
        dt = datetime.utcfromtimestamp(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _compute_level(debug: bool) -> int:
    return logging.DEBUG if debug else logging.INFO


def init_logging(
    *,
    debug: bool = True,
    level: Optional[int] = None,
    to_stdout: bool = True,
    include_process: bool = False,
    include_module: bool = True,
) -> None:
    """
    Initialize global logging configuration for the app.
    Safe to call multiple times (idempotent handler setup).

    Parameters
    ----------
    debug:
        If True, sets the default level to DEBUG.
    level:
        Explicit logging level override. If None, derived from debug.
    to_stdout:
        If True, logs to stdout (works well in Streamlit + Docker).
    include_process:
        If True, adds process id in log format.
    include_module:
        If True, adds module:line in log format.
    """
    lvl = level if level is not None else _compute_level(debug)
    cfg = LoggingConfig(
        debug=debug,
        level=lvl,
        to_stdout=to_stdout,
        include_process=include_process,
        include_module=include_module,
    )

    root_logger = logging.getLogger(_LOGGER_NAME_ROOT)
    root_logger.setLevel(cfg.level)
    root_logger.propagate = False  # prevent double printing via root handlers

    # Prevent adding handlers multiple times (Streamlit reruns!)
    if getattr(root_logger, "_stresslab_configured", False):
        # Still update level if user changed env/debug
        root_logger.setLevel(cfg.level)
        for h in list(root_logger.handlers):
            h.setLevel(cfg.level)
        return

    if cfg.to_stdout:
        handler = logging.StreamHandler(sys.stdout)
    else:
        # Fallback: stderr (but stdout is preferred for containers)
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(cfg.level)

    # Build format
    # Example:
    # 2026-01-14T17:32:18.123Z | INFO  | stresslab.ui.app:87 | message...
    parts = ["%(asctime)s", "%(levelname)-5s"]

    if cfg.include_process:
        parts.append("pid=%(process)d")

    # Always include logger name, but keep it compact
    # logger name will be like "stresslab.ui.app"
    parts.append("%(name)s")

    if cfg.include_module:
        parts.append("%(module)s:%(lineno)d")

    fmt = " | ".join(parts) + " | %(message)s"

    formatter = _UTCFormatter(fmt=fmt)
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

    # Mark configured
    setattr(root_logger, "_stresslab_configured", True)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger under the stresslab namespace.

    If name is None, returns the root stresslab logger.
    If name already starts with "stresslab", it is used as-is.
    Otherwise, it is prefixed as "stresslab.<name>".
    """
    if not name:
        return logging.getLogger(_LOGGER_NAME_ROOT)

    if name.startswith(_LOGGER_NAME_ROOT):
        return logging.getLogger(name)

    return logging.getLogger(f"{_LOGGER_NAME_ROOT}.{name}")


def log_exception(
    logger: logging.Logger,
    msg: str,
    *,
    exc: Optional[BaseException] = None,
    level: int = logging.ERROR,
    extra_context: Optional[dict] = None,
) -> None:
    """
    Log an exception with traceback, safely.

    Parameters
    ----------
    logger:
        Logger to use.
    msg:
        Human-friendly message.
    exc:
        The exception instance. If None, uses the current exception info.
    level:
        Logging level (ERROR by default).
    extra_context:
        Optional dict of extra context to print.
    """
    try:
        if extra_context:
            ctx = " ".join([f"{k}={v!r}" for k, v in extra_context.items()])
            msg_full = f"{msg} | {ctx}"
        else:
            msg_full = msg

        if exc is not None:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.log(level, f"{msg_full}\n{tb}")
        else:
            # Uses current exception
            logger.log(level, msg_full, exc_info=True)
    except Exception:
        # Last-resort: never crash the app due to logging
        try:
            print(f"[stresslab.logging] Failed to log exception: {msg}", file=sys.stderr)
        except Exception:
            pass


def set_third_party_log_levels(
    *,
    warnings_level: int = logging.WARNING,
    urllib3_level: int = logging.WARNING,
    matplotlib_level: int = logging.WARNING,
    plotly_level: int = logging.WARNING,
) -> None:
    """
    Reduce noise from common third-party libraries.
    Call this after init_logging().

    Streamlit + data libraries can be extremely chatty otherwise.
    """
    logging.getLogger("warnings").setLevel(warnings_level)
    logging.getLogger("urllib3").setLevel(urllib3_level)
    logging.getLogger("matplotlib").setLevel(matplotlib_level)
    logging.getLogger("plotly").setLevel(plotly_level)


def dump_env_debug(logger: logging.Logger, keys: Optional[list[str]] = None) -> None:
    """
    Helpful debug utility: logs selected environment variables (non-secret).
    Never dump everything by default.

    Example:
        dump_env_debug(logger, ["STRESSLAB_ENV", "STRESSLAB_DEBUG"])
    """
    if not keys:
        return

    safe = {}
    for k in keys:
        v = os.getenv(k)
        safe[k] = v if v is not None else None

    logger.debug("env: %s", safe)
