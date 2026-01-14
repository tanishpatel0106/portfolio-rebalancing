"""
stresslab/data/sources.py
=========================
Production-grade data access layer for StressLab.

Design goals
------------
1) Stable schema:
   - Prices: OHLCV (Open, High, Low, Close, Adj Close, Volume) with DatetimeIndex
   - Returns: convenience functions produce aligned Series/DataFrames
   - Macro: time-aligned to a given index (e.g., trading calendar), with forward-fill rules

2) Reliability:
   - Strong input validation + clear errors
   - Handles MultiIndex outputs from yfinance
   - Guards against empty downloads, delisted tickers, interval limits, timezone drift
   - Provides deterministic cleaning rules (sort index, drop duplicates, infer freq)

3) Performance:
   - Uses st.cache_data when called from Streamlit pages
   - Pure functions where possible for testability

This module does NOT do analytics (risk, VaR, stress) — only data loading/standardization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import importlib.util
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_yf_spec = importlib.util.find_spec("yfinance")
if _yf_spec is not None:
    import yfinance as yf  # type: ignore
else:  # pragma: no cover
    yf = None  # type: ignore

# Optional (macro)
_fred_spec = importlib.util.find_spec("fredapi")
if _fred_spec is not None:
    from fredapi import Fred  # type: ignore
else:  # pragma: no cover
    Fred = None  # type: ignore


# =========================
# TYPES
# =========================

DateLike = Union[str, date, datetime, pd.Timestamp]


# =========================
# CONSTANTS
# =========================

YF_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo"
}

OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# =========================
# ERRORS
# =========================

class DataError(RuntimeError):
    """Base data error."""


class DataEmptyError(DataError):
    """Raised when a provider returns no rows."""


class ProviderUnavailableError(DataError):
    """Raised when required provider deps are missing."""


class ValidationError(DataError):
    """Raised when inputs fail validation."""


# =========================
# HELPERS
# =========================

def _to_timestamp(x: DateLike) -> pd.Timestamp:
    if isinstance(x, pd.Timestamp):
        return x
    if isinstance(x, datetime):
        return pd.Timestamp(x)
    if isinstance(x, date):
        return pd.Timestamp(datetime(x.year, x.month, x.day))
    if isinstance(x, str):
        return pd.Timestamp(x)
    raise ValidationError(f"Unsupported date type: {type(x)}")


def _validate_window(start: DateLike, end: DateLike) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s = _to_timestamp(start)
    e = _to_timestamp(end)
    if pd.isna(s) or pd.isna(e):
        raise ValidationError("start/end must parse to valid dates.")
    if s >= e:
        raise ValidationError(f"Invalid window: start ({s}) must be before end ({e}).")
    return s, e


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise DataError(f"Failed to convert index to DatetimeIndex: {e}")
    # normalize tz (yfinance can be tz-aware)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize yfinance output:
    - Flatten MultiIndex
    - Keep OHLCV
    - Ensure float columns, Volume as float/int
    - Sort + drop duplicates
    """
    if df is None or len(df) == 0:
        raise DataEmptyError("No rows returned by provider.")

    # If MultiIndex columns (rare: group_by='ticker'), flatten.
    if isinstance(df.columns, pd.MultiIndex):
        # Common form: ('Close', 'AAPL') etc — choose first level if possible
        # But yfinance also returns (symbol, field) in some modes.
        # We'll attempt robust flattening:
        if df.columns.nlevels == 2:
            # Try: columns = level 0 if it contains OHLCV; else level 1
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))
            if any(c in OHLCV_COLS for c in lvl0):
                df.columns = df.columns.get_level_values(0)
            elif any(c in OHLCV_COLS for c in lvl1):
                df.columns = df.columns.get_level_values(1)
            else:
                df.columns = ["_".join(map(str, c)) for c in df.columns.to_list()]
        else:
            df.columns = ["_".join(map(str, c)) for c in df.columns.to_list()]

    df = _ensure_datetime_index(df)

    # Ensure sorted and unique index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Select columns if present; yfinance can omit Adj Close for FX
    cols_present = [c for c in OHLCV_COLS if c in df.columns]
    if not cols_present:
        raise DataError(f"Provider returned columns {list(df.columns)}, none of {OHLCV_COLS} found.")

    df = df[cols_present].copy()

    # Ensure numeric
    for c in cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop fully empty rows after coercion
    df = df.dropna(how="all")
    if df.empty:
        raise DataEmptyError("All rows became NaN after numeric coercion.")

    return df


def _infer_trading_calendar(df: pd.DataFrame) -> str:
    """
    Best-effort: infer frequency string for display/logging.
    Not used for resampling decisions.
    """
    idx = df.index
    if len(idx) < 5:
        return "unknown"
    try:
        freq = pd.infer_freq(idx)
        return freq or "unknown"
    except Exception:
        return "unknown"


# =========================
# PRICES (YFINANCE)
# =========================

@dataclass(frozen=True)
class PriceRequest:
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp
    interval: str
    auto_adjust: bool = False
    actions: bool = False


def yf_download_ohlcv(req: PriceRequest) -> pd.DataFrame:
    """
    Raw yfinance call + standardization.

    Raises:
        ProviderUnavailableError, ValidationError, DataEmptyError, DataError
    """
    if yf is None:
        raise ProviderUnavailableError("yfinance is not installed/available in this environment.")
    if not req.ticker or not isinstance(req.ticker, str):
        raise ValidationError("ticker must be a non-empty string.")
    if req.interval not in YF_INTERVALS:
        raise ValidationError(f"Unsupported interval '{req.interval}'. Must be one of: {sorted(YF_INTERVALS)}")
    if req.start >= req.end:
        raise ValidationError("start must be < end.")

    df = _yf_download_with_retry(req)

    df = _clean_ohlcv(df)

    # FX often has Volume=0; keep it but don't force dropping.
    return df


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((DataError, DataEmptyError, ValueError)),
)
def _yf_download_with_retry(req: PriceRequest) -> pd.DataFrame:
    """Download with retry/backoff to improve resiliency."""
    df = yf.download(
        req.ticker,
        start=req.start,
        end=req.end,
        interval=req.interval,
        auto_adjust=req.auto_adjust,
        actions=req.actions,
        progress=False,
        group_by="column",
        threads=True,
    )
    return df


def get_prices(
    ticker: str,
    start: DateLike,
    end: DateLike,
    timeframe: str = "Daily",
    *,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper mapping a friendly timeframe -> yfinance interval.

    timeframe:
        "Daily", "Weekly", "Monthly", "Quarterly",
        "1H", "4H", "1D" (aliases)
    """
    s, e = _validate_window(start, end)

    tf_map = {
        "Daily": "1d",
        "1D": "1d",
        "Weekly": "1wk",
        "Monthly": "1mo",
        "Quarterly": "3mo",

        # NOTE: yfinance supports 1h but not a true '4h' interval consistently for all tickers.
        # In your earlier app you used "4h" — yfinance supports it for some assets.
        # We'll keep "4h" but validate.
        "1H": "1h",
        "4H": "4h",
        "Hourly": "1h",
    }

    if timeframe not in tf_map:
        raise ValidationError(f"Unknown timeframe '{timeframe}'. Options: {sorted(tf_map.keys())}")

    interval = tf_map[timeframe]
    if interval not in YF_INTERVALS:
        raise ValidationError(f"Timeframe maps to unsupported yfinance interval '{interval}'")

    req = PriceRequest(
        ticker=ticker.strip(),
        start=s,
        end=e,
        interval=interval,
        auto_adjust=auto_adjust,
        actions=False,
    )
    df = yf_download_ohlcv(req)
    return df


def close_series(df_ohlcv: pd.DataFrame) -> pd.Series:
    if "Close" not in df_ohlcv.columns:
        raise DataError("Close column not found in OHLCV.")
    s = pd.to_numeric(df_ohlcv["Close"], errors="coerce")
    s.name = "close"
    return s


def returns(series: pd.Series, periods: int = 1) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    r = s.pct_change(periods=periods)
    r.name = f"ret_{periods}"
    return r


# =========================
# MACRO (FRED)
# =========================

@dataclass(frozen=True)
class FredRequest:
    api_key: str
    series_map: Dict[str, str]  # column_name -> series_id
    start: pd.Timestamp
    end: pd.Timestamp


def fred_load_series(
    api_key: str,
    series_id: str,
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Load a single FRED series via fredapi.
    Returns a float Series with DatetimeIndex.
    """
    if Fred is None:
        raise ProviderUnavailableError("fredapi is not installed/available.")

    if not api_key or not isinstance(api_key, str):
        raise ValidationError("FRED api_key must be a non-empty string.")
    if not series_id or not isinstance(series_id, str):
        raise ValidationError("series_id must be a non-empty string.")

    fred = Fred(api_key=api_key)
    s = fred.get_series(series_id)

    if s is None or len(s) == 0:
        raise DataEmptyError(f"FRED returned no data for series_id={series_id}")

    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_convert(None)

    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    if start is not None:
        s = s.loc[s.index >= start]
    if end is not None:
        s = s.loc[s.index <= end]

    if s.empty:
        raise DataEmptyError(f"FRED series {series_id} has no rows in requested window.")
    s.name = series_id
    return s


def get_macro_frame_fred(
    req: FredRequest,
    *,
    buffer_days: int = 400,
) -> pd.DataFrame:
    """
    Load multiple FRED series into a single DataFrame.

    buffer_days: extend backwards so we can compute pct_change, yoy, etc. later.
    """
    if buffer_days < 0:
        raise ValidationError("buffer_days must be >= 0.")

    start_buf = req.start - pd.Timedelta(days=buffer_days)

    data = {}
    for col, series_id in req.series_map.items():
        s = fred_load_series(
            req.api_key,
            series_id,
            start=start_buf,
            end=req.end,
        )
        s.name = col
        data[col] = s

    df = pd.DataFrame(data).sort_index()
    if df.empty:
        raise DataEmptyError("Macro frame is empty after loading series.")
    return df


def align_to_index(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    method: str = "ffill",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reindex macro data to match a trading index.

    method: "ffill", "bfill", "nearest", None
    limit: optional max consecutive fills
    """
    if not isinstance(target_index, pd.DatetimeIndex):
        raise ValidationError("target_index must be a DatetimeIndex.")
    if df is None or df.empty:
        raise DataEmptyError("Cannot align an empty macro dataframe.")
    df = _ensure_datetime_index(df).sort_index()

    # Expand df index union target index to allow proper forward-filling
    union = df.index.union(target_index)
    df2 = df.reindex(union).sort_index()

    if method is None:
        df2 = df2.reindex(target_index)
        return df2

    if method not in {"ffill", "bfill", "nearest"}:
        raise ValidationError("method must be one of: ffill, bfill, nearest, None")

    if method == "nearest":
        # nearest needs tolerance sometimes; keep simple and deterministic
        df2 = df2.reindex(target_index, method="nearest")
    else:
        df2 = df2.fillna(method=method, limit=limit).reindex(target_index)

    return df2


# =========================
# YAHOO MACRO (VIX/SPX etc.)
# =========================

def yf_close(
    ticker: str,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
    *,
    col_name: Optional[str] = None,
) -> pd.Series:
    """
    Fetch Close series for a ticker from yfinance, standardized.
    Useful for VIX/SPX etc.
    """
    s, e = _validate_window(start, end)
    if interval not in YF_INTERVALS:
        raise ValidationError(f"Unsupported interval '{interval}' for yf_close.")
    req = PriceRequest(ticker=ticker.strip(), start=s, end=e, interval=interval, auto_adjust=False)
    df = yf_download_ohlcv(req)
    close = close_series(df)
    if col_name:
        close.name = col_name
    else:
        close.name = ticker
    return close


def get_macro_yahoo_basic(
    start: DateLike,
    end: DateLike,
    *,
    interval: str = "1d",
    include_vix: bool = True,
    include_spx: bool = True,
) -> pd.DataFrame:
    """
    Pull a minimal set of macro proxies from Yahoo:
    - VIX (^VIX)
    - SPX (^GSPC)
    Returns a DataFrame indexed by date.
    """
    s, e = _validate_window(start, end)
    series = {}

    if include_vix:
        series["vix"] = yf_close("^VIX", s, e, interval=interval, col_name="vix")
    if include_spx:
        series["spx"] = yf_close("^GSPC", s, e, interval=interval, col_name="spx")

    df = pd.DataFrame(series).sort_index()
    if df.empty:
        raise DataEmptyError("Yahoo macro frame returned empty.")
    return df


# =========================
# HIGH-LEVEL: BUILD MACRO FOR A TRADING INDEX
# =========================

def build_macro_for_trading_index(
    trading_index: pd.DatetimeIndex,
    start: DateLike,
    end: DateLike,
    *,
    fred_api_key: Optional[str] = None,
    fred_series_map: Optional[Dict[str, str]] = None,
    yahoo_interval: str = "1d",
    include_yahoo_vix_spx: bool = True,
) -> pd.DataFrame:
    """
    Build a macro dataframe aligned to the trading index.

    - If fred_api_key + series_map provided, loads FRED series and aligns.
    - Optionally joins Yahoo proxies (vix/spx).
    - Forward-fills to match market dates.

    This intentionally does NOT compute derived features (inflation diff, gdp diffs).
    That belongs in analytics/features modules.
    """
    s, e = _validate_window(start, end)

    parts: List[pd.DataFrame] = []

    if fred_api_key and fred_series_map:
        df_fred = get_macro_frame_fred(
            FredRequest(api_key=fred_api_key, series_map=fred_series_map, start=s, end=e)
        )
        df_fred_aligned = align_to_index(df_fred, trading_index, method="ffill")
        parts.append(df_fred_aligned)

    if include_yahoo_vix_spx:
        df_y = get_macro_yahoo_basic(s, e, interval=yahoo_interval, include_vix=True, include_spx=True)
        df_y_aligned = align_to_index(df_y, trading_index, method="ffill")
        parts.append(df_y_aligned)

    if not parts:
        raise ValidationError("No macro sources enabled (provide FRED key+map and/or enable Yahoo proxies).")

    macro = pd.concat(parts, axis=1).sort_index()
    # Deduplicate columns if overlap
    macro = macro.loc[:, ~macro.columns.duplicated(keep="last")]

    # Final ffill to remove gaps from merges
    macro = macro.ffill()

    return macro


# =========================
# DIAGNOSTICS
# =========================

def summarize_frame(df: pd.DataFrame) -> Dict[str, object]:
    """
    Useful to show in UI: row count, date range, inferred frequency.
    """
    if df is None or df.empty:
        return {"rows": 0, "start": None, "end": None, "freq": "unknown", "cols": []}
    df = _ensure_datetime_index(df)
    return {
        "rows": int(len(df)),
        "start": df.index.min(),
        "end": df.index.max(),
        "freq": _infer_trading_calendar(df),
        "cols": list(df.columns),
        "na_perc": float(df.isna().mean().mean()) if df.shape[1] > 0 else 0.0,
    }
