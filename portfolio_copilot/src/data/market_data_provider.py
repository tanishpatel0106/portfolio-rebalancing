from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd
import yfinance as yf

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


class YFinanceMarketDataProvider:
    def __init__(self) -> None:
        self.source = "yfinance"

    def get_latest_prices(self, tickers: Iterable[str]) -> pd.DataFrame:
        tickers = [t.upper().strip() for t in tickers if t]
        if not tickers:
            return pd.DataFrame(columns=["ticker", "price", "timestamp", "source"])
        if st is not None:
            return _get_latest_prices_cached(tuple(sorted(set(tickers))))
        return _get_latest_prices_raw(tuple(sorted(set(tickers))))

    def get_ohlcv(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if st is not None:
            if interval == "1d":
                return _get_ohlcv_cached_daily(ticker, start, end, interval)
            return _get_ohlcv_cached_intraday(ticker, start, end, interval)
        return _get_ohlcv_raw(ticker, start, end, interval)

    def get_close_history(
        self,
        tickers: Iterable[str],
        start: dt.date,
        end: dt.date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        tickers = [t.upper().strip() for t in tickers if t]
        if not tickers:
            return pd.DataFrame(columns=["date", "ticker", "close"])
        if st is not None:
            if interval == "1d":
                return _get_close_history_cached_daily(tuple(sorted(set(tickers))), start, end, interval)
            return _get_close_history_cached_intraday(tuple(sorted(set(tickers))), start, end, interval)
        return _get_close_history_raw(tuple(sorted(set(tickers))), start, end, interval)


def _parse_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    history = history.copy()
    history.reset_index(inplace=True)
    history.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    if "Datetime" in history.columns:
        history.rename(columns={"Datetime": "date"}, inplace=True)
    history["date"] = pd.to_datetime(history["date"]).dt.date
    return history[["date", "open", "high", "low", "close", "volume"]]


@st.cache_data(ttl=60) if st is not None else (lambda x: x)
def _get_latest_prices_cached(tickers: tuple[str, ...]) -> pd.DataFrame:
    return _get_latest_prices_raw(tickers)


def _get_latest_prices_raw(tickers: tuple[str, ...]) -> pd.DataFrame:
    data = yf.download(
        tickers=list(tickers),
        period="5d",
        interval="1d",
        group_by="ticker",
        progress=False,
        auto_adjust=False,
    )
    rows = []
    timestamp = dt.datetime.utcnow().isoformat()
    if len(tickers) == 1:
        ticker = tickers[0]
        if data.empty or "Close" not in data:
            return pd.DataFrame(columns=["ticker", "price", "timestamp", "source"])
        price = float(data["Close"].dropna().iloc[-1])
        rows.append({"ticker": ticker, "price": price, "timestamp": timestamp, "source": "yfinance"})
    else:
        for ticker in tickers:
            if (ticker, "Close") not in data.columns:
                continue
            close_series = data[(ticker, "Close")].dropna()
            if close_series.empty:
                continue
            price = float(close_series.iloc[-1])
            rows.append({"ticker": ticker, "price": price, "timestamp": timestamp, "source": "yfinance"})
    return pd.DataFrame(rows)


@st.cache_data(ttl=21600) if st is not None else (lambda x: x)
def _get_ohlcv_cached_daily(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    return _get_ohlcv_raw(ticker, start, end, interval)


@st.cache_data(ttl=600) if st is not None else (lambda x: x)
def _get_ohlcv_cached_intraday(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    return _get_ohlcv_raw(ticker, start, end, interval)


def _get_ohlcv_raw(ticker: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
    history = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
    )
    return _parse_history(history)


@st.cache_data(ttl=21600) if st is not None else (lambda x: x)
def _get_close_history_cached_daily(
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pd.DataFrame:
    return _get_close_history_raw(tickers, start, end, interval)


@st.cache_data(ttl=600) if st is not None else (lambda x: x)
def _get_close_history_cached_intraday(
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pd.DataFrame:
    return _get_close_history_raw(tickers, start, end, interval)


def _get_close_history_raw(
    tickers: tuple[str, ...],
    start: dt.date,
    end: dt.date,
    interval: str,
) -> pd.DataFrame:
    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    rows = []
    if len(tickers) == 1:
        ticker = tickers[0]
        close_series = data["Close"].dropna()
        for date, value in close_series.items():
            rows.append({"date": pd.to_datetime(date).date(), "ticker": ticker, "close": float(value)})
    else:
        for ticker in tickers:
            if (ticker, "Close") not in data.columns:
                continue
            close_series = data[(ticker, "Close")].dropna()
            for date, value in close_series.items():
                rows.append({"date": pd.to_datetime(date).date(), "ticker": ticker, "close": float(value)})
    return pd.DataFrame(rows)
