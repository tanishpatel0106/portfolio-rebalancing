from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def ohlcv_chart(df: pd.DataFrame, signals: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume", yaxis="y2", opacity=0.3))

    if signals is not None and not signals.empty:
        if "fast_ma" in signals.columns:
            fig.add_trace(go.Scatter(x=signals.index, y=signals["fast_ma"], mode="lines", name="MA Fast"))
        if "slow_ma" in signals.columns:
            fig.add_trace(go.Scatter(x=signals.index, y=signals["slow_ma"], mode="lines", name="MA Slow"))
        if "signal" in signals.columns:
            buys = signals[signals["signal"] > 0]
            sells = signals[signals["signal"] < 0]
            fig.add_trace(go.Scatter(x=buys.index, y=df.set_index("date").loc[buys.index]["close"], mode="markers", marker_symbol="triangle-up", marker_color="green", name="Buy"))
            fig.add_trace(go.Scatter(x=sells.index, y=df.set_index("date").loc[sells.index]["close"], mode="markers", marker_symbol="triangle-down", marker_color="red", name="Sell"))

    fig.update_layout(
        yaxis_title="Price",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Volume"),
        legend=dict(orientation="h"),
        margin=dict(l=30, r=30, t=30, b=30),
    )
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["drawdown"], fill="tozeroy", mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", yaxis_title="Drawdown")
    return fig


def returns_histogram(returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns, nbinsx=40, name="Returns", opacity=0.7))
    fig.update_layout(title="Return Distribution", xaxis_title="Return", yaxis_title="Frequency")
    return fig


def allocation_pie(df: pd.DataFrame, label: str, value: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=df[label], values=df[value], hole=0.45))
    fig.update_layout(title=title)
    return fig


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[x], y=df[y]))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    return fig
