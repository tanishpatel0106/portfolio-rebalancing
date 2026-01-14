"""
stresslab/analytics/regimes.py
==============================

Regime detection + regime analytics for StressLab.

Design goals
------------
- Pure analytics (NO Streamlit)
- Deterministic + production-oriented:
  - strict validation
  - stable output schemas
  - no hidden global state
- Works *with* stress.py (does not require modifying stress.py)

What this module provides
-------------------------
1) Regime detection on a portfolio return stream (or directly on asset returns):
   - Volatility regimes (rolling vol quantiles)
   - Drawdown regimes (rolling max drawdown quantiles)
   - Correlation regimes (rolling average correlation quantiles)
   - Composite regimes (weighted z-score blend of multiple features)

2) Regime post-processing:
   - Run-length minimum regime length enforcement
   - Segment labeling (contiguous blocks)

3) Output:
   - regime_series: date -> regime label (segment-aware)
   - regime_base_series: date -> base regime label (e.g., Calm/Normal/Stress)
   - features: per-date feature dataframe used for detection
   - regime_stats: per-regime summary table
   - regimes_for_mc: Dict[str, Tuple[slice, prob]] compatible with
        stress.regime_mixture_simulation_terminal(...)
     NOTE: Because stress.py’s regime mixture expects contiguous time slices,
     this module emits *segment regimes* (each is a contiguous block).

Recommended integration
-----------------------
- UI imports stress.py for risk/stress/MC pages.
- Regime page imports this regimes.py to:
    - compute regimes
    - show regime timeline + stats
    - feed regimes_for_mc into stress.regime_mixture_simulation_terminal(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Import from stress.py (same package)
from .stress import (
    AnalyticsError,
    ValidationError,
    PortfolioSpec,
    _ensure_df,
    _ensure_series,
    _ensure_datetime_index,
    _sorted_unique_index,
    _as_float_df,
    align_returns_and_portfolio,
    portfolio_returns,
    equity_curve_from_returns,
    drawdown as dd_series,
)


# =========================
# DATA STRUCTURES
# =========================

RegimeMethod = Literal["volatility", "drawdown", "correlation", "composite"]
RegimeLabels = Tuple[str, str, str]  # (low, mid, high) e.g. ("Calm","Normal","Stress")


@dataclass(frozen=True)
class RegimeConfig:
    method: RegimeMethod = "volatility"

    # Rolling window for feature computation (in trading days)
    window: int = 63  # ~3 months

    # Quantile thresholds for 3-bucket regimes (low / mid / high)
    # low_bucket: feature <= q_low
    # high_bucket: feature >= q_high
    # mid_bucket: otherwise
    q_low: float = 0.33
    q_high: float = 0.67

    # Base labels for 3-bucket regimes
    labels: RegimeLabels = ("Calm", "Normal", "Stress")

    # Feature smoothing (EWMA) to reduce flicker
    # If None: no smoothing
    ewma_span: Optional[int] = 10

    # Minimum length (days) for any regime segment after labeling.
    # Segments shorter than this are merged into neighbors.
    min_segment_len: int = 10

    # Composite configuration (only for method="composite")
    # features used: vol, dd, avg_corr (if available)
    composite_weights: Optional[Dict[str, float]] = None  # {"vol":0.5,"dd":0.3,"corr":0.2}

    # Correlation feature details
    corr_target_universe: Optional[Sequence[str]] = None  # if you want corr on subset only

    # Numerical guards
    min_obs: int = 252  # require at least this many return rows for stable regime estimation


@dataclass(frozen=True)
class RegimeResult:
    """
    Canonical output for regimes.

    regime_series:
      - segment-aware labels (e.g., "Stress#03") where each label is a contiguous block.
      - good for producing regimes_for_mc slices.
    regime_base_series:
      - base label only (e.g., "Stress")
      - good for coloring time-series plots.
    features:
      - per-date feature dataframe used in detection.
    regime_stats:
      - per-regime (segment) stats table.
    regimes_for_mc:
      - dict: regime_segment_name -> (slice(start,end), prob)
      - directly compatible with stress.regime_mixture_simulation_terminal(...)
    """
    regime_series: pd.Series
    regime_base_series: pd.Series
    features: pd.DataFrame
    regime_stats: pd.DataFrame
    regimes_for_mc: Dict[str, Tuple[slice, float]]


# =========================
# VALIDATION + HELPERS
# =========================

def _validate_config(cfg: RegimeConfig) -> RegimeConfig:
    if cfg.window <= 5:
        raise ValidationError("RegimeConfig.window must be > 5.")
    if not (0.0 < cfg.q_low < 1.0) or not (0.0 < cfg.q_high < 1.0):
        raise ValidationError("RegimeConfig.q_low and q_high must be in (0,1).")
    if not (cfg.q_low < cfg.q_high):
        raise ValidationError("RegimeConfig.q_low must be < q_high.")
    if cfg.ewma_span is not None and cfg.ewma_span <= 1:
        raise ValidationError("RegimeConfig.ewma_span must be None or > 1.")
    if cfg.min_segment_len < 1:
        raise ValidationError("RegimeConfig.min_segment_len must be >= 1.")
    if cfg.min_obs < 50:
        raise ValidationError("RegimeConfig.min_obs should be >= 50.")
    if cfg.method == "composite":
        if cfg.composite_weights is None or not isinstance(cfg.composite_weights, dict) or len(cfg.composite_weights) == 0:
            raise ValidationError("composite method requires composite_weights dict, e.g. {'vol':0.5,'dd':0.3,'corr':0.2}.")
        for k, v in cfg.composite_weights.items():
            if k not in ("vol", "dd", "corr"):
                raise ValidationError("composite_weights keys must be among {'vol','dd','corr'}.")
            if not isinstance(v, (int, float, np.floating)):
                raise ValidationError("composite_weights values must be numeric.")
    return cfg


def _zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = float(x.mean())
    sig = float(x.std(ddof=1))
    if sig <= 1e-18:
        return x * 0.0
    return (x - mu) / sig


def _ewma_smooth(x: pd.Series, span: Optional[int]) -> pd.Series:
    if span is None:
        return x
    return x.ewm(span=int(span), adjust=False).mean()


def _run_length_encode(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    """
    Return list of (start_idx, end_idx_inclusive, label) runs.
    """
    if len(labels) == 0:
        return []
    runs: List[Tuple[int, int, str]] = []
    start = 0
    cur = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != cur:
            runs.append((start, i - 1, cur))
            start = i
            cur = labels[i]
    runs.append((start, len(labels) - 1, cur))
    return runs


def _merge_short_runs(
    base_labels: List[str],
    feature: np.ndarray,
    *,
    min_len: int,
) -> List[str]:
    """
    Merge regime runs shorter than min_len into neighboring run.

    Strategy:
      - For a short run, merge into the neighbor (left or right) whose feature-mean is closest.
      - Edge runs merge into the only available neighbor.
    """
    if min_len <= 1:
        return base_labels

    labels = list(base_labels)
    n = len(labels)
    if n == 0:
        return labels

    changed = True
    while changed:
        changed = False
        runs = _run_length_encode(labels)
        if len(runs) <= 1:
            break

        for k, (s, e, lab) in enumerate(runs):
            run_len = e - s + 1
            if run_len >= min_len:
                continue

            # identify neighbors
            left = runs[k - 1] if k - 1 >= 0 else None
            right = runs[k + 1] if k + 1 < len(runs) else None

            # compute current run feature mean
            cur_mean = float(np.nanmean(feature[s:e + 1])) if (e >= s) else np.nan

            # choose merge target
            if left is None and right is None:
                continue
            elif left is None:
                target_label = right[2]
            elif right is None:
                target_label = left[2]
            else:
                ls, le, llab = left
                rs, re, rlab = right
                left_mean = float(np.nanmean(feature[ls:le + 1]))
                right_mean = float(np.nanmean(feature[rs:re + 1]))
                if np.isnan(cur_mean):
                    target_label = llab  # default
                else:
                    # closest mean
                    if abs(cur_mean - left_mean) <= abs(cur_mean - right_mean):
                        target_label = llab
                    else:
                        target_label = rlab

            # apply merge: overwrite short segment
            for i in range(s, e + 1):
                labels[i] = target_label

            changed = True
            break

    return labels


def _segment_labels(base_labels: Sequence[str], labels: RegimeLabels) -> Tuple[List[str], List[str]]:
    """
    Convert base labels into segment-aware labels.

    Example:
      base:  Calm Calm Normal Normal Stress Stress Normal ...
      seg :  Calm#01 Calm#01 Normal#01 Normal#01 Stress#01 Stress#01 Normal#02 ...

    Returns:
      segment_labels, base_labels_copy
    """
    base = list(base_labels)
    seg = []
    counters = {labels[0]: 0, labels[1]: 0, labels[2]: 0}

    runs = _run_length_encode(base)
    for (s, e, lab) in runs:
        counters[lab] = counters.get(lab, 0) + 1
        tag = f"{lab}#{counters[lab]:02d}"
        seg.extend([tag] * (e - s + 1))

    return seg, base


def _avg_pairwise_corr(cov: np.ndarray) -> float:
    """
    Average off-diagonal correlation implied by covariance.
    """
    cov = np.array(cov, dtype=float, copy=False)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    invd = 1.0 / d
    corr = cov * invd[:, None] * invd[None, :]
    # off-diagonal mean
    n = corr.shape[0]
    if n <= 1:
        return 1.0
    off = corr[~np.eye(n, dtype=bool)]
    off = off[np.isfinite(off)]
    if off.size == 0:
        return float("nan")
    return float(np.mean(np.clip(off, -1.0, 1.0)))


# =========================
# FEATURE COMPUTATION
# =========================

def compute_regime_features(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    cfg: RegimeConfig,
) -> Tuple[pd.Series, pd.DataFrame, List[str], np.ndarray]:
    """
    Compute the portfolio return series and per-date features.

    Features produced (columns may include):
      - vol: rolling std of portfolio returns (annualized)
      - dd: rolling max drawdown over window (in return space)
      - avg_corr: rolling average pairwise correlation of assets (if N>=2)

    Returns:
      pr (portfolio return series),
      feat_df (indexed by date),
      assets (aligned),
      weights (aligned)
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    returns_df = _ensure_datetime_index(returns_df, "returns_df")
    returns_df = _sorted_unique_index(returns_df)
    returns_df = _as_float_df(returns_df)

    cfg = _validate_config(cfg)

    # Align + portfolio series (return mode for regimes)
    rets, w, assets = align_returns_and_portfolio(
        returns_df,
        portfolio,
        min_non_na_frac=0.95,
        drop_dates_with_any_na=True,
    )

    if len(rets) < cfg.min_obs:
        raise ValidationError(f"Need at least {cfg.min_obs} observations for regime estimation; got {len(rets)}.")

    pr = portfolio_returns(rets, w, notional_mode=False)
    pr = pd.to_numeric(pr, errors="coerce").dropna()
    if len(pr) < cfg.min_obs:
        raise ValidationError("Portfolio return series too short after NA cleaning for regimes.")

    # Align rets to pr index (should match)
    rets = rets.loc[pr.index].copy()

    # --- vol feature ---
    vol = pr.rolling(cfg.window).std(ddof=1) * np.sqrt(252.0)
    vol.name = "vol"

    # --- drawdown feature (rolling max drawdown in return-space) ---
    # Use equity curve from returns, then drawdown series, then rolling max.
    eq = equity_curve_from_returns(pr, starting_capital=1.0)
    dd = dd_series(eq)  # peak - equity
    # convert to drawdown fraction relative to peak (more interpretable)
    peak = eq.cummax().replace(0.0, np.nan)
    dd_frac = (dd / peak).replace([np.inf, -np.inf], np.nan)
    dd_roll = dd_frac.rolling(cfg.window).max()
    dd_roll.name = "dd"

    # --- correlation feature (rolling avg corr) ---
    avg_corr = None
    # allow subset universe
    corr_cols = list(assets)
    if cfg.corr_target_universe is not None:
        corr_cols = [c for c in corr_cols if c in set(cfg.corr_target_universe)]
    if len(corr_cols) >= 2:
        # compute rolling cov then avg corr
        X = rets[corr_cols].to_numpy(dtype=float)
        idx = rets.index

        # rolling covariance each date: O(T*N^2) but window is moderate; acceptable for UI scale.
        # Optimize later if needed.
        vals = np.full(len(idx), np.nan, dtype=float)
        wlen = int(cfg.window)
        for t in range(wlen - 1, len(idx)):
            block = X[t - wlen + 1 : t + 1, :]
            if not np.isfinite(block).all():
                continue
            cov = np.cov(block, rowvar=False, ddof=1)
            vals[t] = _avg_pairwise_corr(cov)

        avg_corr = pd.Series(vals, index=idx, name="avg_corr")

    # Build feature DF (drop early NaNs)
    feat_cols = [vol, dd_roll]
    if avg_corr is not None:
        feat_cols.append(avg_corr)

    feat = pd.concat(feat_cols, axis=1)
    # Optional smoothing to reduce regime flicker
    for c in feat.columns:
        feat[c] = _ewma_smooth(feat[c], cfg.ewma_span)

    # Drop rows where the key feature for the chosen method is NA
    key = "vol" if cfg.method == "volatility" else ("dd" if cfg.method == "drawdown" else ("avg_corr" if cfg.method == "correlation" else None))
    if cfg.method in ("volatility", "drawdown", "correlation"):
        if key not in feat.columns:
            raise ValidationError(f"Feature '{key}' not available for method={cfg.method}.")
        feat = feat.dropna(subset=[key])
        pr = pr.loc[feat.index]
    else:
        # composite: require vol+dd; corr optional
        req = ["vol", "dd"]
        for rcol in req:
            if rcol not in feat.columns:
                raise ValidationError("Composite regimes require at least vol and dd features.")
        feat = feat.dropna(subset=req)
        pr = pr.loc[feat.index]

    if len(feat) < cfg.min_obs // 2:
        raise ValidationError("Too few feature rows after dropping NaNs; increase history or reduce window.")

    return pr, feat, assets, w


# =========================
# REGIME ASSIGNMENT
# =========================

def _assign_3bucket(feature: pd.Series, labels: RegimeLabels, q_low: float, q_high: float) -> List[str]:
    x = pd.to_numeric(feature, errors="coerce").dropna()
    if x.empty:
        raise ValidationError("Feature series is empty; cannot assign regimes.")
    lo = float(x.quantile(q_low))
    hi = float(x.quantile(q_high))

    out = []
    for v in pd.to_numeric(feature, errors="coerce").to_numpy(dtype=float):
        if not np.isfinite(v):
            out.append(labels[1])  # default mid for NA
        elif v <= lo:
            out.append(labels[0])
        elif v >= hi:
            out.append(labels[2])
        else:
            out.append(labels[1])
    return out


def _assign_composite(feat_df: pd.DataFrame, cfg: RegimeConfig) -> Tuple[List[str], pd.Series]:
    """
    Composite score = weighted zscore(vol) + weighted zscore(dd) + weighted zscore(corr?) .
    """
    w = dict(cfg.composite_weights or {})
    # Normalize weights to sum to 1 (absolute)
    s = float(np.sum([abs(float(v)) for v in w.values()])) + 1e-18
    for k in list(w.keys()):
        w[k] = float(w[k]) / s

    score = 0.0
    pieces = {}
    if "vol" in w:
        pieces["z_vol"] = _zscore(feat_df["vol"])
        score = score + w["vol"] * pieces["z_vol"]
    if "dd" in w:
        pieces["z_dd"] = _zscore(feat_df["dd"])
        score = score + w["dd"] * pieces["z_dd"]
    if "corr" in w:
        if "avg_corr" not in feat_df.columns:
            # if corr not available, treat as zero (still deterministic)
            pieces["z_corr"] = pd.Series(0.0, index=feat_df.index)
        else:
            pieces["z_corr"] = _zscore(feat_df["avg_corr"])
        score = score + w["corr"] * pieces["z_corr"]

    score = pd.Series(score, index=feat_df.index, name="composite_score")
    # Smooth score as well (optional)
    score = _ewma_smooth(score, cfg.ewma_span)

    base = _assign_3bucket(score, cfg.labels, cfg.q_low, cfg.q_high)
    return base, score


# =========================
# REGIME STATS + MC SLICES
# =========================

def _compute_regime_stats(
    pr: pd.Series,
    feat: pd.DataFrame,
    seg_series: pd.Series,
    base_series: pd.Series,
) -> pd.DataFrame:
    """
    Per segment regime stats.
    """
    pr = _ensure_series(pr, "portfolio_returns")
    pr = _ensure_datetime_index(pr, "portfolio_returns")
    pr = pd.to_numeric(pr, errors="coerce")

    # Align everything
    idx = pr.index.intersection(seg_series.index).intersection(feat.index)
    pr = pr.loc[idx]
    feat = feat.loc[idx]
    seg = seg_series.loc[idx]
    base = base_series.loc[idx]

    rows = []
    for seg_name in seg.unique():
        mask = (seg == seg_name)
        if mask.sum() == 0:
            continue

        r = pr.loc[mask].dropna()
        if r.empty:
            continue

        start = r.index.min()
        end = r.index.max()

        mu = float(r.mean())
        sig = float(r.std(ddof=1))
        ann_mu = mu * 252.0
        ann_sig = sig * np.sqrt(252.0)
        sharpe = float(ann_mu / ann_sig) if ann_sig > 1e-18 else 0.0

        # simple skew/kurt (moment-based)
        x = r.to_numpy(dtype=float)
        if x.size >= 10 and np.std(x, ddof=1) > 1e-18:
            z = (x - np.mean(x)) / (np.std(x, ddof=1) + 1e-18)
            skew = float(np.mean(z**3))
            kurt_ex = float(np.mean(z**4) - 3.0)
        else:
            skew = float("nan")
            kurt_ex = float("nan")

        # regime drawdown (within that segment)
        eq = equity_curve_from_returns(r.fillna(0.0), starting_capital=1.0)
        dd = dd_series(eq)
        peak = eq.cummax().replace(0.0, np.nan)
        dd_frac = (dd / peak).replace([np.inf, -np.inf], np.nan)
        max_dd = float(dd_frac.max()) if dd_frac.notna().any() else float("nan")

        # feature means
        fmean = feat.loc[mask].mean(numeric_only=True)

        rows.append({
            "Regime": str(seg_name),
            "BaseRegime": str(base.loc[mask].iloc[0]),
            "Start": start,
            "End": end,
            "Days": int(mask.sum()),
            "Prob": float(mask.mean()),
            "Mean(daily)": mu,
            "Std(daily)": sig,
            "AnnReturn": ann_mu,
            "AnnVol": ann_sig,
            "Sharpe": sharpe,
            "Skew": skew,
            "ExcessKurtosis": kurt_ex,
            "MaxDD": max_dd,
            "Feature_vol": float(fmean.get("vol", np.nan)),
            "Feature_dd": float(fmean.get("dd", np.nan)),
            "Feature_avg_corr": float(fmean.get("avg_corr", np.nan)),
            "Feature_composite_score": float(fmean.get("composite_score", np.nan)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise AnalyticsError("Failed to compute regime stats: produced empty table.")
    # stable ordering: chronological by start
    df = df.sort_values(["Start", "Regime"]).reset_index(drop=True)
    return df


def _build_regimes_for_mc(seg_series: pd.Series) -> Dict[str, Tuple[slice, float]]:
    """
    Build dict: segment_label -> (slice(start,end), prob)

    Compatible with stress.regime_mixture_simulation_terminal(...)

    NOTE:
      stress.regime_mixture_simulation_terminal expects each regime to be defined
      by a contiguous time slice. Our segment labels ensure contiguity.
    """
    seg_series = _ensure_series(seg_series, "regime_series")
    seg_series = _ensure_datetime_index(seg_series, "regime_series")

    idx = seg_series.index
    seg = seg_series.astype(str)

    total = float(len(seg))
    if total <= 0:
        raise ValidationError("Empty regime_series; cannot build regimes_for_mc.")

    regimes: Dict[str, Tuple[slice, float]] = {}
    for seg_name in seg.unique():
        mask = (seg == seg_name)
        if mask.sum() == 0:
            continue
        start = idx[mask].min()
        end = idx[mask].max()
        prob = float(mask.sum() / total)
        regimes[str(seg_name)] = (slice(start, end), prob)

    # Normalize probs (numerical safety)
    s = float(sum(v[1] for v in regimes.values()))
    if s <= 0:
        raise ValidationError("Regime probabilities sum to zero.")
    regimes = {k: (v[0], v[1] / s) for k, v in regimes.items()}

    return regimes


# =========================
# PUBLIC API
# =========================

def detect_regimes(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    *,
    cfg: RegimeConfig,
) -> RegimeResult:
    """
    End-to-end regime detection.

    Steps:
      1) Align returns to portfolio
      2) Compute features
      3) Assign base 3-bucket regimes
      4) Enforce minimum segment length
      5) Create segment-aware labels
      6) Compute stats + regimes_for_mc

    Returns RegimeResult.
    """
    cfg = _validate_config(cfg)
    pr, feat, assets, w = compute_regime_features(returns_df, portfolio, cfg)

    # Assign regimes
    if cfg.method == "volatility":
        base = _assign_3bucket(feat["vol"], cfg.labels, cfg.q_low, cfg.q_high)
        key_feature = feat["vol"].to_numpy(dtype=float)
    elif cfg.method == "drawdown":
        base = _assign_3bucket(feat["dd"], cfg.labels, cfg.q_low, cfg.q_high)
        key_feature = feat["dd"].to_numpy(dtype=float)
    elif cfg.method == "correlation":
        if "avg_corr" not in feat.columns:
            raise ValidationError("correlation method requires avg_corr feature (need >=2 assets).")
        base = _assign_3bucket(feat["avg_corr"], cfg.labels, cfg.q_low, cfg.q_high)
        key_feature = feat["avg_corr"].to_numpy(dtype=float)
    elif cfg.method == "composite":
        base, score = _assign_composite(feat, cfg)
        feat = feat.copy()
        feat["composite_score"] = score
        key_feature = score.to_numpy(dtype=float)
    else:
        raise ValidationError("Unknown regime method.")

    # Enforce minimum run length (merge short segments)
    base_merged = _merge_short_runs(base, key_feature, min_len=int(cfg.min_segment_len))

    # Segment-aware labels (contiguous blocks)
    seg_labels, base_labels = _segment_labels(base_merged, cfg.labels)

    seg_series = pd.Series(seg_labels, index=feat.index, name="regime")
    base_series = pd.Series(base_labels, index=feat.index, name="base_regime")

    # Stats + MC slices
    stats = _compute_regime_stats(pr.loc[feat.index], feat, seg_series, base_series)
    regimes_for_mc = _build_regimes_for_mc(seg_series)

    return RegimeResult(
        regime_series=seg_series,
        regime_base_series=base_series,
        features=feat,
        regime_stats=stats,
        regimes_for_mc=regimes_for_mc,
    )


def summarize_regimes_by_base(regime_result: RegimeResult) -> pd.DataFrame:
    """
    Convenience summary aggregated by base regime (Calm/Normal/Stress),
    instead of segment regimes (Stress#01, Stress#02, ...).

    Returns a DataFrame with one row per base regime.
    """
    rr = regime_result
    base = rr.regime_base_series.astype(str)
    idx = base.index.intersection(rr.features.index)

    base = base.loc[idx]
    feat = rr.features.loc[idx]

    rows = []
    for lab in sorted(base.unique()):
        mask = (base == lab)
        if mask.sum() == 0:
            continue
        fmean = feat.loc[mask].mean(numeric_only=True)
        rows.append({
            "BaseRegime": lab,
            "Days": int(mask.sum()),
            "Prob": float(mask.mean()),
            "Feature_vol": float(fmean.get("vol", np.nan)),
            "Feature_dd": float(fmean.get("dd", np.nan)),
            "Feature_avg_corr": float(fmean.get("avg_corr", np.nan)),
            "Feature_composite_score": float(fmean.get("composite_score", np.nan)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise AnalyticsError("Base regime summary produced empty table.")
    out = out.sort_values("BaseRegime").reset_index(drop=True)
    return out


def label_regimes_from_series(
    feature_series: pd.Series,
    *,
    cfg: RegimeConfig,
) -> RegimeResult:
    """
    If you already computed a feature series externally (e.g. VIX),
    this function labels regimes on that series alone and returns a RegimeResult-like object.

    Notes:
      - regime_stats here will be feature-only (no portfolio stats)
      - regimes_for_mc is still segment-based on contiguous blocks
    """
    cfg = _validate_config(cfg)
    feature_series = _ensure_series(feature_series, "feature_series")
    feature_series = _ensure_datetime_index(feature_series, "feature_series")
    feature_series = pd.to_numeric(feature_series, errors="coerce")

    x = _ewma_smooth(feature_series, cfg.ewma_span).dropna()
    if len(x) < cfg.min_obs // 2:
        raise ValidationError("Not enough observations in feature_series for regime labeling.")

    base = _assign_3bucket(x, cfg.labels, cfg.q_low, cfg.q_high)
    key = x.to_numpy(dtype=float)
    base_merged = _merge_short_runs(base, key, min_len=int(cfg.min_segment_len))

    seg_labels, base_labels = _segment_labels(base_merged, cfg.labels)

    seg_series = pd.Series(seg_labels, index=x.index, name="regime")
    base_series = pd.Series(base_labels, index=x.index, name="base_regime")

    feat = pd.DataFrame({"feature": x}, index=x.index)

    # Minimal stats table
    rows = []
    total = float(len(x))
    for seg_name in seg_series.unique():
        mask = (seg_series == seg_name)
        sub = x.loc[mask]
        rows.append({
            "Regime": str(seg_name),
            "BaseRegime": str(base_series.loc[mask].iloc[0]),
            "Start": sub.index.min(),
            "End": sub.index.max(),
            "Days": int(mask.sum()),
            "Prob": float(mask.sum() / total),
            "FeatureMean": float(sub.mean()),
            "FeatureStd": float(sub.std(ddof=1)),
        })
    stats = pd.DataFrame(rows).sort_values(["Start", "Regime"]).reset_index(drop=True)
    regimes_for_mc = _build_regimes_for_mc(seg_series)

    return RegimeResult(
        regime_series=seg_series,
        regime_base_series=base_series,
        features=feat,
        regime_stats=stats,
        regimes_for_mc=regimes_for_mc,
    )
