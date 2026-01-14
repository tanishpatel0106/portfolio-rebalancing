"""
stresslab/analytics/stress.py
=============================
Core stress testing + risk analytics engine for StressLab.

This module is intentionally "math heavy" and production-oriented:
- Robust input validation
- Stable output schemas
- Deterministic computations (seeded MC)
- Clear separation between:
  (a) data alignment/prep
  (b) risk estimation (VaR/ES)
  (c) scenario stress (historical, parametric, macro-proxy)
  (d) Monte Carlo simulation (bootstrap + parametric)
  (e) reporting tables for Streamlit UI

No Streamlit code in this file.

Concepts supported
------------------
1) Portfolio model:
   - holdings: weights (sum to 1) or notional exposures
   - pricing: close prices (or returns provided directly)
   - pnl: linear approximation via returns (sufficient for equity/FX/beta portfolios)

2) Risk:
   - Historical simulation VaR / ES
   - Parametric (Normal) VaR / ES
   - Cornish-Fisher VaR (optional) for skew/kurt adjustment
   - Rolling risk time-series (for charts)

3) Stress:
   - Historical worst-k windows
   - Custom scenario shocks by asset
   - Factor / macro proxy shocks: shock SPX, VIX etc and map to portfolio via betas
   - Correlation stress: inflate correlations toward 1 (or specified target), re-estimate risk

4) Monte Carlo:
   - Parametric MVN with covariance shrinkage options
   - Block bootstrap (to preserve autocorr)
   - Regime-mix simulation (mixture of covariances)

This file will be used by UI and by unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# =========================
# ERRORS
# =========================

class AnalyticsError(RuntimeError):
    """Base analytics error."""


class ValidationError(AnalyticsError):
    """Raised when inputs fail validation."""


# =========================
# UTILITIES
# =========================

Number = Union[int, float, np.floating]


def _ensure_df(x, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame.")
    if x.empty:
        raise ValidationError(f"{name} cannot be empty.")
    return x


def _ensure_series(x, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise ValidationError(f"{name} must be a pandas Series.")
    if x.empty:
        raise ValidationError(f"{name} cannot be empty.")
    return x


def _ensure_datetime_index(obj: Union[pd.DataFrame, pd.Series], name: str) -> Union[pd.DataFrame, pd.Series]:
    if not isinstance(obj.index, pd.DatetimeIndex):
        try:
            obj = obj.copy()
            obj.index = pd.to_datetime(obj.index)
        except Exception as e:
            raise ValidationError(f"{name} index must be DatetimeIndex (or convertible): {e}")
    if getattr(obj.index, "tz", None) is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_convert(None)
    return obj


def _sorted_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _as_float_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _intersect_columns(prices: pd.DataFrame, assets: Sequence[str]) -> List[str]:
    cols = [a for a in assets if a in prices.columns]
    if not cols:
        raise ValidationError("No requested assets were found in the price/return dataframe columns.")
    return cols


def _validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, (int, float, np.floating)):
        raise ValidationError("alpha must be numeric.")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValidationError("alpha must be in (0, 1). Example: 0.05 for 95% VaR.")
    return a


def _nan_guard(x: np.ndarray, msg: str) -> np.ndarray:
    if not np.isfinite(x).all():
        raise AnalyticsError(msg)
    return x


def _cov_shrink_lw(sample_cov: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf style diagonal shrinkage (simple deterministic approximation).
    This is not sklearn's exact LW but a robust shrink toward diagonal for stability.

    cov_shrunk = (1 - delta) * S + delta * D
    where D = diag(S)
    delta chosen based on ratio of off-diagonal energy.
    """
    S = np.array(sample_cov, dtype=float, copy=True)
    p = S.shape[0]
    if S.shape[0] != S.shape[1]:
        raise ValidationError("Covariance must be square.")

    D = np.diag(np.diag(S))
    off = S - D
    off_energy = float(np.sum(off**2))
    total_energy = float(np.sum(S**2)) + 1e-12
    # Heuristic shrinkage intensity: more off-diagonal noise => more shrink
    delta = np.clip(off_energy / total_energy, 0.0, 0.95)
    return (1.0 - delta) * S + delta * D


def _nearest_psd(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest Positive Semi-Definite matrix
    via eigenvalue clipping.
    """
    A = np.array(A, dtype=float, copy=True)
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w_clipped = np.clip(w, eps, None)
    return (V * w_clipped) @ V.T


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.array(cov, dtype=float, copy=False)
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    invd = 1.0 / d
    corr = cov * invd[:, None] * invd[None, :]
    corr = np.clip(corr, -1.0, 1.0)
    corr[np.diag_indices_from(corr)] = 1.0
    return corr


def _cov_from_corr(corr: np.ndarray, vol: np.ndarray) -> np.ndarray:
    vol = np.array(vol, dtype=float, copy=False)
    return corr * vol[:, None] * vol[None, :]


# =========================
# DATA STRUCTURES
# =========================

@dataclass(frozen=True)
class PortfolioSpec:
    """
    Portfolio definition.

    weights:
      - dict {asset: weight} that should sum to 1 (we can normalize if requested)
      - if notional_mode=True, treat as notionals (e.g., $ exposure) and compute P&L in dollars
    """
    weights: Dict[str, float]
    normalize_weights: bool = True
    notional_mode: bool = False  # if True: weights are notionals, pnl is in dollars; else returns are weighted returns


@dataclass(frozen=True)
class RiskConfig:
    alpha: float = 0.05
    method: Literal["historical", "parametric_normal", "cornish_fisher"] = "historical"
    annualization_factor: int = 252
    ewma_lambda: Optional[float] = None  # if provided: use EWMA volatility/cov
    cov_shrink: Optional[Literal["lw_diag"]] = "lw_diag"  # None or "lw_diag"


@dataclass(frozen=True)
class MonteCarloConfig:
    n_sims: int = 50_000
    horizon_days: int = 10
    seed: int = 42
    method: Literal["parametric_mvn", "bootstrap", "block_bootstrap", "regime_mixture"] = "parametric_mvn"
    block_size: int = 5  # for block bootstrap
    cov_shrink: Optional[Literal["lw_diag"]] = "lw_diag"
    use_nearest_psd: bool = True


@dataclass(frozen=True)
class Scenario:
    """
    Generic scenario description.

    Types:
      - "asset_shock": direct return shocks by asset (e.g. {"SPY": -0.05, "TLT": +0.02})
      - "factor_beta": apply shocks to factors and map via betas (portfolio must provide betas)
      - "corr_stress": inflate correlations (needs returns history)
    """
    name: str
    kind: Literal["asset_shock", "factor_beta", "corr_stress", "historical_window"]
    # for asset_shock
    shocks: Optional[Dict[str, float]] = None
    # for factor_beta
    factor_shocks: Optional[Dict[str, float]] = None
    # for corr_stress
    corr_target: Optional[float] = None  # e.g., 0.75 -> push average corr upward
    corr_blend: Optional[float] = None   # blend weight toward target corr
    # for historical_window
    window_days: Optional[int] = None
    pick: Optional[Literal["worst", "best"]] = "worst"


# =========================
# RETURNS / ALIGNMENT
# =========================

def compute_returns_from_prices(
    prices: pd.DataFrame,
    *,
    method: Literal["simple", "log"] = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Compute returns from price levels.

    prices: DataFrame indexed by date, columns assets, float
    method:
      - "simple": pct_change
      - "log": log returns
    """
    prices = _ensure_df(prices, "prices")
    prices = _ensure_datetime_index(prices, "prices")
    prices = _sorted_unique_index(prices)
    prices = _as_float_df(prices)

    if method == "simple":
        rets = prices.pct_change()
    elif method == "log":
        rets = np.log(prices).diff()
    else:
        raise ValidationError("method must be 'simple' or 'log'.")

    rets = rets.replace([np.inf, -np.inf], np.nan)
    if dropna:
        rets = rets.dropna(how="all")
    return rets


def align_returns_and_portfolio(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    *,
    min_non_na_frac: float = 0.95,
    drop_dates_with_any_na: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Align returns to the portfolio universe and return:
      - aligned returns [T x N]
      - weights vector [N]
      - assets list [N]

    min_non_na_frac:
      - For each asset, require at least this fraction of non-NA dates.
    drop_dates_with_any_na:
      - If True: drop dates where any asset is NA after filtering.
      - If False: fill remaining NA with 0 (NOT recommended for real risk; but can be used)
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    returns_df = _ensure_datetime_index(returns_df, "returns_df")
    returns_df = _sorted_unique_index(returns_df)
    returns_df = _as_float_df(returns_df)

    if not portfolio.weights:
        raise ValidationError("portfolio.weights cannot be empty.")

    assets = list(portfolio.weights.keys())
    assets = _intersect_columns(returns_df, assets)

    rets = returns_df[assets].copy()

    # asset-level NA screening
    non_na = rets.notna().mean()
    keep_assets = [a for a in assets if float(non_na[a]) >= float(min_non_na_frac)]
    if not keep_assets:
        raise ValidationError(
            "All assets failed NA coverage threshold. "
            "Try lowering min_non_na_frac or extending history."
        )

    rets = rets[keep_assets].copy()

    if drop_dates_with_any_na:
        rets = rets.dropna(how="any")
    else:
        rets = rets.fillna(0.0)

    if rets.empty:
        raise ValidationError("No return rows remain after alignment/NA filtering.")

    # weights vector
    w = np.array([float(portfolio.weights[a]) for a in keep_assets], dtype=float)
    if portfolio.normalize_weights and not portfolio.notional_mode:
        s = float(np.sum(w))
        if abs(s) < 1e-12:
            raise ValidationError("Sum of weights is zero; cannot normalize.")
        w = w / s

    return rets, w, keep_assets


def portfolio_returns(
    aligned_returns: pd.DataFrame,
    weights: np.ndarray,
    *,
    notional_mode: bool = False,
    starting_capital: float = 1.0,
) -> pd.Series:
    """
    Compute portfolio P&L series.

    If notional_mode=False:
      - portfolio return = R_t · w
      - starting_capital is not applied (returns series)

    If notional_mode=True:
      - treat weights as notionals in dollars
      - pnl_t = sum_i (notional_i * return_{t,i})
      - if you want equity curve, integrate pnl onto starting_capital.
    """
    aligned_returns = _ensure_df(aligned_returns, "aligned_returns")
    if aligned_returns.shape[1] != len(weights):
        raise ValidationError("weights length must match aligned_returns columns.")
    R = aligned_returns.to_numpy(dtype=float)
    w = np.array(weights, dtype=float)

    if notional_mode:
        pnl = R @ w
        s = pd.Series(pnl, index=aligned_returns.index, name="portfolio_pnl")
        return s
    else:
        pr = R @ w
        s = pd.Series(pr, index=aligned_returns.index, name="portfolio_return")
        return s


def equity_curve_from_returns(
    returns: pd.Series,
    *,
    starting_capital: float = 10_000.0,
) -> pd.Series:
    returns = _ensure_series(returns, "returns")
    returns = _ensure_datetime_index(returns, "returns")
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    eq = (1.0 + r).cumprod() * float(starting_capital)
    eq.name = "equity"
    return eq


def equity_curve_from_pnl(
    pnl: pd.Series,
    *,
    starting_capital: float = 10_000.0,
) -> pd.Series:
    pnl = _ensure_series(pnl, "pnl")
    pnl = _ensure_datetime_index(pnl, "pnl")
    p = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    eq = p.cumsum() + float(starting_capital)
    eq.name = "equity"
    return eq


def drawdown(equity: pd.Series) -> pd.Series:
    equity = _ensure_series(equity, "equity")
    equity = _ensure_datetime_index(equity, "equity")
    e = pd.to_numeric(equity, errors="coerce").fillna(method="ffill").fillna(0.0)
    peak = e.cummax()
    dd = peak - e
    dd.name = "drawdown"
    return dd


# =========================
# VAR / ES
# =========================

def var_es_historical(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Historical (empirical) VaR and ES at level alpha.

    Returns:
      VaR (positive number representing loss quantile),
      ES  (positive number representing mean loss beyond VaR)
    """
    a = _validate_alpha(alpha)
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size < 50:
        raise ValidationError("Need at least 50 return observations for stable VaR/ES.")
    # Losses: L = -R
    L = -r
    q = np.quantile(L, 1.0 - (1.0 - a))  # equivalently quantile at (1-a) for losses
    # For clarity: VaR_alpha means P(L > VaR)=alpha? Typically VaR at 95% => alpha=0.05 => quantile 0.95
    var = float(np.quantile(L, 1.0 - a))
    tail = L[L >= var]
    es = float(np.mean(tail)) if tail.size > 0 else float(var)
    return var, es


def var_es_parametric_normal(returns: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Parametric Normal VaR/ES.

    If R ~ N(mu, sigma^2), then loss L = -R.
    VaR_alpha (loss) at (1-alpha) quantile:
      VaR = -(mu + sigma z_alpha) where z_alpha = Phi^{-1}(alpha) ??? careful sign.
    We'll compute on losses directly:
      L ~ N(-mu, sigma^2)
      VaR = mean_L + sigma * z_{1-alpha}
      ES = mean_L + sigma * phi(z_{1-alpha}) / alpha
    """
    a = _validate_alpha(alpha)
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size < 50:
        raise ValidationError("Need at least 50 return observations for stable VaR/ES.")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig <= 1e-18:
        return 0.0, 0.0

    # z for loss tail at probability (1-alpha)
    from math import sqrt, exp, pi
    # Inverse normal quantile using numpy's erf approximation via scipy not available.
    # Use a robust approximation (Acklam) implemented below.
    z = _norm_ppf(1.0 - a)
    # standard normal pdf
    phi = (1.0 / sqrt(2.0 * pi)) * exp(-0.5 * z * z)

    mean_L = -mu
    var = mean_L + sig * z
    es = mean_L + sig * (phi / a)
    return float(var), float(es)


def var_cornish_fisher(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Cornish-Fisher VaR approximation to adjust for skewness/kurtosis.
    Returns VaR on losses.

    z_cf = z + (1/6)(z^2-1)S + (1/24)(z^3-3z)K - (1/36)(2z^3-5z)S^2
    where S=skew, K=excess kurtosis

    VaR = mean_L + sigma * z_cf
    """
    a = _validate_alpha(alpha)
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if r.size < 100:
        raise ValidationError("Need at least 100 observations for Cornish-Fisher stability.")
    mu = float(np.mean(r))
    sig = float(np.std(r, ddof=1))
    if sig <= 1e-18:
        return 0.0

    # moments
    x = (r - mu) / (sig + 1e-18)
    S = float(np.mean(x**3))
    K = float(np.mean(x**4) - 3.0)

    z = _norm_ppf(1.0 - a)
    z2 = z*z
    z3 = z2*z

    z_cf = (
        z
        + (1.0/6.0) * (z2 - 1.0) * S
        + (1.0/24.0) * (z3 - 3.0*z) * K
        - (1.0/36.0) * (2.0*z3 - 5.0*z) * (S*S)
    )

    mean_L = -mu
    var = mean_L + sig * z_cf
    return float(var)


def compute_var_es(
    returns: pd.Series,
    config: RiskConfig,
) -> Dict[str, float]:
    """
    Unified VaR/ES interface returning dict:
      {"VaR": ..., "ES": ..., "mu": ..., "sigma": ...}
    """
    a = _validate_alpha(config.alpha)
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.size < 50:
        raise ValidationError("Need >= 50 observations for risk estimation.")

    mu = float(r.mean())
    sig = float(r.std(ddof=1))

    if config.method == "historical":
        var, es = var_es_historical(r, a)
    elif config.method == "parametric_normal":
        var, es = var_es_parametric_normal(r, a)
    elif config.method == "cornish_fisher":
        var = var_cornish_fisher(r, a)
        # ES under CF is non-trivial; keep ES as historical tail mean for robustness
        # (production choice: avoid fake precision)
        _, es = var_es_historical(r, a)
    else:
        raise ValidationError("Unknown risk method.")

    return {"VaR": float(var), "ES": float(es), "mu": mu, "sigma": sig}


def rolling_var_series(
    returns: pd.Series,
    config: RiskConfig,
    window: int = 252,
) -> pd.DataFrame:
    """
    Rolling VaR/ES time series for charting.
    Returns DataFrame with columns ["VaR", "ES"] indexed by date.
    """
    returns = _ensure_series(returns, "returns")
    returns = _ensure_datetime_index(returns, "returns")
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < window + 10:
        raise ValidationError("Not enough observations for rolling risk.")

    out = []
    idx = []
    for i in range(window, len(r) + 1):
        slice_ = r.iloc[i - window:i]
        res = compute_var_es(slice_, config)
        out.append([res["VaR"], res["ES"]])
        idx.append(r.index[i - 1])

    df = pd.DataFrame(out, columns=["VaR", "ES"], index=pd.DatetimeIndex(idx))
    return df


# =========================
# SCENARIO STRESS
# =========================

def apply_asset_shock(
    weights: np.ndarray,
    assets: List[str],
    shocks: Dict[str, float],
    *,
    notional_mode: bool = False,
) -> Dict[str, float]:
    """
    Compute portfolio shock P&L / return under direct asset return shocks.

    If notional_mode=False: return = sum w_i * shock_i
    If notional_mode=True: pnl = sum notional_i * shock_i
    """
    if shocks is None or not isinstance(shocks, dict) or len(shocks) == 0:
        raise ValidationError("shocks must be a non-empty dict of {asset: shock_return}.")
    if len(weights) != len(assets):
        raise ValidationError("weights/assets mismatch.")

    shock_vec = np.zeros(len(assets), dtype=float)
    for i, a in enumerate(assets):
        if a in shocks:
            shock_vec[i] = float(shocks[a])
        else:
            shock_vec[i] = 0.0

    if notional_mode:
        pnl = float(np.dot(weights, shock_vec))
        return {"pnl": pnl, "return": np.nan}
    else:
        ret = float(np.dot(weights, shock_vec))
        return {"return": ret, "pnl": np.nan}


def estimate_factor_betas(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    add_intercept: bool = True,
    ridge_lambda: float = 0.0,
) -> pd.DataFrame:
    """
    Estimate betas of each asset to factors via OLS (optionally ridge).

    Returns DataFrame:
      index=asset, columns=[factor1, factor2, ..., "intercept"(optional)]
    """
    asset_returns = _ensure_df(asset_returns, "asset_returns")
    factor_returns = _ensure_df(factor_returns, "factor_returns")
    asset_returns = _ensure_datetime_index(asset_returns, "asset_returns")
    factor_returns = _ensure_datetime_index(factor_returns, "factor_returns")

    # align on intersection dates
    idx = asset_returns.index.intersection(factor_returns.index)
    if len(idx) < 50:
        raise ValidationError("Need at least 50 overlapping observations to estimate betas.")

    Y = asset_returns.loc[idx].dropna(how="any")
    X = factor_returns.loc[idx].reindex(Y.index).dropna(how="any")

    # final align
    idx2 = Y.index.intersection(X.index)
    Y = Y.loc[idx2]
    X = X.loc[idx2]
    if len(idx2) < 50:
        raise ValidationError("Not enough clean overlapping observations after NA drops.")

    # Build design matrix
    Xmat = X.to_numpy(dtype=float)
    if add_intercept:
        Xmat = np.column_stack([np.ones(len(Xmat)), Xmat])
        cols = ["intercept"] + list(X.columns)
    else:
        cols = list(X.columns)

    # Precompute (X'X + λI)^-1 X'
    XtX = Xmat.T @ Xmat
    if ridge_lambda > 0:
        XtX = XtX + ridge_lambda * np.eye(XtX.shape[0])
    XtX_inv = np.linalg.inv(XtX)
    B = XtX_inv @ Xmat.T  # (k x T)

    betas = {}
    for asset in Y.columns:
        y = Y[asset].to_numpy(dtype=float)
        b = B @ y
        betas[asset] = b

    beta_df = pd.DataFrame(betas, index=cols).T
    return beta_df


def apply_factor_shock(
    weights: np.ndarray,
    assets: List[str],
    factor_shocks: Dict[str, float],
    betas: pd.DataFrame,
    *,
    notional_mode: bool = False,
    include_intercept: bool = False,
) -> Dict[str, float]:
    """
    Map factor shocks to asset returns via betas, then compute portfolio impact.

    asset_shock_i = sum_f beta_{i,f} * shock_f
    """
    if betas is None or betas.empty:
        raise ValidationError("betas must be a non-empty DataFrame.")
    if factor_shocks is None or not isinstance(factor_shocks, dict) or len(factor_shocks) == 0:
        raise ValidationError("factor_shocks must be a non-empty dict.")
    if len(weights) != len(assets):
        raise ValidationError("weights/assets mismatch.")

    # Determine beta columns to use
    beta_cols = list(betas.columns)
    if not include_intercept and "intercept" in beta_cols:
        beta_cols = [c for c in beta_cols if c != "intercept"]

    # Build shock vector aligned to beta_cols
    svec = np.zeros(len(beta_cols), dtype=float)
    for j, f in enumerate(beta_cols):
        svec[j] = float(factor_shocks.get(f, 0.0))

    # Compute implied asset shocks
    shock_vec = np.zeros(len(assets), dtype=float)
    for i, a in enumerate(assets):
        if a not in betas.index:
            raise ValidationError(f"Missing betas for asset '{a}'.")
        b = betas.loc[a, beta_cols].to_numpy(dtype=float)
        shock_vec[i] = float(np.dot(b, svec))

    if notional_mode:
        pnl = float(np.dot(weights, shock_vec))
        return {"pnl": pnl, "return": np.nan}
    else:
        ret = float(np.dot(weights, shock_vec))
        return {"return": ret, "pnl": np.nan}


def worst_historical_window(
    portfolio_ret: pd.Series,
    window_days: int = 10,
    *,
    pick: Literal["worst", "best"] = "worst",
) -> Dict[str, object]:
    """
    Identify worst (or best) cumulative return window in history.

    Uses simple cumulative: (1+r).prod - 1 over rolling window.
    """
    portfolio_ret = _ensure_series(portfolio_ret, "portfolio_ret")
    portfolio_ret = _ensure_datetime_index(portfolio_ret, "portfolio_ret")
    r = pd.to_numeric(portfolio_ret, errors="coerce").dropna()

    if len(r) < window_days + 5:
        raise ValidationError("Not enough data for historical window search.")

    roll = (1.0 + r).rolling(window_days).apply(lambda x: np.prod(x) - 1.0, raw=True)
    roll = roll.dropna()

    if roll.empty:
        raise ValidationError("Rolling window computation produced empty output.")

    if pick == "worst":
        end = roll.idxmin()
        val = float(roll.loc[end])
    elif pick == "best":
        end = roll.idxmax()
        val = float(roll.loc[end])
    else:
        raise ValidationError("pick must be 'worst' or 'best'.")

    end_loc = r.index.get_loc(end)
    start_loc = end_loc - window_days + 1
    start = r.index[start_loc]

    return {
        "window_days": int(window_days),
        "pick": pick,
        "start": start,
        "end": end,
        "cumulative_return": val,
    }


def corr_stress_cov(
    cov: np.ndarray,
    *,
    target_corr: float = 0.75,
    blend: float = 0.50,
    method: Literal["toward_target", "toward_one"] = "toward_target",
) -> np.ndarray:
    """
    Stress covariance by inflating correlations.

    Steps:
      1) corr = corr(cov)
      2) corr_stressed = (1-blend)*corr + blend*target_matrix
      3) cov_stressed = corr_stressed * vol_i * vol_j

    target_matrix:
      - toward_target: all off-diagonals = target_corr
      - toward_one: all off-diagonals = 1.0
    """
    cov = np.array(cov, dtype=float, copy=True)
    if cov.shape[0] != cov.shape[1]:
        raise ValidationError("cov must be square.")
    if not (0.0 <= blend <= 1.0):
        raise ValidationError("blend must be in [0,1].")
    if not (-1.0 <= target_corr <= 1.0):
        raise ValidationError("target_corr must be in [-1,1].")

    corr = _corr_from_cov(cov)
    vol = np.sqrt(np.clip(np.diag(cov), 1e-18, None))

    p = corr.shape[0]
    if method == "toward_target":
        T = np.full((p, p), float(target_corr), dtype=float)
        np.fill_diagonal(T, 1.0)
    elif method == "toward_one":
        T = np.ones((p, p), dtype=float)
        np.fill_diagonal(T, 1.0)
    else:
        raise ValidationError("Unknown method for corr stress.")

    corr_s = (1.0 - blend) * corr + blend * T
    corr_s = np.clip(corr_s, -1.0, 1.0)
    np.fill_diagonal(corr_s, 1.0)

    cov_s = _cov_from_corr(corr_s, vol)

    # Project to PSD for safe simulation
    cov_s = _nearest_psd(cov_s)
    return cov_s


# =========================
# COVARIANCE ESTIMATION
# =========================

def sample_mean_cov(
    returns_df: pd.DataFrame,
    *,
    ewma_lambda: Optional[float] = None,
    cov_shrink: Optional[Literal["lw_diag"]] = "lw_diag",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate mean vector and covariance matrix.

    If ewma_lambda is provided:
      - mu = EWMA mean
      - cov = EWMA covariance
    Else:
      - mu = sample mean
      - cov = sample covariance

    cov_shrink:
      - "lw_diag": diagonal shrink for stability
      - None
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    returns_df = _ensure_datetime_index(returns_df, "returns_df")
    returns_df = returns_df.dropna(how="any")
    if len(returns_df) < 50:
        raise ValidationError("Need >= 50 rows for covariance estimation.")

    X = returns_df.to_numpy(dtype=float)
    T, N = X.shape

    if ewma_lambda is None:
        mu = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False, ddof=1)
    else:
        lam = float(ewma_lambda)
        if not (0.0 < lam < 1.0):
            raise ValidationError("ewma_lambda must be in (0,1). Typical: 0.94.")
        # weights: newest has highest weight
        w = np.array([(1 - lam) * (lam ** (T - 1 - t)) for t in range(T)], dtype=float)
        w = w / (np.sum(w) + 1e-18)

        mu = np.sum(X * w[:, None], axis=0)
        Xc = X - mu[None, :]
        cov = (Xc.T * w) @ Xc  # weighted covariance

    cov = np.array(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)

    if cov_shrink == "lw_diag":
        cov = _cov_shrink_lw(cov)
    elif cov_shrink is None:
        pass
    else:
        raise ValidationError("cov_shrink must be None or 'lw_diag'.")

    cov = _nearest_psd(cov)
    mu = np.array(mu, dtype=float)
    return mu, cov


# =========================
# MONTE CARLO
# =========================

def simulate_portfolio_paths_parametric(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    *,
    n_sims: int,
    horizon_days: int,
    seed: int,
    notional_mode: bool = False,
    use_nearest_psd: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Simulate horizon_days returns via MVN(mu, cov) iid each day.

    Outputs:
      - sim_portfolio_returns: (n_sims, horizon_days)
      - sim_terminal: (n_sims,) terminal cumulative return (or pnl if notional_mode)
      - sim_cum_paths: (n_sims, horizon_days) cumulative path
    """
    if n_sims <= 0 or horizon_days <= 0:
        raise ValidationError("n_sims and horizon_days must be positive.")
    mu = np.array(mu, dtype=float)
    cov = np.array(cov, dtype=float)
    w = np.array(weights, dtype=float)

    if cov.shape[0] != cov.shape[1]:
        raise ValidationError("cov must be square.")
    if len(mu) != cov.shape[0] or len(w) != cov.shape[0]:
        raise ValidationError("mu/cov/weights dimensions mismatch.")

    if use_nearest_psd:
        cov = _nearest_psd(cov)

    rng = np.random.default_rng(int(seed))

    # draw daily returns for assets
    # shape: (n_sims, horizon_days, N)
    N = cov.shape[0]
    R = rng.multivariate_normal(mean=mu, cov=cov, size=(n_sims, horizon_days))

    # portfolio return each day
    # (n_sims, horizon_days)
    port = np.einsum("thn,n->th", R, w)

    if notional_mode:
        # interpret w as notionals, so port is pnl per day in dollars only if R is return;
        # but einsum already uses notional*return, so port = pnl/day.
        pnl = port
        cum = np.cumsum(pnl, axis=1)
        terminal = cum[:, -1]
        return {
            "sim_portfolio": pnl,
            "sim_cum_paths": cum,
            "sim_terminal": terminal,
        }
    else:
        # compound
        cum = np.cumprod(1.0 + port, axis=1) - 1.0
        terminal = cum[:, -1]
        return {
            "sim_portfolio": port,
            "sim_cum_paths": cum,
            "sim_terminal": terminal,
        }


def bootstrap_simulation_terminal(
    portfolio_returns: pd.Series,
    *,
    n_sims: int,
    horizon_days: int,
    seed: int,
    block_size: int = 1,
) -> np.ndarray:
    """
    Bootstrap simulation of terminal cumulative return for the portfolio.

    block_size=1 => iid bootstrap
    block_size>1 => block bootstrap (preserves short autocorrelation)

    Output:
      terminal cumulative return array shape (n_sims,)
    """
    pr = _ensure_series(portfolio_returns, "portfolio_returns")
    pr = pd.to_numeric(pr, errors="coerce").dropna()
    if len(pr) < 200:
        raise ValidationError("Bootstrap MC needs >= 200 observations for stability.")
    if block_size <= 0:
        raise ValidationError("block_size must be >= 1.")

    x = pr.to_numpy(dtype=float)
    T = len(x)

    rng = np.random.default_rng(int(seed))
    terminals = np.zeros(n_sims, dtype=float)

    if block_size == 1:
        # iid
        idx = rng.integers(0, T, size=(n_sims, horizon_days))
        samples = x[idx]  # (n_sims, horizon_days)
        terminals = np.prod(1.0 + samples, axis=1) - 1.0
        return terminals

    # block bootstrap: sample starting indices for blocks
    n_blocks = int(np.ceil(horizon_days / block_size))
    max_start = T - block_size
    if max_start <= 1:
        raise ValidationError("Time series too short for chosen block_size.")

    for s in range(n_sims):
        path = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, max_start))
            path.extend(x[start:start + block_size].tolist())
        path = np.array(path[:horizon_days], dtype=float)
        terminals[s] = float(np.prod(1.0 + path) - 1.0)

    return terminals


def regime_mixture_simulation_terminal(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    *,
    n_sims: int,
    horizon_days: int,
    seed: int,
    regimes: Dict[str, Tuple[slice, float]],  # name -> (time slice, probability)
    ewma_lambda: Optional[float] = None,
    cov_shrink: Optional[Literal["lw_diag"]] = "lw_diag",
) -> np.ndarray:
    """
    Regime mixture simulation:
      - Define regimes by time slices in returns_df (e.g., crisis vs calm)
      - Estimate mu/cov per regime
      - For each simulation path, sample a regime for each day according to regime probabilities
      - Simulate daily returns from that regime's MVN

    Output:
      terminal cumulative return array shape (n_sims,)
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    returns_df = returns_df.dropna(how="any")
    if len(returns_df) < 300:
        raise ValidationError("Need >= 300 observations for regime mixture stability.")
    if not regimes or not isinstance(regimes, dict):
        raise ValidationError("regimes must be a non-empty dict {name: (slice, prob)}.")

    # normalize probs
    names = list(regimes.keys())
    probs = np.array([float(regimes[n][1]) for n in names], dtype=float)
    if np.any(probs < 0):
        raise ValidationError("Regime probabilities must be non-negative.")
    if probs.sum() <= 0:
        raise ValidationError("Sum of regime probabilities must be > 0.")
    probs = probs / probs.sum()

    # estimate per regime
    mu_map = {}
    cov_map = {}
    for n in names:
        slc = regimes[n][0]
        df_r = returns_df.loc[slc].copy()
        if len(df_r) < 100:
            raise ValidationError(f"Regime '{n}' has < 100 observations; widen slice.")
        mu, cov = sample_mean_cov(df_r, ewma_lambda=ewma_lambda, cov_shrink=cov_shrink)
        mu_map[n] = mu
        cov_map[n] = cov

    rng = np.random.default_rng(int(seed))
    terminals = np.zeros(n_sims, dtype=float)

    for s in range(n_sims):
        cum = 1.0
        for _ in range(horizon_days):
            regime = rng.choice(names, p=probs)
            mu = mu_map[regime]
            cov = cov_map[regime]
            r = rng.multivariate_normal(mean=mu, cov=cov)
            port_r = float(np.dot(r, weights))
            cum *= (1.0 + port_r)
        terminals[s] = cum - 1.0

    return terminals


# =========================
# END-TO-END RISK + STRESS API
# =========================

def portfolio_risk_report(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    *,
    risk: RiskConfig,
    starting_capital: float = 10_000.0,
) -> Dict[str, object]:
    """
    Compute core portfolio series + point-in-time risk metrics.

    Returns dict with:
      - assets
      - weights
      - portfolio_return_series
      - equity_curve
      - drawdown
      - metrics: {VaR, ES, mu, sigma, Sharpe, Vol, MaxDD, TotalReturn, ...}
    """
    rets, w, assets = align_returns_and_portfolio(
        returns_df,
        portfolio,
        min_non_na_frac=0.95,
        drop_dates_with_any_na=True,
    )

    pr = portfolio_returns(rets, w, notional_mode=portfolio.notional_mode)

    if portfolio.notional_mode:
        eq = equity_curve_from_pnl(pr, starting_capital=starting_capital)
    else:
        eq = equity_curve_from_returns(pr, starting_capital=starting_capital)

    dd = drawdown(eq)
    risk_res = compute_var_es(pr if not portfolio.notional_mode else (pr / starting_capital), risk)

    # Additional metrics
    # If pnl-mode, approximate "return series" by pnl/capital for volatility/sharpe
    if portfolio.notional_mode:
        r_for_stats = (pr / float(starting_capital)).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        r_for_stats = pr.replace([np.inf, -np.inf], np.nan).dropna()

    ann = float(risk.annualization_factor)
    mu = float(r_for_stats.mean())
    vol = float(r_for_stats.std(ddof=1)) * np.sqrt(ann)
    ann_ret = float(mu) * ann
    sharpe = float(ann_ret / vol) if vol > 1e-18 else 0.0

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else np.nan
    max_dd = float(dd.max())

    metrics = {
        "Annualized Return": ann_ret,
        "Annualized Volatility": vol,
        "Sharpe Ratio": sharpe,
        "Total Return": total_return,
        "Max Drawdown ($)": max_dd,
        "VaR (alpha)": float(risk_res["VaR"]),
        "ES (alpha)": float(risk_res["ES"]),
        "Mean (daily)": float(risk_res["mu"]),
        "Std (daily)": float(risk_res["sigma"]),
        "Alpha": float(risk.alpha),
        "Method": risk.method,
        "Rows Used": int(len(rets)),
    }

    return {
        "assets": assets,
        "weights": w,
        "aligned_returns": rets,
        "portfolio_series": pr,
        "equity_curve": eq,
        "drawdown": dd,
        "metrics": metrics,
    }


def run_scenarios(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    scenarios: Sequence[Scenario],
    *,
    factor_returns: Optional[pd.DataFrame] = None,
    betas: Optional[pd.DataFrame] = None,
    starting_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Execute scenario set and return a normalized scenario results table.

    Output schema:
      columns: ["Scenario", "Kind", "ImpactType", "ImpactValue", "Unit", "Details"]
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    rets, w, assets = align_returns_and_portfolio(returns_df, portfolio, min_non_na_frac=0.95, drop_dates_with_any_na=True)

    # Build base for historical window scenario
    pr = portfolio_returns(rets, w, notional_mode=portfolio.notional_mode)

    rows = []
    for sc in scenarios:
        if sc.kind == "asset_shock":
            res = apply_asset_shock(w, assets, sc.shocks or {}, notional_mode=portfolio.notional_mode)
            if portfolio.notional_mode:
                rows.append([sc.name, sc.kind, "PnL", res["pnl"], "USD", f"assets={len(assets)}"])
            else:
                rows.append([sc.name, sc.kind, "Return", res["return"], "fraction", f"assets={len(assets)}"])

        elif sc.kind == "factor_beta":
            if betas is None:
                if factor_returns is None:
                    raise ValidationError("factor_returns is required to estimate betas for factor_beta scenarios.")
                # estimate betas from history
                betas_est = estimate_factor_betas(rets, factor_returns.loc[rets.index], add_intercept=True, ridge_lambda=1e-6)
            else:
                betas_est = betas

            res = apply_factor_shock(
                w, assets, sc.factor_shocks or {}, betas_est,
                notional_mode=portfolio.notional_mode,
                include_intercept=False,
            )
            if portfolio.notional_mode:
                rows.append([sc.name, sc.kind, "PnL", res["pnl"], "USD", "mapped via betas"])
            else:
                rows.append([sc.name, sc.kind, "Return", res["return"], "fraction", "mapped via betas"])

        elif sc.kind == "corr_stress":
            # stress covariance and compute new VaR/ES on portfolio under MVN
            target = float(sc.corr_target if sc.corr_target is not None else 0.75)
            blend = float(sc.corr_blend if sc.corr_blend is not None else 0.50)
            mu, cov = sample_mean_cov(rets, ewma_lambda=None, cov_shrink="lw_diag")
            cov_s = corr_stress_cov(cov, target_corr=target, blend=blend, method="toward_target")

            # Parametric portfolio std with stressed cov
            port_mu = float(np.dot(mu, w))
            port_var = float(w.T @ cov_s @ w)
            port_sig = float(np.sqrt(max(port_var, 0.0)))

            # Convert to 1-day loss VaR at alpha via normal approx
            # L ~ N(-mu_p, sig_p^2) => VaR = mean_L + sig*z_{1-alpha}
            a = 0.05
            z = _norm_ppf(1.0 - a)
            var = (-port_mu) + port_sig * z

            rows.append([sc.name, sc.kind, "VaR", float(var), "fraction", f"target_corr={target}, blend={blend}"])

        elif sc.kind == "historical_window":
            wd = int(sc.window_days or 10)
            info = worst_historical_window(pr if not portfolio.notional_mode else (pr / starting_capital), window_days=wd, pick=sc.pick or "worst")
            rows.append([sc.name, sc.kind, "CumulativeReturn", float(info["cumulative_return"]), "fraction", f"{info['start']}→{info['end']}"])

        else:
            raise ValidationError(f"Unknown scenario kind: {sc.kind}")

    df = pd.DataFrame(rows, columns=["Scenario", "Kind", "ImpactType", "ImpactValue", "Unit", "Details"])
    return df


def monte_carlo_report(
    returns_df: pd.DataFrame,
    portfolio: PortfolioSpec,
    *,
    mc: MonteCarloConfig,
    risk_alpha: float = 0.05,
) -> Dict[str, object]:
    """
    Run Monte Carlo simulation and return:
      - terminals (array)
      - stats (VaR/ES of terminal distribution, mean, std, percentiles)
      - config echo
    """
    returns_df = _ensure_df(returns_df, "returns_df")
    rets, w, assets = align_returns_and_portfolio(returns_df, portfolio, min_non_na_frac=0.95, drop_dates_with_any_na=True)

    pr = portfolio_returns(rets, w, notional_mode=False)  # use returns for bootstrap too
    a = _validate_alpha(risk_alpha)

    if mc.method == "parametric_mvn":
        mu, cov = sample_mean_cov(rets, ewma_lambda=None, cov_shrink=mc.cov_shrink)
        sim = simulate_portfolio_paths_parametric(
            mu, cov, w,
            n_sims=mc.n_sims,
            horizon_days=mc.horizon_days,
            seed=mc.seed,
            notional_mode=False,
            use_nearest_psd=mc.use_nearest_psd,
        )
        terminals = sim["sim_terminal"]

    elif mc.method == "bootstrap":
        terminals = bootstrap_simulation_terminal(pr, n_sims=mc.n_sims, horizon_days=mc.horizon_days, seed=mc.seed, block_size=1)

    elif mc.method == "block_bootstrap":
        terminals = bootstrap_simulation_terminal(pr, n_sims=mc.n_sims, horizon_days=mc.horizon_days, seed=mc.seed, block_size=mc.block_size)

    elif mc.method == "regime_mixture":
        # default example: last 30% = "high vol", first 70% = "normal"
        n = len(rets)
        cut = int(0.7 * n)
        regimes = {
            "normal": (slice(rets.index[0], rets.index[cut - 1]), 0.7),
            "stress": (slice(rets.index[cut], rets.index[-1]), 0.3),
        }
        terminals = regime_mixture_simulation_terminal(
            rets, w,
            n_sims=mc.n_sims,
            horizon_days=mc.horizon_days,
            seed=mc.seed,
            regimes=regimes,
            ewma_lambda=None,
            cov_shrink=mc.cov_shrink,
        )
    else:
        raise ValidationError("Unknown MC method.")

    terminals = np.array(terminals, dtype=float)
    terminals = terminals[np.isfinite(terminals)]
    if terminals.size < 100:
        raise ValidationError("MC produced too few valid terminal samples.")

    # Loss distribution
    L = -terminals
    var = float(np.quantile(L, 1.0 - a))
    tail = L[L >= var]
    es = float(np.mean(tail)) if tail.size > 0 else float(var)

    pct = {
        "p01": float(np.quantile(terminals, 0.01)),
        "p05": float(np.quantile(terminals, 0.05)),
        "p50": float(np.quantile(terminals, 0.50)),
        "p95": float(np.quantile(terminals, 0.95)),
        "p99": float(np.quantile(terminals, 0.99)),
    }

    stats = {
        "Horizon Days": int(mc.horizon_days),
        "Sims": int(terminals.size),
        "Mean": float(np.mean(terminals)),
        "Std": float(np.std(terminals, ddof=1)),
        "VaR (terminal loss)": var,
        "ES (terminal loss)": es,
        **pct,
    }

    return {
        "assets": assets,
        "weights": w,
        "terminals": terminals,
        "stats": stats,
        "config": mc,
    }


# =========================
# NORMAL QUANTILE (ACKLAM APPROX)
# =========================

def _norm_ppf(p: float) -> float:
    """
    Inverse CDF of standard normal via Peter J. Acklam approximation.
    Valid for 0 < p < 1.
    """
    if not (0.0 < p < 1.0):
        raise ValidationError("ppf input p must be in (0,1).")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ]

    # Define break-points
    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = np.sqrt(-2.0 * np.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return float(num / den)

    if p > phigh:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return float(num / den)

    q = p - 0.5
    r = q*q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4]) * r + 1.0)
    return float(num / den)
