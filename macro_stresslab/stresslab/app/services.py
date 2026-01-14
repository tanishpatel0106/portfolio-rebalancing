"""
stresslab/app/services.py
========================

Application/service layer for StressLab.

This module is the *only* layer the Streamlit UI should talk to.

Responsibilities
----------------
1) Orchestrate:
   data acquisition (optional) -> preprocessing -> analytics -> UI-ready payloads

2) Provide stable, high-level "use case" functions:
   - build_risk_payload()
   - build_scenario_payload()
   - build_regime_payload()
   - build_monte_carlo_payload()

3) Enforce consistent output schemas so UI pages remain simple and robust.

Notes
-----
- No Streamlit imports here.
- This file intentionally does not "do math" itself; it delegates to analytics modules.
- Designed to be unit-testable:
  * functions are pure given inputs (unless you pass a data_provider)
  * deterministic where MC seed is set in configs

Integration points
------------------
- stresslab.analytics.stress (your math-heavy engine)
- stresslab.analytics.regimes (regime detection + summaries)
- stresslab.data.sources (optional: data fetch/provider)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from stresslab.utils.logging import get_logger

from stresslab.analytics import stress as st

# regimes.py exists in your tree; import directly.
# If something changes later, this explicit import will fail loudly (preferred).
from stresslab.analytics import regimes as rg

# Optional provider module: your project has stresslab/data/sources.py
# We import it as optional to keep services usable with direct DataFrame inputs.
try:
    from stresslab.data import sources as src  # type: ignore
except Exception:  # pragma: no cover
    src = None  # type: ignore


log = get_logger(__name__)


# =========================
# TYPES / DATA PROVIDER
# =========================

PriceOrReturnDF = pd.DataFrame


class ServiceError(RuntimeError):
    """Base error for service-layer failures."""


class ProviderError(ServiceError):
    """Raised when a data provider fails or returns invalid data."""


class InputError(ServiceError):
    """Raised when user/UI inputs are invalid at the service boundary."""


@dataclass(frozen=True)
class DataRequest:
    """
    Canonical request for data acquisition.

    You can use this with a data_provider (callable) or with stresslab.data.sources (if implemented).

    The service layer *does not assume* tickers are equities; it’s generic.
    """
    tickers: Sequence[str]
    start: Optional[Union[str, pd.Timestamp]] = None
    end: Optional[Union[str, pd.Timestamp]] = None
    field: str = "close"  # "close" is common; provider decides meaning
    frequency: Optional[str] = None  # e.g., "1D"
    extra: Optional[Dict[str, Any]] = None  # provider-specific args


# A data_provider is any callable that takes DataRequest and returns a price DataFrame.
DataProvider = Callable[[DataRequest], pd.DataFrame]


# =========================
# INTERNAL VALIDATION HELPERS
# =========================

def _ensure_df(x: Any, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame.")
    if x.empty:
        raise InputError(f"{name} cannot be empty.")
    return x


def _ensure_datetime_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise InputError(f"{name} must have a DatetimeIndex (or convertible). Error: {e}")
    # normalize timezone if present
    if getattr(df.index, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df


def _sorted_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def _as_float_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _validate_assets_nonempty(assets: Sequence[str]) -> List[str]:
    if assets is None:
        raise InputError("assets/tickers cannot be None.")
    assets_list = [str(a) for a in assets if str(a).strip() != ""]
    if not assets_list:
        raise InputError("assets/tickers cannot be empty.")
    return assets_list


def _infer_is_prices(df: pd.DataFrame) -> bool:
    """
    Heuristic: if values are mostly > 0 and large-ish, likely prices; if mostly small, likely returns.
    This is only used when user doesn't specify; we keep it conservative.

    If unsure, user should explicitly pass `input_kind='prices'|'returns'`.
    """
    x = df.to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    med = float(np.median(np.abs(x)))
    # Returns typically have median abs << 1 (e.g., 0.005), while prices usually > 1.
    return med > 1.0


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = _ensure_df(prices, "prices")
    prices = _ensure_datetime_index(prices, "prices")
    prices = _sorted_unique_index(prices)
    prices = _as_float_df(prices)
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.dropna(how="all")
    if prices.empty:
        raise InputError("Prices became empty after cleaning.")
    return prices


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    returns = _ensure_df(returns, "returns")
    returns = _ensure_datetime_index(returns, "returns")
    returns = _sorted_unique_index(returns)
    returns = _as_float_df(returns)
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all")
    if returns.empty:
        raise InputError("Returns became empty after cleaning.")
    return returns


# =========================
# DATA LOADING / PREP
# =========================

def load_prices(
    req: DataRequest,
    *,
    data_provider: Optional[DataProvider] = None,
) -> pd.DataFrame:
    """
    Load prices using:
      1) explicit data_provider(req) if provided, else
      2) stresslab.data.sources if available and supports a compatible interface.

    Returns DataFrame indexed by date, columns tickers.
    """
    tickers = _validate_assets_nonempty(req.tickers)
    req = DataRequest(
        tickers=tickers,
        start=req.start,
        end=req.end,
        field=req.field,
        frequency=req.frequency,
        extra=req.extra or {},
    )

    if data_provider is not None:
        try:
            df = data_provider(req)
        except Exception as e:
            raise ProviderError(f"data_provider failed: {e}") from e
        return _clean_prices(df)

    if src is None:
        raise ProviderError(
            "No data_provider provided and stresslab.data.sources could not be imported. "
            "Either pass data_provider=... or implement sources.py fetch function."
        )

    # We try a few common function names to keep this flexible without cutting corners.
    candidates = [
        "load_prices",
        "get_prices",
        "fetch_prices",
        "prices",
    ]
    for fn_name in candidates:
        fn = getattr(src, fn_name, None)
        if callable(fn):
            try:
                df = fn(
                    tickers=tickers,
                    start=req.start,
                    end=req.end,
                    field=req.field,
                    frequency=req.frequency,
                    **(req.extra or {}),
                )
                return _clean_prices(df)
            except TypeError:
                # provider signature mismatch; keep trying other candidates
                continue
            except Exception as e:
                raise ProviderError(f"sources.{fn_name} failed: {e}") from e

    raise ProviderError(
        "stresslab.data.sources is available, but no compatible price loader was found. "
        "Expected one of: load_prices/get_prices/fetch_prices/prices."
    )


def returns_from_input(
    data: PriceOrReturnDF,
    *,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Convert either prices or returns input into *returns* DataFrame.
    - If input_kind is "prices": compute returns
    - If input_kind is "returns": clean and return
    - If input_kind is None: infer via heuristic

    Uses stress.py compute_returns_from_prices for consistency.
    """
    data = _ensure_df(data, "data")
    data = _ensure_datetime_index(data, "data")
    data = _sorted_unique_index(data)
    data = _as_float_df(data)

    kind = input_kind
    if kind is not None:
        kind = str(kind).strip().lower()
        if kind not in {"prices", "returns"}:
            raise InputError("input_kind must be one of: None, 'prices', 'returns'.")
    else:
        kind = "prices" if _infer_is_prices(data) else "returns"

    if kind == "prices":
        prices = _clean_prices(data)
        rets = st.compute_returns_from_prices(prices, method=return_method, dropna=dropna)
        return _clean_returns(rets)
    else:
        return _clean_returns(data)


# =========================
# HIGH-LEVEL PAYLOAD BUILDERS
# =========================

def build_risk_payload(
    data: PriceOrReturnDF,
    portfolio: st.PortfolioSpec,
    *,
    risk: st.RiskConfig,
    starting_capital: float = 10_000.0,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
) -> Dict[str, Any]:
    """
    UI-ready payload for the Risk page.

    Output keys (stable):
      - "returns_df": aligned return matrix used for analytics (T x N)
      - "assets", "weights"
      - "portfolio_series" (returns or pnl)
      - "equity_curve"
      - "drawdown"
      - "metrics" (dict)
      - "rolling_var" (DataFrame with columns VaR/ES)
    """
    returns_df = returns_from_input(
        data,
        input_kind=input_kind,
        return_method=return_method,
        dropna=True,
    )

    base = st.portfolio_risk_report(
        returns_df,
        portfolio,
        risk=risk,
        starting_capital=starting_capital,
    )

    # Rolling VaR series: always computed on *return-like* series
    # If pnl mode, convert to return proxy by dividing by capital.
    if portfolio.notional_mode:
        pr_for_rolling = (base["portfolio_series"] / float(starting_capital)).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        pr_for_rolling = base["portfolio_series"].replace([np.inf, -np.inf], np.nan).dropna()

    rolling = st.rolling_var_series(pr_for_rolling, risk, window=risk.annualization_factor)

    payload: Dict[str, Any] = {
        "returns_df": base["aligned_returns"],
        "assets": base["assets"],
        "weights": base["weights"],
        "portfolio_series": base["portfolio_series"],
        "equity_curve": base["equity_curve"],
        "drawdown": base["drawdown"],
        "metrics": base["metrics"],
        "rolling_var": rolling,
    }
    return payload


def build_scenario_payload(
    data: PriceOrReturnDF,
    portfolio: st.PortfolioSpec,
    scenarios: Sequence[st.Scenario],
    *,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
    factor_returns: Optional[pd.DataFrame] = None,
    betas: Optional[pd.DataFrame] = None,
    starting_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    UI-ready payload for the Stress/Scenarios page.

    Output keys:
      - "scenario_table": DataFrame (Scenario, Kind, ImpactType, ImpactValue, Unit, Details)
      - "base_portfolio_series"
      - "aligned_returns"
      - "assets", "weights"
    """
    returns_df = returns_from_input(
        data,
        input_kind=input_kind,
        return_method=return_method,
        dropna=True,
    )

    scenario_table = st.run_scenarios(
        returns_df=returns_df,
        portfolio=portfolio,
        scenarios=scenarios,
        factor_returns=factor_returns,
        betas=betas,
        starting_capital=starting_capital,
    )

    # Build aligned + base portfolio series for charts
    aligned_rets, w, assets = st.align_returns_and_portfolio(
        returns_df,
        portfolio,
        min_non_na_frac=0.95,
        drop_dates_with_any_na=True,
    )
    base_series = st.portfolio_returns(aligned_rets, w, notional_mode=portfolio.notional_mode)

    return {
        "scenario_table": scenario_table,
        "base_portfolio_series": base_series,
        "aligned_returns": aligned_rets,
        "assets": assets,
        "weights": w,
    }


def build_monte_carlo_payload(
    data: PriceOrReturnDF,
    portfolio: st.PortfolioSpec,
    *,
    mc: st.MonteCarloConfig,
    risk_alpha: float = 0.05,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
) -> Dict[str, Any]:
    """
    UI-ready payload for Monte Carlo page.

    Output keys:
      - "terminals": np.ndarray
      - "stats": dict
      - "assets", "weights"
      - "config": MonteCarloConfig
    """
    returns_df = returns_from_input(
        data,
        input_kind=input_kind,
        return_method=return_method,
        dropna=True,
    )

    mc_res = st.monte_carlo_report(
        returns_df=returns_df,
        portfolio=portfolio,
        mc=mc,
        risk_alpha=risk_alpha,
    )

    return {
        "terminals": mc_res["terminals"],
        "stats": mc_res["stats"],
        "assets": mc_res["assets"],
        "weights": mc_res["weights"],
        "config": mc_res["config"],
    }


def build_regime_payload(
    data: PriceOrReturnDF,
    portfolio: st.PortfolioSpec,
    *,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
    starting_capital: float = 10_000.0,
    # regime config knobs (passed through to regimes.py)
    regime_method: str = "hmm_gaussian",
    regime_k: int = 2,
    feature_set: str = "portfolio_default",
    lookback: Optional[int] = None,
    min_obs: int = 252,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    UI-ready payload for the Regimes page.

    This function intentionally *does not* assume a specific regimes.py API beyond
    a minimal expected surface. If your regimes.py exports different names,
    update ONLY this service function; UI stays stable.

    Expected outputs (stable keys):
      - "regime_series": pd.Series of regime labels indexed by date
      - "regime_table": pd.DataFrame of per-regime summary stats
      - "features": pd.DataFrame used for regime modeling (for charts/debug)
      - "base_portfolio_series": pd.Series (returns or pnl)
      - "equity_curve": pd.Series
      - "aligned_returns": pd.DataFrame
      - "assets", "weights"
      - "meta": dict of model settings / diagnostics
    """
    returns_df = returns_from_input(
        data,
        input_kind=input_kind,
        return_method=return_method,
        dropna=True,
    )

    aligned_rets, w, assets = st.align_returns_and_portfolio(
        returns_df,
        portfolio,
        min_non_na_frac=0.95,
        drop_dates_with_any_na=True,
    )

    base_series = st.portfolio_returns(aligned_rets, w, notional_mode=portfolio.notional_mode)

    if portfolio.notional_mode:
        equity = st.equity_curve_from_pnl(base_series, starting_capital=starting_capital)
        # for regime features, always use a return-like series
        port_ret = (base_series / float(starting_capital)).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        equity = st.equity_curve_from_returns(base_series, starting_capital=starting_capital)
        port_ret = base_series.replace([np.inf, -np.inf], np.nan).dropna()

    # --- Regime feature engineering & detection via regimes.py ---
    # We support two common patterns:
    # (A) rg.build_features(port_ret, ...) + rg.detect_regimes(features, ...)
    # (B) rg.detect_regimes_from_returns(port_ret, ...)
    #
    # Your regimes.py SHOULD provide at least one of these patterns.
    features: Optional[pd.DataFrame] = None
    regime_series: Optional[pd.Series] = None
    meta: Dict[str, Any] = {}

    # Guard rails
    if len(port_ret) < int(min_obs):
        raise InputError(f"Not enough observations for regime detection. Need >= {min_obs}.")

    # Try Pattern A
    build_features = getattr(rg, "build_features", None)
    detect_regimes = getattr(rg, "detect_regimes", None)

    if callable(build_features) and callable(detect_regimes):
        features = build_features(
            port_ret=port_ret,
            method=feature_set,
            lookback=lookback,
        )
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ServiceError("regimes.build_features returned empty/invalid features DataFrame.")

        regime_series, meta = detect_regimes(
            features=features,
            method=regime_method,
            k=regime_k,
            seed=seed,
        )
        if not isinstance(regime_series, pd.Series) or regime_series.empty:
            raise ServiceError("regimes.detect_regimes returned empty/invalid regime Series.")

    else:
        # Try Pattern B
        detect_from_returns = getattr(rg, "detect_regimes_from_returns", None)
        if callable(detect_from_returns):
            regime_series, features, meta = detect_from_returns(
                port_ret=port_ret,
                method=regime_method,
                k=regime_k,
                feature_set=feature_set,
                lookback=lookback,
                seed=seed,
            )
        else:
            raise ServiceError(
                "regimes.py does not expose expected functions. "
                "Expected either (build_features + detect_regimes) OR detect_regimes_from_returns."
            )

    # Summary table (per regime)
    summarize = getattr(rg, "regime_summary_table", None)
    if callable(summarize):
        regime_table = summarize(port_ret=port_ret, regimes=regime_series)
    else:
        # Fallback: compute basic summary here (still service-level reporting is okay)
        tmp = pd.DataFrame({"ret": port_ret, "regime": regime_series}).dropna()
        grp = tmp.groupby("regime")["ret"]
        regime_table = pd.DataFrame(
            {
                "count": grp.size(),
                "mean": grp.mean(),
                "std": grp.std(ddof=1),
                "min": grp.min(),
                "max": grp.max(),
            }
        ).reset_index().rename(columns={"regime": "Regime"})
        regime_table = regime_table.sort_values("Regime").reset_index(drop=True)

    payload: Dict[str, Any] = {
        "regime_series": regime_series,
        "regime_table": regime_table,
        "features": features,
        "base_portfolio_series": base_series,
        "equity_curve": equity,
        "aligned_returns": aligned_rets,
        "assets": assets,
        "weights": w,
        "meta": meta,
    }
    return payload


# =========================
# OPTIONAL: ONE-SHOT END-TO-END WRAPPER
# =========================

def build_all_payloads(
    data: PriceOrReturnDF,
    portfolio: st.PortfolioSpec,
    *,
    risk: st.RiskConfig,
    scenarios: Optional[Sequence[st.Scenario]] = None,
    mc: Optional[st.MonteCarloConfig] = None,
    include_regimes: bool = False,
    input_kind: Optional[str] = None,
    return_method: str = "simple",
    starting_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Convenience wrapper for an "All-in-one" page or for integration tests.

    Returns:
      {
        "risk": ...,
        "scenarios": ... (optional),
        "mc": ... (optional),
        "regimes": ... (optional)
      }
    """
    out: Dict[str, Any] = {}

    out["risk"] = build_risk_payload(
        data=data,
        portfolio=portfolio,
        risk=risk,
        starting_capital=starting_capital,
        input_kind=input_kind,
        return_method=return_method,
    )

    if scenarios is not None:
        out["scenarios"] = build_scenario_payload(
            data=data,
            portfolio=portfolio,
            scenarios=scenarios,
            input_kind=input_kind,
            return_method=return_method,
            starting_capital=starting_capital,
        )

    if mc is not None:
        out["mc"] = build_monte_carlo_payload(
            data=data,
            portfolio=portfolio,
            mc=mc,
            input_kind=input_kind,
            return_method=return_method,
        )

    if include_regimes:
        out["regimes"] = build_regime_payload(
            data=data,
            portfolio=portfolio,
            input_kind=input_kind,
            return_method=return_method,
            starting_capital=starting_capital,
        )

    return out
