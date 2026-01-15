"""
stresslab/utils/schema.py
========================
Typed configuration models using Pydantic.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Holding(BaseModel):
    asset_id: str
    asset_name: str
    asset_type: str
    currency: str
    quantity: Optional[float] = None
    price: Optional[float] = None
    notional: Optional[float] = None
    weight: Optional[float] = None
    sector: Optional[str] = None
    region: Optional[str] = None


class PortfolioConfig(BaseModel):
    name: str
    base_currency: str = Field(default="USD")
    description: Optional[str] = None
    holdings: List[Holding]


class ScenarioShockConfig(BaseModel):
    price_shocks: Dict[str, float] = Field(default_factory=dict)
    vol_shocks: Dict[str, float] = Field(default_factory=dict)
    rates_shifts: Dict[str, float] = Field(default_factory=dict)
    fx_shocks: Dict[str, float] = Field(default_factory=dict)
    corr_shocks: Dict[str, float] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    shocks: ScenarioShockConfig
    regime_rule: Optional[str] = None
    regime_transition: Optional[str] = None


class RunConfig(BaseModel):
    lookback_days: int = 756
    frequency: str = "D"
    risk_free_rate: float = 0.0
    confidence_level: float = 0.95
    cov_method: str = "sample"
    ewma_lambda: Optional[float] = None
    shrinkage: Optional[str] = "lw_diag"
    mc_sims: int = 5000
    mc_seed: int = 42
