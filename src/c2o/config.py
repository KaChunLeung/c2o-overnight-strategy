"""Config loading and validation for the C2O pipeline.

Public API:
    load_config(path, overrides_path=None, repo_root=None) -> Config
    Config (frozen dataclass tree; every behaviour-bearing number lives here, not in code)

The config file is the single source of truth for paths and numeric constants. Code receives a
``Config`` object and never reads YAML or environment state on its own. Private helpers are prefixed ``_``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunCfg:
    seed: int
    fast: bool
    fast_eval_years: list[int]
    smoke_instruments: int = 0   # >0 keeps only the top-N instruments by market cap (fast verification)


@dataclass(frozen=True)
class WindowCfg:
    start_date: str
    cutoff: str
    universe_size: int
    min_history_days: int
    residual_tolerance: float


@dataclass(frozen=True)
class CapacityCfg:
    aum_levels: list[float]
    planning_basket_names: int
    participation_cap: float
    impact_k: float
    price_floor: float
    mcap_floor: float
    vol_min_ann: float
    vol_max_ann: float
    rolling_window: int
    earn_window_after: int


@dataclass(frozen=True)
class BorrowCfg:
    trading_days_per_year: int
    annual_rates: dict[str, float]
    dsi_moderate: float
    dsi_high: float
    dtcn_moderate: float
    dtcn_high: float
    ddtcn_moderate: float
    ddtcn_high: float
    moderate_pct: float
    high_pct: float
    small_mcap_for_htb: float
    low_adv_for_htb: float
    stress_score_moderate: int
    stress_score_high: int
    hard_exclude_tier: str


@dataclass(frozen=True)
class AlphaCfg:
    signal_aum: float
    first_eval_year: int
    last_eval_year: int
    ridge_lambda: float
    min_daily_names: int
    min_train_obs: int
    rolling_ic_window: int
    zscore_clip: float
    hgb_train_sample: int
    hgb: dict[str, Any]
    base_features: list[str]
    new_features: list[str]

    @property
    def raw_features(self) -> list[str]:
        return list(self.base_features) + list(self.new_features)

    @property
    def z_features(self) -> list[str]:
        return [f"z_{f}" for f in self.raw_features]


@dataclass(frozen=True)
class PortfolioCfg:
    commission_bps_per_leg: float
    slippage_bps_per_leg: float
    headline_quantile: float
    headline_weighting: str
    headline_score: str
    htb_exclude: bool
    gate_q: float
    benchmark_aum: float
    min_basket_names: int
    concentration_sweep: list[float]
    weighting_grid: list[str]
    stress_windows: dict[str, list[str]]

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * (self.commission_bps_per_leg + self.slippage_bps_per_leg)


@dataclass(frozen=True)
class Paths:
    inputs_dir: Path
    intermediary_dir: Path
    outputs_dir: Path
    files: dict[str, str]

    def input(self, key: str) -> Path:
        """Absolute path of a declared input file (raises if undeclared)."""
        if key not in self.files:
            raise KeyError(f"input file '{key}' is not declared in config.paths.files")
        return self.inputs_dir / self.files[key]

    def intermediary(self, name: str) -> Path:
        return self.intermediary_dir / name


@dataclass(frozen=True)
class Config:
    run: RunCfg
    paths: Paths
    window: WindowCfg
    capacity: CapacityCfg
    borrow: BorrowCfg
    alpha: AlphaCfg
    portfolio: PortfolioCfg
    raw: dict[str, Any]


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``over`` onto ``base`` (returns a new dict)."""
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _build_paths(d: dict[str, Any], repo_root: Path) -> Paths:
    return Paths(
        inputs_dir=repo_root / d["inputs_dir"],
        intermediary_dir=repo_root / d["intermediary_dir"],
        outputs_dir=repo_root / d["outputs_dir"],
        files=dict(d["files"]),
    )


def load_config(path: str | Path = "config/default.yaml",
                overrides_path: str | Path | None = None,
                repo_root: Path | None = None) -> Config:
    """Load and validate the config tree from YAML, optionally overlaying an overrides file.

    Paths are resolved relative to the repository root. No I/O beyond reading the YAML file(s).
    """
    root = repo_root or _REPO_ROOT
    with open(root / path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if overrides_path is not None:
        with open(root / overrides_path, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})

    cfg = Config(
        run=RunCfg(**data["run"]),
        paths=_build_paths(data["paths"], root),
        window=WindowCfg(**data["window"]),
        capacity=CapacityCfg(**data["capacity"]),
        borrow=BorrowCfg(**data["borrow"]),
        alpha=AlphaCfg(**data["alpha"]),
        portfolio=PortfolioCfg(**data["portfolio"]),
        raw=data,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail loudly on the invariants the rest of the pipeline assumes."""
    if not (0.0 < cfg.capacity.participation_cap < 1.0):
        raise ValueError("capacity.participation_cap must be in (0, 1)")
    if cfg.window.cutoff < cfg.window.start_date:
        raise ValueError("window.cutoff precedes window.start_date")
    if set(cfg.borrow.annual_rates) != {"A", "B", "C"}:
        raise ValueError("borrow.annual_rates must define tiers A, B, C")
    if not (0.0 < cfg.portfolio.headline_quantile <= 0.5):
        raise ValueError("portfolio.headline_quantile must be in (0, 0.5]")
    if cfg.alpha.signal_aum > min(cfg.capacity.aum_levels):
        raise ValueError("alpha.signal_aum should be the most permissive (<= min AUM level)")
