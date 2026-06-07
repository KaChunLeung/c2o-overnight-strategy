"""All filesystem access for the C2O pipeline (the only module that touches the disk).

Public API:
    load_prices / load_earnings / load_earnings_transfo / load_short_interest / load_cheapness
    load_gics / load_regime / load_sp500_tr
    read_intermediary / write_intermediary
    new_run_dir / write_table / write_figure / write_manifest

Inputs are read-only and filtered to ``window.cutoff`` on load (anti-leakage). Schema is validated at
the boundary. Private helpers are prefixed ``_``.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import Config

_PRICE_COLUMNS = ["ticker", "instrument_id", "date", "open", "high", "low", "close",
                  "adjusted_close", "volume", "market_cap", "status"]
_CHEAPNESS_COLUMNS = ["instrument_id", "date", "valuation_score", "quality_score", "health_score",
                      "momentum_score", "score_velocity", "value_trap"]


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"input '{name}' is missing required columns: {missing}")


def load_prices(cfg: Config) -> pd.DataFrame:
    """Daily OHLCV + market cap, cutoff-filtered. Keeps pre-2010 rows for lookback features."""
    df = pd.read_parquet(cfg.paths.input("prices"), columns=_PRICE_COLUMNS)
    _require_columns(df, _PRICE_COLUMNS, "prices")
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"] <= pd.Timestamp(cfg.window.cutoff)].copy()
    if cfg.run.smoke_instruments > 0:                       # fast verification: top-N names by market cap
        keep = (df.groupby("instrument_id")["market_cap"].max()
                .nlargest(cfg.run.smoke_instruments).index)
        df = df.loc[df["instrument_id"].isin(keep)].copy()
    return df


def load_earnings(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.paths.input("earnings_calendar"))
    df["reporting_date"] = pd.to_datetime(df["reporting_date"])
    return df.loc[df["reporting_date"] <= pd.Timestamp(cfg.window.cutoff)].copy()


def load_short_interest(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.paths.input("short_interest"))
    df = df.rename(columns={"stock_id": "instrument_id", "date": "si_available_date"})
    df["si_available_date"] = pd.to_datetime(df["si_available_date"])
    return df.loc[df["si_available_date"] <= pd.Timestamp(cfg.window.cutoff)].copy()


def load_cheapness(cfg: Config) -> pd.DataFrame:
    """Cheapness/quality scores, merged later on the previous trading day (point-in-time)."""
    df = pd.read_parquet(cfg.paths.input("cheapness"), columns=_CHEAPNESS_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df.loc[df["date"] <= pd.Timestamp(cfg.window.cutoff)].copy()


def load_earnings_transfo(cfg: Config) -> pd.DataFrame:
    """Analyst-revision / earnings-surprise features (flow sleeve), merged as-of the previous trading day.

    ``stock_id`` is the same identifier space as ``prices.instrument_id`` (verified). Cutoff-filtered.
    """
    df = pd.read_parquet(cfg.paths.input("earnings_transfo"))
    df = df.rename(columns={"stock_id": "instrument_id", "date": "earn_feat_date"})
    df["earn_feat_date"] = pd.to_datetime(df["earn_feat_date"])
    return df.loc[df["earn_feat_date"] <= pd.Timestamp(cfg.window.cutoff)].copy()


def load_gics(cfg: Config) -> pd.DataFrame:
    """Static GICS sector map (instrument_id -> sector). Used for sector-neutral construction."""
    df = pd.read_parquet(cfg.paths.input("gics"), columns=["instrument_id", "sector"])
    return df.drop_duplicates("instrument_id").copy()


def load_regime(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.paths.input("regime"))
    df["date"] = pd.to_datetime(df["date"])
    return df.loc[df["date"] <= pd.Timestamp(cfg.window.cutoff), ["date", "regime"]].copy()


def load_sp500_tr(cfg: Config) -> pd.Series:
    """S&P 500 Total Return daily returns (the QuantStats benchmark), cutoff-filtered."""
    df = pd.read_parquet(cfg.paths.input("sp500_tr"), columns=["date", "adjusted_close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"] <= pd.Timestamp(cfg.window.cutoff)].sort_values("date").set_index("date")
    return df["adjusted_close"].pct_change().rename("SP500_TR")


def write_intermediary(cfg: Config, df: pd.DataFrame, name: str) -> Path:
    """Cache a step handoff in data/intermediary/ (disposable; regenerated on re-run)."""
    cfg.paths.intermediary_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.intermediary(name)
    df.to_parquet(path)
    return path


def read_intermediary(cfg: Config, name: str) -> pd.DataFrame:
    return pd.read_parquet(cfg.paths.intermediary(name))


def new_run_dir(cfg: Config, run_id: str | None = None) -> Path:
    """Create data/outputs/<run_id>/{tables,figures,reports} and return the run directory."""
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.outputs_dir / run_id
    for sub in ("tables", "figures", "reports"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_table(run_dir: Path, df: pd.DataFrame, name: str, index: bool = True) -> Path:
    path = run_dir / "tables" / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def write_figure(run_dir: Path, fig, name: str) -> Path:
    path = run_dir / "figures" / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_manifest(run_dir: Path, cfg: Config, extra: dict | None = None) -> Path:
    """Drop a self-describing manifest (config, git SHA, host, time) into the run directory."""
    manifest = {
        "run_dir": run_dir.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "config": cfg.raw,
    }
    if extra:
        manifest.update(extra)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
