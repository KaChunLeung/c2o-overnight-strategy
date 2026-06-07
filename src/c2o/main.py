"""Single entry point for the C2O pipeline.

``main(cfg)`` runs the five steps in order, writes a self-describing run directory under data/outputs/,
and returns a summary dict. Run modes are CLI flags, not parallel scripts:

    python -m c2o.main                       # full run with config/default.yaml
    python -m c2o.main --overrides config/fast.yaml   # fast smoke run (subset of OOS years)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from . import io
from .alpha import build_alpha
from .borrow import build_borrow
from .capacity import build_capacity
from .config import Config, load_config
from .panel import build_panel
from .portfolio import build_trade_panel, run_strategy
from .reporting import write_all

logger = logging.getLogger("c2o")


def _step(name: str, start: float) -> None:
    logger.info("%-28s done (%6.1fs elapsed)", name, time.time() - start)


def _book_kwargs(overrides: dict) -> dict:
    """Map a portfolio override dict (e.g. the frontier block) to run_strategy keyword arguments."""
    return dict(score=overrides.get("headline_score"),
                quantile=overrides.get("headline_quantile"),
                weighting=overrides.get("headline_weighting"),
                selection_mode=overrides.get("selection_mode"),
                cost_buffer_c=overrides.get("cost_buffer_c"),
                neutralize_sector=overrides.get("neutralize_sector"),
                neutralize_beta=overrides.get("neutralize_beta"),
                vol_target=overrides.get("vol_target_enabled"),
                vol_target_ann=overrides.get("vol_target_ann"))


def _run_book(cfg: Config, trade_panel, capacity, borrow, overrides: dict | None = None) -> dict:
    """Run one configured book (headline = config defaults; frontier = overrides) at every AUM level."""
    kw = _book_kwargs(overrides or {})
    return {aum: run_strategy(cfg, trade_panel, capacity.eligibility_by_aum[aum],
                              borrow.short_eligibility_by_aum[aum], aum, **kw)
            for aum in cfg.capacity.aum_levels}


def main(cfg: Config) -> dict:
    """Run the full pipeline and return a summary dict (also persisted in the run manifest)."""
    t0 = time.time()
    logger.info("C2O pipeline starting (fast=%s, cutoff=%s)", cfg.run.fast, cfg.window.cutoff)

    prices = io.load_prices(cfg)
    earnings = io.load_earnings(cfg)
    short_interest = io.load_short_interest(cfg)
    _step("load inputs", t0)

    panel = build_panel(cfg, prices, earnings, short_interest)
    del prices
    _step("step1 panel", t0)

    capacity = build_capacity(cfg, panel)
    panel.prices = pd.DataFrame()          # free the 8M-row price frame; not needed downstream
    panel.panel = pd.DataFrame()
    _step("step2 capacity", t0)

    borrow = build_borrow(cfg, capacity, panel.short_interest_daily)
    _step("step3 borrow", t0)

    alpha = build_alpha(cfg, borrow.borrow_panel, capacity.eligibility_by_aum,
                        io.load_cheapness(cfg), io.load_regime(cfg),
                        io.load_earnings_transfo(cfg), earnings, panel.calendar)
    del earnings
    _step("step4b alpha", t0)

    sp500_tr = io.load_sp500_tr(cfg)
    trade_panel = build_trade_panel(cfg, alpha.scores, borrow.borrow_panel,
                                    gics=io.load_gics(cfg), market_ret=sp500_tr)
    headline_runs = _run_book(cfg, trade_panel, capacity, borrow)
    frontier_runs = _run_book(cfg, trade_panel, capacity, borrow, cfg.portfolio.frontier)
    _step("step5 portfolio", t0)

    run_dir = io.new_run_dir(cfg)
    summary = write_all(cfg, run_dir, panel, capacity, borrow, alpha, trade_panel,
                        headline_runs, frontier_runs, sp500_tr)
    io.write_manifest(run_dir, cfg, extra=summary)
    _step("reporting", t0)

    logger.info("DONE in %.0fs -> %s | net Sharpe @250M = %.3f (gross %.3f) | tear-sheet=%s",
                time.time() - t0, run_dir, summary["net_sharpe_250M"],
                summary["gross_sharpe_250M"], summary["tearsheet_written"])
    return {**summary, "run_dir": str(run_dir)}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C2O close-to-open overnight strategy pipeline")
    p.add_argument("--config", default="config/default.yaml", help="base config YAML")
    p.add_argument("--overrides", default=None, help="optional overrides YAML (e.g. config/fast.yaml)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def cli(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    cfg = load_config(args.config, args.overrides)
    return main(cfg)


if __name__ == "__main__":
    cli()
