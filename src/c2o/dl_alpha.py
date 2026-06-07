"""Documented ML experiment (OFF the production path): does a neural net beat the HGB reversal sleeve?

Motivation (course material: Lectures 6-7 sequence models / attention; the TFT reading; the Adam reading):
the brief is an ML-in-finance task, so we benchmark a neural forecaster against the gradient-boosted tree
that ships in production. We compare, on the identical walk-forward OOS protocol and feature panel:

    * HGB         — the production reversal model (gradient-boosted trees).
    * MLP         — a feed-forward neural net on the same cross-sectional features (Adam optimiser).
    * MLP-seq     — the MLP plus a flattened 5-session lag window of overnight/close-to-close returns,
                    a feed-forward proxy for a sequence model.

NOTE ON TOOLING: PyTorch has no wheels for the Python 3.14 toolchain in this environment, so a full
GRU/attention network could not be trained here. We use scikit-learn's ``MLPRegressor`` (trained with the
Adam optimiser of Kingma & Ba, 2015) as the neural family and argue in the report why a deeper recurrent /
attention model is unlikely to change the conclusion: the binding constraint is execution cost, not forecast
accuracy (the HGB already delivers IC t-stat ~8 gross). Run manually:

    python -m c2o.dl_alpha                       # full protocol (slow)
    python -m c2o.dl_alpha --overrides config/fast.yaml
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import io
from .alpha import _build_feature_panel, _cross_sectional_zscore
from .borrow import build_borrow
from .capacity import build_capacity
from .config import Config, load_config
from .metrics import daily_ic, summarize_ic
from .panel import build_panel

logger = logging.getLogger("c2o.dl")

_SEQ_BASE = ["r_on", "r_cc"]          # series whose recent history forms the "sequence" window
_SEQ_LAGS = 5


def _build_frame(cfg: Config) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Replicate the front of build_alpha to get the scored feature panel, plus a lag-window block."""
    prices = io.load_prices(cfg)
    earnings = io.load_earnings(cfg)
    short_interest = io.load_short_interest(cfg)
    panel = build_panel(cfg, prices, earnings, short_interest)
    del prices
    capacity = build_capacity(cfg, panel)
    panel.prices = pd.DataFrame()
    borrow = build_borrow(cfg, capacity, panel.short_interest_daily)
    sig = _build_feature_panel(cfg, borrow.borrow_panel, capacity.eligibility_by_aum[cfg.alpha.signal_aum],
                               io.load_cheapness(cfg), io.load_regime(cfg),
                               io.load_earnings_transfo(cfg), earnings, panel.calendar)
    sig = sig.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    # flattened lag window (feed-forward proxy for a sequence model)
    seq_cols = []
    g = sig.groupby("instrument_id", group_keys=False)
    for base in _SEQ_BASE:
        for k in range(1, _SEQ_LAGS + 1):
            col = f"seq_{base}_l{k}"
            sig[col] = g[base].shift(k)
            seq_cols.append(col)

    for f in cfg.alpha.raw_features:
        sig[f"z_{f}"] = _cross_sectional_zscore(sig, f, cfg.alpha.zscore_clip)
    for c in seq_cols:                                  # standardise the sequence block too
        sig[f"z_{c}"] = _cross_sectional_zscore(sig, c, cfg.alpha.zscore_clip)
    sig["target_rank_cs"] = sig.groupby("date")["target_r_on_next"].rank(pct=True) - 0.5
    sig["valid_day"] = sig.groupby("date")["target_r_on_next"].transform("count") >= cfg.alpha.min_daily_names
    z_feats = cfg.alpha.z_features
    z_seq = [f"z_{c}" for c in seq_cols]
    return sig, z_feats, z_seq


def _mlp() -> object:
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                     alpha=1e-3, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                     n_iter_no_change=6, random_state=7))


def _walk_forward(cfg: Config, sig: pd.DataFrame, z_feats: list[str], z_seq: list[str]) -> pd.DataFrame:
    a = cfg.alpha
    rng = np.random.default_rng(cfg.run.seed)
    eval_years = (y for y in cfg.run.fast_eval_years) if cfg.run.fast else range(a.first_eval_year, a.last_eval_year + 1)
    for col in ["dl_hgb", "dl_mlp", "dl_mlpseq"]:
        sig[col] = np.nan
    raw = a.raw_features
    for yr in eval_years:
        train_m = sig["valid_day"] & (sig["year"] < yr) & sig["target_rank_cs"].notna()
        test_m = sig["year"].eq(yr)
        if test_m.sum() == 0 or train_m.sum() < a.min_train_obs:
            continue
        tr = sig.loc[train_m]
        if len(tr) > a.hgb_train_sample:
            tr = tr.loc[rng.choice(tr.index.to_numpy(), a.hgb_train_sample, replace=False)]
        y = tr["target_rank_cs"].to_numpy("float64")
        te = sig.loc[test_m]
        t0 = time.time()

        hgb = HistGradientBoostingRegressor(random_state=cfg.run.seed, **a.hgb)
        hgb.fit(tr[raw].to_numpy("float32"), y)
        sig.loc[test_m, "dl_hgb"] = hgb.predict(te[raw].to_numpy("float32"))

        mlp = _mlp().fit(tr[z_feats].fillna(0.0).to_numpy("float32"), y)
        sig.loc[test_m, "dl_mlp"] = mlp.predict(te[z_feats].fillna(0.0).to_numpy("float32"))

        mlp2 = _mlp().fit(tr[z_feats + z_seq].fillna(0.0).to_numpy("float32"), y)
        sig.loc[test_m, "dl_mlpseq"] = mlp2.predict(te[z_feats + z_seq].fillna(0.0).to_numpy("float32"))
        logger.info("year %d: fit HGB/MLP/MLP-seq on %d rows (%.0fs)", yr, len(tr), time.time() - t0)
    return sig


def main(cfg: Config) -> pd.DataFrame:
    """Run the bake-off and write step4b_dl_benchmark.csv into a fresh run directory."""
    t0 = time.time()
    sig, z_feats, z_seq = _build_frame(cfg)
    logger.info("feature frame built (%d rows, %.0fs)", len(sig), time.time() - t0)
    sig = _walk_forward(cfg, sig, z_feats, z_seq)
    rows = []
    for label, col in [("HGB (production)", "dl_hgb"), ("MLP (feed-forward, Adam)", "dl_mlp"),
                       ("MLP + 5-session lag window", "dl_mlpseq")]:
        ic = daily_ic(sig.loc[sig["valid_day"]], col, "target_r_on_next", cfg.alpha.min_daily_names)
        rows.append({"model": label, **summarize_ic(ic)})
    table = pd.DataFrame(rows)
    run_dir = io.new_run_dir(cfg)
    io.write_table(run_dir, table, "step4b_dl_benchmark", index=False)
    io.write_manifest(run_dir, cfg, extra={"experiment": "dl_benchmark"})
    logger.info("DL benchmark -> %s\n%s", run_dir, table.to_string(index=False))
    return table


def cli(argv: list[str] | None = None) -> pd.DataFrame:
    p = argparse.ArgumentParser(description="C2O neural vs HGB alpha benchmark (documented experiment)")
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--overrides", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    return main(load_config(args.config, args.overrides))


if __name__ == "__main__":
    cli()
