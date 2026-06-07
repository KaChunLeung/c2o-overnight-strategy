"""DEV-ONLY construction lab (not part of the shipped pipeline).

Builds the trade panel once, caches it, then sweeps portfolio-construction variants in seconds so we can
find what actually lifts net Sharpe before baking it into config. Run:

    PYTHONPATH=src python tools/experiment.py            # build cache (once, ~11 min) then run battery
    PYTHONPATH=src python tools/experiment.py --battery  # re-run battery from cache (seconds)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from c2o import io
from c2o.alpha import build_alpha
from c2o.borrow import build_borrow
from c2o.capacity import build_capacity
from c2o.config import load_config
from c2o.metrics import annualised_sharpe, max_drawdown
from c2o.panel import build_panel
from c2o.portfolio import build_trade_panel, run_strategy

CACHE = Path("data/intermediary")
AUM = 250_000_000.0
_OFF = dict(selection_mode="quantile", neutralize_sector=False, neutralize_beta=False, vol_target=False)


def build_cache(cfg):
    prices = io.load_prices(cfg); earnings = io.load_earnings(cfg); si = io.load_short_interest(cfg)
    panel = build_panel(cfg, prices, earnings, si); del prices
    capacity = build_capacity(cfg, panel); panel.prices = pd.DataFrame()
    borrow = build_borrow(cfg, capacity, panel.short_interest_daily)
    alpha = build_alpha(cfg, borrow.borrow_panel, capacity.eligibility_by_aum, io.load_cheapness(cfg),
                        io.load_regime(cfg), io.load_earnings_transfo(cfg), earnings, panel.calendar)
    tp = build_trade_panel(cfg, alpha.scores, borrow.borrow_panel,
                           gics=io.load_gics(cfg), market_ret=io.load_sp500_tr(cfg))
    CACHE.mkdir(parents=True, exist_ok=True)
    tp.to_parquet(CACHE / "exp_trade_panel.parquet")
    capacity.eligibility_by_aum[AUM].to_parquet(CACHE / "exp_elig250.parquet")
    borrow.short_eligibility_by_aum[AUM][["instrument_id", "date", "short_eligibility"]].to_parquet(CACHE / "exp_shortelig250.parquet")
    print("cache written:", tp.shape)


def _vol_target(net: pd.Series, target_ann=0.06, lb=42, clip=(0.5, 2.0)) -> pd.Series:
    td = target_ann / np.sqrt(252)
    realised = net.rolling(lb, min_periods=lb // 2).std().shift(1)
    scale = (td / realised).clip(*clip).fillna(1.0)
    return net * scale


def _summ(net: pd.Series) -> dict:
    return dict(sharpe=annualised_sharpe(net), ann=net.mean() * 252 * 100,
                vol=net.std(ddof=1) * np.sqrt(252) * 100, maxdd=max_drawdown(net) * 100)


def battery(cfg):
    tp = pd.read_parquet(CACHE / "exp_trade_panel.parquet")
    elig = pd.read_parquet(CACHE / "exp_elig250.parquet")
    se = pd.read_parquet(CACHE / "exp_shortelig250.parquet")

    def book(score, q, weighting="equal"):
        d = run_strategy(cfg, tp, elig, se, AUM, score=score, quantile=q, weighting=weighting, **_OFF)
        return d["net_ret"]

    rows = []

    def add(name, net):
        rows.append({"variant": name, **_summ(net)})

    # 1) single-sleeve baselines at the headline quantile
    for sc in ["score_ens", "score_flow", "score_combined"]:
        add(f"{sc} q2% (every day)", book(sc, 0.02))

    # 2) reversal quantile sweep
    for q in [0.01, 0.015, 0.02, 0.03, 0.05]:
        add(f"reversal q{q*100:g}%", book("score_ens", q))

    # 3) flow quantile sweep
    for q in [0.01, 0.02, 0.03]:
        add(f"flow q{q*100:g}%", book("score_flow", q))

    # 4) TWO-BOOK return-level combination (reversal + flow), various splits & flow quantiles
    rev = book("score_ens", 0.02)
    for fq in [0.02, 0.03, 0.05]:
        fl = book("score_flow", fq)
        df = pd.concat([rev.rename("r"), fl.rename("f")], axis=1).fillna(0.0)
        for wflow in [0.3, 0.4, 0.5]:
            add(f"2book rev2%+flow{fq*100:g}% (wf={wflow})", (1 - wflow) * df["r"] + wflow * df["f"])

    # 5) vol-target on the best singles and the combo
    add("reversal q2% + voltarget", _vol_target(rev))
    fl3 = book("score_flow", 0.03)
    combo = (0.6 * rev.reindex(rev.index).fillna(0) + 0.4 * fl3.reindex(rev.index).fillna(0))
    add("2book(0.6/0.4) ", combo)
    add("2book(0.6/0.4) + voltarget", _vol_target(combo))

    # 6) inverse-vol weighting
    add("reversal q2% invvol", book("score_ens", 0.02, "invvol"))

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    out.to_csv(CACHE / "exp_results.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", action="store_true", help="run battery from existing cache only")
    args = ap.parse_args()
    cfg = load_config()
    if not args.battery:
        build_cache(cfg)
    battery(cfg)
