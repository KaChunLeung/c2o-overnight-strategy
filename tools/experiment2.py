"""DEV-ONLY decisive battery (reads cache from experiment.py): concentration, edge-scaling, short-leg fix.

Battery 1 showed tail concentration is the lever (reversal q1% >> q2%) and that the flow sleeve fails as a
standalone tail book. Here we test the remaining high-value ideas:
  A. fine quantile sweep with annual stability (is q1% / q0.75% robust or overfit?);
  B. edge-proportional gross scaling / day gating (the only lever that can cut cost where edge is thin);
  C. surgical use of flow on the SHORT leg only (short reversal-bottom names unless flow says they'll rise);
  D. the winner at $50M / $250M / $1B (v1 Sharpe rose with AUM as the cap forces low-vol names).

    PYTHONPATH=src python tools/experiment2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from c2o import io
from c2o.config import load_config
from c2o.metrics import annualised_sharpe, max_drawdown
from c2o.portfolio import run_strategy

CACHE = Path("data/intermediary")
_OFF = dict(selection_mode="quantile", neutralize_sector=False, neutralize_beta=False, vol_target=False)


def _summ(net: pd.Series) -> dict:
    ann_yr = net.groupby(net.index.year).apply(lambda x: (1 + x).prod() - 1)
    return dict(sharpe=annualised_sharpe(net), ann=net.mean() * 252 * 100,
                vol=net.std(ddof=1) * np.sqrt(252) * 100, maxdd=max_drawdown(net) * 100,
                pos_yrs=f"{int((ann_yr > 0).sum())}/{len(ann_yr)}", frac_on=float((net.abs() > 1e-12).mean()))


def _pred_spread(fr, score, q):
    out = {}
    for date, day in fr.groupby("date", sort=True):
        if len(day) < 80:
            continue
        rank = day[score].rank(pct=True)
        out[date] = day.loc[rank >= 1 - q, "expected_edge"].mean() - day.loc[rank <= q, "expected_edge"].mean()
    return pd.Series(out).sort_index()


def _vol_target(net, target_ann=0.05, lb=63, clip=(0.5, 2.0)):
    td = target_ann / np.sqrt(252)
    realised = net.rolling(lb, min_periods=lb // 2).std().shift(1)
    return net * (td / realised).clip(*clip).fillna(1.0)


def main():
    cfg = load_config()
    tp = pd.read_parquet(CACHE / "exp_trade_panel.parquet")
    elig = pd.read_parquet(CACHE / "exp_elig250.parquet")
    se = pd.read_parquet(CACHE / "exp_shortelig250.parquet")
    fr = (tp.merge(elig[["instrument_id", "date", "eligibility"]], on=["instrument_id", "date"], how="left")
            .merge(se, on=["instrument_id", "date"], how="left"))
    fr = fr.loc[fr["eligibility"].eq("OK")].copy()

    def book(score, q, weighting="equal", short_e=se):
        return run_strategy(cfg, tp, elig, short_e, 250e6, score=score, quantile=q, weighting=weighting, **_OFF)["net_ret"]

    rows, add = [], None
    rows = []
    add = lambda name, net: rows.append({"variant": name, **_summ(net)})

    # ---- A. fine quantile sweep + stability ----
    for q in [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02]:
        add(f"A reversal q{q*100:g}%", book("score_ens", q))

    # ---- B. edge scaling / gating on reversal q1% ----
    rev1 = book("score_ens", 0.01)
    pred = _pred_spread(fr, "score_ens", 0.01).reindex(rev1.index)
    gd = fr.groupby("date")["gap_disp"].first().reindex(rev1.index)
    valid = pred.notna() & rev1.notna()
    print(f"[check] corr(pred_spread, net_ret)={np.corrcoef(pred[valid].rank(), rev1[valid].rank())[0,1]:.3f}; "
          f"corr(gap_disp, net_ret)={np.corrcoef(gd[valid].rank(), rev1[valid].rank())[0,1]:.3f}")
    for sig_name, sig in [("predspread", pred), ("gapdisp", gd)]:
        for keep in [0.4, 0.6, 0.8]:
            thr = sig.shift(1).expanding(min_periods=150).quantile(1 - keep)
            f = (sig >= thr).astype(float).fillna(1.0)
            add(f"B rev1% gate top{int(keep*100)}% {sig_name}", f * rev1)
    add("B rev1% + voltarget(5%,63d)", _vol_target(rev1))

    # ---- C. flow fixes the short leg: short reversal-bottom names only if flow does NOT rank them high ----
    flow_rank = fr.groupby("date")["score_flow"].rank(pct=True)
    fr2 = fr.assign(flow_rank=flow_rank)
    for thr_flow in [0.5, 0.7]:
        se_mod = se.merge(fr2[["instrument_id", "date", "flow_rank"]], on=["instrument_id", "date"], how="left")
        se_mod["short_eligibility"] = np.where(
            se_mod["short_eligibility"].eq("OK") & (se_mod["flow_rank"].fillna(0) > thr_flow),
            "FLOW_VETO", se_mod["short_eligibility"])
        se_mod = se_mod[["instrument_id", "date", "short_eligibility"]]
        add(f"C rev1% short-veto flow>{thr_flow}", book("score_ens", 0.01, short_e=se_mod))

    out = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(out.to_string(index=False))
    out.to_csv(CACHE / "exp2_results.csv", index=False)


if __name__ == "__main__":
    main()
