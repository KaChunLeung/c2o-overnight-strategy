"""Step 5 — ranking to a dollar-neutral overnight portfolio with the Section 6.3 cost schedule.

Public API:
    build_trade_panel(cfg, scores, borrow_panel) -> DataFrame
    run_strategy(cfg, trade_panel, eligibility, short_eligibility, aum, ...) -> daily DataFrame
    perf_summary(cfg, daily, aum) -> dict

Positions are sized with the brief's participation-cap water-fill (cap each name at 5% of ADV, redistribute
the excess pro-rata, reduce gross if the basket cannot absorb the target). The overnight book is fully
liquidated each morning, so the round trip (MOC entry + MOO exit) is charged every night. Helpers are ``_``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .metrics import annualised_sharpe, max_drawdown


def build_trade_panel(cfg: Config, scores: pd.DataFrame, borrow_panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (instrument, date) with score, label, risk, ADV, borrow cost and gap dispersion."""
    cols = ["instrument_id", "date", "r_on", "vol20_ann", "adv20", "borrow_daily_rate", "borrow_tier"]
    tp = scores.merge(borrow_panel[cols], on=["instrument_id", "date"], how="left")
    tp["vol20_ann"] = tp["vol20_ann"].fillna(tp["vol20_ann"].median())
    tp["adv20"] = tp["adv20"].fillna(tp["adv20"].median())
    gc_daily = cfg.borrow.annual_rates["A"] / cfg.borrow.trading_days_per_year
    tp["borrow_daily_rate"] = tp["borrow_daily_rate"].fillna(gc_daily)
    tp["borrow_tier"] = tp["borrow_tier"].fillna("A")
    gap = tp.groupby("date")["r_on"].std().rename("gap_disp")          # ex-ante dispersion (observable 15:50)
    tp = tp.merge(gap, on="date", how="left")
    return tp.loc[tp["target_r_on_next"].notna()].copy()


def _waterfill_alloc(weights: np.ndarray, caps: np.ndarray, capital: float) -> np.ndarray:
    """Allocate ``capital`` over names with target proportions ``weights`` under per-name dollar ``caps``.

    Excess from capped names is redistributed pro-rata to the rest (Section 6.2). The sum is < capital iff
    every name is at its cap (gross then reduced).
    """
    w = np.asarray(weights, float)
    caps = np.asarray(caps, float)
    alloc = np.zeros(len(w))
    free = np.ones(len(w), bool)
    remaining = float(capital)
    for _ in range(64):
        wf = w * free
        sw = wf.sum()
        if sw <= 0 or remaining <= 1e-9:
            break
        prop = np.where(free, wf / sw * remaining, 0.0)
        over = free & (prop > caps)
        if not over.any():
            alloc[free] = prop[free]
            break
        alloc[over] = caps[over]
        remaining -= caps[over].sum()
        free &= ~over
    return alloc


def _side_weights(side: pd.DataFrame, weighting: str, score: str) -> np.ndarray:
    if weighting == "invvol":
        w = (1.0 / side["vol20_ann"].clip(lower=0.05)).to_numpy()
    elif weighting == "score":
        w = (side[score] - 0.5).abs().to_numpy() + 1e-9
    else:
        w = np.ones(len(side))
    return w / w.sum()


def run_strategy(cfg: Config, trade_panel: pd.DataFrame, eligibility: pd.DataFrame,
                 short_eligibility: pd.DataFrame, aum: float, quantile: float | None = None,
                 weighting: str | None = None, score: str | None = None,
                 htb_exclude: bool | None = None, gate_q: float | None = None) -> pd.DataFrame:
    """Daily dollar-neutral overnight L/S backtest at one AUM. Returns a per-day diagnostics frame."""
    p = cfg.portfolio
    quantile = p.headline_quantile if quantile is None else quantile
    weighting = p.headline_weighting if weighting is None else weighting
    score = p.headline_score if score is None else score
    htb_exclude = p.htb_exclude if htb_exclude is None else htb_exclude
    gate_q = p.gate_q if gate_q is None else gate_q
    comm = p.commission_bps_per_leg * 2 * 1e-4
    slip = p.slippage_bps_per_leg * 2 * 1e-4
    side_capital = aum / 2.0

    fr = (trade_panel.merge(eligibility[["instrument_id", "date", "eligibility"]], on=["instrument_id", "date"], how="left")
                     .merge(short_eligibility[["instrument_id", "date", "short_eligibility"]], on=["instrument_id", "date"], how="left"))
    fr = fr.loc[fr["eligibility"].eq("OK")].copy()
    fr["cap_usd"] = cfg.capacity.participation_cap * fr["adv20"]

    recs = []
    for date, day in fr.groupby("date", sort=True):
        if len(day) < cfg.alpha.min_daily_names:
            continue
        rank = day[score].rank(pct=True)
        long_mask = rank >= (1 - quantile)
        short_mask = rank <= quantile
        if htb_exclude:
            short_mask = short_mask & day["short_eligibility"].eq("OK")
        longs, shorts = day[long_mask], day[short_mask]
        if len(longs) < p.min_basket_names or len(shorts) < p.min_basket_names:
            continue
        al = _waterfill_alloc(_side_weights(longs, weighting, score), longs["cap_usd"].to_numpy(), side_capital)
        ash = _waterfill_alloc(_side_weights(shorts, weighting, score), shorts["cap_usd"].to_numpy(), side_capital)
        deployed = min(al.sum(), ash.sum())                       # enforce dollar neutrality
        al = al * (deployed / al.sum()) if al.sum() > 0 else al
        ash = ash * (deployed / ash.sum()) if ash.sum() > 0 else ash
        gross = al.sum() + ash.sum()
        pnl = (al * longs["target_r_on_next"].to_numpy()).sum() - (ash * shorts["target_r_on_next"].to_numpy()).sum()
        borrow = (ash * shorts["borrow_daily_rate"].to_numpy()).sum()
        max_adv = float(np.max(np.concatenate([al / (longs["adv20"].to_numpy() + 1.0),
                                               ash / (shorts["adv20"].to_numpy() + 1.0)]))) if gross > 0 else 0.0
        recs.append(dict(date=date, gross_pnl=pnl, commission=comm * gross, slippage=slip * gross,
                         borrow=borrow, gross_usd=gross, n_long=len(longs), n_short=len(shorts),
                         gap_disp=day["gap_disp"].iloc[0], max_pos_pct_adv=max_adv))

    cols = ["gross_pnl", "commission", "slippage", "borrow", "gross_usd", "n_long", "n_short",
            "gap_disp", "max_pos_pct_adv"]
    if not recs:                                            # no tradable days (e.g. truncated smoke universe)
        empty = pd.DataFrame(columns=cols + ["traded", "gross_ret", "net_ret", "turnover", "aum"])
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    daily = pd.DataFrame(recs).set_index("date").sort_index()
    if gate_q > 0:
        thr = daily["gap_disp"].shift(1).rolling(cfg.borrow.trading_days_per_year, min_periods=60).quantile(gate_q)
        daily["traded"] = (daily["gap_disp"] >= thr).fillna(True)
    else:
        daily["traded"] = True
    gp = daily["gross_pnl"] / aum
    cost = (daily["commission"] + daily["slippage"] + daily["borrow"]) / aum
    daily["gross_ret"] = np.where(daily["traded"], gp, 0.0)
    daily["net_ret"] = np.where(daily["traded"], gp - cost, 0.0)
    daily["turnover"] = np.where(daily["traded"], daily["gross_usd"] / aum, 0.0)
    daily["aum"] = aum
    return daily


def perf_summary(cfg: Config, daily: pd.DataFrame, aum: float) -> dict[str, float]:
    """Headline metrics for one AUM run (costs in bps of AUM per traded day)."""
    tr = daily["traded"]
    nt = max(int(tr.sum()), 1)
    return {
        "AUM": f"{aum / 1e6:.0f}M",
        "net_ann_ret": daily["net_ret"].mean() * 252,
        "net_ann_vol": daily["net_ret"].std(ddof=1) * np.sqrt(252),
        "net_sharpe": annualised_sharpe(daily["net_ret"]),
        "gross_sharpe": annualised_sharpe(daily["gross_ret"]),
        "max_drawdown": max_drawdown(daily["net_ret"]),
        "daily_turnover": daily.loc[tr, "turnover"].mean(),
        "avg_gross_util": daily.loc[tr, "gross_usd"].mean() / aum,
        "frac_days_full_gross": float((daily.loc[tr, "gross_usd"] / aum >= 0.999).mean()),
        "frac_days_traded": float(tr.mean()),
        "avg_max_pos_pct_adv": daily.loc[tr, "max_pos_pct_adv"].mean(),
        "comm_bps": daily.loc[tr, "commission"].sum() / aum / nt * 1e4,
        "slip_bps": daily.loc[tr, "slippage"].sum() / aum / nt * 1e4,
        "borrow_bps": daily.loc[tr, "borrow"].sum() / aum / nt * 1e4,
    }
