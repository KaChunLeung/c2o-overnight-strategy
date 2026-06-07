"""Step 5 — ranking to a dollar-neutral overnight portfolio with the Section 6.3 cost schedule.

Public API:
    build_trade_panel(cfg, scores, borrow_panel, gics=None, market_ret=None) -> DataFrame
    run_strategy(cfg, trade_panel, eligibility, short_eligibility, aum, ...) -> daily DataFrame
    perf_summary(cfg, daily, aum) -> dict

Two selection modes share one sizing/cost engine:
  * ``quantile``    — long the top fraction, short the bottom fraction of a score (the v1 book).
  * ``cost_aware``  — trade only names whose calibrated cross-sectional EXCESS edge clears the round-trip
    cost (+borrow for shorts). The book size adapts to the daily edge; thin nights are skipped, which is
    where the slippage drag is cut. An optional sector/beta neutralization shrinks vol, and a vol-target
    overlay scales daily gross to a constant risk budget.

Positions are sized with the brief's participation-cap water-fill (cap each name at 5% of ADV, redistribute
the excess pro-rata, reduce gross if the basket cannot absorb the target). The overnight book is fully
liquidated each morning, so the round trip (MOC entry + MOO exit) is charged every night. Helpers are ``_``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .metrics import annualised_sharpe, max_drawdown


def _trailing_beta(borrow_panel: pd.DataFrame, market_ret: pd.Series, lookback: int) -> pd.DataFrame:
    """Per-(instrument, date) trailing market beta of close-to-close returns, lagged one session."""
    bp = borrow_panel[["instrument_id", "date", "r_cc"]].copy()
    bp["mkt"] = bp["date"].map(market_ret)
    bp = bp.sort_values(["instrument_id", "date"])
    g = bp.groupby("instrument_id")
    rc, mk = g["r_cc"].shift(1), g["mkt"].shift(1)
    mp = max(lookback // 2, 20)
    by = bp["instrument_id"]
    roll = lambda s: s.groupby(by).transform(lambda x: x.rolling(lookback, min_periods=mp).mean())
    mean_rc, mean_mk = roll(rc), roll(mk)
    cov = roll(rc * mk) - mean_rc * mean_mk
    var = roll(mk * mk) - mean_mk * mean_mk
    bp["beta"] = (cov / var.replace(0, np.nan)).clip(-3.0, 3.0)
    return bp[["instrument_id", "date", "beta"]]


def build_trade_panel(cfg: Config, scores: pd.DataFrame, borrow_panel: pd.DataFrame,
                      gics: pd.DataFrame | None = None, market_ret: pd.Series | None = None) -> pd.DataFrame:
    """One row per (instrument, date) with scores, edge, risk, ADV, borrow, sector, beta and gap dispersion."""
    cols = ["instrument_id", "date", "r_on", "vol20_ann", "adv20", "borrow_daily_rate", "borrow_tier"]
    tp = scores.merge(borrow_panel[cols], on=["instrument_id", "date"], how="left")
    tp["vol20_ann"] = tp["vol20_ann"].fillna(tp["vol20_ann"].median())
    tp["adv20"] = tp["adv20"].fillna(tp["adv20"].median())
    gc_daily = cfg.borrow.annual_rates["A"] / cfg.borrow.trading_days_per_year
    tp["borrow_daily_rate"] = tp["borrow_daily_rate"].fillna(gc_daily)
    tp["borrow_tier"] = tp["borrow_tier"].fillna("A")

    if gics is not None:
        tp = tp.merge(gics, on="instrument_id", how="left")
    tp["sector"] = tp["sector"].fillna(-1).astype(int) if "sector" in tp else -1
    if market_ret is not None:
        tp = tp.merge(_trailing_beta(borrow_panel, market_ret, cfg.portfolio.beta_lookback),
                      on=["instrument_id", "date"], how="left")
    tp["beta"] = tp["beta"].fillna(1.0) if "beta" in tp else 1.0

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


def _neutralize(day: pd.DataFrame, sig: pd.Series, neut_sector: bool, neut_beta: bool) -> pd.Series:
    """Remove sector and/or market-beta tilts from the selection signal (cross-sectionally, within a day)."""
    s = sig.astype(float).copy()
    if neut_sector and "sector" in day:
        s = s - s.groupby(day["sector"]).transform("mean")
    if neut_beta and "beta" in day:
        b = day["beta"].fillna(1.0).to_numpy()
        bb = b - b.mean()
        denom = float((bb * bb).sum())
        if denom > 0:
            coef = float((bb * (s - s.mean()).to_numpy()).sum()) / denom
            s = s - coef * pd.Series(bb, index=s.index)
    return s


def _side_weights(side: pd.DataFrame, weighting: str, wmag: np.ndarray) -> np.ndarray:
    """Within-leg target proportions: equal, inverse-vol, or proportional to conviction magnitude."""
    if weighting == "invvol":
        w = (1.0 / side["vol20_ann"].clip(lower=0.05)).to_numpy()
    elif weighting == "score":
        w = np.abs(wmag) + 1e-9
    else:
        w = np.ones(len(side))
    return w / w.sum()


def _cost_aware_masks(day: pd.DataFrame, sel: pd.Series, c: float, rt_ret: float,
                      htb_exclude: bool) -> tuple[pd.Series, pd.Series]:
    """Per-name bar: long if excess edge > c*round_trip; short if edge < -(c*round_trip + borrow)."""
    bar = c * rt_ret
    long_mask = sel > bar
    short_mask = sel < -(bar + day["borrow_daily_rate"].to_numpy())
    if htb_exclude:
        short_mask = short_mask & day["short_eligibility"].eq("OK")
    return long_mask, short_mask


def _quantile_masks(day: pd.DataFrame, score: str, quantile: float, htb_exclude: bool) -> tuple[pd.Series, pd.Series]:
    rank = day[score].rank(pct=True)
    long_mask = rank >= (1 - quantile)
    short_mask = rank <= quantile
    if htb_exclude:
        short_mask = short_mask & day["short_eligibility"].eq("OK")
    return long_mask, short_mask


def _apply_vol_target(daily: pd.DataFrame, cfg: Config, target_ann: float) -> pd.DataFrame:
    """Scale daily gross (and hence pnl + cost) to a constant net-vol target using trailing realised vol."""
    p = cfg.portfolio
    target_daily = target_ann / np.sqrt(cfg.borrow.trading_days_per_year)
    realised = daily["net_ret"].rolling(p.vol_target_lookback, min_periods=p.vol_target_lookback // 2).std().shift(1)
    scale = (target_daily / realised).clip(p.vol_target_clip[0], p.vol_target_clip[1]).fillna(1.0)
    daily["vol_scale"] = scale
    for col in ["gross_ret", "net_ret", "turnover"]:
        daily[col] = daily[col] * scale
    for col in ["commission", "slippage", "borrow", "gross_usd"]:
        daily[col] = daily[col] * scale
    return daily


def run_strategy(cfg: Config, trade_panel: pd.DataFrame, eligibility: pd.DataFrame,
                 short_eligibility: pd.DataFrame, aum: float, quantile: float | None = None,
                 weighting: str | None = None, score: str | None = None,
                 htb_exclude: bool | None = None, gate_q: float | None = None,
                 selection_mode: str | None = None, cost_buffer_c: float | None = None,
                 neutralize_sector: bool | None = None, neutralize_beta: bool | None = None,
                 vol_target: bool | None = None, vol_target_ann: float | None = None) -> pd.DataFrame:
    """Daily dollar-neutral overnight L/S backtest at one AUM. Returns a per-day diagnostics frame."""
    p = cfg.portfolio
    quantile = p.headline_quantile if quantile is None else quantile
    weighting = p.headline_weighting if weighting is None else weighting
    score = p.headline_score if score is None else score
    htb_exclude = p.htb_exclude if htb_exclude is None else htb_exclude
    gate_q = p.gate_q if gate_q is None else gate_q
    selection_mode = p.selection_mode if selection_mode is None else selection_mode
    cost_buffer_c = p.cost_buffer_c if cost_buffer_c is None else cost_buffer_c
    neutralize_sector = p.neutralize_sector if neutralize_sector is None else neutralize_sector
    neutralize_beta = p.neutralize_beta if neutralize_beta is None else neutralize_beta
    vol_target = p.vol_target_enabled if vol_target is None else vol_target
    vol_target_ann = p.vol_target_ann if vol_target_ann is None else vol_target_ann

    comm = p.commission_bps_per_leg * 2 * 1e-4
    slip = p.slippage_bps_per_leg * 2 * 1e-4
    rt_ret = p.round_trip_bps * 1e-4
    side_capital = aum / 2.0
    has_edge = selection_mode == "cost_aware" and "expected_edge" in trade_panel.columns

    fr = (trade_panel.merge(eligibility[["instrument_id", "date", "eligibility"]], on=["instrument_id", "date"], how="left")
                     .merge(short_eligibility[["instrument_id", "date", "short_eligibility"]], on=["instrument_id", "date"], how="left"))
    fr = fr.loc[fr["eligibility"].eq("OK")].copy()
    fr["cap_usd"] = cfg.capacity.participation_cap * fr["adv20"]
    min_names = p.min_edge_names if has_edge else p.min_basket_names

    recs = []
    for date, day in fr.groupby("date", sort=True):
        if len(day) < cfg.alpha.min_daily_names:
            continue
        if has_edge and day["expected_edge"].notna().any():
            sel = _neutralize(day, day["expected_edge"].fillna(0.0), neutralize_sector, neutralize_beta)
            long_mask, short_mask = _cost_aware_masks(day, sel, cost_buffer_c, rt_ret, htb_exclude)
            wmag = sel
        else:                                              # quantile (also the cost-aware fallback pre-calibration)
            base = day[score]
            if neutralize_sector or neutralize_beta:
                base = _neutralize(day, base, neutralize_sector, neutralize_beta)
                day = day.assign(_neut=base)
                long_mask, short_mask = _quantile_masks(day, "_neut", quantile, htb_exclude)
            else:
                long_mask, short_mask = _quantile_masks(day, score, quantile, htb_exclude)
            wmag = (day[score] - 0.5)
        longs, shorts = day[long_mask], day[short_mask]
        if len(longs) < min_names or len(shorts) < min_names:
            recs.append(dict(date=date, gross_pnl=0.0, commission=0.0, slippage=0.0, borrow=0.0,
                             gross_usd=0.0, n_long=0, n_short=0, gap_disp=day["gap_disp"].iloc[0],
                             max_pos_pct_adv=0.0))
            continue
        al = _waterfill_alloc(_side_weights(longs, weighting, wmag[long_mask].to_numpy()),
                              longs["cap_usd"].to_numpy(), side_capital)
        ash = _waterfill_alloc(_side_weights(shorts, weighting, wmag[short_mask].to_numpy()),
                               shorts["cap_usd"].to_numpy(), side_capital)
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
        empty = pd.DataFrame(columns=cols + ["traded", "gross_ret", "net_ret", "turnover", "aum", "vol_scale"])
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    daily = pd.DataFrame(recs).set_index("date").sort_index()
    base_traded = daily["gross_usd"] > 0
    if gate_q > 0:
        thr = daily["gap_disp"].shift(1).rolling(cfg.borrow.trading_days_per_year, min_periods=60).quantile(gate_q)
        daily["traded"] = (base_traded & (daily["gap_disp"] >= thr).fillna(True))
    else:
        daily["traded"] = base_traded
    gp = daily["gross_pnl"] / aum
    cost = (daily["commission"] + daily["slippage"] + daily["borrow"]) / aum
    daily["gross_ret"] = np.where(daily["traded"], gp, 0.0)
    daily["net_ret"] = np.where(daily["traded"], gp - cost, 0.0)
    daily["turnover"] = np.where(daily["traded"], daily["gross_usd"] / aum, 0.0)
    daily["aum"] = aum
    daily["vol_scale"] = 1.0
    if vol_target:
        daily = _apply_vol_target(daily, cfg, vol_target_ann)
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
        "avg_names_per_side": float(daily.loc[tr, ["n_long", "n_short"]].mean().mean()) if nt else 0.0,
    }
