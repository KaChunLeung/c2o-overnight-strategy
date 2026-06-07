"""Step 4 / 4b — the cross-sectional overnight alpha (multi-sleeve, walk-forward).

Public API:
    AlphaResult
    build_alpha(cfg, borrow_panel, eligibility_by_aum, cheapness, regime, earnings_transfo,
                earnings_cal, calendar) -> AlphaResult

Scores the AUM-agnostic signal universe (base-OK at the most permissive AUM) so capacity can be applied
per-AUM downstream. Every feature is observable by 15:50 ET on day t; training is walk-forward expanding.

Two orthogonal sleeves are produced and combined:
  * ``score_ens``  — the reversal/price sleeve: rank average of a ridge (linear) + HGB (non-linear).
  * ``score_flow`` — the fundamental-flow sleeve: an HGB on analyst-revision / earnings-surprise features
    (merged as-of the prior trading day). Directional, so it lets the short leg short genuinely-falling
    names rather than drift-fighters.
  * ``score_combined`` — IC-weighted (walk-forward) rank average of the two sleeves.
A walk-forward calibration maps ``score_combined`` to an expected cross-sectional **excess** overnight
return (``expected_edge``, return units) so the portfolio can select names whose edge clears the round-trip
cost. Private helpers are ``_``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import Config
from .panel import TradingCalendar
from .metrics import daily_ic, summarize_ic

_CHEAPNESS_RENAME = {"valuation_score": "valuation_score_lag", "quality_score": "quality_score_lag",
                     "health_score": "health_score_lag", "momentum_score": "momentum_score_lag",
                     "score_velocity": "score_velocity_lag", "value_trap": "value_trap_lag"}

# raw analyst-revision / surprise columns in earnings_transfo (renamed to ``*_lag`` once merged as-of t-1)
_EARN_TRANSFO_COLS = ["epsp", "epsf", "reps1", "repsf4", "sue", "inesp", "inesn",
                      "reps41", "repsfs", "repsfl", "nspc5", "deps"]


@dataclass
class AlphaResult:
    scores: pd.DataFrame                 # instrument_id, date, year, score_*, expected_edge, target_r_on_next
    ic_compare: pd.DataFrame
    ic_by_year: pd.DataFrame
    rolling_ic: pd.DataFrame
    sleeve_ic: pd.DataFrame              # per-sleeve mean IC / t / hit-rate (reversal, flow, combined)
    sleeve_corr: pd.DataFrame            # cross-sectional rank correlation between sleeves


def _cross_sectional_zscore(frame: pd.DataFrame, col: str, clip: float) -> pd.Series:
    x = frame[col].replace([np.inf, -np.inf], np.nan).astype(float)
    grp = x.groupby(frame["date"])
    z = (x - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0.0).astype("float32")


def _asof_merge_prior_day(sig: pd.DataFrame, right: pd.DataFrame, right_on: str,
                          value_cols: list[str]) -> pd.DataFrame:
    """As-of (backward) merge ``right`` onto ``sig`` keyed on the prior trading day, per instrument.

    Returns a frame aligned to ``sig.index`` (rows with a NaT decision date get NaN). This guarantees no
    same-day leakage regardless of the record's intraday timing (e.g. AMC vs BMO earnings).
    """
    left = sig.loc[sig["decision_asof_date"].notna(), ["instrument_id", "decision_asof_date"]]
    left = left.sort_values(["decision_asof_date", "instrument_id"]).reset_index()
    r = right.sort_values([right_on, "instrument_id"])
    merged = pd.merge_asof(left, r, by="instrument_id", left_on="decision_asof_date",
                           right_on=right_on, direction="backward")
    return merged.set_index("index")[value_cols].reindex(sig.index)


def _add_flow_features(sig: pd.DataFrame, earnings_transfo: pd.DataFrame,
                       earnings_cal: pd.DataFrame) -> pd.DataFrame:
    """Attach analyst-revision / surprise features (``*_lag``) and the post-earnings clock, as-of t-1."""
    et = earnings_transfo.rename(columns={c: f"{c}_lag" for c in _EARN_TRANSFO_COLS})
    et = et.dropna(subset=["earn_feat_date"]).drop_duplicates(["instrument_id", "earn_feat_date"], keep="last")
    lag_cols = [f"{c}_lag" for c in _EARN_TRANSFO_COLS]
    flow = _asof_merge_prior_day(sig, et, "earn_feat_date", lag_cols)
    for c in lag_cols:
        sig[c] = flow[c].to_numpy()

    cal_dates = (earnings_cal.rename(columns={"stock_id": "instrument_id"})
                 .dropna(subset=["reporting_date"])[["instrument_id", "reporting_date"]]
                 .drop_duplicates())
    last_rep = _asof_merge_prior_day(sig, cal_dates, "reporting_date", ["reporting_date"])["reporting_date"]
    dse = (sig["decision_asof_date"] - last_rep).dt.days
    sig["days_since_earn"] = dse.clip(lower=0, upper=250)
    return sig


def _build_feature_panel(cfg: Config, borrow_panel: pd.DataFrame, eligibility: pd.DataFrame,
                         cheapness: pd.DataFrame, regime: pd.DataFrame,
                         earnings_transfo: pd.DataFrame, earnings_cal: pd.DataFrame,
                         cal: TradingCalendar) -> pd.DataFrame:
    """Add point-in-time features + the next-overnight target on the full per-instrument series, then
    restrict to the signal universe. Features are computed BEFORE filtering so lags/rolling windows use the
    true previous sessions (not the gappy eligible-only series), and no valid next-session target is lost.
    """
    sig = borrow_panel.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    sig["decision_asof_date"] = sig["date"].map(cal.previous)
    sig["next_trading_date"] = sig["date"].map(cal.next)
    g = sig.groupby("instrument_id", group_keys=False)

    # target: next session's overnight return (only when the next row is the consecutive session)
    sig["target_date"] = g["date"].shift(-1)
    sig["target_r_on_next"] = g["r_on"].shift(-1)
    sig.loc[sig["target_date"].ne(sig["next_trading_date"]), "target_r_on_next"] = np.nan

    # return features (today's gap is observable; the rest are lagged)
    sig["r_on_today"] = sig["r_on"]
    sig["r_on_lag1"] = g["r_on"].shift(1)
    sig["r_id_lag1"] = g["r_id"].shift(1)
    sig["r_cc_lag1"] = g["r_cc"].shift(1)
    sig["r_on_5d_lag"] = g["r_on"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    sig["r_cc_5d_lag"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).sum())
    sig["r_cc_20d_lag"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).sum())
    sig["r_cc_2d_lag"] = g["r_cc"].transform(lambda s: s.shift(1).rolling(2, min_periods=2).sum())
    sig["log_adv20"] = np.log1p(sig["adv20"].where(sig["adv20"] > 0))
    sig["log_ranking_mcap"] = np.log1p(sig["ranking_market_cap"].where(sig["ranking_market_cap"] > 0))

    on_vol = g["r_on"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
    sig["gap_z"] = (sig["r_on"] / on_vol.replace(0, np.nan)).clip(-10, 10)
    rng = np.log(sig["adj_high"].where(sig["adj_high"] > 0)) - np.log(sig["adj_low"].where(sig["adj_low"] > 0))
    sig["range_lag1"] = rng.groupby(sig["instrument_id"]).shift(1)
    dv_lag = sig["dollar_volume"].groupby(sig["instrument_id"]).shift(1)
    sig["amihud_lag1"] = np.log1p((sig["r_cc_lag1"].abs() / (dv_lag + 1.0) * 1e9).clip(lower=0))
    sig["vol_shock"] = np.log((dv_lag + 1.0) / (sig["adv20"] + 1.0)).clip(-3, 3)

    # cheapness/quality + regime, merged on the previous trading day (no same-day leakage)
    ch = cheapness.rename(columns={"date": "decision_asof_date", **_CHEAPNESS_RENAME})
    ch = ch.drop_duplicates(["instrument_id", "decision_asof_date"])
    ch["value_trap_lag"] = ch["value_trap_lag"].astype(float)
    sig = sig.merge(ch, on=["instrument_id", "decision_asof_date"], how="left")
    rg = regime.rename(columns={"date": "decision_asof_date", "regime": "market_regime"}).drop_duplicates("decision_asof_date")
    sig = sig.merge(rg, on="decision_asof_date", how="left")
    sig["market_regime"] = sig["market_regime"].fillna("Pre-2016 / unavailable")

    # fundamental-flow features (analyst revisions / surprise / PEAD clock), as-of the prior trading day
    sig = _add_flow_features(sig, earnings_transfo, earnings_cal)

    # restrict to the signal universe LAST (features/target already use the full continuous series)
    sig = sig.merge(eligibility[["instrument_id", "date", "eligibility"]], on=["instrument_id", "date"], how="left")
    return sig.loc[sig["eligibility"].eq("OK")].reset_index(drop=True)


def _fit_ridge(train: pd.DataFrame, z_features: list[str], lam: float) -> np.ndarray:
    tr = train.loc[train["target_rank_cs"].notna(), z_features + ["target_rank_cs"]].replace([np.inf, -np.inf], np.nan).dropna()
    X, y = tr[z_features].to_numpy("float64"), tr["target_rank_cs"].to_numpy("float64")
    y = y - y.mean()
    Xd = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xd.shape[1]); P[0, 0] = 0.0
    return np.linalg.solve(Xd.T @ Xd + lam * P, Xd.T @ y)


def _predict_ridge(frame: pd.DataFrame, z_features: list[str], beta: np.ndarray) -> np.ndarray:
    X = frame[z_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64")
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _walk_forward(cfg: Config, sig: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window ridge + HGB per OOS year; add score_ridge/score_hgb (in place)."""
    a = cfg.alpha
    rng = np.random.default_rng(cfg.run.seed)
    eval_years = (a for a in cfg.run.fast_eval_years) if cfg.run.fast else range(a.first_eval_year, a.last_eval_year + 1)
    sig["score_ridge"] = np.nan
    sig["score_hgb"] = np.nan
    for yr in eval_years:
        train_m = sig["valid_day"] & (sig["year"] < yr) & sig["target_rank_cs"].notna()
        test_m = sig["year"].eq(yr)
        if test_m.sum() == 0 or train_m.sum() < a.min_train_obs:
            continue
        tr = sig.loc[train_m]
        beta = _fit_ridge(tr, a.z_features, a.ridge_lambda)
        sig.loc[test_m, "score_ridge"] = _predict_ridge(sig.loc[test_m], a.z_features, beta)
        tr_h = tr
        if len(tr_h) > a.hgb_train_sample:
            tr_h = tr_h.loc[rng.choice(tr_h.index.to_numpy(), a.hgb_train_sample, replace=False)]
        hgb = HistGradientBoostingRegressor(random_state=cfg.run.seed, **a.hgb)
        hgb.fit(tr_h[a.raw_features].to_numpy("float32"), tr_h["target_rank_cs"].to_numpy("float64"))
        sig.loc[test_m, "score_hgb"] = hgb.predict(sig.loc[test_m, a.raw_features].to_numpy("float32"))
    return sig


def _walk_forward_flow(cfg: Config, sig: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window HGB on the fundamental-flow features (NaN-native) -> ``score_flow`` (in place)."""
    a = cfg.alpha
    feats = [f for f in a.flow_features if f in sig.columns]
    rng = np.random.default_rng(cfg.run.seed + 1)
    eval_years = (y for y in cfg.run.fast_eval_years) if cfg.run.fast else range(a.first_eval_year, a.last_eval_year + 1)
    sig["score_flow"] = np.nan
    if not feats:
        return sig
    for yr in eval_years:
        train_m = sig["valid_day"] & (sig["year"] < yr) & sig["target_rank_cs"].notna()
        test_m = sig["year"].eq(yr)
        if test_m.sum() == 0 or train_m.sum() < a.min_train_obs:
            continue
        tr = sig.loc[train_m]
        if len(tr) > a.hgb_train_sample:
            tr = tr.loc[rng.choice(tr.index.to_numpy(), a.hgb_train_sample, replace=False)]
        hgb = HistGradientBoostingRegressor(random_state=cfg.run.seed, **a.hgb)
        hgb.fit(tr[feats].to_numpy("float32"), tr["target_rank_cs"].to_numpy("float64"))
        sig.loc[test_m, "score_flow"] = hgb.predict(sig.loc[test_m, feats].to_numpy("float32"))
    return sig


def _combine_sleeves(cfg: Config, sig: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward IC-weighted rank-average of the reversal and flow sleeves -> ``score_combined``."""
    a = cfg.alpha
    sig["rank_reversal"] = sig.groupby("date")["score_ens"].rank(pct=True)
    sig["rank_flow"] = sig.groupby("date")["score_flow"].rank(pct=True)
    has_flow = sig["score_flow"].notna()
    if a.sleeve_combine == "reversal_only" or not has_flow.any():
        sig["score_combined"] = sig["rank_reversal"]
        return sig
    if a.sleeve_combine == "equal":
        sig["score_combined"] = sig[["rank_reversal", "rank_flow"]].mean(axis=1)
        sig["score_combined"] = sig["score_combined"].fillna(sig["rank_reversal"])
        return sig

    # ic_weighted: weight each sleeve by the mean of its IC over *prior* OOS years only (no peeking)
    yic = {}
    for label, col in [("rev", "score_ens"), ("flow", "score_flow")]:
        ic = daily_ic(sig.loc[sig["valid_day"]], col, "target_r_on_next", a.min_daily_names).reset_index()
        ic["year"] = ic["date"].dt.year
        yic[label] = ic.groupby("year")["ic"].mean()
    sig["score_combined"] = np.nan
    years = sorted(sig.loc[has_flow, "year"].unique())
    for yr in years:
        prev = [y for y in years if y < yr]
        wr = max(float(yic["rev"].reindex(prev).mean()), 0.0) if prev else 1.0
        wf = max(float(yic["flow"].reindex(prev).mean()), 0.0) if prev else 1.0
        tot = wr + wf if (wr + wf) > 0 else 1.0
        m = sig["year"].eq(yr)
        sig.loc[m, "score_combined"] = (wr * sig.loc[m, "rank_reversal"] + wf * sig.loc[m, "rank_flow"]) / tot
    sig["score_combined"] = sig["score_combined"].fillna(sig["rank_reversal"])
    return sig


def _calibrate_edge(cfg: Config, sig: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward map ``score_combined`` -> expected cross-sectional EXCESS overnight return (return units).

    The drift cancels in a dollar-neutral book, so the cost-relevant quantity is each name's excess vs the
    daily cross-section. Per OOS year we bin *prior*-year (score, realised excess) pairs and map this year's
    scores through that monotone bucket curve. Years with no prior signal get NaN (portfolio falls back).
    """
    nb = cfg.alpha.edge_calib_buckets
    daily_mean = sig.groupby("date")["target_r_on_next"].transform("mean")
    sig["target_excess"] = sig["target_r_on_next"] - daily_mean
    sig["expected_edge"] = np.nan
    cal_mask = sig["valid_day"] & sig["score_combined"].notna() & sig["target_excess"].notna()
    years = sorted(sig.loc[sig["score_combined"].notna(), "year"].unique())
    for yr in years:
        train = sig.loc[cal_mask & (sig["year"] < yr), ["score_combined", "target_excess"]]
        if len(train) < cfg.alpha.min_train_obs:
            continue
        edges = np.quantile(train["score_combined"], np.linspace(0, 1, nb + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
        b = pd.cut(train["score_combined"], edges, labels=False, include_lowest=True)
        bucket_excess = train["target_excess"].groupby(b).mean()
        test_m = sig["year"].eq(yr) & sig["score_combined"].notna()
        tb = pd.cut(sig.loc[test_m, "score_combined"], edges, labels=False, include_lowest=True)
        sig.loc[test_m, "expected_edge"] = tb.map(bucket_excess).to_numpy()
    return sig


def _sleeve_diagnostics(cfg: Config, sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sleeve mean IC and the cross-sectional rank correlation between sleeves (diversification proof)."""
    rows = []
    for label, col in [("reversal", "score_ens"), ("flow", "score_flow"), ("combined", "score_combined")]:
        ic = daily_ic(sig.loc[sig["valid_day"]], col, "target_r_on_next", cfg.alpha.min_daily_names)
        rows.append({"sleeve": label, **summarize_ic(ic)})
    sleeve_ic = pd.DataFrame(rows)
    both = sig.loc[sig["rank_reversal"].notna() & sig["rank_flow"].notna(), ["rank_reversal", "rank_flow"]]
    sleeve_corr = both.corr(method="spearman")
    return sleeve_ic, sleeve_corr


def _evaluate(cfg: Config, sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IC comparison across models, IC by year, and the rolling-IC series for the combined alpha."""
    rows = []
    for label, col in [("ridge", "score_ridge"), ("HGB", "score_hgb"), ("ensemble (reversal)", "score_ens"),
                       ("flow", "score_flow"), ("combined", "score_combined")]:
        ic = daily_ic(sig.loc[sig["valid_day"]], col, "target_r_on_next", cfg.alpha.min_daily_names)
        s = summarize_ic(ic)
        rows.append({"alpha": label, **s})
    ic_compare = pd.DataFrame(rows)

    ens = daily_ic(sig.loc[sig["valid_day"]], "score_ens", "target_r_on_next", cfg.alpha.min_daily_names).reset_index()
    ens["year"] = ens["date"].dt.year
    by_year = ens.groupby("year")["ic"].agg(["mean", "std", "count"])
    ens = ens.sort_values("date")
    ens["rolling_ic"] = ens["ic"].rolling(cfg.alpha.rolling_ic_window, min_periods=40).mean()
    return ic_compare, by_year, ens[["date", "ic", "rolling_ic"]]


def build_alpha(cfg: Config, borrow_panel: pd.DataFrame, eligibility_by_aum: dict[float, pd.DataFrame],
                cheapness: pd.DataFrame, regime: pd.DataFrame, earnings_transfo: pd.DataFrame,
                earnings_cal: pd.DataFrame, cal: TradingCalendar) -> AlphaResult:
    """Run Step 4b end to end: two sleeves, their combination, the edge calibration, and IC diagnostics."""
    sig = _build_feature_panel(cfg, borrow_panel, eligibility_by_aum[cfg.alpha.signal_aum],
                               cheapness, regime, earnings_transfo, earnings_cal, cal)
    for f in cfg.alpha.raw_features:
        sig[f"z_{f}"] = _cross_sectional_zscore(sig, f, cfg.alpha.zscore_clip)
    sig["target_rank_cs"] = sig.groupby("date")["target_r_on_next"].rank(pct=True) - 0.5
    sig["valid_day"] = sig.groupby("date")["target_r_on_next"].transform("count") >= cfg.alpha.min_daily_names

    sig = _walk_forward(cfg, sig)                                   # reversal sleeve: ridge + HGB
    sig["score_ens"] = (sig.groupby("date")["score_ridge"].rank(pct=True)
                        + sig.groupby("date")["score_hgb"].rank(pct=True)) / 2.0
    sig = _walk_forward_flow(cfg, sig)                              # flow sleeve: HGB on revisions/surprise
    sig = _combine_sleeves(cfg, sig)                                # IC-weighted combination
    sig = _calibrate_edge(cfg, sig)                                 # score_combined -> expected excess edge

    ic_compare, by_year, rolling = _evaluate(cfg, sig)
    sleeve_ic, sleeve_corr = _sleeve_diagnostics(cfg, sig)
    keep = ["instrument_id", "date", "year", "score_ridge", "score_hgb", "score_ens", "score_flow",
            "score_combined", "expected_edge", "target_r_on_next", "market_regime"]
    scores = sig.loc[sig["score_combined"].notna(), keep].copy()
    return AlphaResult(scores=scores, ic_compare=ic_compare, ic_by_year=by_year, rolling_ic=rolling,
                       sleeve_ic=sleeve_ic, sleeve_corr=sleeve_corr)
