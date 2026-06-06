"""Step 4 / 4b — the cross-sectional overnight alpha (ridge + gradient-boosting ensemble).

Public API:
    AlphaResult
    build_alpha(cfg, borrow_panel, eligibility_by_aum, cheapness, regime, calendar) -> AlphaResult

Scores the AUM-agnostic signal universe (base-OK at the most permissive AUM) so capacity can be applied
per-AUM downstream. Every feature is observable by 15:50 ET on day t; training is walk-forward expanding.
The ensemble is the cross-sectional rank average of a ridge (linear) and an HGB (non-linear) model.
Private helpers are ``_``.
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


@dataclass
class AlphaResult:
    scores: pd.DataFrame                 # instrument_id, date, year, score_ridge/hgb/ens, target_r_on_next
    ic_compare: pd.DataFrame
    ic_by_year: pd.DataFrame
    rolling_ic: pd.DataFrame


def _cross_sectional_zscore(frame: pd.DataFrame, col: str, clip: float) -> pd.Series:
    x = frame[col].replace([np.inf, -np.inf], np.nan).astype(float)
    grp = x.groupby(frame["date"])
    z = (x - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0.0).astype("float32")


def _build_feature_panel(cfg: Config, borrow_panel: pd.DataFrame, eligibility: pd.DataFrame,
                         cheapness: pd.DataFrame, regime: pd.DataFrame, cal: TradingCalendar) -> pd.DataFrame:
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


def _evaluate(cfg: Config, sig: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """IC comparison across models, IC by year, and the rolling-IC series for the ensemble."""
    rows = []
    for label, col in [("ridge", "score_ridge"), ("HGB", "score_hgb"), ("ensemble", "score_ens")]:
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
                cheapness: pd.DataFrame, regime: pd.DataFrame, cal: TradingCalendar) -> AlphaResult:
    """Run Step 4b end to end and return the AUM-agnostic ensemble scores plus IC diagnostics."""
    sig = _build_feature_panel(cfg, borrow_panel, eligibility_by_aum[cfg.alpha.signal_aum],
                               cheapness, regime, cal)
    for f in cfg.alpha.raw_features:
        sig[f"z_{f}"] = _cross_sectional_zscore(sig, f, cfg.alpha.zscore_clip)
    sig["target_rank_cs"] = sig.groupby("date")["target_r_on_next"].rank(pct=True) - 0.5
    sig["valid_day"] = sig.groupby("date")["target_r_on_next"].transform("count") >= cfg.alpha.min_daily_names

    sig = _walk_forward(cfg, sig)
    sig["score_ens"] = (sig.groupby("date")["score_ridge"].rank(pct=True)
                        + sig.groupby("date")["score_hgb"].rank(pct=True)) / 2.0
    ic_compare, by_year, rolling = _evaluate(cfg, sig)
    scores = sig.loc[sig["score_ens"].notna(),
                     ["instrument_id", "date", "year", "score_ridge", "score_hgb", "score_ens",
                      "target_r_on_next", "market_regime"]].copy()
    return AlphaResult(scores=scores, ic_compare=ic_compare, ic_by_year=by_year, rolling_ic=rolling)
