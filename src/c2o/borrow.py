"""Step 3 — short-leg borrow overlay: a point-in-time hard-to-borrow proxy and tier assignment.

Public API:
    BorrowResult
    build_borrow(cfg, capacity_result, short_interest_daily) -> BorrowResult

Combines lagged short-interest features (dsi, dtcn, ddtcn) with daily cross-sectional percentiles and a
size/liquidity context flag into an additive stress score, mapped to tiers A/B/C. Tier C is excluded from
new short entries; A/B are charged the Section 6.3 borrow rate. Private helpers are ``_``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config
from .capacity import CapacityResult

_SHORT_REASON_COLUMNS = ["OK", "HTB_EXCLUDE", "ADV_FAIL", "MCAP_FAIL", "PRICE_FAIL", "VOL_FAIL", "EARN_WINDOW"]


@dataclass
class BorrowResult:
    borrow_panel: pd.DataFrame
    short_eligibility_by_aum: dict[float, pd.DataFrame]
    tier_distribution: pd.DataFrame


def _percentile_flags(df: pd.DataFrame, col: str, cfg: Config) -> tuple[pd.Series, pd.Series]:
    """Moderate/high component flags combining absolute thresholds and daily cross-sectional percentiles."""
    b = cfg.borrow
    rank = df.groupby("date")[col].rank(pct=True)
    abs_mod = {"dsi": b.dsi_moderate, "dtcn": b.dtcn_moderate, "ddtcn": b.ddtcn_moderate}[col]
    abs_high = {"dsi": b.dsi_high, "dtcn": b.dtcn_high, "ddtcn": b.ddtcn_high}[col]
    moderate = (df[col] >= abs_mod) | (rank >= b.moderate_pct)
    high = (df[col] >= abs_high) | (rank >= b.high_pct)
    return moderate, high


def _assign_tiers(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute the additive borrow-stress score and map it to tiers A/B/C with borrow rates."""
    b = cfg.borrow
    dsi_m, dsi_h = _percentile_flags(df, "dsi", cfg)
    dtcn_m, dtcn_h = _percentile_flags(df, "dtcn", cfg)
    ddtcn_m, ddtcn_h = _percentile_flags(df, "ddtcn", cfg)
    context = (df["dsi"] >= b.dsi_moderate) & ((df["ranking_market_cap"] < b.small_mcap_for_htb)
                                               | (df["adv20"] < b.low_adv_for_htb))
    score = sum(f.fillna(False).astype(int) for f in
                [dsi_m, dtcn_m, ddtcn_m, context, dsi_h, dtcn_h, ddtcn_h])
    high_conf = (score >= b.stress_score_high) | (dsi_h & dtcn_h)
    mod_conf = (score >= b.stress_score_moderate) | (dsi_m & dtcn_m)

    df["borrow_stress_score"] = score
    df["borrow_tier"] = "A"
    df.loc[mod_conf.fillna(False), "borrow_tier"] = "B"
    df.loc[high_conf.fillna(False), "borrow_tier"] = "C"
    df["short_hard_exclude"] = df["borrow_tier"].eq(b.hard_exclude_tier)
    df["borrow_annual_rate"] = df["borrow_tier"].map(b.annual_rates)
    df["borrow_daily_rate"] = df["borrow_annual_rate"] / b.trading_days_per_year
    df["borrow_daily_bps"] = df["borrow_daily_rate"] * 10_000
    return df


def build_borrow(cfg: Config, cap: CapacityResult, short_interest_daily: pd.DataFrame) -> BorrowResult:
    """Build the borrow panel and the per-AUM short-leg eligibility tables."""
    si_cols = ["instrument_id", "date", "si_available_date", "dsi", "dtcn", "ddtcn"]
    borrow_panel = cap.step2_panel.merge(short_interest_daily[si_cols], on=["instrument_id", "date"], how="left")
    borrow_panel["borrow_data_missing"] = borrow_panel[["dsi", "dtcn", "ddtcn"]].isna().any(axis=1)
    borrow_panel = _assign_tiers(borrow_panel, cfg)

    treat = borrow_panel[["instrument_id", "date", "borrow_tier", "borrow_stress_score",
                          "borrow_annual_rate", "borrow_daily_rate", "borrow_daily_bps",
                          "short_hard_exclude"]]
    short_elig = {}
    for aum, elig in cap.eligibility_by_aum.items():
        se = elig.merge(treat, on=["instrument_id", "date"], how="left")
        se["borrow_tier"] = se["borrow_tier"].fillna("A")
        se["borrow_annual_rate"] = se["borrow_tier"].map(cfg.borrow.annual_rates)
        se["borrow_daily_rate"] = se["borrow_annual_rate"] / cfg.borrow.trading_days_per_year
        se["short_eligibility"] = se["eligibility"]
        se.loc[se["eligibility"].eq("OK") & se["short_hard_exclude"].fillna(False), "short_eligibility"] = "HTB_EXCLUDE"
        short_elig[aum] = se

    tier_dist = (borrow_panel.groupby(["year", "borrow_tier"]).size().unstack(fill_value=0)
                 .reindex(columns=["A", "B", "C"], fill_value=0))
    return BorrowResult(borrow_panel=borrow_panel, short_eligibility_by_aum=short_elig, tier_distribution=tier_dist)
