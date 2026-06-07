"""Step 2 — capacity-aware eligibility per AUM level.

Public API:
    CapacityResult
    build_capacity(cfg, panel_result) -> CapacityResult

Liquidity/risk features (ADV20, annualised vol, Roll spread) are computed on the full price history and
shifted one trading day, so a 15:50 decision on day t uses only data through t-1. The per-(stock, date)
eligibility label records the single binding reason. Private helpers are ``_``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .panel import PanelResult

_REASON_COLUMNS = ["OK", "ADV_FAIL", "MCAP_FAIL", "PRICE_FAIL", "VOL_FAIL", "EARN_WINDOW"]


@dataclass
class CapacityResult:
    step2_panel: pd.DataFrame
    eligibility_by_aum: dict[float, pd.DataFrame]
    binding_by_aum: dict[float, pd.DataFrame]


def _roll_spread_bps(returns: pd.Series, window: int) -> pd.Series:
    """Roll (1984) effective-spread proxy in bps from lagged daily-return autocovariance."""
    past, past_lag = returns.shift(1), returns.shift(2)
    cov = past.rolling(window, min_periods=window).cov(past_lag)
    return 2.0 * np.sqrt((-cov).clip(lower=0)) * 10_000


def _liquidity_features(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing, point-in-time ADV20 / annualised vol / Roll spread / previous close."""
    src = prices.sort_values(["instrument_id", "date"]).copy()
    grp = src.groupby("instrument_id")
    src["prev_close"] = grp["close"].shift(1)
    src["adv20"] = grp["dollar_volume"].transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    src["vol20_ann"] = grp["r_cc"].transform(lambda s: s.shift(1).rolling(window, min_periods=window).std() * np.sqrt(252))
    src["roll_spread_bps"] = grp["r_cc"].transform(lambda s: _roll_spread_bps(s, window))
    return src[["instrument_id", "date", "prev_close", "adv20", "vol20_ann", "roll_spread_bps"]]


def _assign_eligibility(panel: pd.DataFrame, aum: float, cfg: Config) -> pd.DataFrame:
    """Per-(stock, date) eligibility label at one AUM. Priority: MCAP -> PRICE -> ADV -> VOL -> EARN."""
    cap = cfg.capacity
    target = aum / cap.planning_basket_names
    required_adv = target / cap.participation_cap
    max_at_cap = cap.participation_cap * panel["adv20"]
    cap_binds = max_at_cap < target

    reason = pd.Series("OK", index=panel.index, dtype="object")
    reason[panel["ranking_market_cap"].isna() | (panel["ranking_market_cap"] < cap.mcap_floor)] = "MCAP_FAIL"
    reason[(reason == "OK") & (panel["prev_close"].isna() | (panel["prev_close"] < cap.price_floor))] = "PRICE_FAIL"
    reason[(reason == "OK") & (panel["adv20"].isna() | cap_binds)] = "ADV_FAIL"
    vol_bad = panel["vol20_ann"].isna() | (panel["vol20_ann"] < cap.vol_min_ann) | (panel["vol20_ann"] > cap.vol_max_ann)
    reason[(reason == "OK") & vol_bad] = "VOL_FAIL"
    reason[(reason == "OK") & panel["is_earn_window"]] = "EARN_WINDOW"

    out = panel[["date", "year", "instrument_id", "ticker", "adv20", "prev_close",
                 "vol20_ann", "ranking_market_cap", "is_earn_window"]].copy()
    out["AUM"] = aum
    out["target_position_per_stock"] = target
    out["max_position_at_cap"] = max_at_cap
    out["required_adv20"] = required_adv
    out["cap_would_bind_at_equal_weight"] = cap_binds
    out["eligibility"] = reason
    return out


def build_capacity(cfg: Config, pr: PanelResult) -> CapacityResult:
    """Build the Step 2 working panel and the eligibility/binding tables for every AUM level."""
    features = _liquidity_features(pr.prices, cfg.capacity.rolling_window)
    ranking = (pr.yearly_universe[["year", "instrument_id", "market_cap"]]
               .rename(columns={"market_cap": "ranking_market_cap"}))
    step2 = (pr.panel.merge(ranking, on=["year", "instrument_id"], how="left")
                     .merge(features, on=["instrument_id", "date"], how="left")
                     .merge(pr.earn_window[["instrument_id", "date", "is_earn_window",
                                            "earnings_timing", "earnings_reporting_date"]],
                            on=["instrument_id", "date"], how="left"))
    step2["is_earn_window"] = step2["is_earn_window"].eq(True)   # NaN/absent -> False (no fillna downcast warning)
    step2 = step2.sort_values(["instrument_id", "date"]).reset_index(drop=True)

    eligibility, binding = {}, {}
    for aum in cfg.capacity.aum_levels:
        elig = _assign_eligibility(step2, aum, cfg)
        eligibility[aum] = elig
        binding[aum] = (elig.groupby(["year", "eligibility"]).size().unstack(fill_value=0)
                        .reindex(columns=_REASON_COLUMNS, fill_value=0))
    return CapacityResult(step2_panel=step2, eligibility_by_aum=eligibility, binding_by_aum=binding)
