"""Unit tests for Step 3 borrow-tier assignment (tiers, rates, Tier-C exclusion)."""
import numpy as np
import pandas as pd

from c2o.borrow import _assign_tiers


def _panel(cfg):
    # 27 benign tied names (low SI) + one Tier-C and one Tier-B candidate on one date
    d = pd.Timestamp("2020-06-01")
    low = [{"dsi": 0.01, "dtcn": 1.0, "ddtcn": 0.0} for _ in range(27)]
    rows = low + [
        {"dsi": 0.25, "dtcn": 15.0, "ddtcn": 6.0},   # both high -> Tier C
        {"dsi": 0.12, "dtcn": 8.0, "ddtcn": 0.0},    # both moderate -> Tier B
    ]
    df = pd.DataFrame(rows)
    df["date"] = d
    df["ranking_market_cap"] = 5e10      # large => context flag off
    df["adv20"] = 1e9
    return df


def test_tiers_and_rates(cfg):
    out = _assign_tiers(_panel(cfg), cfg)
    tiers = out["borrow_tier"].tolist()
    assert tiers[-2] == "C" and tiers[-1] == "B"
    assert (out["borrow_tier"].iloc[:27] == "A").all()
    rate = out.groupby("borrow_tier")["borrow_daily_rate"].first()
    assert abs(rate["A"] - 0.0040 / 252) < 1e-12
    assert abs(rate["B"] - 0.0200 / 252) < 1e-12
    assert abs(rate["C"] - 0.0800 / 252) < 1e-12


def test_tier_c_marked_for_exclusion(cfg):
    out = _assign_tiers(_panel(cfg), cfg)
    assert bool(out.loc[out["borrow_tier"] == "C", "short_hard_exclude"].all())
    assert not bool(out.loc[out["borrow_tier"] != "C", "short_hard_exclude"].any())
