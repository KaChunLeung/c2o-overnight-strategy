"""Unit tests for Step 5: participation-cap water-fill, dollar-neutrality, and the cost schedule."""
import numpy as np
import pandas as pd

from c2o.portfolio import _waterfill_alloc, run_strategy


def test_waterfill_no_cap_is_proportional():
    alloc = _waterfill_alloc(np.array([1, 1, 1, 1.0]), np.array([1e9] * 4), 100.0)
    assert np.allclose(alloc, 25.0) and abs(alloc.sum() - 100.0) < 1e-9


def test_waterfill_redistributes_excess():
    # name 0 capped at 10; the other 90 flows to name 1
    alloc = _waterfill_alloc(np.array([1.0, 1.0]), np.array([10.0, 1e9]), 100.0)
    assert abs(alloc[0] - 10.0) < 1e-9 and abs(alloc[1] - 90.0) < 1e-9


def test_waterfill_reduces_gross_when_all_capped():
    alloc = _waterfill_alloc(np.array([1.0, 1.0]), np.array([10.0, 20.0]), 100.0)
    assert alloc.sum() < 100.0 and np.allclose(alloc, [10.0, 20.0])


def _synthetic_panel(n=200, date="2020-06-01"):
    d = pd.Timestamp(date)
    df = pd.DataFrame({
        "instrument_id": np.arange(n), "date": d,
        "score_ens": np.linspace(0, 1, n), "vol20_ann": 0.3, "adv20": 1e12,  # ADV huge => no cap binding
        "borrow_daily_rate": 0.0, "borrow_tier": "A", "gap_disp": 0.02,
    })
    # top names rise overnight (+1%), bottom names fall (-1%) => positive L/S spread
    df["target_r_on_next"] = np.where(df["score_ens"] >= 0.5, 0.01, -0.01)
    df["r_on"] = 0.0
    elig = df[["instrument_id", "date"]].assign(eligibility="OK")
    short_elig = df[["instrument_id", "date"]].assign(short_eligibility="OK")
    return df, elig, short_elig


def test_run_strategy_dollar_neutral_and_cost(cfg):
    tp, elig, se = _synthetic_panel()
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, quantile=0.10, weighting="equal", gate_q=0.0)
    row = d.iloc[0]
    aum = 250_000_000.0
    # full deployment with symmetric caps => gross == AUM (=> long $ == short $ == AUM/2)
    assert abs(row["gross_usd"] - aum) / aum < 1e-6
    # Section 6.3: commission 1 bps + slippage 3 bps of gross per night
    assert abs(row["commission"] - 1e-4 * aum) / aum < 1e-9
    assert abs(row["slippage"] - 3e-4 * aum) / aum < 1e-9
    # gross overnight L/S return is +1% (top +1, bottom -1, dollar-neutral) and net subtracts 4 bps
    assert abs(row["gross_ret"] - 0.01) < 1e-6
    assert abs(row["net_ret"] - (0.01 - 4e-4)) < 1e-6


def test_htb_exclusion_drops_short_names(cfg):
    tp, elig, se = _synthetic_panel()
    # mark the entire bottom decile as hard-to-borrow excluded
    bottom = tp["score_ens"] <= tp["score_ens"].quantile(0.10)
    se.loc[bottom, "short_eligibility"] = "HTB_EXCLUDE"
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, quantile=0.10, htb_exclude=True)
    # with the bottom decile excluded there are too few shorts -> day skipped (empty result)
    assert d.empty
