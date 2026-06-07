"""Unit tests for Step 5: water-fill, dollar-neutrality, cost schedule, and the v2 construction levers."""
import numpy as np
import pandas as pd

from c2o.portfolio import _waterfill_alloc, _neutralize, _apply_vol_target, run_strategy


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
    score = np.linspace(0, 1, n)
    df = pd.DataFrame({
        "instrument_id": np.arange(n), "date": d,
        "score_ens": score, "score_flow": score, "score_combined": score,
        "vol20_ann": 0.3, "adv20": 1e12,                # ADV huge => no cap binding
        "borrow_daily_rate": 0.0, "borrow_tier": "A", "gap_disp": 0.02,
        "sector": np.tile(np.arange(5), n // 5), "beta": 1.0,
        # expected excess edge spanning +/-10 bps so a c=1 (4 bps) bar selects the tails
        "expected_edge": np.linspace(-10e-4, 10e-4, n),
    })
    # top names rise overnight (+1%), bottom names fall (-1%) => positive L/S spread
    df["target_r_on_next"] = np.where(df["score_combined"] >= 0.5, 0.01, -0.01)
    df["r_on"] = 0.0
    elig = df[["instrument_id", "date"]].assign(eligibility="OK")
    short_elig = df[["instrument_id", "date"]].assign(short_eligibility="OK")
    return df, elig, short_elig


_QUANT = dict(selection_mode="quantile", score="score_ens", neutralize_sector=False,
              neutralize_beta=False, vol_target=False)


def test_run_strategy_dollar_neutral_and_cost(cfg):
    tp, elig, se = _synthetic_panel()
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, quantile=0.10, weighting="equal", gate_q=0.0, **_QUANT)
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
    bottom = tp["score_ens"] <= tp["score_ens"].quantile(0.10)
    se.loc[bottom, "short_eligibility"] = "HTB_EXCLUDE"
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, quantile=0.10, htb_exclude=True, **_QUANT)
    # with the bottom decile excluded there are too few shorts -> flat day (no gross deployed)
    assert (d["gross_usd"] == 0).all()


def test_cost_aware_only_trades_names_above_the_bar(cfg):
    tp, elig, se = _synthetic_panel(n=200)
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, selection_mode="cost_aware",
                     weighting="equal", neutralize_sector=False, vol_target=False)
    row = d.iloc[0]
    # bar = c*round_trip = 4 bps excess; longs = (edge > 4 bps), shorts = (edge < -4 bps)
    n_above = int((tp["expected_edge"] > 4e-4).sum())
    n_below = int((tp["expected_edge"] < -4e-4).sum())
    assert row["n_long"] == n_above and row["n_short"] == n_below
    assert 0 < row["n_long"] < len(tp)                      # genuinely selective, not the whole book
    assert abs(row["gross_usd"] - 250_000_000.0) / 250_000_000.0 < 1e-6   # dollar-neutral, fully deployed


def test_cost_aware_skips_day_with_no_edge(cfg):
    tp, elig, se = _synthetic_panel(n=200)
    tp["expected_edge"] = 1e-5                              # everything below the 4 bps bar
    d = run_strategy(cfg, tp, elig, se, aum=250_000_000.0, selection_mode="cost_aware",
                     neutralize_sector=False, vol_target=False)
    assert (d["gross_usd"] == 0).all() and (~d["traded"]).all()
    assert d["net_ret"].iloc[0] == 0.0                     # no trade => no cost, no pnl


def test_neutralize_removes_sector_mean():
    day = pd.DataFrame({"sector": [0, 0, 1, 1, 2, 2], "beta": 1.0})
    sig = pd.Series([1.0, 3.0, 10.0, 14.0, -2.0, -4.0], index=day.index)
    out = _neutralize(day, sig, neut_sector=True, neut_beta=False)
    within = out.groupby(day["sector"]).mean()
    assert within.abs().max() < 1e-9                        # zero net signal within every sector


def test_vol_target_scales_toward_budget(cfg):
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(0)
    daily = pd.DataFrame(index=idx)
    daily.index.name = "date"
    daily["net_ret"] = rng.normal(0, 0.02, len(idx))       # ~32% ann vol, well above the 6% target
    for c in ["gross_ret", "turnover", "commission", "slippage", "borrow", "gross_usd"]:
        daily[c] = 1.0
    out = _apply_vol_target(daily.copy(), cfg, target_ann=0.06)
    realised_after = out["net_ret"].std(ddof=1) * np.sqrt(252)
    assert (out["vol_scale"] <= 1.0 + 1e-9).all()          # high-vol stream is de-levered
    assert realised_after < 0.02 * np.sqrt(252)            # vol pulled down toward the budget
