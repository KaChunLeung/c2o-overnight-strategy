"""Unit tests for Step 4b: cross-sectional standardisation and flow-feature anti-leakage."""
import numpy as np
import pandas as pd

from c2o.alpha import _cross_sectional_zscore, _add_flow_features, _EARN_TRANSFO_COLS


def test_zscore_is_per_date_standardised():
    rng = np.random.default_rng(0)
    frames = []
    for d in pd.bdate_range("2020-01-01", periods=2):
        frames.append(pd.DataFrame({"date": d, "x": rng.normal(5, 3, 500)}))
    df = pd.concat(frames, ignore_index=True)
    df["z"] = _cross_sectional_zscore(df, "x", clip=5.0)
    by_day = df.groupby("date")["z"]
    assert by_day.mean().abs().max() < 1e-6        # mean ~ 0 each day
    assert (by_day.std(ddof=0) - 1.0).abs().max() < 0.05   # std ~ 1 (clip barely bites)


def test_zscore_constant_column_is_zero():
    df = pd.DataFrame({"date": pd.Timestamp("2020-01-01"), "x": np.ones(100)})
    z = _cross_sectional_zscore(df, "x", clip=5.0)
    assert (z == 0).all()      # zero variance -> 0, not NaN/inf


def test_zscore_clips_outliers():
    df = pd.DataFrame({"date": pd.Timestamp("2020-01-01"),
                       "x": np.concatenate([np.zeros(99), [1e6]])})
    z = _cross_sectional_zscore(df, "x", clip=5.0)
    assert z.max() <= 5.0 and z.min() >= -5.0


def test_flow_features_are_asof_prior_day_no_leakage():
    """A revision published on the trade date must NOT enter that day's features (only <= t-1 records)."""
    days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])  # D0, D1, D2 (consecutive sessions)
    sig = pd.DataFrame({"instrument_id": [1, 1], "date": [days[1], days[2]],
                        "decision_asof_date": [days[0], days[1]]})
    et = pd.DataFrame({"instrument_id": [1, 1], "earn_feat_date": [days[1], days[2]]})
    for c in _EARN_TRANSFO_COLS:
        et[c] = np.nan
    et.loc[et["earn_feat_date"] == days[1], "deps"] = 0.5   # known on D1
    et.loc[et["earn_feat_date"] == days[2], "deps"] = 9.9   # on D2 (the second row's trade date)
    cal = pd.DataFrame({"stock_id": [1], "reporting_date": [days[1]]})

    out = _add_flow_features(sig.copy(), et, cal)
    # row 0 trades on D1, decides as-of D0 -> no record <= D0 -> NaN
    assert np.isnan(out["deps_lag"].iloc[0])
    # row 1 trades on D2, decides as-of D1 -> sees the D1 revision (0.5), never the same-day D2 value (9.9)
    assert out["deps_lag"].iloc[1] == 0.5
    assert (out["deps_lag"] != 9.9).all()
