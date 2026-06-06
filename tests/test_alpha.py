"""Unit tests for Step 4b cross-sectional standardisation (per-date, robust to constants)."""
import numpy as np
import pandas as pd

from c2o.alpha import _cross_sectional_zscore


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
