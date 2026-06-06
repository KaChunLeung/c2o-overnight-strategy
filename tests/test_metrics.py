"""Unit tests for the shared metrics helpers."""
import numpy as np
import pandas as pd

from c2o.metrics import annualised_sharpe, max_drawdown, daily_ic, summarize_ic


def test_sharpe_matches_manual():
    r = pd.Series([0.001, -0.0005, 0.0008, 0.0002, -0.0003])
    expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
    assert abs(annualised_sharpe(r) - expected) < 1e-12


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(annualised_sharpe(pd.Series([0.0, 0.0, 0.0])))


def test_max_drawdown_sign_and_value():
    # +10% then -50% -> trough at 0.55 of peak 1.1 => -50%
    dd = max_drawdown(pd.Series([0.10, -0.50, 0.0]))
    assert dd <= 0 and abs(dd - (-0.5)) < 1e-9


def test_daily_ic_perfect_rank():
    # score perfectly ranks the target each day -> IC == 1
    rows = []
    for d in pd.bdate_range("2020-01-01", periods=3):
        for i in range(100):
            rows.append({"date": d, "score": i, "y": i + 0.0})
    df = pd.DataFrame(rows)
    ic = daily_ic(df, "score", "y", min_names=80)
    assert np.allclose(ic.values, 1.0)
    s = summarize_ic(ic)
    assert s["n_days"] == 3 and s["hit_rate"] == 1.0
