"""Unit tests for Step 1: calendar, return reconciliation, corporate-action consistency, earnings window."""
import numpy as np
import pandas as pd

from c2o.panel import TradingCalendar, _add_returns, _clean_prices, _build_earnings_window


def _calendar():
    days = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    return TradingCalendar(pd.DatetimeIndex(days))


def test_calendar_next_previous_position():
    cal = _calendar()
    assert cal.next("2020-01-03") == pd.Timestamp("2020-01-06")
    assert cal.previous("2020-01-06") == pd.Timestamp("2020-01-03")
    assert cal.next("2020-01-07") is pd.NaT          # nothing after the last day
    assert cal.previous("2020-01-02") is pd.NaT
    assert cal.position("2020-01-06") == 2
    assert cal.position("2020-01-04") is None         # not a trading day


def test_returns_reconcile_through_a_split():
    # raw close halves on a 2:1 split but adjusted_close is continuous => returns must still reconcile
    raw = pd.DataFrame({
        "ticker": "X", "instrument_id": 1,
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
        "open": [100.0, 101.0, 51.0], "high": [102, 103, 52], "low": [99, 100, 50],
        "close": [101.0, 102.0, 51.5], "adjusted_close": [101.0, 102.0, 103.0],
        "volume": [1e6, 1e6, 2e6], "market_cap": [1e10, 1e10, 1e10], "status": "1",
    })
    df = _add_returns(_clean_prices(raw))
    resid = ((1 + df["r_on"]) * (1 + df["r_id"]) - 1 - df["r_cc"]).abs().max()
    assert resid < 1e-12
    # overnight return is computed on the adjusted scale (no spurious -50% split jump)
    assert df["r_on"].dropna().abs().max() < 0.10


def test_clean_prices_dedup_keeps_active_status():
    raw = pd.DataFrame({
        "ticker": ["OLD", "NEW"], "instrument_id": [1, 1],
        "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "open": [10, 10], "high": [10, 10], "low": [10, 10], "close": [10, 10],
        "adjusted_close": [10, 10], "volume": [1, 1], "market_cap": [1e9, 1e9], "status": ["0", "1"],
    })
    out = _clean_prices(raw)
    assert len(out) == 1 and out.iloc[0]["ticker"] == "NEW"


def test_earnings_window_amc_vs_bmo():
    cal = _calendar()
    earn = pd.DataFrame({"stock_id": [1, 2],
                         "reporting_date": pd.to_datetime(["2020-01-06", "2020-01-06"]),
                         "before_after_market": ["after", "before"]})
    win = _build_earnings_window(earn, cal, after=0).set_index("instrument_id")
    # AMC on 01-06 -> exclude decision date 01-06; BMO on 01-06 -> exclude prior session 01-03
    assert win.loc[1, "date"] == pd.Timestamp("2020-01-06")
    assert win.loc[2, "date"] == pd.Timestamp("2020-01-03")
