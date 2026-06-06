"""Unit tests for Step 2 eligibility labelling (priority order and the ADV participation cap)."""
import pandas as pd

from c2o.capacity import _assign_eligibility


def _panel(**over):
    base = dict(adv20=1e12, prev_close=50.0, vol20_ann=0.30, ranking_market_cap=5e9, is_earn_window=False)
    base.update(over)
    return pd.DataFrame([{"date": pd.Timestamp("2020-06-01"), "year": 2020, "instrument_id": 1,
                          "ticker": "X", **base}])


def test_each_reason(cfg):
    aum = 250_000_000.0
    assert _assign_eligibility(_panel(), aum, cfg).iloc[0]["eligibility"] == "OK"
    assert _assign_eligibility(_panel(ranking_market_cap=1e9), aum, cfg).iloc[0]["eligibility"] == "MCAP_FAIL"
    assert _assign_eligibility(_panel(prev_close=2.0), aum, cfg).iloc[0]["eligibility"] == "PRICE_FAIL"
    assert _assign_eligibility(_panel(adv20=1e6), aum, cfg).iloc[0]["eligibility"] == "ADV_FAIL"
    assert _assign_eligibility(_panel(vol20_ann=2.0), aum, cfg).iloc[0]["eligibility"] == "VOL_FAIL"
    assert _assign_eligibility(_panel(is_earn_window=True), aum, cfg).iloc[0]["eligibility"] == "EARN_WINDOW"


def test_adv_cap_binds_harder_at_higher_aum(cfg):
    # ADV20 = 50M: cap-binds when 5%*ADV (=2.5M) < target (=AUM/200). Holds at 1B (5M) but not 50M (0.25M).
    out_small = _assign_eligibility(_panel(adv20=50e6), 50_000_000.0, cfg).iloc[0]
    out_big = _assign_eligibility(_panel(adv20=50e6), 1_000_000_000.0, cfg).iloc[0]
    assert out_small["eligibility"] == "OK"
    assert out_big["eligibility"] == "ADV_FAIL"


def test_priority_mcap_before_price(cfg):
    out = _assign_eligibility(_panel(ranking_market_cap=1e9, prev_close=1.0), 250_000_000.0, cfg).iloc[0]
    assert out["eligibility"] == "MCAP_FAIL"     # mcap checked before price
