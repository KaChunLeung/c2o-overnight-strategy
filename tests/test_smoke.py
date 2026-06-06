"""Integration smoke test: run the whole pipeline on a truncated universe and assert sane headline output.

Marked ``slow`` (a few minutes; needs data/inputs). Run with:  pytest -m slow
"""
import numpy as np
import pytest

from c2o.config import load_config
from c2o.main import main


@pytest.mark.slow
def test_pipeline_smoke():
    cfg = load_config("config/default.yaml", "config/fast.yaml")
    summary = main(cfg)
    assert np.isfinite(summary["net_sharpe_250M"])
    assert summary["gross_sharpe_250M"] > 0.5      # the gross overnight alpha is genuinely present
    assert summary["tearsheet_written"] is True
