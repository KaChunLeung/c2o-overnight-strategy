"""Shared fixtures and markers. `pytest` runs fast unit tests; `pytest -m slow` adds the integration run."""
import pytest

from c2o.config import load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end pipeline run (minutes; needs data/inputs)")


@pytest.fixture(scope="session")
def cfg():
    return load_config()
