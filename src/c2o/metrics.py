"""Shared performance/skill metrics (neutral helpers used by alpha and portfolio).

Public API: annualised_sharpe, max_drawdown, daily_ic, summarize_ic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PERIODS_PER_YEAR = 252


def annualised_sharpe(returns: pd.Series, periods: int = _PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe of a daily return series (0 mean/vol guarded)."""
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    return float(np.sqrt(periods) * r.mean() / sd) if len(r) > 1 and sd > 0 else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of the compounded daily return series (negative number)."""
    cum = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def daily_ic(frame: pd.DataFrame, score_col: str, target_col: str, min_names: int) -> pd.Series:
    """Daily cross-sectional Spearman IC between score and realised target."""
    e = frame.loc[frame[score_col].notna() & frame[target_col].notna(), ["date", score_col, target_col]]

    def _one(day: pd.DataFrame) -> float:
        if len(day) < min_names:
            return np.nan
        return day[score_col].rank().corr(day[target_col].rank())

    return e.groupby("date").apply(_one, include_groups=False).dropna().rename("ic")


def summarize_ic(ic: pd.Series) -> dict[str, float]:
    """Mean IC, its t-stat, positive-day hit rate and day count."""
    v = pd.Series(ic).dropna()
    n = len(v)
    mean, sd = (v.mean(), v.std(ddof=1)) if n > 1 else (np.nan, np.nan)
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    return {"mean_ic": float(mean), "t_stat": float(t),
            "hit_rate": float((v > 0).mean()) if n else float("nan"), "n_days": int(n)}
