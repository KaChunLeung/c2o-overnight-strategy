"""Step 1 — build the survivorship-aware daily panel and the return objects.

Public API:
    TradingCalendar         : trading-day lookups (next / previous / position)
    PanelResult             : typed handoff for downstream steps
    build_panel(cfg, prices, earnings) -> PanelResult

Returns are computed on the adjusted-close scale so corporate actions are consistent across open/close,
and (1+r_on)(1+r_id) == 1+r_cc by construction. The yearly universe is the top-N by market cap at the
prior year-end, frozen within the year (point-in-time, survivorship-aware). Private helpers are ``_``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


@dataclass(frozen=True)
class TradingCalendar:
    """Sorted trading days with next/previous lookups (strictly after / before a date)."""
    days: pd.DatetimeIndex

    def next(self, date) -> pd.Timestamp:
        pos = self.days.searchsorted(pd.Timestamp(date), side="right")
        return pd.NaT if pos >= len(self.days) else self.days[pos]

    def previous(self, date) -> pd.Timestamp:
        pos = self.days.searchsorted(pd.Timestamp(date), side="left") - 1
        return pd.NaT if pos < 0 else self.days[pos]

    def position(self, date):
        ts = pd.Timestamp(date)
        pos = self.days.searchsorted(ts, side="left")
        return pos if pos < len(self.days) and self.days[pos] == ts else None


@dataclass
class PanelResult:
    prices: pd.DataFrame              # cleaned full-history prices with returns + quality flags
    panel: pd.DataFrame              # frozen-universe daily panel within [start, cutoff]
    calendar: TradingCalendar
    yearly_universe: pd.DataFrame
    earn_window: pd.DataFrame
    short_interest_daily: pd.DataFrame
    reconciliation: pd.DataFrame
    universe_counts: pd.DataFrame


def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """De-duplicate (instrument, date), keeping the active/current security record."""
    df = prices.copy()
    df["status_priority"] = df["status"].astype(str).eq("1").astype(int)
    df = (df.sort_values(["instrument_id", "date", "status_priority", "ticker"],
                         ascending=[True, True, False, True])
            .drop_duplicates(["instrument_id", "date"], keep="first")
            .drop(columns="status_priority")
            .sort_values(["instrument_id", "date"]).reset_index(drop=True))
    return df


def _add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add adjusted OHLC, overnight/intraday/close-to-close returns, dollar volume, quality flags."""
    df = prices
    df["adjustment_factor"] = df["adjusted_close"] / df["close"]
    ohlc = df[["open", "high", "low", "close", "adjusted_close"]]
    df["nonpositive_price"] = (ohlc <= 0).any(axis=1) | ohlc.isna().any(axis=1)
    df["severe_price_scale_error"] = ((df["close"] > 10_000) & (df["adjusted_close"] < 1_000)
                                      & (df["adjustment_factor"] < 0.05))
    df["mcap_ranking_ok"] = (df["market_cap"].notna() & (df["market_cap"] > 0)
                             & ~df["severe_price_scale_error"])
    valid = (~df["nonpositive_price"]) & np.isfinite(df["adjustment_factor"])
    df.loc[~valid, "adjustment_factor"] = np.nan
    df["adj_open"] = df["open"] * df["adjustment_factor"]
    df["adj_high"] = df["high"] * df["adjustment_factor"]
    df["adj_low"] = df["low"] * df["adjustment_factor"]
    df["adj_close"] = df["adjusted_close"]
    df["prev_adj_close"] = df.groupby("instrument_id")["adj_close"].shift(1)
    df["r_on"] = df["adj_open"] / df["prev_adj_close"] - 1.0
    df["r_id"] = df["adj_close"] / df["adj_open"] - 1.0
    df["r_cc"] = df["adj_close"] / df["prev_adj_close"] - 1.0
    df["dollar_volume"] = df["close"] * df["volume"]
    df["return_reconciliation_residual"] = (1.0 + df["r_on"]) * (1.0 + df["r_id"]) - 1.0 - df["r_cc"]
    return df


def _reconciliation_summary(df: pd.DataFrame, tol: float) -> pd.DataFrame:
    valid = df[["r_on", "r_id", "r_cc", "return_reconciliation_residual"]].dropna()
    failed = valid["return_reconciliation_residual"].abs() > tol
    return pd.DataFrame({"valid_stock_days": [len(valid)], "failed_stock_days": [int(failed.sum())],
                         "failed_fraction": [float(failed.mean())],
                         "max_abs_residual": [float(valid["return_reconciliation_residual"].abs().max())],
                         "tolerance": [tol]})


def _build_earnings_window(earnings: pd.DataFrame, cal: TradingCalendar, after: int) -> pd.DataFrame:
    """Per-(instrument, decision-date) flag for sessions whose overnight crosses an earnings event.

    AMC on D -> exclude decision date D; BMO on D -> exclude D-1; missing -> exclude both. Then extend
    ``after`` trading days for post-earnings drift.
    """
    rows = []
    for r in earnings[["stock_id", "reporting_date", "before_after_market"]].dropna(subset=["reporting_date"]).itertuples(index=False):
        timing = "missing" if pd.isna(r.before_after_market) else str(r.before_after_market).lower()
        rep = pd.Timestamp(r.reporting_date)
        if timing == "before":
            bases = [cal.previous(rep)]
        elif timing == "after":
            bases = [rep]
        else:
            bases = [cal.previous(rep), rep]
        for base in bases:
            pos = cal.position(base)
            if pos is None:
                continue
            for off in range(after + 1):
                if pos + off < len(cal.days):
                    rows.append((r.stock_id, cal.days[pos + off], timing, rep))
    out = pd.DataFrame(rows, columns=["instrument_id", "date", "earnings_timing", "earnings_reporting_date"])
    out = out.drop_duplicates(["instrument_id", "date"])
    out["is_earn_window"] = True
    return out


def _build_short_interest_daily(prices: pd.DataFrame, short_interest: pd.DataFrame,
                                cal: TradingCalendar, cfg: Config) -> pd.DataFrame:
    """Daily short-interest, as-of the previous trading day (no same-day leakage)."""
    start, cutoff = pd.Timestamp(cfg.window.start_date), pd.Timestamp(cfg.window.cutoff)
    dates = (prices.loc[prices["date"].between(start, cutoff), ["instrument_id", "date"]]
             .drop_duplicates().sort_values(["instrument_id", "date"]).copy())
    dates["decision_asof_date"] = dates["date"].map(cal.previous)
    dates = dates.dropna(subset=["decision_asof_date"])
    si = short_interest.sort_values(["si_available_date", "instrument_id"]).reset_index(drop=True)
    return pd.merge_asof(
        dates.sort_values(["decision_asof_date", "instrument_id"]).reset_index(drop=True),
        si, left_on="decision_asof_date", right_on="si_available_date",
        by="instrument_id", direction="backward")


def _build_yearly_universe(prices: pd.DataFrame, cal: TradingCalendar, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-N by market cap at prior year-end (>= min history), frozen per calendar year."""
    first_seen = prices.groupby("instrument_id")["date"].min()
    last_seen = prices.groupby("instrument_id")["date"].max()
    records, audit = [], []
    for year in range(pd.Timestamp(cfg.window.start_date).year, pd.Timestamp(cfg.window.cutoff).year + 1):
        ydays = cal.days[(cal.days >= pd.Timestamp(f"{year}-01-01")) & (cal.days <= pd.Timestamp(f"{year}-12-31"))]
        if len(ydays) == 0:
            continue
        year_start, year_end = ydays[0], ydays[-1]
        ranking_date = cal.previous(year_start)
        snap_all = prices.loc[prices["date"] == ranking_date].copy()
        enough = first_seen <= ranking_date - pd.Timedelta(days=cfg.window.min_history_days)
        snap_all["enough_history"] = snap_all["instrument_id"].map(enough).fillna(False)
        snap = snap_all.loc[snap_all["enough_history"] & snap_all["mcap_ranking_ok"]].copy()
        snap = (snap.sort_values("market_cap", ascending=False)
                    .drop_duplicates("instrument_id", keep="first").head(cfg.window.universe_size))
        snap["year"] = year
        snap["universe_start"], snap["universe_end"] = year_start, year_end
        snap["last_seen"] = snap["instrument_id"].map(last_seen)
        snap["active_end"] = snap[["universe_end", "last_seen"]].min(axis=1)
        snap["mid_year_exit"] = snap["last_seen"] < year_end
        records.append(snap[["year", "universe_start", "universe_end", "active_end",
                             "instrument_id", "ticker", "market_cap", "mid_year_exit"]])
        audit.append({"year": year, "ranking_date": ranking_date, "selected_names": snap["instrument_id"].nunique(),
                      "mid_year_exits": int(snap["mid_year_exit"].sum()),
                      "median_selected_mcap": snap["market_cap"].median()})
    uni = pd.concat(records, ignore_index=True)
    return uni, pd.DataFrame(audit)


def build_panel(cfg: Config, prices: pd.DataFrame, earnings: pd.DataFrame,
                short_interest: pd.DataFrame) -> PanelResult:
    """Run Step 1 end to end and return the typed panel handoff."""
    prices = _add_returns(_clean_prices(prices))
    cal = TradingCalendar(pd.DatetimeIndex(sorted(prices["date"].unique())))
    recon = _reconciliation_summary(prices, cfg.window.residual_tolerance)

    yearly_universe, universe_counts = _build_yearly_universe(prices, cal, cfg)
    start, cutoff = pd.Timestamp(cfg.window.start_date), pd.Timestamp(cfg.window.cutoff)
    panel = prices.loc[prices["date"].between(start, cutoff)].copy()
    panel["year"] = panel["date"].dt.year
    panel = panel.merge(yearly_universe[["year", "instrument_id", "universe_start", "active_end"]],
                        on=["year", "instrument_id"], how="inner")
    panel = panel.loc[panel["date"].between(panel["universe_start"], panel["active_end"])].copy()

    earn_window = _build_earnings_window(earnings, cal, cfg.capacity.earn_window_after)
    si_daily = _build_short_interest_daily(prices, short_interest, cal, cfg)
    return PanelResult(prices=prices, panel=panel, calendar=cal, yearly_universe=yearly_universe,
                       earn_window=earn_window, short_interest_daily=si_daily,
                       reconciliation=recon, universe_counts=universe_counts)
