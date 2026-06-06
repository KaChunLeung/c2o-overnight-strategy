"""Step 5 reporting — assemble tables/figures/tear-sheet for one run.

Public API:
    write_all(cfg, run_dir, panel, capacity, borrow, alpha, trade_panel, headline_runs, sp500_tr) -> dict

Computes the gross->net decomposition, concentration sweep, annual stability, stress windows and
robustness grid, writes them as CSV tables and PNG figures under the run directory, and renders the
QuantStats HTML tear-sheet at the benchmark AUM. Private helpers are ``_``. All figures carry their
binding assumptions (AUM, basket, participation cap, cost schedule) in the title.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .io import write_table, write_figure
from .metrics import annualised_sharpe, max_drawdown
from .portfolio import run_strategy, perf_summary


def _decomposition(daily: pd.DataFrame, aum: float) -> pd.DataFrame:
    """Sharpe lost at each cost stage: gross -> commission -> slippage -> borrow (= net)."""
    stages = {
        "gross": daily["gross_ret"],
        "after commission": daily["gross_ret"] - daily["commission"] / aum,
        "after +slippage": daily["gross_ret"] - (daily["commission"] + daily["slippage"]) / aum,
        "after +borrow (=net)": daily["net_ret"],
    }
    out = pd.Series({k: annualised_sharpe(v) for k, v in stages.items()}).to_frame("sharpe")
    out["sharpe_lost"] = out["sharpe"].diff().fillna(0.0).abs()
    return out


def _concentration_sweep(cfg: Config, trade_panel, elig, short_elig, aum: float) -> pd.DataFrame:
    rows = []
    for q in cfg.portfolio.concentration_sweep:
        d = run_strategy(cfg, trade_panel, elig, short_elig, aum, quantile=q, weighting="equal")
        p = perf_summary(cfg, d, aum)
        rows.append({"quantile": f"{q * 100:.0f}%", "gross_sharpe": p["gross_sharpe"],
                     "net_sharpe": p["net_sharpe"], "net_ann_pct": p["net_ann_ret"] * 100,
                     "max_dd_pct": p["max_drawdown"] * 100, "avg_gross_util": p["avg_gross_util"],
                     "names_per_side": int(d["n_long"].mean())})
    return pd.DataFrame(rows)


def _annual_breakdown(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.year)["net_ret"]
    out = pd.DataFrame({"net_return_pct": g.apply(lambda x: ((1 + x).prod() - 1) * 100),
                        "net_sharpe": g.apply(annualised_sharpe), "days": g.size()})
    return out


def _stress_windows(cfg: Config, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, (a, b) in cfg.portfolio.stress_windows.items():
        w = daily.loc[a:b, "net_ret"]
        if len(w) == 0:
            continue
        rows.append({"window": name, "days": len(w), "net_return_pct": ((1 + w).prod() - 1) * 100,
                     "ann_sharpe": annualised_sharpe(w), "max_dd_pct": max_drawdown(w) * 100})
    return pd.DataFrame(rows)


def _robustness_grid(cfg: Config, trade_panel, elig, short_elig, aum: float) -> pd.DataFrame:
    q = cfg.portfolio.headline_quantile
    variants = [("ensemble | equal | no-gate (HEADLINE)", dict(score="score_ens", weighting="equal", gate_q=0.0)),
                ("ridge | equal | no-gate", dict(score="score_ridge", weighting="equal", gate_q=0.0)),
                ("HGB | equal | no-gate", dict(score="score_hgb", weighting="equal", gate_q=0.0)),
                ("ensemble | inverse-vol | no-gate", dict(score="score_ens", weighting="invvol", gate_q=0.0)),
                ("ensemble | equal | dispersion-gate", dict(score="score_ens", weighting="equal", gate_q=0.5))]
    rows = []
    for label, kw in variants:
        d = run_strategy(cfg, trade_panel, elig, short_elig, aum, quantile=q, **kw)
        p = perf_summary(cfg, d, aum)
        ann = d.groupby(d.index.year)["net_ret"].apply(lambda x: (1 + x).prod() - 1)
        rows.append({"variant": label, "net_sharpe": p["net_sharpe"], "net_ann_pct": p["net_ann_ret"] * 100,
                     "max_dd_pct": p["max_drawdown"] * 100, "positive_years": f"{int((ann > 0).sum())}/{len(ann)}",
                     "pct_days_traded": p["frac_days_traded"] * 100})
    return pd.DataFrame(rows)


def write_tearsheet(cfg: Config, strat: pd.Series, bench: pd.Series, path: Path) -> bool:
    """Render the QuantStats HTML tear-sheet (strategy vs SP500_TR). Returns success."""
    import quantstats as qs
    bench = bench.reindex(strat.index).fillna(0.0)
    try:
        qs.reports.html(strat, benchmark=bench, output=str(path),
                        title="C2O Overnight Long/Short @ 250M vs S&P 500 TR (OOS)")
        return True
    except Exception as exc:  # noqa: BLE001 - record and continue; tear-sheet is one of several outputs
        print(f"  [warn] QuantStats HTML failed ({type(exc).__name__}: {exc}); metrics still written")
        return False


def _fig_gross_vs_net(daily: pd.DataFrame, aum: float):
    cum_g, cum_n = (1 + daily["gross_ret"]).cumprod(), (1 + daily["net_ret"]).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(cum_g.index, cum_g.values, label="Gross (pre-cost)", color="#2f6f8f", lw=1.8)
    ax.plot(cum_n.index, cum_n.values, label="Net (4 bps round trip + borrow)", color="#9b3d3d", lw=1.8)
    ax.axhline(1.0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Decile @ {aum/1e6:.0f}M: gross alpha is real, net turns negative\n"
                 "(dollar-neutral, top/bottom 10%, equal-weight, 5% ADV cap, Section 6.3 costs)")
    ax.set_ylabel("Growth of $1"); ax.legend(); ax.grid(alpha=0.25)
    return fig


def _fig_rolling_ic(rolling_ic: pd.DataFrame, window: int):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rolling_ic["date"], rolling_ic["rolling_ic"], color="#2f6f8f", lw=1.7)
    ax.axhline(0, color="black", lw=1, alpha=0.5)
    ax.set_title(f"Ensemble alpha: {window}-day rolling mean IC (OOS, signal universe = base-OK @ widest AUM)")
    ax.set_ylabel("Rolling mean IC"); ax.grid(alpha=0.25)
    return fig


def write_all(cfg: Config, run_dir: Path, panel, capacity, borrow, alpha, trade_panel,
              headline_runs: dict[float, pd.DataFrame], sp500_tr: pd.Series) -> dict:
    """Write every table, figure and the tear-sheet; return a small summary for the manifest."""
    bench_aum = cfg.portfolio.benchmark_aum
    elig = capacity.eligibility_by_aum[bench_aum]
    short_elig = borrow.short_eligibility_by_aum[bench_aum]
    daily_bench = headline_runs[bench_aum]

    # Step 1-4 diagnostics
    write_table(run_dir, panel.reconciliation, "step1_reconciliation", index=False)
    write_table(run_dir, panel.universe_counts, "step1_universe_counts", index=False)
    write_table(run_dir, capacity.binding_by_aum[bench_aum], "step2_binding_250M")
    write_table(run_dir, borrow.tier_distribution, "step3_borrow_tier_distribution")
    write_table(run_dir, alpha.ic_compare, "step4b_ic_compare", index=False)
    write_table(run_dir, alpha.ic_by_year, "step4b_ic_by_year")

    # Step 5 headline + analytics
    headline = pd.DataFrame([perf_summary(cfg, headline_runs[a], a) for a in cfg.capacity.aum_levels])
    write_table(run_dir, headline, "step5_headline_3aum", index=False)
    decile = run_strategy(cfg, trade_panel, elig, short_elig, bench_aum, quantile=0.10, weighting="equal")
    write_table(run_dir, _decomposition(daily_bench, bench_aum), "step5_gross_to_net_decomposition")
    write_table(run_dir, _concentration_sweep(cfg, trade_panel, elig, short_elig, bench_aum), "step5_concentration_sweep", index=False)
    write_table(run_dir, _annual_breakdown(daily_bench), "step5_annual_breakdown")
    write_table(run_dir, _stress_windows(cfg, daily_bench), "step5_stress_windows", index=False)
    write_table(run_dir, _robustness_grid(cfg, trade_panel, elig, short_elig, bench_aum), "step5_robustness_grid", index=False)

    # figures
    write_figure(run_dir, _fig_gross_vs_net(decile, bench_aum), "decile_gross_vs_net_250M")
    write_figure(run_dir, _fig_rolling_ic(alpha.rolling_ic, cfg.alpha.rolling_ic_window), "ensemble_rolling_ic")
    plt.close("all")

    # tear-sheet
    strat = daily_bench["net_ret"].copy()
    strat.index = pd.to_datetime(strat.index)
    strat.name = "C2O_Overnight_250M"
    ts_ok = write_tearsheet(cfg, strat, sp500_tr, run_dir / "reports" / "C2O_tearsheet_250M.html")

    headline_250 = headline.loc[headline["AUM"] == f"{bench_aum/1e6:.0f}M"].iloc[0]
    return {"net_sharpe_250M": float(headline_250["net_sharpe"]),
            "gross_sharpe_250M": float(headline_250["gross_sharpe"]),
            "tearsheet_written": ts_ok,
            "headline": headline.to_dict(orient="records")}
