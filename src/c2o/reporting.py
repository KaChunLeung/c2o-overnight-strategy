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


_QUANT = dict(selection_mode="quantile", neutralize_sector=False, neutralize_beta=False, vol_target=False)


def _concentration_sweep(cfg: Config, trade_panel, elig, short_elig, aum: float) -> pd.DataFrame:
    """The original quantile concentration story (cost-aware selection turned off), for context."""
    rows = []
    for q in cfg.portfolio.concentration_sweep:
        d = run_strategy(cfg, trade_panel, elig, short_elig, aum, quantile=q, weighting="equal", **_QUANT)
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
    """Net Sharpe across the alpha source x concentration grid (transparency on the two tuned choices)."""
    rows = []
    for score, slabel in [("score_ens", "ensemble"), ("score_hgb", "HGB"), ("score_combined", "combined(+flow)")]:
        for q in [0.02, 0.0125, 0.01]:
            d = run_strategy(cfg, trade_panel, elig, short_elig, aum, score=score, weighting="equal",
                             quantile=q, **_QUANT)
            p = perf_summary(cfg, d, aum)
            ann = d.groupby(d.index.year)["net_ret"].apply(lambda x: (1 + x).prod() - 1)
            tag = ""
            if score == cfg.portfolio.headline_score and abs(q - cfg.portfolio.headline_quantile) < 1e-9:
                tag = "  <- HEADLINE"
            if score == "score_hgb" and abs(q - 0.01) < 1e-9:
                tag = "  <- FRONTIER"
            rows.append({"alpha": slabel, "quantile_pct": q * 100, "net_sharpe": p["net_sharpe"],
                         "net_ann_pct": p["net_ann_ret"] * 100, "max_dd_pct": p["max_drawdown"] * 100,
                         "positive_years": f"{int((ann > 0).sum())}/{len(ann)}", "note": tag})
    return pd.DataFrame(rows)


def _sharpe_ladder(cfg: Config, trade_panel, elig, short_elig, aum: float,
                   headline_daily: pd.DataFrame, frontier_daily: pd.DataFrame) -> pd.DataFrame:
    """The investigation: concentration is the lever; the sophisticated overlays do not beat it.

    Rows 0-1 show the only move that lifts net Sharpe (tighter tails). Rows L1-L4 apply each sophisticated
    lever ON TOP of the concentrated headline and report ``delta_vs_headline`` (<=0 means it did not help).
    """
    qh = cfg.portfolio.headline_quantile
    base = perf_summary(cfg, run_strategy(cfg, trade_panel, elig, short_elig, aum,
                                          score="score_ens", weighting="equal", quantile=0.02, **_QUANT), aum)
    head = perf_summary(cfg, headline_daily, aum)
    front = perf_summary(cfg, frontier_daily, aum)

    def row(step, p, ref=None):
        return {"step": step, "net_sharpe": p["net_sharpe"], "net_ann_pct": p["net_ann_ret"] * 100,
                "max_dd_pct": p["max_drawdown"] * 100, "avg_names_per_side": p["avg_names_per_side"],
                "delta_vs_headline": (p["net_sharpe"] - ref) if ref is not None else np.nan}

    levers = [
        ("L1  + flow blend (combined score)", dict(score="score_combined", weighting="equal", quantile=qh, **_QUANT)),
        ("L2  + cost-aware per-name selection", dict(score="score_ens", weighting="equal",
                                                     selection_mode="cost_aware", neutralize_sector=False,
                                                     neutralize_beta=False, vol_target=False)),
        ("L3  + sector-neutralization", dict(score="score_ens", weighting="equal", quantile=qh,
                                             selection_mode="quantile", neutralize_sector=True,
                                             neutralize_beta=False, vol_target=False)),
        ("L4  + vol-target overlay", dict(score="score_ens", weighting="equal", quantile=qh,
                                          selection_mode="quantile", neutralize_sector=False,
                                          neutralize_beta=False, vol_target=True)),
        ("L5  + inverse-vol weighting", dict(score="score_ens", weighting="invvol", quantile=qh, **_QUANT)),
    ]
    rows = [row(f"0  reversal ensemble, quantile 2% (v1 headline)", base),
            row(f"1  concentrate to {qh*100:g}% tails = HEADLINE", head)]
    for label, kw in levers:
        p = perf_summary(cfg, run_strategy(cfg, trade_panel, elig, short_elig, aum, **kw), aum)
        rows.append(row(label, p, ref=head["net_sharpe"]))
    rows.append(row("F  FRONTIER (pure HGB, 1% tails)", front))
    return pd.DataFrame(rows)


def _sleeve_return_corr(cfg: Config, trade_panel, elig, short_elig, aum: float) -> tuple[pd.DataFrame, dict]:
    """Per-sleeve net-return streams: their correlation matrix and the series (for the equity-curve figure)."""
    runs = {}
    for label, sc in [("reversal", "score_ens"), ("flow", "score_flow"), ("combined", "score_combined")]:
        d = run_strategy(cfg, trade_panel, elig, short_elig, aum, score=sc, weighting="equal",
                         quantile=cfg.portfolio.headline_quantile, **_QUANT)
        runs[label] = d["net_ret"]
    return pd.DataFrame(runs).corr(), runs


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
    ax.set_title(f"Reversal ensemble alpha: {window}-day rolling mean IC (OOS, signal universe = base-OK @ widest AUM)")
    ax.set_ylabel("Rolling mean IC"); ax.grid(alpha=0.25)
    return fig


def _fig_sharpe_ladder(ladder: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#9b3d3d" if s < 0 else "#2f6f8f" for s in ladder["net_sharpe"]]
    ax.bar(range(len(ladder)), ladder["net_sharpe"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1.0, color="#3a7d3a", lw=1.0, ls="--", alpha=0.7, label="Sharpe = 1.0")
    ax.set_xticks(range(len(ladder)))
    ax.set_xticklabels([s.split("  ")[0] if "  " in s else s for s in ladder["step"]], rotation=0)
    ax.set_title("Net-Sharpe ladder @ benchmark AUM: each construction lever's marginal contribution\n"
                 "(dollar-neutral overnight L/S, full Section 6.3 costs, OOS)")
    ax.set_ylabel("Net Sharpe"); ax.legend(); ax.grid(alpha=0.25, axis="y")
    return fig


def _fig_performance(daily: pd.DataFrame, bench: pd.Series, aum: float):
    """Tear-sheet core: cumulative net return vs S&P 500 TR (top) and the underwater drawdown (bottom)."""
    r = daily["net_ret"]
    cum = (1 + r).cumprod()
    b = bench.reindex(r.index).fillna(0.0)
    cumb = (1 + b).cumprod()
    under = cum / cum.cummax() - 1.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), height_ratios=[2.2, 1], sharex=True)
    ax1.plot(cum.index, cum.values, color="#2f6f8f", lw=1.8, label="C2O headline (net)")
    ax1.plot(cumb.index, cumb.values, color="#999999", lw=1.3, label="S&P 500 TR")
    ax1.axhline(1.0, color="black", lw=0.7, alpha=0.5)
    ax1.set_yscale("log"); ax1.set_ylabel("Growth of \\$1 (log)"); ax1.legend(loc="upper left")
    ax1.set_title(f"Headline @ {aum/1e6:.0f}M vs S&P 500 TR --- net of full Section 6.3 costs (OOS 2015--2024).\n"
                  "Near-flat to the index (market-neutral) but with a steady, low-vol, positively-skewed climb.")
    ax2.fill_between(under.index, under.values * 100, 0, color="#9b3d3d", alpha=0.6)
    ax2.set_ylabel("Drawdown %"); ax2.grid(alpha=0.25); ax1.grid(alpha=0.25)
    return fig


def _fig_monthly_heatmap(daily: pd.DataFrame):
    """Year x month net-return heatmap (the classic tear-sheet panel)."""
    r = daily["net_ret"]
    m = ((1 + r).resample("ME").prod() - 1) * 100
    tab = m.to_frame("ret")
    tab["year"] = tab.index.year; tab["month"] = tab.index.month
    piv = tab.pivot_table(index="year", columns="month", values="ret")
    piv = piv.reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="black" if abs(v) < vmax * 0.6 else "white")
    ax.set_title("Monthly net return % (headline @ 250M). Losses are shallow and not clustered; "
                 "2021--2024 are the strongest.")
    fig.colorbar(im, ax=ax, shrink=0.8, label="net return %")
    return fig


def _fig_distribution(daily: pd.DataFrame):
    """Daily net-return histogram, annotating the positive skew (the strategy's distinctive feature)."""
    r = (daily["net_ret"] * 100).dropna()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(r, bins=80, color="#2f6f8f", alpha=0.85)
    ax.axvline(r.mean(), color="#9b3d3d", lw=1.5, label=f"mean {r.mean():.2f}%")
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.set_title(f"Distribution of daily net returns (headline @ 250M). Skew = {r.skew():.2f} (positive --- a "
                 "right tail), unlike the negative skew typical of equity L/S.")
    ax.set_xlabel("daily net return %"); ax.set_ylabel("days"); ax.legend(); ax.grid(alpha=0.25)
    return fig


def _fig_concentration(cfg: Config, sweep: pd.DataFrame):
    """Net (and gross) Sharpe vs tail concentration --- the central 'concentration is the lever' chart."""
    q = [x * 100 for x in cfg.portfolio.concentration_sweep]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(q, sweep["net_sharpe"], "o-", color="#2f6f8f", lw=2, label="net Sharpe")
    ax.plot(q, sweep["gross_sharpe"], "s--", color="#999999", lw=1.3, label="gross Sharpe")
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    qh = cfg.portfolio.headline_quantile * 100
    ax.axvline(qh, color="#3a7d3a", lw=1.2, ls=":", label=f"headline {qh:g}%")
    ax.invert_xaxis()
    ax.set_xlabel("top/bottom quantile traded (%)"); ax.set_ylabel("Sharpe")
    ax.set_title("Net Sharpe is hump-shaped in concentration: a broad 1--1.5% plateau, not a spike.\n"
                 "Gross Sharpe is roughly flat --- the lift is cost-density, not a stronger raw signal.")
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def _fig_sharpe_vs_aum(headline: pd.DataFrame, frontier: pd.DataFrame):
    """Net Sharpe vs AUM for headline + frontier --- the capacity 'sweet spot' (hump) story."""
    x = list(range(len(headline)))
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.plot(x, headline["net_sharpe"], "o-", color="#2f6f8f", lw=2, label="headline (ensemble 1.25%)")
    ax.plot(x, frontier["net_sharpe"], "s--", color="#b5651d", lw=1.6, label="frontier (HGB 1%)")
    ax.axhline(1.0, color="#3a7d3a", lw=1.0, ls="--", alpha=0.7, label="Sharpe = 1.0")
    ax.set_xticks(x); ax.set_xticklabels(headline["AUM"])
    ax.set_xlabel("AUM"); ax.set_ylabel("net Sharpe")
    ax.set_title("Net Sharpe peaks at \\$250M, not at the extremes: the 5% ADV cap performs variance reduction\n"
                 "in the middle of the range; the aggressive 1% frontier collapses at \\$1B as the book thins.")
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def _fig_sleeve_curves(runs: dict[str, pd.Series]):
    """Cumulative net returns of the reversal / flow / combined books at the headline concentration."""
    fig, ax = plt.subplots(figsize=(11, 4.3))
    colors = {"reversal": "#2f6f8f", "flow": "#b5651d", "combined": "#6a4c93"}
    for label, r in runs.items():
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, lw=1.7, color=colors.get(label, None), label=f"{label} book")
    ax.axhline(1.0, color="black", lw=0.7, alpha=0.5)
    ax.set_ylabel("Growth of \\$1 (net)")
    ax.set_title("Per-sleeve net equity @ 250M (top/bottom 1.25%): the flow book is barely profitable at the "
                 "tails\nand drags the combined book below the reversal book --- broad IC is not tail edge.")
    ax.legend(); ax.grid(alpha=0.25)
    return fig


def write_all(cfg: Config, run_dir: Path, panel, capacity, borrow, alpha, trade_panel,
              headline_runs: dict[float, pd.DataFrame], frontier_runs: dict[float, pd.DataFrame],
              sp500_tr: pd.Series) -> dict:
    """Write every table, figure and the tear-sheet; return a small summary for the manifest."""
    bench_aum = cfg.portfolio.benchmark_aum
    elig = capacity.eligibility_by_aum[bench_aum]
    short_elig = borrow.short_eligibility_by_aum[bench_aum]
    daily_bench = headline_runs[bench_aum]
    daily_frontier = frontier_runs[bench_aum]

    # Step 1-4 diagnostics
    write_table(run_dir, panel.reconciliation, "step1_reconciliation", index=False)
    write_table(run_dir, panel.universe_counts, "step1_universe_counts", index=False)
    write_table(run_dir, capacity.binding_by_aum[bench_aum], "step2_binding_250M")
    write_table(run_dir, borrow.tier_distribution, "step3_borrow_tier_distribution")
    write_table(run_dir, alpha.ic_compare, "step4b_ic_compare", index=False)
    write_table(run_dir, alpha.ic_by_year, "step4b_ic_by_year")
    write_table(run_dir, alpha.sleeve_ic, "step4b_sleeve_ic", index=False)
    write_table(run_dir, alpha.sleeve_corr, "step4b_sleeve_corr")

    # Step 5 headline + frontier + analytics
    headline = pd.DataFrame([perf_summary(cfg, headline_runs[a], a) for a in cfg.capacity.aum_levels])
    frontier = pd.DataFrame([perf_summary(cfg, frontier_runs[a], a) for a in cfg.capacity.aum_levels])
    write_table(run_dir, headline, "step5_headline_3aum", index=False)
    write_table(run_dir, frontier, "step5_frontier_3aum", index=False)
    ladder = _sharpe_ladder(cfg, trade_panel, elig, short_elig, bench_aum, daily_bench, daily_frontier)
    write_table(run_dir, ladder, "step5_sharpe_ladder", index=False)
    sleeve_corr, sleeve_runs = _sleeve_return_corr(cfg, trade_panel, elig, short_elig, bench_aum)
    write_table(run_dir, sleeve_corr, "step5_sleeve_return_corr")
    decile = run_strategy(cfg, trade_panel, elig, short_elig, bench_aum, quantile=0.10, weighting="equal", **_QUANT)
    write_table(run_dir, _decomposition(daily_bench, bench_aum), "step5_gross_to_net_decomposition")
    sweep = _concentration_sweep(cfg, trade_panel, elig, short_elig, bench_aum)
    write_table(run_dir, sweep, "step5_concentration_sweep", index=False)
    write_table(run_dir, _annual_breakdown(daily_bench), "step5_annual_breakdown")
    write_table(run_dir, _stress_windows(cfg, daily_bench), "step5_stress_windows", index=False)
    write_table(run_dir, _robustness_grid(cfg, trade_panel, elig, short_elig, bench_aum), "step5_robustness_grid", index=False)

    # figures (each supports a specific argument in the report)
    write_figure(run_dir, _fig_performance(daily_bench, sp500_tr, bench_aum), "headline_performance_250M")
    write_figure(run_dir, _fig_monthly_heatmap(daily_bench), "monthly_returns_heatmap_250M")
    write_figure(run_dir, _fig_distribution(daily_bench), "return_distribution_250M")
    write_figure(run_dir, _fig_concentration(cfg, sweep), "concentration_curve")
    write_figure(run_dir, _fig_sharpe_vs_aum(headline, frontier), "sharpe_vs_aum")
    write_figure(run_dir, _fig_sleeve_curves(sleeve_runs), "sleeve_equity_curves")
    write_figure(run_dir, _fig_gross_vs_net(decile, bench_aum), "decile_gross_vs_net_250M")
    write_figure(run_dir, _fig_rolling_ic(alpha.rolling_ic, cfg.alpha.rolling_ic_window), "combined_rolling_ic")
    write_figure(run_dir, _fig_sharpe_ladder(ladder), "sharpe_ladder")
    plt.close("all")

    # tear-sheet (headline strategy at benchmark AUM)
    strat = daily_bench["net_ret"].copy()
    strat.index = pd.to_datetime(strat.index)
    strat.name = "C2O_Overnight_250M"
    ts_ok = write_tearsheet(cfg, strat, sp500_tr, run_dir / "reports" / "C2O_tearsheet_250M.html")

    headline_250 = headline.loc[headline["AUM"] == f"{bench_aum/1e6:.0f}M"].iloc[0]
    frontier_250 = frontier.loc[frontier["AUM"] == f"{bench_aum/1e6:.0f}M"].iloc[0]
    return {"net_sharpe_250M": float(headline_250["net_sharpe"]),
            "gross_sharpe_250M": float(headline_250["gross_sharpe"]),
            "frontier_sharpe_250M": float(frontier_250["net_sharpe"]),
            "tearsheet_written": ts_ok,
            "headline": headline.to_dict(orient="records"),
            "frontier": frontier.to_dict(orient="records")}
