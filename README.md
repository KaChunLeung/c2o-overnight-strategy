# C2O — Close-to-Open Overnight Long/Short

A daily-frequency, dollar-neutral, US large-cap equity strategy that holds positions **only overnight**:
long stocks bought at today's closing auction and sold at tomorrow's opening auction; shorts the mirror.
The portfolio is rebuilt every trading day. Built strictly from daily data (OHLCV, fundamentals,
earnings/borrow flags), with a single anti-leakage cutoff so the marker can re-run on held-out 2025–2026.

## What it does (five steps)

1. **Panel** — adjusted-close returns (`r_on`, `r_id`, `r_cc`) that reconcile to machine precision; a
   survivorship-aware top-1000 universe frozen yearly; point-in-time earnings (BMO/AMC) and short-interest lags.
2. **Capacity** — per-(stock, date, AUM) eligibility under ADV / price / market-cap / volatility / earnings
   filters and a 5% participation cap.
3. **Borrow** — a point-in-time hard-to-borrow proxy → tiers A/B/C; Tier-C excluded from shorts.
4. **Alpha** — a ridge + gradient-boosting reversal **ensemble** plus an orthogonal **fundamental-flow sleeve**
   (analyst revisions / earnings surprise), scored walk-forward on an AUM-agnostic signal universe
   (reversal IC ≈ 0.022, t ≈ 7; flow IC ≈ 0.014, rank-corr 0.11 — genuinely orthogonal).
5. **Portfolio** — dollar-neutral baskets, participation-cap **water-fill** sizing, the Section 6.3 cost
   schedule; a **disciplined headline** and an aggressive **frontier**, a Sharpe-ladder ablation, performance
   at 50M / 250M / 1B, and a QuantStats tear-sheet.

## Headline result (OOS 2015–2024, full Section 6.3 costs)

The search for net Sharpe was run as a **controlled experiment**: every plausible lever (flow blend, cost-aware
per-name selection, edge-proportional gross scaling, sector/beta neutralization, vol-targeting, inverse-vol, a
neural model) was tested and **only tail concentration helped** — so the headline is the reversal ensemble at
the concentration optimum (1.25% tails). This roughly **doubles** the naive 2% result.

| AUM | net Sharpe (headline 1.25%) | frontier (HGB 1%) | gross Sharpe | net ann |
|----|----|----|----|----|
| 50M  | 0.32 | 0.74 | ~1.6 | ~2.0% |
| 250M | **0.71** | **0.80** | ~1.8 | ~3–5% |
| 1B   | 0.58 | 0.05* | ~1.4 | ~0.9% |

*The aggressive 1% frontier collapses at 1B as the book thins to ~5 names — a capacity limit of concentration.
**Net Sharpe > 1 was not reached and is not claimed**: the 4 bps cost on the *forced gross* round trip is a
near-fixed ~1.3 Sharpe-units of drag; reaching 1.0 needs gross Sharpe ~2.4 (we have ~1.8) or relaxing a mandate
constraint. See [docs/C2O_Steps4b-5_Worklog.md](docs/C2O_Steps4b-5_Worklog.md) and
[report/C2O_report.tex](report/C2O_report.tex) for the full investigation, including the failed levers.

## Run it

```bash
pip install -e .                       # installs the c2o package + deps
python -m c2o.main                     # full run -> data/outputs/<run_id>/ (headline + frontier + ablation)
python -m c2o.main --overrides config/fast.yaml   # fast smoke run (truncated universe, ~3 min)
python -m c2o.dl_alpha                 # documented neural-vs-tree benchmark (off the production path)
pytest                                 # fast unit tests (26 tests, < 5 s)
pytest -m slow                         # + the end-to-end integration smoke
PYTHONPATH=src python tools/experiment.py   # dev construction lab (caches the trade panel, sweeps variants)
```

Outputs land in `data/outputs/<run_id>/`: `tables/*.csv`, `figures/*.png`,
`reports/C2O_tearsheet_250M.html`, and a self-describing `manifest.json` (config + git SHA + versions).

## Held-out evaluation

Everything keys off one parameter, `window.cutoff` (default `2024-12-31`) in
[config/default.yaml](config/default.yaml). To evaluate 2025–2026, move the cutoff; no code changes.

## Layout

```
config/        default.yaml (all params), fast.yaml (smoke overrides)
src/c2o/       config, io, panel, capacity, borrow, alpha, portfolio, metrics, reporting, main, dl_alpha
tests/         26 unit tests + one slow integration smoke
tools/         experiment*.py — dev construction lab (not part of the shipped pipeline)
data/inputs/   read-only source parquets (gitignored)
data/outputs/  per-run deliverables (gitignored)
notebooks/     exploratory companion (full Steps 1-5 narrative)
report/        C2O_report.tex (the investigation), innovation_declaration.md, figures/
docs/          brief, coding guidelines, worklog
```

See [AGENTS.md](AGENTS.md) for conventions, gotchas, and the verification command.
