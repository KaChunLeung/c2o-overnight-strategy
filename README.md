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
4. **Alpha** — a ridge + gradient-boosting **ensemble** on point-in-time features, scored walk-forward on an
   AUM-agnostic signal universe (mean IC ≈ 0.022, t ≈ 7, gross L/S Sharpe ≈ 1.6).
5. **Portfolio** — dollar-neutral quantile baskets, participation-cap **water-fill** sizing, the Section 6.3
   cost schedule, performance at 50M / 250M / 1B, and a QuantStats tear-sheet.

## Headline result (OOS 2015–2024, top/bottom 2%, full costs)

| AUM | net Sharpe | gross Sharpe | net ann | maxDD | gross util |
|----|-----------|-------------|---------|-------|-----------|
| 50M | ~0.25 | ~1.9 | ~1.5% | ~-13% | ~0.93 |
| 250M | ~0.39 | ~1.8 | ~1.9% | ~-7% | ~0.65 |
| 1B | ~0.37 | ~1.4 | ~0.8% | ~-5% | ~0.23 |

The gross overnight alpha is strong; the mandated 4 bps/night round trip on a forced daily round trip, plus
dollar-neutrality stripping the overnight drift, leaves the strategy only marginally net-positive. See
[docs/C2O_Steps4b-5_Worklog.md](docs/C2O_Steps4b-5_Worklog.md) for the full economic diagnosis.

## Run it

```bash
pip install -e .                       # installs the c2o package + deps
python -m c2o.main                     # full run -> data/outputs/<run_id>/ (tables, figures, tear-sheet, manifest)
python -m c2o.main --overrides config/fast.yaml   # fast smoke run (truncated universe, ~3 min)
pytest                                 # fast unit tests (verification, < 1 min)
pytest -m slow                         # + the end-to-end integration smoke
```

Outputs land in `data/outputs/<run_id>/`: `tables/*.csv`, `figures/*.png`,
`reports/C2O_tearsheet_250M.html`, and a self-describing `manifest.json` (config + git SHA + versions).

## Held-out evaluation

Everything keys off one parameter, `window.cutoff` (default `2024-12-31`) in
[config/default.yaml](config/default.yaml). To evaluate 2025–2026, move the cutoff; no code changes.

## Layout

```
config/        default.yaml (all params), fast.yaml (smoke overrides)
src/c2o/       config, io, panel, capacity, borrow, alpha, portfolio, metrics, reporting, main
tests/         unit tests + one slow integration smoke
data/inputs/   read-only source parquets (gitignored)
data/outputs/  per-run deliverables (gitignored)
notebooks/     exploratory companion (full Steps 1-5 narrative)
docs/          brief, coding guidelines, worklog
```

See [AGENTS.md](AGENTS.md) for conventions, gotchas, and the verification command.
