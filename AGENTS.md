# AGENTS.md — conventions, gotchas, commands

Things a new contributor (human or AI) cannot infer from the code alone. Keep it short.

## Commands
- **Verification (inner loop, < 1 min):** `pytest` — fast unit tests for every step.
- **Integration smoke (~3 min):** `pytest -m slow` or `python -m c2o.main --overrides config/fast.yaml`.
- **Full run:** `python -m c2o.main` → writes `data/outputs/<run_id>/`.
- Install: `pip install -e .` (or `pip install -e ".[dev]"` for pytest).

## Non-obvious decisions (and why)
- **Out-of-sample window is 2015–2024, not 2010–2024.** The alpha is walk-forward; 2010–2014 is the training
  burn-in, so no OOS signal exists there. The tear-sheet necessarily begins in 2015.
- **The alpha is scored on the most permissive universe (`alpha.signal_aum` = 50M base-OK).** It is
  AUM-agnostic; capacity (ADV cap, borrow) is applied per-AUM only in Step 5. This is why `signal_aum` must be
  ≤ the smallest AUM level (validated in `config.py`).
- **Turnover is structural, not a tunable.** The overnight mandate forces a full MOC→MOO round trip every
  night; signal smoothing does NOT reduce cost. Net edge = ½·(L−S) − 4 bps − borrow, so the headline
  concentrates into the tails (`portfolio.headline_quantile` = 2%); the concentration sweep shows this is a
  monotone economic effect, not a tuned quantile.
- **The cost-aware dispersion gate (`portfolio.gate_q`) is OFF by default.** It helps wide baskets but is
  unstable year-to-year at the 2% tails; kept in the robustness grid, not the headline.

## Gotchas
- **Memory:** the full run holds the 8M-row price panel plus ~3M-row feature panels (~6–8 GB peak). `main()`
  frees `panel.prices`/`panel.panel` after Step 2. Do not run the monolithic notebook headless on a low-RAM box.
- **Python 3.14** is the tested interpreter (`.python-version`). `quantstats` works but emits deprecation
  warnings; they are filtered, not suppressed at the cause (it is a third-party issue).
- **Inputs are large and gitignored** (`data/inputs/*.parquet`, ~1 GB). See `data/inputs/README.md` for
  provenance. The pipeline reads them read-only; nothing is written back to `data/inputs/`.
- **Held-out run:** move `window.cutoff` in `config/default.yaml` to 2026-12-31 (or pass an overrides YAML);
  no code changes are needed. No data after the cutoff is read, even for plotting.

## Conventions
- All paths and behaviour-bearing numbers live in `config/*.yaml`; code carries no path or magic-number literals.
- `io.py` is the only module that touches the filesystem. Steps are pure functions with typed handoffs.
- Fixed seed (`run.seed`). Outputs are per-run under `data/outputs/<run_id>/` and safe to delete.
- Coding rules: `docs/coding_architecture_guidelines.pdf`.
