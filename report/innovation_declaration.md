# C2O — Innovation Declaration (one page)

Where our judgement went beyond what an off-the-shelf assistant would have proposed. We used LLM tooling as a
coding assistant; the choices below are ours, and they shaped the entire submission. The unifying idea: we
treated the search for net Sharpe as a **controlled experiment** and shipped the **negative results**.

### 1. We pursued the brief's "Sharpe > 1" ambition seriously — and reported, honestly, that we did not reach it.
The easy path is to bolt on models and overlays and present whichever number looks best. We instead
pre-registered a slate of levers — a fundamental-flow alpha sleeve, a neural model, cost-aware per-name
selection, edge-proportional gross scaling, sector/beta neutralization, vol-targeting, inverse-vol weighting —
and **measured each against a concentrated baseline**, finding and documenting that none of them beat it. We
roughly **doubled** the v1 net Sharpe (0.38 → **0.71** headline, **0.80** frontier at \$250M) by correctly
locating the concentration optimum, and we state plainly that net Sharpe ≈ 0.7–0.8 is the **honest ceiling** of
this mandate. The discipline of shipping the failed levers — not burying them — is the core contribution.

### 2. The cost-structure diagnosis that *predicts* every failed lever.
We showed the mandated 4 bps is charged on the **gross** book, on a **forced** nightly liquidation. That single
fact has three consequences we then verified: (a) **per-name "cost-aware" selection cannot reduce cost** —
splitting the same gross over 5 or 50 names costs the same (it traded ~10% of nights and *lowered* Sharpe); (b)
**turnover reduction is inapplicable** (the overnight mandate forces the round trip nightly), so the standard
smoothing trick is moot; (c) the **only** way to cut cost is to scale gross down on thin nights — which requires
*timing* the good nights, and we measured the correlation between the alpha's predicted day-edge and the
realised next-night return at **≈ −0.003**: the good nights are not predictable. An off-the-shelf assistant
would have "reduced turnover" or "added a regime gate"; we showed, with numbers, why neither can work here.

### 3. An orthogonal sleeve — and the "broad IC ≠ tail edge" lesson.
We built a genuinely orthogonal fundamental-flow sleeve (analyst revisions / earnings surprise; rank-corr only
0.11 to the price sleeve, strongest exactly where the price alpha has decayed). Blending it **raises
full-cross-section IC (0.022 → 0.024) yet lowers the net Sharpe of the tail book**. The non-obvious lesson:
a strategy that trades only the extreme ~1% should *not* dilute its strongest names with a milder signal —
higher average rank-skill is not the same as a better tail. This reframes how sleeves should (and should not)
be combined for concentrated books, and it is why we keep the flow sleeve documented but out of the headline.

### 4. AUM-agnostic signal, capacity applied at construction — surfacing that scale is the variance lever.
Rather than re-fit per AUM, we score one signal universe and let the participation-cap water-fill express
capacity, giving an honest three-AUM table where the only thing that changes is deployable capital. This
surfaced a non-obvious result: net Sharpe **peaks at \$250M**, where the 5% ADV cap forces the book into the
most liquid, lowest-volatility names — the capacity constraint itself performs the variance reduction the cost
structure otherwise denies us — while aggressive concentration (the 1% frontier) *collapses* at \$1B as the
book thins to ~5 names. Scale is not merely a cost; under this mandate it is the main risk-adjustment lever.

*(Honest tooling note: PyTorch has no wheels for the Python 3.14 toolchain here, so the neural benchmark uses a
scikit-learn MLP trained with Adam; we argue why a deeper recurrent/attention model would not change the
conclusion, since the binding constraint is execution cost, not forecast accuracy.)*
