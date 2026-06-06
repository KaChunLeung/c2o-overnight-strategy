# C2O — Innovation Declaration (one page)

Where our judgement went beyond what an off-the-shelf LLM would have proposed. We used LLM tooling as a
coding assistant; the three choices below are ours, and they shaped the entire submission.

### 1. We reframed the task as cost-structural, not predictive — and rejected the obvious "turnover" fix.
The default LLM move on a high-turnover strategy is "reduce turnover with signal smoothing / rebalance bands."
We measured, up front, that this is **inapplicable here**: the brief mandates an overnight-only book that is
fully liquidated every morning, so a full MOC→MOO round trip (4 bps) is incurred *every night regardless of
the signal* — smoothing changes which names trade, not the gross traded, so it cannot cut cost. Recognising
this redirected all our effort from "a bigger model" to the two levers that actually move the net: widening
the cross-sectional spread and concentrating into the tails. This is the single most important judgement in the
project and it is the opposite of the generic recommendation.

### 2. We diagnosed the long/short drift asymmetry as the reason the strategy is hard.
We decomposed the two legs and found that the **long** leg clears the cost per name (top names earn ≈6 bps
overnight, riding the universal overnight drift of the brief's Section 1), while the **short** leg *fights*
that same drift (even the lowest-ranked names drift ≈+2 bps overnight, so shorting them loses ≈6 bps net).
Dollar-neutrality then guarantees the profitable drift cancels and only the sub-cost half-spread remains.
This turns a vague "the alpha loses to costs" into a precise, defensible economic statement, and it *predicts*
the result we then verified: net Sharpe is monotone in tail-extremity (decile −1.0 → 2% +0.4), because only the
extreme tails carry a spread above the fixed cost. We report this honest ~0.4 rather than an inflated number,
exactly as the brief rewards.

### 3. We made the alpha AUM-agnostic and pushed capacity entirely into portfolio construction.
Rather than re-fitting a model per AUM level, we score one signal universe (base-eligible at the most
permissive AUM) and let the participation-cap **water-fill** express capacity as falling gross utilisation
(0.93 → 0.65 → 0.22 across \$50M/\$250M/\$1B). This yields an honest, apples-to-apples three-AUM table where
the only thing that changes is deployable capital — and it surfaced a non-obvious result we would have missed
otherwise: risk-adjusted performance is *not* degraded by scale (net Sharpe actually rises to +0.58 at \$1B)
because the binding cap forces the book into the most liquid, lowest-volatility names. Scale costs absolute
return and deployed capital, not Sharpe.

*(A self-inflicted but instructive catch: when porting the notebook to the package we initially computed lag
features after filtering to the eligible set, which silently flattened the tail edge. We caught it because the
package reproduces the notebook's numbers under test — a concrete payoff of the reproducibility discipline.)*
