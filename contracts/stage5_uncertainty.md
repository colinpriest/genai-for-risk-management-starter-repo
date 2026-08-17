# Contract: Stage 5 (Uncertainty) → dashboard and report

`stage5_uncertainty.py` is a **stub**. This contract is the specification you build to.

**File:** `data/processed/uncertainty.json`
**Format:** JSON, not parquet — the layers have different shapes and do not form one table.

```json
{
  "specification": {"endog": "log_rv", "n_regimes": 3, "text_features": ["..."]},
  "layer1_parameter": {
    "n_draws_used": 48, "n_requested": 60,
    "draw_outcomes": {"ok": 48, "not_converged": 9, "degenerate": 3, "exception": 0},
    "excluded_from_interval": 12, "block_length_days": 63,
    "regime0_ann_vol_p5_p50_p95": [0.090, 0.102, 0.123],
    "regime2_ann_vol_p5_p50_p95": [0.222, 0.298, 0.598]
  },
  "layer2_model": [{"spec": "...", "k_regimes": 3, "aic": -5352.1, "converged": true}],
  "layer3_llm": {"financial_conditions_concern": {"within_doc_sd_mean": 0.072,
                                                  "between_doc_sd": 0.136,
                                                  "signal_to_noise": 1.9}},
  "layer4_human": "produced by the cross-review exercise, not by this code"
}
```

## Layer 1 — parameter uncertainty

Moving-block bootstrap. Blocks, not iid resampling: an iid bootstrap destroys the volatility
clustering the regime model exists to detect. 63 trading days is about one quarter.

**Three requirements, and all three are marked.**

1. **Fit the same specification as stage 4** — same endog, same exog, same regime count.
   Bootstrapping a different model measures the uncertainty of a model you are not reporting.
2. **Re-order the regimes inside every draw**, on empirical realised volatility. Each draw is a
   fresh fit and labels its regimes independently. This is trap 4, and it is silent: pooling
   unordered draws gives a tidy interval that has mixed regimes together.
3. **Record how many draws failed to converge**, and report `n_draws` against `n_requested`.
   A quietly-dropped third of the draws is a finding about model stability.

Report a p5/p50/p95 per regime. If two regimes' intervals overlap, say so — it qualifies any
claim that they are distinct states.

## Layer 2 — model uncertainty

Refit under alternative defensible specifications and report the spread. At minimum: with and
without text. Do not add regime counts as a specification axis - the count is fixed at three
for the reasons given in `config.py`.

**Use the same fitter and search settings for every spec.** Weaker settings on one specification
measure how well the optimiser did, not model uncertainty — the two differ by thousands of AIC on
this data. **Reject non-converged fits** rather than reporting their AIC.

**Do not compare raw AIC across different dependent variables.** Returns and log realised
volatility are different response scales, so their likelihoods are not on a common footing.
Compare *within* a response definition (text versus no text), and compare response definitions
only on an out-of-sample score expressed on a common target.

## Layer 3 — LLM uncertainty

From `riskvoice_scores.parquet`. Per construct, report the mean within-document sd, the
between-document sd, and their ratio. **The ratio is what matters**: a construct whose
between-document spread barely exceeds its own measurement noise is not discriminating.

`embedding_spread` measures whether the calls agreed for the same *reasons*, which the numeric
sd cannot see.

**It must appear in `layer3_llm`, not merely exist in stage 3's output.** Report, per construct
or pooled: mean, p90 and max of `embedding_spread`, and the share of documents above 0.4. A
document where the numbers agree and the rationales do not is the case this layer exists to
find, and an aggregate of standard deviations alone cannot show it.

**Name this layer accurately.** It is output stability under a stated sampling protocol at
temperature 1.0. It is not calibrated epistemic uncertainty, and it must not be presented as a
confidence interval on the truth.

## Layer 4 — human agreement

**Not produced by this file.** It comes from the cross-review exercise in section 8.2 and lives in
`data/processed/agreement/`. Reference it here so the report can assemble all four.

**Agreed by:** ______________ (stage 5) and ______________ (dashboard/report)  **Date:** ______

**Change log**

| Date | Change | Why | Agreed by |
|---|---|---|---|
| | | | |
