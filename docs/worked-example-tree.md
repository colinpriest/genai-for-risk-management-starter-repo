# A worked tree, end to end — the unassessed migration scenario

**Read this before you write the four Shock prompts.** It is a static record of one
complete `branch -> evaluate -> prune -> expand` cycle, produced by the supplied
framework on the scenario you are *not* marked on:

> Sharp cut to net overseas migration

Nothing here is an answer to either assessed scenario, and reading it costs no API
calls. What it shows you is the SHAPE of the thing you are about to build: what a
branch record contains, what the evaluator does to it, why most branches die, and
what survives to carry a number.

---

## What one run produced

| | |
|---|---:|
| Branches generated | 46 |
| Scored by the evaluator | 46 |
| Cut by the pruning threshold | 12 |
| Blocked at the grounding gate | 34 |
| Quotations verified verbatim | 5 |

**Nothing is marked `keep` here, and that is not the framework rejecting everything.**
`--worked` deliberately stops BEFORE the human review step, because a demonstration
that handed you reviewed branches would be handing you the judgement you are assessed
on. Every branch therefore sits at `evidence_status: neither` - the Australian leg is
`unreviewed` because nobody has reviewed it. In your own run, that is the column your
`CHANNEL_REVIEWS` entries change.

Read `kept_by` for the reason each branch is where it is. Two very different things
live in that column: `threshold` means the evaluator scored it below the pruning bar,
while `grounding: ...` means it failed the evidence gate no matter how it scored.

### Where the branches stood on evidence

| Status | Branches |
|---|---:|
| `generated` | 46 |
| `grounded` | 5 |
| `ungrounded` | 20 |
| `not_in_sources` | 21 |

---

## Three branches, in full

One whose quotation was verified, one the evaluator scored below the threshold, and
one that cited something not in the sources at all. The field names are exactly the
ones you will fill in `CHANNEL_REVIEWS`.

### A branch whose quotation was VERIFIED

*The quoted words really are in the cited document - that is all `quote_verified` means. Whether those words SUPPORT the claim is the judgement you sign in CHANNEL_REVIEWS, and it is the one the marks are for. Read the `source_quote` against the `mechanism` and decide for yourself.*

| Field | Value |
|---|---|
| `channel` | financial stability concern |
| `channel_type` | financial_stability |
| `mechanism` | Decreased migration exacerbates financial vulnerability in the economy, increasing concerns over financial stability. |
| `direction` | easing |
| `horizon` | weeks |
| `n_sd` | 4.0 |
| `depth` | 0 |
| `score` | 0.8 |
| `keep` | False |
| `kept_by` | grounding: neither (global quote_unreviewed, australian unreviewed) |
| `source` | [imf-weo-update-2026-07.pdf p.10] |
| `source_quote` | Durable peace agreements could rapidly restore global trade routes and supply chains. |
| `quote_verified` | True |
| `evidence_status` | neither |
| `pathway` | rises -> financial stability worse -> easing |
| `note` | Further analysis on local financial instability would be required. \| eval: The relationship between reduced migration and financial stability is strong; concerns about financial vulnerability are highly credible. \| BARRED FROM THE SHOCK: neither. Global leg quote_unreviewed; Australian leg unrevie ... |

### A branch the EVALUATOR scored below the threshold

*`kept_by: threshold` means the model's own credibility score killed it. This is the pile your pruning audit samples: the evaluator is the same model that wrote the branch, so somebody has to check what it threw away.*

| Field | Value |
|---|---|
| `channel` | wage pressure |
| `channel_type` | labour_supply |
| `mechanism` | With fewer workers available due to capped migration, employers compete for a smaller labor pool, driving wages higher. |
| `direction` | tightening |
| `horizon` | 1-2q |
| `n_sd` | 1.5 |
| `depth` | 1 |
| `score` | 0.4 |
| `keep` | False |
| `kept_by` | threshold |
| `source` | [not in sources] |
| `quote_verified` | False |
| `evidence_status` | neither |
| `pathway` | falls -> inflation up -> tightening |
| `note` | While the initial reduction in labor supply didn't raise inflation expectations significantly, the resulting wage gains could eventually translate into higher consumer prices, necessitating a tightening response from the RBA. \| CITATION NOT VERIFIED: the citation names a passage that was never retr ... |

### A branch that cited something NOT IN THE SOURCES

*The most common failure by a wide margin here. The model produced a plausible mechanism and attached a citation that does not resolve to any document you supplied. It is barred automatically - but notice that the MECHANISM may still be sound. Barred is not refuted.*

| Field | Value |
|---|---|
| `channel` | labour_supply reduction |
| `channel_type` | labour_supply |
| `mechanism` | The shrinking population due to reduced migration leads to a smaller workforce, which constricts supply-side capacity and creates upward pressure on wages. |
| `direction` | easing |
| `horizon` | 1-2q |
| `n_sd` | -0.9 |
| `depth` | 0 |
| `score` | 0.7 |
| `keep` | False |
| `kept_by` | grounding: neither (global not_in_sources, australian unreviewed) |
| `source` | not in sources |
| `quote_verified` | False |
| `evidence_status` | neither |
| `pathway` | falls -> supply capacity down -> easing |
| `note` | An analysis of the impact on labour supply and wages would be required to quantify this. \| eval: The reduction in labour supply impacts growth and wage dynamics; however, the pathway needs more rigorous analysis for real effects. \| BARRED FROM THE SHOCK: neither. Global leg not_in_sources; Austral ... |

---

## What to take from it

1. **A branch is a pathway, not a sentence.** proxy movement -> first-round effect ->
   policy implication -> direction, with a signed size. A branch prompt that returns
   opinions produces branches that cannot be reviewed or shocked.
2. **The citation is checked, and most fail.** `quote_verified` is a verbatim
   substring test against the document. A verified quote still has to SUPPORT the
   claim, and that judgement is yours.
3. **Restatements are barred automatically.** One canonical branch carries each claim;
   the rest vote nowhere. This is why your review load is mechanism groups, not
   branches.
4. **An unmodellable branch is a result.** The honest output of this stage includes
   channels nobody could sign a number for.

When you have read this, write the four prompts (Step 3), then run `--discover` on
your own scenarios to get the review template.

---

*Generated from the supplied framework's own output on the unassessed scenario. To
regenerate it after a framework change, re-run the worked scenario and re-render.*
