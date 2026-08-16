# Contract: Stage 3 (Risk-voice extraction) → Stage 4 (Regime model)

**WORKED EXAMPLE — copy this format for your own contracts.**

This one is filled in for you because stage 3 and stage 4 are both supplied. The contracts you
write in Week 2 for your own hand-overs must be this specific.

**File:** `data/processed/riskvoice_scores.parquet`
**Grain:** one row per RBA meeting
**Row count:** equals the number of documents scored, in date order, no gaps

| Column | Type | Range | Meaning | Missing means |
|---|---|---|---|---|
| `meeting_date` | date | — | Date of the meeting the minutes describe | Never missing. Primary key. |
| `financial_conditions_concern` | float | 0 to 1 | Mean across the parallel calls | Never missing |
| `financial_conditions_concern_sd` | float | ≥ 0 | Standard deviation across those calls. **This is the LLM uncertainty layer** — do not drop it. | Missing if fewer than 2 valid calls |
| `downside_risk_emphasis` | float | 0 to 1 | Mean across the parallel calls | Never missing |
| `downside_risk_emphasis_sd` | float | ≥ 0 | Standard deviation across those calls | Missing if fewer than 2 valid calls |
| `global_risk_salience` | float | 0 to 1 | Mean across the parallel calls | Never missing |
| `global_risk_salience_sd` | float | ≥ 0 | Standard deviation across those calls | Missing if fewer than 2 valid calls |
| `vigilance` | float | 0 to 1 | Mean across the parallel calls | Never missing |
| `vigilance_sd` | float | ≥ 0 | Standard deviation across those calls | Missing if fewer than 2 valid calls |
| `uncertainty_language` | float | 0 to 1 | Mean across the parallel calls | Never missing |
| `uncertainty_language_sd` | float | ≥ 0 | Standard deviation across those calls | Missing if fewer than 2 valid calls |
| `policy_stance` | float | 0 to 1 | The deliberate contrast field, not one of the five constructs | Never missing |
| `policy_stance_sd` | float | ≥ 0 | Standard deviation across those calls | Missing if fewer than 2 valid calls |
| `n_calls_valid` | int | 0–10 | Calls returning valid structured output | Never missing. Below 10 means the means use fewer calls. |
| `embedding_spread` | float | ≥ 0 | Semantic spread across responses | Missing if `n_calls_valid` < 3 |

**What stage 4 does with this.** Only the columns named in `stage4.TEXT_FEATURES` enter the
model. Choosing which of the five belong there is your decision — see section 5.3 of the brief.
Scores are forward-filled between meetings, so a missing row is not neutral: it silently extends
the previous meeting's reading.

**Checked by:** `tests/test_contracts.py`, which verifies the grain, the ranges, and that every
construct actually discriminates between documents.

**Agreed by:** ______________ (stage 3) and ______________ (stage 4)  **Date:** ______

**Change log**

| Date | Change | Why | Agreed by |
|---|---|---|---|
| | | | |
