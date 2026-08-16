# Contract: Stage 3 (Sentiment) → Stage 4 (Regime model)

**WORKED EXAMPLE — copy this format for your own contracts.**

**File:** `data/processed/sentiment_scores.parquet`
**Grain:** one row per RBA meeting
**Row count:** equals the number of documents scored, in date order, no gaps

| Column | Type | Range | Meaning | Missing means |
|---|---|---|---|---|
| `meeting_date` | date | — | Date of the meeting the minutes describe | Never missing. Primary key. |
| `sentiment_mean` | float | −1 to +1 | Mean across the parallel calls. Negative = dovish, positive = hawkish. | Never missing |
| `sentiment_sd` | float | ≥ 0 | Standard deviation across the calls | Never missing |
| `n_calls_valid` | int | 0–10 | Calls returning valid structured output | Never missing. Below 10 means the mean uses fewer calls. |
| `embedding_spread` | float | ≥ 0 | Semantic spread across responses | Missing if `n_calls_valid` < 3 |

**Agreed by:** ______________ (stage 3) and ______________ (stage 4)  **Date:** ______

**Change log**

| Date | Change | Why | Agreed by |
|---|---|---|---|
| | | | |
