"""
Stage 3 — sentiment

OWNER: <put your name here>

PURPOSE
    Score each document's monetary policy stance using N_PARALLEL_CALLS separate API calls
    with structured Pydantic outputs. Record the spread across calls.

READS
    data/processed/documents.parquet

WRITES
    data/processed/sentiment_scores.parquet  +  raw responses in data/processed/llm_raw/

CONTRACT
    See contracts/stage3_sentiment.md
    Agree it in Week 2 and commit it BEFORE writing this code.
    tests/test_contracts.py will check your output against it.
"""
from __future__ import annotations
import pandas as pd
import config


def run() -> pd.DataFrame:
    """Produce this stage's output and write it to the path in the contract.

    Returns the DataFrame as well, so run_pipeline.py can chain stages in memory.
    """
    raise NotImplementedError("Stage 3 not implemented yet")


# HINTS
#   - Save EVERY raw response. The harness and your uncertainty work both need them,
#     and reproducibility is marked.
#   - Use Pydantic for structured output, as the reference does.
#   - Some calls will fail or return unparseable output. Count them; do not silently drop.
#   - Expose a function with this signature so the supplied harness can call it:
#
#         def score_text(text: str) -> float
#
#     The harness in harness/ takes it as an argument.


if __name__ == "__main__":
    df = run()
    print(df.head())
    print(f"rows: {len(df)}")
