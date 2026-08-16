"""
Stage 5 — uncertainty

OWNER: <put your name here>

PURPOSE
    Three uncertainty layers from the reference: parameter (bootstrap), model (ensemble),
    and LLM (semantic spread across the parallel calls).

READS
    data/processed/regimes.parquet, sentiment_scores.parquet, llm_raw/

WRITES
    data/processed/uncertainty.parquet

CONTRACT
    See contracts/stage5_uncertainty.md
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
    raise NotImplementedError("Stage 5 not implemented yet")


# HINTS
#   - The FOURTH layer - human agreement - is NOT computed here. It comes from your
#     cross-review work (section 5.2) and lives in data/processed/agreement/.
#   - Embeddings run locally via sentence-transformers. No API cost.
#   - Report intervals, not point estimates. Most RBA decisions are holds, so accuracy
#     is meaningless here (section 5.1 of the brief).


if __name__ == "__main__":
    df = run()
    print(df.head())
    print(f"rows: {len(df)}")
