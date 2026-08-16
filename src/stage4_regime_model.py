"""
Stage 4 — regime model

OWNER: <put your name here>

PURPOSE
    3-regime Markov switching model on market volatility, with sentiment as a feature.
    Per-regime VaR, Expected Shortfall and tail risk.

READS
    data/processed/market_data.parquet, data/processed/sentiment_scores.parquet

WRITES
    data/processed/regimes.parquet

CONTRACT
    See contracts/stage4_regime_model.md
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
    raise NotImplementedError("Stage 4 not implemented yet")


# HINTS
#   - THE REGIME IS VOLATILITY. Sentiment is an INPUT. If you find yourself modelling
#     "sentiment regimes", stop and re-read section 1 of the brief.
#   - N_REGIMES is fixed at 3. Do not tune it.
#   - Sentiment is observed at meeting frequency; market data is daily. Decide how you
#     align them and document the choice - it is a real modelling decision.
#   - statsmodels MarkovRegression. Check convergence; it does not always converge.


if __name__ == "__main__":
    df = run()
    print(df.head())
    print(f"rows: {len(df)}")
