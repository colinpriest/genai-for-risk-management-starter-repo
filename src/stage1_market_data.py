"""
Stage 1 — market data

OWNER: <put your name here>

PURPOSE
    Download ASX 200 daily prices and Australian macro indicators. Produce a clean daily series
    with the volatility measures the regime model consumes.

READS
    yfinance (config.MARKET_TICKER), plus macro sources you choose

WRITES
    data/processed/market_data.parquet

CONTRACT
    See contracts/stage1_market_data.md
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
    raise NotImplementedError("Stage 1 not implemented yet")


# HINTS
#   - The reference uses S&P 500. You use ^AXJO. Check for differences in trading
#     calendar and holidays before assuming the series align.
#   - Justify your macro indicator choices in the report. "The reference used X" is
#     not a justification when X is a US series with no Australian equivalent.
#   - Decide and DOCUMENT how you handle missing days. It affects the regime model.


if __name__ == "__main__":
    df = run()
    print(df.head())
    print(f"rows: {len(df)}")
