"""
Stage 2 — documents

OWNER: <put your name here>

PURPOSE
    Load the RBA minutes corpus, extract clean text from the HTML, and build the retrieval
    step that selects relevant passages for scoring.

READS
    data/raw/rba-minutes/*.html  (211 files, supplied)

WRITES
    data/processed/documents.parquet  and a retrieval index

CONTRACT
    See contracts/stage2_documents.md
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
    raise NotImplementedError("Stage 2 not implemented yet")


# HINTS
#   - Filenames are rba-minutes-YYYY-MM-DD.html. The date is the MEETING date.
#   - Strip navigation, headers and footers before scoring. Check what your text
#     extraction actually returns on one file before running all 211.
#   - Retrieval, not full-context stuffing. The brief requires this (section 5.2).
#   - Document length varies a lot across the corpus. Look at the distribution.


if __name__ == "__main__":
    df = run()
    print(df.head())
    print(f"rows: {len(df)}")
