"""
Stage 5 — uncertainty

OWNER: <put your name here>

PURPOSE
    Three uncertainty layers from the reference: parameter (bootstrap), model (ensemble),
    and LLM (semantic spread across the parallel calls).

READS
    data/processed/regimes.parquet, riskvoice_scores.parquet, llm_raw/

WRITES
    data/processed/uncertainty.json

CONTRACT
    See contracts/stage5_uncertainty.md (supplied)
    Agree it in Week 2 and commit it BEFORE writing this code.
    tests/test_contracts.py will check your output against it.
"""
from __future__ import annotations
import pandas as pd
import config


def run() -> dict:
    """Produce this stage's output and write it to the path in the contract.

    Returns the uncertainty DICT (written to uncertainty.json). It is not a DataFrame:
    the three layers have different shapes and do not form one table.
    """
    raise NotImplementedError("Stage 5 not implemented yet")


# HINTS
#   - The FOURTH layer - human agreement - is NOT computed here. It comes from your
#     cross-review work (section 8.2) and lives in data/processed/agreement/.
#   - Embeddings run locally via sentence-transformers. No API cost.
#   - Report intervals, not point estimates. There is no accuracy figure to report: this
#     model does not classify anything against a known answer. Regimes are LATENT - there is
#     no labelled "true regime" for a given day - so any accuracy-style number you produce is
#     measuring agreement with your own fit, not correctness (section 8.1 of the brief).
#     Report per-regime intervals, expected durations, and whether the regimes remain
#     distinguishable once uncertainty is accounted for.


if __name__ == "__main__":
    # run() returns the four-layer DICT specified in contracts/stage5_uncertainty.md, not a
    # DataFrame. An earlier version called .head() on it, which works only while run() is an
    # unimplemented stub and crashes the moment you implement the contract.
    import json as _json
    result = run()
    print(_json.dumps({k: v for k, v in result.items() if not k.startswith("_")},
                      indent=2, default=str)[:2000])
