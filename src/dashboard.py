"""
Dashboard — interactive Plotly output

OWNER: <put your name here>

PURPOSE
    Produce outputs/dashboard.html: Australian rates and market volatility against the
    inferred regime and all five extracted risk-voice constructs, with uncertainty shown.

READS
    data/processed/regimes.parquet, riskvoice_scores.parquet, uncertainty.json
    (uncertainty is JSON, not parquet - the layers have different shapes and do
     not form one table. See contracts/stage5_uncertainty.md.)

WRITES
    outputs/dashboard.html
"""
from __future__ import annotations
import config


def build() -> None:
    raise NotImplementedError("Dashboard not implemented yet")


# HINTS
#   - Your audience is a senior manager, not a modeller (section 7 of the brief).
#   - Show uncertainty. A regime call with no interval is a false promise.
#   - Compare against the reference's dashboard.html for layout ideas.

if __name__ == "__main__":
    build()
