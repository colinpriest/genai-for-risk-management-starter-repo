"""
Dashboard — interactive Plotly output

OWNER: <put your name here>

PURPOSE
    Produce outputs/dashboard.html: Australian rates and market volatility against the
    inferred regime and the extracted RBA sentiment, with uncertainty shown.

READS
    data/processed/regimes.parquet, sentiment_scores.parquet, uncertainty.parquet

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
