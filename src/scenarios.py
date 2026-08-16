"""
Geopolitical scenarios — SHARED WORK

LEAD: <put a name here>

PURPOSE
    Generate a candidate pool of scenarios anchored in the external source library,
    have the model rank them for relevance to Australia, select three with DIFFERENT
    transmission routes, and push each through the pipeline.

READS
    data/raw/geopolitical/ (GPR index), plus the linked external sources
    data/processed/regimes.parquet  (the response function)

WRITES
    data/processed/scenarios.json  - selected AND rejected, with the model's reasons
"""
from __future__ import annotations
import config

# The five transmission routes from section 5.4 Rule 3 of the brief.
# Your three selected scenarios must use three DIFFERENT routes.
TRANSMISSION_ROUTES = [
    "commodity_prices",
    "supply_chain_inflation",
    "currency_and_capital_flows",
    "confidence_and_demand",
    "labour_market",
]


def generate_pool():
    """Generate candidates anchored in the external sources. NOT from RBA documents."""
    raise NotImplementedError


def rank_relevance(pool):
    """Ask the model to rank relevance to Australia. Your job is to CHECK this ranking."""
    raise NotImplementedError


def pool_diversity(pool) -> float:
    """Mean pairwise embedding distance across the pool. Report it whatever it says."""
    raise NotImplementedError


# HINTS
#   - Rule 1: scenarios come from the external library, never from RBA minutes.
#     RBA minutes are the RESPONSE function, not the scenario source.
#   - Rule 3 is a hard requirement. Three supply-shock scenarios are one scenario.
#   - Save the REJECTED candidates too. They are evidence of the search.

if __name__ == "__main__":
    pool = generate_pool()
    print(f"pool size: {len(pool)}  diversity: {pool_diversity(pool):.3f}")
