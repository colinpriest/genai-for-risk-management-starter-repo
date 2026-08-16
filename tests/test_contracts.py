"""Turn interface contracts into tests.

Run after any stage produces output:

    pytest tests/ -v

Each test skips if the stage has not run yet, so this is safe from Week 2 onward.
"""
from __future__ import annotations
import pytest
import pandas as pd
import config

# (file, required columns, grain description)
CONTRACTS = {
    "market_data.parquet": (["date"], "one row per trading day"),
    "documents.parquet": (["meeting_date", "text"], "one row per RBA meeting"),
    "sentiment_scores.parquet": (
        ["meeting_date", "sentiment_mean", "sentiment_sd", "n_calls_valid"],
        "one row per RBA meeting"),
    "regimes.parquet": (["date", "regime"], "one row per trading day"),
}


def _load(name):
    p = config.DATA_PROCESSED / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return pd.read_parquet(p)


@pytest.mark.parametrize("name,spec", CONTRACTS.items())
def test_required_columns(name, spec):
    cols, _ = spec
    df = _load(name)
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"{name} is missing required columns: {missing}"


def test_sentiment_ranges():
    df = _load("sentiment_scores.parquet")
    assert df["sentiment_mean"].between(-1, 1).all(), "sentiment_mean outside [-1, 1]"
    assert (df["sentiment_sd"] >= 0).all(), "negative standard deviation"
    assert df["n_calls_valid"].between(0, config.N_PARALLEL_CALLS).all()
    assert df["meeting_date"].is_unique, "duplicate meeting_date - grain is one row per meeting"
    assert df["meeting_date"].is_monotonic_increasing, "rows must be in date order"


def test_sentiment_never_silently_empty():
    """A row with no valid calls must not carry a sentiment score."""
    df = _load("sentiment_scores.parquet")
    bad = df[(df["n_calls_valid"] == 0) & df["sentiment_mean"].notna()]
    assert bad.empty, f"{len(bad)} rows have a score but zero valid calls"


def test_regimes_count():
    df = _load("regimes.parquet")
    n = df["regime"].nunique()
    assert n <= config.N_REGIMES, f"found {n} regimes; N_REGIMES is {config.N_REGIMES}"


def test_corpus_present():
    n = len(list(config.RBA_MINUTES_DIR.glob("*.html")))
    assert n >= 200, f"expected ~211 RBA minutes, found {n}"
