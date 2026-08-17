"""Turn interface contracts into tests.

Run after any stage produces output:

    pytest tests/ -v

Each test skips if the stage has not run yet, so this is safe from Week 2 onward.

These cover the SUPPLIED stages only. As you agree your own interface contracts in Week 2,
add a test here for each one — a contract nobody checks is a comment, not a contract.
"""
from __future__ import annotations
import pytest
import pandas as pd
import config

from src.stage3_riskvoice import FIELD_DESCRIPTIONS

CONSTRUCTS = list(FIELD_DESCRIPTIONS.keys())

# (file, required columns, grain description)
CONTRACTS = {
    "market_data.parquet": (["date"], "one row per trading day"),
    "documents.parquet": (["meeting_date", "text_full"], "one row per RBA meeting"),
    "riskvoice_scores.parquet": (
        ["meeting_date", "n_calls_valid"] + CONSTRUCTS + [f"{c}_sd" for c in CONSTRUCTS],
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


def test_riskvoice_grain():
    df = _load("riskvoice_scores.parquet")
    assert df["meeting_date"].is_unique, "duplicate meeting_date - grain is one row per meeting"
    assert df["meeting_date"].is_monotonic_increasing, "rows must be in date order"
    assert df["n_calls_valid"].between(0, config.N_PARALLEL_CALLS).all()


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_construct_ranges(construct):
    df = _load("riskvoice_scores.parquet")
    ok = df[construct].dropna()
    assert ok.between(0, 1).all(), f"{construct} outside [0, 1]"
    assert (df[f"{construct}_sd"].dropna() >= 0).all(), f"{construct}_sd is negative"


@pytest.mark.parametrize("construct", CONSTRUCTS)
def test_construct_discriminates(construct):
    """A construct that returns the same value for every document carries no information.

    This is not a bug in the code - it means the field description needs rewriting.
    """
    df = _load("riskvoice_scores.parquet")
    spread = df[construct].std()
    assert spread > config.MIN_CONSTRUCT_SPREAD, (
        f"{construct} has a between-document sd of {spread:.4f} (< "
        f"{config.MIN_CONSTRUCT_SPREAD}): it is returning "
        f"essentially the same value for every document. Rewrite its field description "
        f"in src/stage3_riskvoice.py so that high and low are clearly distinguished.")


def test_embedding_spread_present():
    """The LLM layer promises SEMANTIC spread, not just numeric spread."""
    df = _load("riskvoice_scores.parquet")
    assert "embedding_spread" in df.columns, (
        "embedding_spread missing. stage3 computes it from the per-call rationales; if it is "
        "absent, the contract's semantic-spread column is unfulfilled.")
    ok = df["embedding_spread"].dropna()
    assert (ok >= 0).all(), "embedding_spread is a distance and cannot be negative"


def test_never_silently_empty():
    """A row with no valid calls must not carry a score."""
    df = _load("riskvoice_scores.parquet")
    for c in CONSTRUCTS:
        bad = df[(df["n_calls_valid"] == 0) & df[c].notna()]
        assert bad.empty, f"{len(bad)} rows have a {c} score but zero valid calls"


def test_regimes_count():
    """All N regimes must be present AND materially occupied.

    `n <= N_REGIMES` passes on a degenerate fit that collapsed to one state, which is exactly
    the failure worth catching.
    """
    df = _load("regimes.parquet")
    labels = set(df["regime"].dropna().astype(int))
    assert labels == set(range(config.N_REGIMES)), (
        f"expected regimes {set(range(config.N_REGIMES))}, found {labels} - a regime with no "
        f"assigned days means the fit collapsed")
    share = df["regime"].value_counts(normalize=True)
    thin = share[share < 0.01]
    assert thin.empty, (f"regime(s) {list(thin.index)} hold under 1% of days; the fit has "
                        f"effectively collapsed to fewer regimes")


def test_text_model_is_the_saved_one():
    """regimes.parquet must come from the model that USED the text features."""
    p = config.DATA_PROCESSED / "regimes_base.parquet"
    if not p.exists():
        pytest.skip("regimes_base.parquet not produced yet")
    a = _load("regimes.parquet")
    b = pd.read_parquet(p)
    same = (a["regime"].values == b["regime"].values).all() if len(a) == len(b) else False
    assert not same, (
        "regimes.parquet is identical to the no-text fit. The saved regimes must come from "
        "the model including TEXT_FEATURES, or your constructs are decorative.")


def test_corpus_present():
    n = len(list(config.RBA_MINUTES_DIR.glob("*.html")))
    assert n >= 200, f"expected ~211 RBA minutes, found {n}"


# =============================================================================================
# SUBMISSION MODE
# =============================================================================================
# Everything above SKIPS when a stage has not run yet, which is right while you are building
# and wrong when you are submitting: an empty repository reports "all tests passed".
#
# Run this before you submit:
#
#     pytest tests/ -q --submission
#
# In submission mode a missing artefact is a FAILURE, not a skip.

def test_submission_artefacts_present(request):
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")

    required = {
        "market_data.parquet":      "stage 1 never produced market data",
        "documents.parquet":        "stage 2 never parsed the corpus",
        "riskvoice_scores.parquet": "stage 3 never scored the documents",
        "regimes.parquet":          "stage 4 never fitted the regime model",
        "regimes_3.parquet":        "stage 4 never fitted the 3-regime comparison",
        "regimes_base.parquet":     "stage 4 never fitted the no-text comparison",
        "uncertainty.json":         "stage 5 is still a stub - the three uncertainty layers "
                                    "are a required part of the submission",
    }
    missing = {n: why for n, why in required.items()
               if not (config.DATA_PROCESSED / n).exists()}
    assert not missing, "Missing required outputs:\n" + "\n".join(
        f"  {n}: {why}" for n, why in missing.items())


def test_submission_prompts_written(request):
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")
    from src.stage3_riskvoice import SYSTEM_PROMPT, FIELD_DESCRIPTIONS
    todo = [k for k, v in FIELD_DESCRIPTIONS.items() if "TODO" in v]
    assert "TODO" not in SYSTEM_PROMPT, "SYSTEM_PROMPT is still the placeholder"
    assert not todo, f"field descriptions still placeholders: {todo}"


def test_submission_regimes_named(request):
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")
    placeholders = [v for v in config.REGIME_NAMES.values() if v.startswith("regime_")]
    assert not placeholders, (
        f"regimes are still unnamed placeholders: {placeholders}. Naming them from your own "
        f"output is marked work - see section 6.1 of the brief.")


def test_submission_agreement_work_present(request):
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")
    d = config.DATA_PROCESSED / "agreement"
    labels = list(d.glob("labels_*.csv")) if d.exists() else []
    assert len(labels) >= 3, (
        f"found {len(labels)} blind labelling files in data/processed/agreement/. The fourth "
        f"uncertainty layer needs independent labels from every team member - see "
        f"team-templates/human-agreement-protocol.md")
