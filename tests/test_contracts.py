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
        "regimes_base.parquet":     "stage 4 never fitted the no-text comparison",
        "model_artifact.json":      "stage 4 never wrote the ordered model artifact that "
                                    "scenarios and explainability read",
        "uncertainty.json":         "stage 5 is still a stub - the three uncertainty layers "
                                    "are a required part of the submission",
        # Sections 8.3, 8.4 and 8.5 carry 45 of the 100 rubric marks between them, and none
        # of them was checked here before. A submission could pass every test with no
        # explainability, no scenarios, no stakeholder work and no dashboard.
        "explainability.json":      "section 8.3 (20 marks) produced no output",
        "historical_calibration.json": "section 8.4 Rule 0 - the historical response function "
                                    "was never computed",
        "scenarios_final.json":     "section 8.4 (15 marks) produced no scenarios",
        "stakeholders.json":        "section 8.5 (10 marks) produced no stakeholder output",
    }
    missing = {n: why for n, why in required.items()
               if not (config.DATA_PROCESSED / n).exists()}
    assert not missing, "Missing required outputs:\n" + "\n".join(
        f"  {n}: {why}" for n, why in missing.items())

    dash = config.OUTPUTS / "dashboard.html"
    assert dash.exists() and dash.stat().st_size > 5000, (
        "outputs/dashboard.html is missing or trivially small. The dashboard is a required "
        "deliverable and carries marks in C1.")


def test_submission_artefact_schemas(request):
    """Present is not the same as populated. An empty JSON file passed the presence check."""
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")
    import json

    # Empty-dict checks passed on {"anything": 1}. These name the keys each artefact must
    # carry, so a file that exists but says nothing fails.
    checks = {
        "model_artifact.json": ["n_regimes", "transition_daily_col_from_row_to", "regimes",
                                "filtered_last", "ordered_to_raw", "features",
                                "endog", "publication_lag_days"],
        "explainability.json": ["perturbation_aggregate", "faithfulness_aggregate",
                                "attribution", "target_construct"],
        "scenarios_final.json": ["selected", "rejected", "pool_diversity",
                                 "ranking_stability"],
        "stakeholders.json": ["personas", "situations", "reactions"],
    }
    for name, keys in checks.items():
        p = config.DATA_PROCESSED / name
        if not p.exists():
            pytest.skip(f"{name} not produced yet")
        obj = json.load(open(p))
        assert obj, f"{name} is empty"
        for k in keys:
            assert k in obj, f"{name} is missing required key '{k}'"

    # Cross-field checks: the shapes have to make sense, not merely exist.
    sc = json.load(open(config.DATA_PROCESSED / "scenarios_final.json"))
    sel = sc["selected"]
    assert len(sel) == 3, f"section 8.4 requires exactly 3 selected scenarios, found {len(sel)}"
    routes = [x.get("transmission_route") for x in sel]
    assert len(set(routes)) == 3, (
        f"the three scenarios use routes {routes}. Rule 3 is a hard requirement: three "
        f"scenarios on fewer than three routes test fewer than three things.")

    ex = json.load(open(config.DATA_PROCESSED / "explainability.json"))
    fa = ex["faithfulness_aggregate"]
    for k in ("mean_delta_named", "mean_delta_unnamed", "mean_gap"):
        assert k in fa, f"explainability.json faithfulness_aggregate is missing '{k}'"
    assert ex["target_construct"], "explainability.json does not say which construct it explains"

    st = json.load(open(config.DATA_PROCESSED / "stakeholders.json"))
    assert len(st["reactions"]) >= 4, (
        f"only {len(st['reactions'])} stakeholder reactions; section 8.5 needs two personas "
        f"across at least three situations")

    art = json.load(open(config.DATA_PROCESSED / "model_artifact.json"))

    # PROVENANCE. Regime count and dependent variable alone do not identify a model: two runs
    # with different features, a different publication lag or a different prompt version are
    # both "3 regimes on log_rv" and are not the same model. A marker must be able to tell
    # which run produced a number.
    for k_ in ("features", "endog", "publication_lag_days", "n_regimes"):
        assert k_ in art, f"model_artifact.json is missing provenance field '{k_}'"
    assert isinstance(art["features"], list) and art["features"], (
        "model_artifact.json records no feature list, so the saved model cannot be identified")
    if art.get("prompt_fingerprint") in (None, ""):
        pytest.fail("model_artifact.json has no prompt_fingerprint - the scores behind this "
                    "model cannot be traced to the prompt version that produced them")

    P = art["transition_daily_col_from_row_to"]
    k = art["n_regimes"]
    assert len(P) == k and all(len(row) == k for row in P), "transition matrix is not k x k"
    sums = [sum(P[r][c] for r in range(k)) for c in range(k)]
    assert all(abs(s - 1.0) < 1e-6 for s in sums), (
        f"transition matrix columns must sum to 1, got {sums}")


def test_submission_config_fingerprint_saved(request):
    """Evidence that the saved outputs came from the settings the report describes.

    The previous provenance test required the text model's HARD regime labels to differ from
    the base model's. That is neither necessary (a good text model can agree with the base
    model on most days) nor sufficient (any perturbation makes labels differ). A recorded
    fingerprint of the settings is the thing that actually establishes provenance.
    """
    if not request.config.getoption("--submission"):
        pytest.skip("development mode; run with --submission to enforce")
    import json
    p = config.DATA_PROCESSED / "model_artifact.json"
    if not p.exists():
        pytest.skip("stage 4 not run")
    art = json.load(open(p))
    assert art["n_regimes"] == config.N_REGIMES, (
        f"model_artifact.json was produced with n_regimes={art['n_regimes']} but config now "
        f"says {config.N_REGIMES}. Re-run stage 4 - your saved outputs and your settings "
        f"disagree, and the report cannot be describing both.")
    assert art["endog"] == config.REGIME_ENDOG, (
        f"artifact endog is {art['endog']}, config says {config.REGIME_ENDOG}. Re-run stage 4.")
    assert art["regime_names"] == [config.REGIME_NAMES[i] for i in range(config.N_REGIMES)], (
        "regime names changed after stage 4 was run. Re-run it so the saved probabilities "
        "carry the names your report uses.")


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
    labels = sorted(d.glob("labels_*.csv")) if d.exists() else []

    # Expected count comes from the TEAM ROSTER, not a hardcoded number.
    roster = _team_roster()
    assert roster, (
        "CONTRIBUTIONS.md has no filled-in roster table, so the number of labelling files "
        "cannot be checked. Fill in the Member column.")
    assert 3 <= len(roster) <= 4, (
        f"CONTRIBUTIONS.md lists {len(roster)} members {sorted(roster)}; teams are 3-4. "
        f"If that table is picking up non-people, fix the table rather than the test.")
    assert len(labels) == len(roster), (
        f"found {len(labels)} labelling files in data/processed/agreement/ but "
        f"CONTRIBUTIONS.md lists {len(roster)} members: {sorted(roster)}. Exactly one file "
        f"per member - see team-templates/human-agreement-protocol.md")

    import pandas as pd
    from src.agreement import VALID_SETS, EXPECTED_PER_SUBSET, LABEL_COLUMNS
    CONSTRUCTS = [c for c in LABEL_COLUMNS
                  if c not in ("meeting_date", "labeller", "set")]

    sets = {}
    for f in labels:
        df = pd.read_csv(f)
        for col in ("meeting_date", "set", *CONSTRUCTS):
            assert col in df.columns, f"{f.name} has no '{col}' column"

        bad = set(df["set"].dropna().astype(str)) - set(VALID_SETS)
        assert not bad, f"{f.name} has invalid set values {sorted(bad)}; expected {VALID_SETS}"

        dates = df["meeting_date"].astype(str)
        assert dates.is_unique, f"{f.name} labels the same document more than once"
        assert len(dates) == 2 * EXPECTED_PER_SUBSET, (
            f"{f.name} has {len(dates)} documents; the protocol specifies "
            f"{EXPECTED_PER_SUBSET} development + {EXPECTED_PER_SUBSET} confirmatory")

        for subset in VALID_SETS:
            n = int((df["set"].astype(str) == subset).sum())
            assert n == EXPECTED_PER_SUBSET, (
                f"{f.name} has {n} '{subset}' documents, expected {EXPECTED_PER_SUBSET}")

        for c in CONSTRUCTS:
            v = pd.to_numeric(df[c], errors="coerce")
            assert v.notna().all(), f"{f.name} has non-numeric or missing values in '{c}'"
            assert v.between(0, 1).all(), (
                f"{f.name} has '{c}' values outside [0, 1]: "
                f"min {v.min()}, max {v.max()}")

        sets[f.name] = set(zip(dates, df["set"].astype(str)))

    # COMPLETE design: everyone labels the same documents, in the same halves. An incomplete
    # matrix makes ICC(2,1) inapplicable, and a document in different halves for different
    # raters silently contaminates the confirmatory set.
    if len(sets) > 1:
        common = set.intersection(*sets.values())
        for name, sset in sets.items():
            missing, extra = common - sset, sset - common
            assert not missing, (
                f"{name} is missing {len(missing)} (document, set) pairs that others have, "
                f"e.g. {sorted(missing)[:3]}. The design is COMPLETE and the development/"
                f"confirmatory split must be identical for every rater.")
            assert not extra, (
                f"{name} has {len(extra)} (document, set) pairs nobody else does, e.g. "
                f"{sorted(extra)[:3]}.")


def _team_roster() -> set[str]:
    """Team members, read from the FIRST table of CONTRIBUTIONS.md.

    Parses only the roster table - the one whose header starts with "Member". An earlier
    version matched any table row, so the shared-work table below it contributed "Report",
    "Presentation" and "Scenario generation" as team members, and the agreement test then
    demanded a labelling file from each of them.

    Returns an empty set if no roster is filled in; the caller treats that as a failure
    rather than defaulting to a number, so this cannot pass vacuously.
    """
    p = config.ROOT / "CONTRIBUTIONS.md"
    if not p.exists():
        return set()
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    # Find the header row of the roster table, then read until the table ends.
    start = None
    for i, line in enumerate(lines):
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in {"member", "name"}:
            start = i + 2                      # skip header and the |---| separator
            break
    if start is None:
        return set()

    names = set()
    for line in lines[start:]:
        if not line.strip().startswith("|"):
            break                              # table ended
        first = line.strip().strip("|").split("|")[0].strip().strip("*")
        if not first:
            continue                           # blank template row
        if first.lower() in {"member", "name", "---"} or set(first) <= set("-: "):
            continue
        names.add(first)
    return names
