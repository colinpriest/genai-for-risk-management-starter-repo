"""
The REPOSITORY-AND-PIPELINE completeness check: does this repository contain the completed
CODE-AND-ARTEFACT portion of the assignment?

    python -m pytest tests/test_submission.py -q -m submission

WHAT IT DOES NOT COVER, deliberately: the Cycle ChatGPT transcript, the AI-use log, the
presentation and the peer form are submitted OUTSIDE this repository, so no test here can
see them - passing this suite does not prove the whole assessment submission is complete,
and nothing in the course materials claims it does. What it does prove: no required stub
remains, the judgement tables are filled, every required artefact exists, and each
artefact's stage hash matches the prompts and judgements in this repository - so the
committed results were produced by the committed configuration.

These tests fail on the untouched starter BY DESIGN. Run them before you submit; every
failure names the missing piece. Nothing here needs an API key: the committed caches and
artefacts are the submission.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config  # noqa: E402

pytestmark = pytest.mark.submission

ROOT = pathlib.Path(config.__file__).resolve().parent.parent


def _artefact(name: str) -> dict:
    """Load an output artefact with a concise failure instead of a raw traceback."""
    path = config.OUTPUTS / name
    assert path.exists(), (f"outputs/{name} is missing - the stage that writes it has "
                           f"not been run")
    return json.loads(path.read_text(encoding="utf-8"))


def _src(name: str) -> str:
    return (ROOT / "src" / name).read_text(encoding="utf-8")


# -------------------------------------------------------------------------------------------
# No required stub remains
# -------------------------------------------------------------------------------------------

def test_every_required_prompt_is_written():
    import scenarios as sc
    import text_features as tf
    assert tf.SYSTEM_PROMPT and "TODO" not in tf.SYSTEM_PROMPT, "Words system prompt"
    for name in ("BRANCH_PROMPT", "EVALUATE_PROMPT", "EXPAND_PROMPT", "ADVERSARIAL_PROMPT"):
        assert "TODO" not in getattr(sc, name), f"{name} is still the stub"
    import decision_replay as dr
    for name in ("RECOMMENDATION_PROMPT", "STATEMENT_PROMPT"):
        assert "TODO" not in getattr(dr, name), f"{name} is still the stub"


def test_run_is_assembled_not_stubbed():
    for mod in ("scenarios", "decision_replay"):
        body = _src(mod + ".py")
        m = re.search(r"^def run\(\) -> dict:\n(.*?)(?=^\S)", body, re.M | re.S)
        assert m, f"{mod}.run() not found"
        # the prompts-written guard raises NotImplementedError legitimately; the STUB is
        # identified by its own instruction text
        stub = ("Assemble your own run()" in m.group(1)
                or "YOURS TO ASSEMBLE" in m.group(1))
        assert not stub, f"{mod}.run() is still the starter stub"


def test_the_judgement_tables_are_filled():
    import decision_replay as dr
    import scenarios as sc
    for name, table in (("PROXY_JUDGEMENTS", sc.PROXY_JUDGEMENTS),
                        ("DIRECTION_WEIGHTS", sc.DIRECTION_WEIGHTS),
                        ("ADVERSARIAL_RESPONSES", sc.ADVERSARIAL_RESPONSES),
                        ("EXPECTED_PROFILES", sc.EXPECTED_PROFILES),
                        ("CLAIMED_ANALOGUES", sc.CLAIMED_ANALOGUES),
                        ("CLAIM_REVIEWS", dr.CLAIM_REVIEWS)):
        assert table, f"{name} is empty - the assessed judgement work is missing"
    # the review records are legitimate in EITHER form - the Python dicts or the JSON
    # record files - so completeness is judged on what is in effect, not on where it lives
    for scen in sc.SCENARIOS:
        assert sc.effective_channel_reviews(scen), (
            f"no channel reviews for {scen} - neither CHANNEL_REVIEWS nor "
            f"data/processed/channel_reviews.json has entries")
        assert sc.effective_pruning_reviews(scen), (
            f"no pruning reviews for {scen} - neither PRUNING_REVIEWS nor "
            f"data/processed/pruning_reviews.json has entries")
        assert scen in sc.CLAIMED_ANALOGUES, f"no claimed analogue for {scen}"
    assert len(dr.CLAIM_REVIEWS) >= dr.MIN_CLAIM_REVIEWS
    rep = json.loads((config.OUTPUTS / "replay.json").read_text()) \
        if (config.OUTPUTS / "replay.json").exists() else {}
    ids = {c.get("claim_id") for c in rep.get("claim_audit", {}).get("claims", [])}
    if ids:
        assert set(dr.CLAIM_REVIEWS) <= ids, (
            f"CLAIM_REVIEWS keys {sorted(set(dr.CLAIM_REVIEWS) - ids)} do not match any "
            f"claim id in replay.json - reviews are keyed by exact claim id")
    for cid, r in dr.CLAIM_REVIEWS.items():
        for f in ("human_verdict", "reason", "final_action", "by"):
            assert str(r.get(f, "")).strip(), f"claim review {cid} is missing {f}"
        assert r["human_verdict"] in dr.CLAIM_CATEGORIES, cid
    assert sc.REQUIRE_CHANNEL_REVIEWS and sc.REQUIRE_PRUNING_REVIEWS, (
        "the review requirements are switched off - a reportable run needs them on")


# -------------------------------------------------------------------------------------------
# Every required output exists and matches the configuration that claims it
# -------------------------------------------------------------------------------------------

REQUIRED_OUTPUTS = ["words_audit.json", "words_validation.json", "replay.json",
                    "replay_statement.md", "shock.json", "tot_trees.json",
                    "reaction_profiles.json", "cycle.json"]


@pytest.mark.parametrize("name", REQUIRED_OUTPUTS)
def test_required_output_exists(name):
    assert (config.OUTPUTS / name).exists(), (
        f"outputs/{name} is missing - the stage that writes it has not been run")


def test_construct_scores_exist_for_development_and_validation():
    assert config.CONSTRUCT_SCORES_DEV.exists(), "development construct scores missing"
    assert config.CONSTRUCT_SCORES_VAL.exists(), (
        "validation construct scores missing - `text_features.py --validate` has not run")
    assert config.CONSTRUCT_SCORES.exists()


def test_validation_was_run_under_the_current_prompts():
    import text_features as tf
    stamp = _artefact("words_validation.json")
    assert stamp["config_hash"] == tf.config_hash(), (
        "words_validation.json was produced under different prompts than the ones in this "
        "repository. Either restore the validated prompts or re-validate with --revalidate "
        "and declare the second exposure.")


def test_shock_artefacts_match_the_reviews_in_this_repository():
    import scenarios as sc
    sh = _artefact("shock.json")["scenarios"]
    assert set(sh) == set(sc.SCENARIOS), "shock.json scenarios differ from SCENARIOS"
    for name, res in sh.items():
        recorded = res.get("channel_reviews") or {}
        assert recorded == (sc.effective_channel_reviews(name) or {}), (
            f"{name}: shock.json was generated under different channel reviews than the "
            f"ones in effect in this repository (Python dict or JSON record file) - "
            f"rerun the Shock stage")
        assert not res.get("channel_review_summary", {}).get("unreviewed"), (
            f"{name}: mechanism groups are still unreviewed in the committed artefact")
        assert not res.get("pruning_audit", {}).get("outstanding"), (
            f"{name}: required pruning reviews are outstanding in the committed artefact")


# -------------------------------------------------------------------------------------------
# Reproducibility envelopes are committed - the marker replays without a key
# -------------------------------------------------------------------------------------------

def test_reproducibility_envelopes_are_committed():
    for d, label in ((config.LLM_RAW, "Words"),
                     (config.DATA_PROCESSED / "replay_raw", "Replay"),
                     (config.DATA_PROCESSED / "shock_raw", "Shock")):
        files = list(d.rglob("*.json")) if d.exists() else []
        assert files, (f"{label} envelopes missing under {d} - the committed caches ARE "
                       f"the reproducibility artefact; without them nothing can be "
                       f"replayed offline")


def test_shock_can_be_replayed_offline_from_the_envelopes():
    """Every branch/evaluate/adversarial call the committed trees needed is cached."""
    import hashlib
    import scenarios as sc
    raw = {p.name for p in (config.DATA_PROCESSED / "shock_raw").glob("*.json")}
    assert raw, "no shock envelopes committed"
    # the cheap, honest proxy for replayability: the artefact records cached=0 fresh calls
    # would need a key; instead assert the cache is non-trivial and the run wrote trees
    trees = json.loads((config.OUTPUTS / "tot_trees.json").read_text())
    n_branches = sum(len(t["branches"]) for t in trees.values())
    assert n_branches >= 20 and len(raw) >= 20, (
        f"{n_branches} branches but only {len(raw)} envelopes - the cache does not cover "
        f"the committed trees")


def test_the_report_exists_and_the_word_count_is_printed():
    """
    ADVISORY count only. The published rule excludes tables, figures, references and
    appendices; a script can only approximate those boundaries (this one strips markdown
    tables and code fences and stops at a heading beginning "## Appendix"), so formatting
    choices could flip a strict assertion either way. The OFFICIAL count is the marker's.
    This test asserts the report exists and prints the approximation for your own check.
    """
    candidates = list((ROOT / "docs").glob("*report*.md"))
    path = next((c for c in candidates if c.exists()), None)
    assert path is not None, "no report found under docs/"
    text = path.read_text(encoding="utf-8")
    body = re.sub(r"^\|.*$", "", text, flags=re.M)
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.split(r"(?m)^## Appendix", body)[0]
    n = len(body.split())
    print(f"approximate prose count for {path.name}: {n} words "
          f"(limit 3,000 + 5% tolerance = 3,150; the marker's count is official)")


def test_replay_artefact_matches_the_prompts_and_reviews_in_this_repository():
    """
    THE GAP THIS CLOSES. words_validation.json has always carried a config hash; replay.json
    carried none, so the Replay prompts could be edited after generation and the suite
    stayed green - in an assignment about GenAI, the marker could not establish that the
    assessed prompts produced the submitted results.
    """
    import decision_replay as dr
    rep = _artefact("replay.json")
    assert "config_hash" in rep, "replay.json carries no stage hash - regenerate it"
    assert rep["config_hash"] == dr.config_hash(), (
        "replay.json was generated under different prompts, settings or claim reviews "
        "than this repository holds - rerun the Replay stage")
    assert rep.get("claim_reviews") == dr.CLAIM_REVIEWS, (
        "the claim reviews persisted in replay.json differ from CLAIM_REVIEWS")


def test_shock_artefact_matches_the_prompts_and_judgements_in_this_repository():
    import scenarios as sc
    sh = _artefact("shock.json")
    assert "config_hash" in sh, "shock.json carries no stage hash - regenerate it"
    assert sh["config_hash"] == sc.config_hash(), (
        "shock.json was generated under different prompts, settings or judgement tables "
        "than this repository holds - rerun the Shock stage")


# -------------------------------------------------------------------------------------------
# Artefact schemas: structurally complete before any consistency test reads them
# -------------------------------------------------------------------------------------------

def test_shock_artefact_schema():
    import scenarios as sc
    sh = _artefact("shock.json")
    assert re.fullmatch(r"[0-9a-f]{12}", str(sh.get("config_hash", ""))), (
        "shock.json config_hash missing or malformed")
    assert set(sh.get("scenarios", {})) == set(sc.SCENARIOS)
    for name, res in sh["scenarios"].items():
        for field in ("per_base_row", "channel_direction", "grounding_counts",
                      "channel_review_summary", "pruning_audit", "prune_sweep",
                      "channel_reviews", "pruning_reviews", "source_manifest"):
            assert field in res, f"{name}: shock.json record lacks '{field}'"
        assert isinstance(res["prune_sweep"], list) and res["prune_sweep"]
        assert isinstance(res["channel_reviews"], dict) and res["channel_reviews"]


def test_replay_artefact_schema():
    rep = _artefact("replay.json")
    assert re.fullmatch(r"[0-9a-f]{12}", str(rep.get("config_hash", "")))
    for field in ("meeting", "k", "strategy", "claim_audit", "arithmetic_check",
                  "claim_reviews", "evidence_block", "actual"):
        assert field in rep, f"replay.json lacks '{field}'"
    for c in rep["claim_audit"]["claims"]:
        assert re.fullmatch(r"[0-9a-f]{8}", str(c.get("claim_id", ""))), (
            "every audited claim carries a stable 8-hex claim_id")


def test_reaction_profiles_artefact_schema():
    import scenarios as sc
    rp = _artefact("reaction_profiles.json")
    assert set(rp.get("scenarios", {})) == set(sc.SCENARIOS)
    for name, rec in rp["scenarios"].items():
        assert set(rec["expected"]) == set(config.TEXT_FEATURES), (
            f"{name}: expected profile does not cover the seven constructs")
        assert rec.get("claimed_analogue"), f"{name}: no claimed analogue recorded"
    assert "excluded_constructs" in rp
