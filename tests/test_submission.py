"""
The REPOSITORY-AND-PIPELINE completeness check: does this repository contain the completed
CODE-AND-ARTEFACT portion of the assignment?

    python -m pytest tests/test_submission.py -q -m submission

WHAT IT DOES NOT COVER, deliberately: the AI-use log, the presentation and the peer form
are submitted OUTSIDE this repository, so no test here can see them - passing this suite
does not prove the whole assessment submission is complete, and nothing in the course
materials claims it does. The Cycle ChatGPT transcript, the meeting records and the Shock
source declaration DO live inside the repository (brief S7 matrix), so their PRESENCE is
checked here; their content is assessed by the marker. What the suite proves: no required
stub remains, the judgement tables are filled, every required artefact and repository
record exists, and each artefact's stage hash matches the prompts and judgements in this
repository - so the committed results were produced by the committed configuration.

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
import text_features as tf  # noqa: E402

pytestmark = pytest.mark.submission

ROOT = pathlib.Path(config.__file__).resolve().parent.parent


def _artefact(name: str) -> dict:
    """
    Load an output artefact; SKIP (not fail) when it is absent. Presence is asserted
    once, by test_required_output_exists - a missing shock.json should read as one
    failure plus "blocked by" skips, not a page of cascading tracebacks.
    """
    path = config.OUTPUTS / name
    if not path.exists():
        pytest.skip(f"blocked by missing outputs/{name} - see "
                    f"test_required_output_exists")
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


def test_the_supplied_runners_are_intact():
    """
    The runners are SUPPLIED framework: students fill prompts, judgement tables and
    reviews, and run() validates and executes them. Composing pipeline control flow is
    not assessed work, so a missing or rewritten runner is a framework-integrity defect,
    not an incomplete answer.
    """
    for mod in ("scenarios", "decision_replay"):
        body = _src(mod + ".py")
        supplied = body[body.index("# SUPPLIED BELOW THIS LINE"):]
        assert re.search(r"^def run\(\) -> dict:", supplied, re.M), (
            f"{mod}.run() is missing from the supplied framework region")
        assert "Assemble your own run()" not in body, f"{mod} still carries the old stub"


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
    import hashlib
    import text_features as tf
    stamp = _artefact("words_validation.json")
    assert stamp["config_hash"] == tf.config_hash(), (
        "words_validation.json was produced under different prompts than the ones in this "
        "repository. Either restore the validated prompts or re-validate with --revalidate "
        "and declare the second exposure.")
    assert stamp.get("scores_sha256") == hashlib.sha256(
        config.CONSTRUCT_SCORES_VAL.read_bytes()).hexdigest(), (
        "the validation stamp describes a different validation table than the one on "
        "disk - the held-out scores were modified after the stamped run")


def test_the_panel_rests_on_the_construct_scores_in_this_repository():
    """
    THE GAP THIS CLOSES. The panel's text tier is built FROM construct_scores.parquet,
    but nothing tied the two together afterwards: a team could edit the scores (or the
    panel) and every downstream consumer still certified. The panel stamps the score
    table's content hash at build time; here it must match the file on disk.
    """
    import hashlib
    prov = config.DATA_PROCESSED / "panel.provenance.json"
    assert prov.exists(), (
        "panel.provenance.json is missing - rebuild the panel under the current "
        "framework (python src/run_all.py)")
    rec = json.loads(prov.read_text(encoding="utf-8"))
    assert config.CONSTRUCT_SCORES.exists(), "construct_scores.parquet is missing"
    cur = hashlib.sha256(config.CONSTRUCT_SCORES.read_bytes()).hexdigest()
    assert rec.get("construct_scores_sha256") == cur, (
        "panel.parquet was built from a DIFFERENT construct-score table than the one "
        "in data/processed - re-run the Words stage and rebuild the panel")


def test_words_audit_is_current_and_the_combined_table_is_its_partials():
    """
    THE GAP THIS CLOSES. Only words_validation.json carried a checked config hash, so a
    team could keep old development scores, revalidate under revised prompts, and submit a
    combined table no single configuration ever produced - with a stale audit beside it.
    """
    import pandas as pd
    import text_features as tf
    audit = _artefact("words_audit.json")
    assert audit["config_hash"] == tf.config_hash(), (
        "words_audit.json was produced under different prompts than the ones in this "
        "repository - re-run the Words development scoring")
    dev = pd.read_parquet(config.CONSTRUCT_SCORES_DEV)
    val = pd.read_parquet(config.CONSTRUCT_SCORES_VAL)
    comb = pd.read_parquet(config.CONSTRUCT_SCORES)
    assert list(dev.columns) == list(val.columns) == list(comb.columns), (
        "the three score tables no longer share one schema - extra or missing columns")
    assert not set(dev["meeting_date"]) & set(val["meeting_date"]), (
        "development and validation partials overlap - they must score disjoint meetings")
    for name, df in (("dev", dev), ("validation", val), ("combined", comb)):
        assert df["meeting_date"].is_unique, f"{name}: duplicated meeting dates"
    docs = tf.load_documents()
    held = set(tf.validation_meetings(docs))
    assert set(val["meeting_date"]) == held, (
        "the validation partial does not cover exactly the held-out meetings")
    assert set(dev["meeting_date"]) == set(docs["meeting_date"]) - held, (
        "the development partial does not cover exactly the non-held-out corpus")
    assert (len(dev), len(val), len(comb)) == (len(docs) - len(held), len(held),
                                               len(docs))
    # THE shared contract does the value checks - all seven constructs AND all seven
    # sd columns, finite scores in [0, 1], whole-number n_calls_valid in bounds - the
    # same one the panel, Replay and Shock apply. Reportable: a submitted combined
    # table must cover exactly the authoritative corpus.
    try:
        config.validate_construct_scores(comb, reportable=True)
        for part in (dev, val):
            config.validate_construct_scores(part, reportable=False)
    except RuntimeError as err:
        raise AssertionError(str(err)) from None
    # IDENTITY, not just configuration: each table must be the exact bytes its
    # provenance certifies, so an edited score file - even one edited consistently
    # across all three tables - no longer certifies itself.
    import hashlib
    partial_sha = {}
    for p in (config.CONSTRUCT_SCORES_DEV, config.CONSTRUCT_SCORES_VAL):
        prov = p.with_suffix("").with_suffix(".provenance.json")
        assert prov.exists(), (f"{prov.name} is missing - each score partial must carry "
                               f"its provenance sidecar")
        rec = json.loads(prov.read_text(encoding="utf-8"))
        assert rec["config_hash"] == tf.call_config_hash(), (
            f"{prov.name} was written under a different scoring configuration")
        partial_sha[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert rec.get("scores_sha256") == partial_sha[p.name], (
            f"{p.name} is not the table its provenance sidecar certifies - the score "
            f"file was modified after it was written; re-run the Words stage")
    cprov = config.CONSTRUCT_SCORES.with_suffix("").with_suffix(".provenance.json")
    assert cprov.exists(), ("construct_scores.provenance.json is missing - re-run the "
                            "Words stage under the current framework")
    crec = json.loads(cprov.read_text(encoding="utf-8"))
    assert crec.get("combined_sha256") == hashlib.sha256(
        config.CONSTRUCT_SCORES.read_bytes()).hexdigest(), (
        "construct_scores.parquet is not the table its provenance certifies - it was "
        "modified after the Words stage wrote it")
    assert crec.get("partials") == partial_sha, (
        "the combined provenance names different partials than the ones on disk - the "
        "partial tables changed after the combined table was built")
    assert audit.get("scores_sha256") == partial_sha[config.CONSTRUCT_SCORES_DEV.name], (
        "words_audit.json describes a different development table than the one on "
        "disk - the audit's quality gates no longer certify these scores")
    expect = (pd.concat([dev, val], ignore_index=True)
                .sort_values("meeting_date").reset_index(drop=True))
    got = comb.sort_values("meeting_date").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expect)


def test_replay_statement_file_matches_the_replay_artefact():
    """replay_statement.md must be the statement replay.json describes, not a replacement."""
    rep = _artefact("replay.json")
    md = (config.OUTPUTS / "replay_statement.md").read_text(encoding="utf-8")
    assert str(rep["meeting"]) in md.splitlines()[0], (
        "replay_statement.md is for a different meeting than replay.json")
    parts = md.split("\n---\n")
    assert len(parts) >= 3, "replay_statement.md no longer has its statement block"
    statement = parts[1].strip()
    norm = " ".join(statement.split())
    assert norm and len(norm.split()) == rep["statement_words"]
    import hashlib
    assert hashlib.sha256(norm.encode("utf-8")).hexdigest() \
        == rep.get("statement_sha256"), (
        "the statement in replay_statement.md is not the statement replay.json was "
        "written from - same word count is not identity, and one of the two files was "
        "replaced after the run")
    m = re.search(r"```\n(\{.*?\})\n```", md, re.S)
    assert m, "the arithmetic block is missing from replay_statement.md"
    assert json.loads(m.group(1)) == rep["arithmetic_check"], (
        "the arithmetic check in replay_statement.md differs from replay.json")


def test_shock_probabilities_and_counts_are_coherent():
    """Value-level invariants, not just field presence."""
    sh = _artefact("shock.json")
    for name, res in sh["scenarios"].items():
        for label, rec in res["per_base_row"].items():
            probs = rec.get("probabilities")
            if rec.get("in_support"):
                assert probs, (f"{name}/{label}: an in-support run must carry "
                               f"probabilities")
            else:
                assert not probs, (f"{name}/{label}: an out-of-support run must carry NO "
                                   f"probabilities - the in-support rule is the point")
            if probs:
                vals = list(probs.values())
                assert all(v == v and 0.0 <= v <= 1.0 for v in vals), (name, label, vals)
                assert abs(sum(vals) - 1.0) < 1e-6, (name, label, sum(vals))
        g = res["grounding_counts"]
        for k in ("both", "global_only", "australian_only", "neither", "n_branches"):
            assert isinstance(g.get(k), int) and g[k] >= 0, (name, k, g.get(k))
        assert g["both"] + g["global_only"] + g["australian_only"] + g["neither"] \
            == g["n_branches"], (name, "grounding counts do not partition the branches")


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
# Required repository records - presence is mechanical, content is the marker's
# -------------------------------------------------------------------------------------------

def test_the_cycle_transcript_is_in_the_repository():
    docs = ROOT / "docs"
    found = [p for p in (docs / "cycle-transcript.md", docs / "cycle-transcript.pdf")
             if p.exists() and p.stat().st_size > 0]
    assert found, (
        "docs/cycle-transcript.md (or .pdf) is missing - the submission matrix places the "
        "Cycle ChatGPT transcript inside the repository, and its absence is a -6 raw-stage-"
        "point validity gate")


#: An EXPLICIT route declaration, on its own line, rather than a keyword hunt through
#: prose. "SUBMISSION ROUTE: Moodle" is a statement a team makes on purpose; the word
#: "Moodle" appearing in a sentence is not.
#: The first line of the signpost this repository SHIPS in docs/meetings/. While it is
#: still there, the file is the template rather than a team's statement.
_SHIPPED_SIGNPOST = "# Meeting records go here"

_ROUTE_RE = re.compile(
    r"^\s*(?:submission[ _-]?)?route\s*[:=]\s*(moodle|repository|repo)\b",
    re.I | re.M)


def _declared_route(declaration) -> str | None:
    """"moodle", "repository", or None when the README declares no route.

    The explicit line wins. Failing that - a team working from an earlier version of the
    template, which asked for prose - a README that plainly says the records went to
    Moodle is still honoured. The prose reading is only ever consulted when there are no
    record files to check, so it can no longer contradict the evidence on disk.
    """
    if not declaration.exists():
        return None
    text = declaration.read_text(encoding="utf-8", errors="ignore")
    m = _ROUTE_RE.search(text)
    if m:
        return "moodle" if m.group(1).lower() == "moodle" else "repository"
    # THE SHIPPED SIGNPOST DECLARES NOTHING. It explains both routes, so it necessarily
    # mentions Moodle, transcripts and minutes - and the prose fallback below read that as
    # a team choosing the Moodle route, which let an untouched starter pass the meeting
    # gate by doing nothing at all. A template is not a declaration.
    if _SHIPPED_SIGNPOST in text:
        return None
    low = text.lower()
    if "moodle" in low and any(w in low for w in ("transcript", "minutes", "records")):
        return "moodle"
    return None


def test_meeting_records_are_paired_transcripts_and_minutes():
    """
    The naming convention from the meeting-records template: for every recorded meeting,
    docs/meetings/ holds <YYYY-MM-DD>-transcript.<ext> AND <YYYY-MM-DD>-minutes.<ext>
    (md, txt, pdf or docx; a per-date subfolder with those filenames also counts). At
    least the two mid-term meetings must be present, each fully paired.
    """
    d = ROOT / "docs" / "meetings"
    files = [p for p in d.rglob("*")
             if p.is_file() and p.suffix.lower() in (".md", ".pdf", ".txt", ".docx")] \
        if d.exists() else []

    # RECORD FILES ARE EVIDENCE; THE README IS A ROUTE DECLARATION. Deciding the route by
    # scanning the README for the word "Moodle" got this backwards. The assignment itself
    # requires a mid-term Moodle delivery, so "mid-term transcripts were submitted on
    # Moodle; all records are also stored here" is an ordinary, accurate sentence - and it
    # made a complete, correctly paired private repository FAIL, because the scan read it
    # as choosing the Moodle-only route and then objected to the records being present.
    #
    # So: if the records are here, they are checked, whatever the README says about where
    # else they went. The declaration is consulted only when there is nothing to check.
    records = [p for p in files if "readme" not in p.name.lower()]
    declaration = d / "README.md"
    route = _declared_route(declaration)

    if not records:
        # THE RESTRICTED-MOODLE ROUTE. The brief offers teams a choice: keep identifiable
        # records in the private repository, or submit them to the restricted Moodle item
        # and declare that here. The declaration is a claim, not evidence: it tells the
        # marker where the records are and puts the team on record as having made them.
        # Staff verify receipt on Moodle. A team that declares records it never submitted
        # has made a false statement rather than slipped through a gap.
        assert route == "moodle", (
            "docs/meetings/ is missing or holds no transcript/minutes files - the meeting "
            "records are a condition of assessment and are read for the individual 10. If "
            "your team submitted them to the restricted Moodle item instead, declare the "
            "route on its own line in docs/meetings/README.md:\n"
            "    SUBMISSION ROUTE: Moodle\n"
            "and this check will accept that declaration (see the brief's submission "
            "matrix).")
        return

    kinds: dict[str, set] = {}
    for p in records:
        dm = re.search(r"\d{4}-\d{2}-\d{2}", str(p.relative_to(d)))
        km = re.search(r"transcript|minutes", p.name, re.I)
        if dm and km:
            kinds.setdefault(dm.group(0), set()).add(km.group(0).lower())
    assert len(kinds) >= 2, (
        f"at least two meetings must be recorded; found dated records for "
        f"{sorted(kinds)} - name files <YYYY-MM-DD>-transcript.* and "
        f"<YYYY-MM-DD>-minutes.*")
    unpaired = {k: v for k, v in kinds.items() if v != {"transcript", "minutes"}}
    assert not unpaired, (
        f"every recorded meeting needs BOTH a transcript and minutes; incomplete: "
        f"{ {k: sorted(v) for k, v in unpaired.items()} }")


def test_the_shock_source_declaration_is_valid_and_matches_the_folder():
    """
    The declaration is always checked. The DOCUMENTS are checked when they are present.

    The context documents are not redistributable, so `.gitignore` correctly keeps them
    out of the repository - which means a marker working from a clone does not have them
    and never will. This test therefore has two modes, and says which one it ran:

      WITH the documents   - full bijection between sources.json and the folder, exactly
                             as a team sees it on their own machine;
      WITHOUT them         - the committed fingerprint record must cover exactly the
                             declared files, which is what makes the Shock stage hash
                             reproducible on a clone (see scenarios.write_context_
                             fingerprints).

    A team always runs the first mode, because they downloaded the documents to do the
    stage at all. Neither mode is a relaxation of the other: the fingerprints are written
    from the real files at run time, so a declaration that did not match the folder could
    never have produced a matching record.
    """
    p = ROOT / "data" / "raw" / "context" / "sources.json"
    assert p.exists(), (
        "data/raw/context/sources.json is missing - every Shock context document must be "
        "declared, and Shock grounding cannot be verified without the declaration")
    import re as _re
    import context_docs
    import scenarios as sc
    declared = context_docs._declared_sources()  # the REAL validator: fields, dates,
    assert declared, "sources.json declares nothing"  # URLs, duplicates all checked

    # `source_files()` returns [] rather than raising, so "are the sources here?" is a
    # question this test can ask. `load_documents()` raises FileNotFoundError when they
    # are absent, which is right for the pipeline and made the fallback below
    # unreachable when it was used here.
    present = {p.name for p in context_docs.source_files()}

    assert sc.CONTEXT_FINGERPRINTS.exists(), (
        f"{sc.CONTEXT_FINGERPRINTS.name} is missing. The Shock runner writes it on every "
        f"run; without it a marker working from a clone cannot verify the sources at all, "
        f"because the documents themselves are not redistributable and are gitignored. "
        f"Re-run the Shock stage.")
    rec = json.loads(sc.CONTEXT_FINGERPRINTS.read_text(encoding="utf-8"))
    recorded = rec.get("documents", {})

    # The record must be well formed, and must cover exactly what sources.json declares.
    for name, digest in recorded.items():
        assert isinstance(digest, str) and _re.fullmatch(r"[0-9a-f]{8,64}", digest), (
            f"context fingerprint for {name!r} is not a hex digest: {digest!r}")
    assert set(recorded) == set(declared), (
        f"the committed context fingerprints do not match sources.json; recorded-only: "
        f"{sorted(set(recorded) - set(declared))}, declared-only: "
        f"{sorted(set(declared) - set(recorded))}")

    if not present:
        # A GIT-ONLY CHECKOUT. This is the marker's normal position, not a degraded one:
        # the sources cannot be committed. The declaration and the fingerprint record are
        # verified above; reading the documents to judge whether a quotation is supported
        # is a separate staff action, done from the team's own source pack.
        return

    assert present == set(declared), (
        f"declarations and downloaded documents must match one-to-one; declared-only: "
        f"{sorted(set(declared) - present)}, undeclared: {sorted(present - set(declared))}")
    live = {p.name: sc.file_fingerprint(p) for p in context_docs.source_files()}
    drifted = {n: (recorded[n], live[n]) for n in live if recorded.get(n) != live[n]}
    assert not drifted, (
        f"these context documents differ from the fingerprints the Shock run recorded: "
        f"{sorted(drifted)}. Either the files were replaced after the run, or the "
        f"artefacts predate them; re-run the Shock stage.")


def test_every_stage_completed_under_the_current_configuration():
    """
    THE GAP THIS CLOSES. A team with a valid artefact whose same-configuration re-run
    FAILED still had the old artefact, a matching hash, and a green suite - the failure
    hid behind the previous output. Each supplied runner now records a stage status:
    "running" at invocation, "complete" only after every output is committed.
    """
    import cycle_model
    import decision_replay as dr
    import scenarios as sc
    for stage, hash_fn in (("words", tf.config_hash),
                           ("cycle", cycle_model.config_hash),
                           ("replay", dr.config_hash),
                           ("shock", sc.config_hash)):
        p = config.OUTPUTS / ".stage_status" / f"{stage}.json"
        assert p.exists(), (f"{stage}: no stage status recorded - the supplied runner "
                            f"has not completed this stage")
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec.get("status") == "complete", (
            f"{stage}: the LATEST attempt did not complete (status "
            f"{rec.get('status')!r}) - a failed re-run must not hide behind an older "
            f"artefact; fix the failure and re-run the stage")
        assert rec.get("config_hash") == hash_fn(), (
            f"{stage}: the last completed run used a different configuration than this "
            f"repository holds - re-run the stage")


def test_every_artefact_rests_on_intact_envelopes():
    """
    THE GAP THIS CLOSES. An artefact and its configuration hash could agree while the
    envelopes underneath had been overwritten by a later run - same-configuration
    cross-run mixing. Each supplied runner commits a ledger of every envelope it served
    or wrote, with a SHA-256 per file; every referenced envelope must still exist,
    hash-match, and MATCH the success/failure state recorded in its ledger, with
    failures permitted only for Replay's bounded benchmark attrition (rec-* records).
    The per-entry rule lives in config.validate_envelope_entry so the negative-path
    suite can exercise it directly.
    """
    n_failed_by_stage = {}
    for name in ("words_development", "words_validation", "replay", "shock"):
        p = config.OUTPUTS / ".stage_status" / f"{name}.envelopes.json"
        assert p.exists(), (f"{name}: no envelope ledger - the supplied runner has not "
                            f"completed this stage under the current framework")
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec.get("entries"), f"{name}: the envelope ledger is empty"
        n_failed = 0
        for e in rec["entries"]:
            try:
                config.validate_envelope_entry(name, e)
            except RuntimeError as err:
                raise AssertionError(str(err)) from None
            if not e.get("ok", True):
                n_failed += 1
        n_failed_by_stage[name] = n_failed
    # the ledger's failure count must agree with what replay.json reports: the failed
    # calls shaped the usable-call denominator and the reliability figures
    rep = _artefact("replay.json")
    api_skips = sum(1 for tag in ("dev_sample", "holdout_sample")
                    for r in rep.get(tag, [])
                    if r.get("skipped") and str(r.get("reason", "")).startswith("api:"))
    assert n_failed_by_stage["replay"] == api_skips, (
        f"the replay ledger records {n_failed_by_stage['replay']} failed calls but "
        f"replay.json reports {api_skips} api-skipped benchmark rows - the artefact "
        f"and its cache history disagree")


def test_the_frozen_structured_inputs_match_their_pinned_hashes():
    """
    Defence in depth behind the semantic input contracts: every vendored structured
    input in data/raw is pinned by SHA-256 in data/raw/checksums.json. The contracts
    catch a corrupt or truncated file by its content; this catches ANY byte drift,
    including edits the contracts happen to tolerate.
    """
    import hashlib
    pin_file = config.DATA_RAW / "checksums.json"
    assert pin_file.exists(), "data/raw/checksums.json is missing - restore data/raw"
    pins = json.loads(pin_file.read_text(encoding="utf-8"))
    assert pins, "the checksum pin file is empty"
    for name, expected in sorted(pins.items()):
        f = config.DATA_RAW / name
        assert f.exists(), f"frozen input {name} is missing - restore data/raw"
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        assert got == expected, (
            f"frozen input {name} does not match its pinned hash - the vendored file "
            f"was modified; restore data/raw")


def test_cycle_artefact_matches_the_claims_and_verdicts_in_this_repository():
    import cycle_model
    cy = _artefact("cycle.json")
    assert cy.get("config_hash") == cycle_model.config_hash(), (
        "cycle.json was generated under different claims, verdicts or model-card "
        "evidence than this repository holds - re-run the Cycle stage")


def test_shock_artefacts_share_one_run_id():
    """The three Shock artefacts are only coherent when written by the same run."""
    sh = _artefact("shock.json")
    trees = _artefact("tot_trees.json")
    rp = _artefact("reaction_profiles.json")
    rid = sh.get("run_id")
    assert rid, "shock.json carries no run_id - regenerate the Shock stage"
    assert rp.get("run_id") == rid, (
        "reaction_profiles.json was written by a different run than shock.json")
    for name, t in trees.items():
        assert t.get("run_id") == rid, (
            f"tot_trees.json[{name!r}] was written by a different run than shock.json")


def test_replay_artefact_invariants():
    import decision_replay as dr
    rep = _artefact("replay.json")
    claims = rep["claim_audit"]["claims"]
    ids = [c["claim_id"] for c in claims]
    assert len(ids) == len(set(ids)), "claim ids must be unique"
    counts = rep["claim_audit"].get("counts", {})
    assert sum(counts.values()) == len(claims), (
        "claim-audit category counts do not reconcile with the claims list")
    assert rep["claim_audit"].get("n_reviewed_by_human", 0) >= dr.MIN_CLAIM_REVIEWS
    assert rep.get("recommendation_coherent") is True, (
        "an incoherent recommendation/size pair must not be the reported result")
    assert rep.get("statement_sha256"), "replay.json carries no statement identity"


def test_shock_headline_comes_from_the_declared_base_row():
    import scenarios as sc
    sh = _artefact("shock.json")
    for name, res in sh["scenarios"].items():
        assert res.get("base_row_date") == sc.BASE_ROWS[sc.HEADLINE_BASE], (
            f"{name}: the headline result does not come from the declared "
            f"HEADLINE_BASE row")


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


def test_the_holdout_result_belongs_to_a_frozen_selection_and_a_recorded_exposure():
    """
    THE GAP THIS CLOSES. The suite checked the Replay prompts and the claim reviews, but
    nothing required the selection freeze or the exposure record to exist at all: a probe
    deleted outputs/replay_selection.json, the embedded "selection" block and both A/B
    prompt hashes, and the suite stayed exactly as green as before.

    AND THE GAP THE FIRST ATTEMPT LEFT. It then read only the list embedded in
    replay.json, so deleting the committed log changed nothing, and it demanded that
    EVERY event in that list carry the current selection hash - which rejected the one
    workflow the design exists to support: expose the holdout, declare a revalidation,
    and expose it again under a changed configuration. Older events with older hashes are
    the history. What must match the current freeze is the event THIS REPORT rests on.
    """
    import decision_replay as dr
    rep = _artefact("replay.json")

    stamp = ROOT / "outputs" / dr.SELECTION_STAMP.name
    assert stamp.exists(), (
        f"outputs/{dr.SELECTION_STAMP.name} is missing - the holdout result has no frozen "
        f"selection to belong to. Run:  python src/decision_replay.py --dev")
    selection = json.loads(stamp.read_text(encoding="utf-8"))

    embedded = rep.get("selection") or {}
    assert embedded, "replay.json carries no embedded selection record - regenerate it"
    assert embedded.get("config_hash") == selection.get("config_hash"), (
        "the selection embedded in replay.json is not the one in "
        f"outputs/{dr.SELECTION_STAMP.name} - one of them is stale")
    assert embedded.get("config_hash") == dr.selection_config_hash(), (
        "the frozen selection does not match this repository's prompts, strategy, k, seed "
        "count, draw indices, output ceiling, benchmark samples or panel - the reported "
        "holdout numbers were produced under a different selection")

    exposure = rep.get("exposure") or {}
    assert exposure.get("events") is not None, (
        "replay.json carries no exposure record - regenerate the Replay stage")
    embedded_events = list(exposure["events"])
    declared_prior = bool(selection.get("prior_exposure_declared"))

    # ---- the COMMITTED LOG is the authority, not the copy inside the artefact --------
    # READ WITH THE RUNTIME'S OWN PARSER AND ACCOUNTING. This check used to parse the log
    # itself and rejected ANY damaged line, while the runtime accepted the lines a lost-log
    # declaration covered - so a team that followed the supported recovery could never
    # submit, and was told to restore the file it had just declared unrecoverable. Both now
    # read the same lines the same way, and only damage NO declaration covers is refused.
    log_path = ROOT / "outputs" / dr.EXPOSURE_LOG.name
    committed = dr.parse_exposure_log(log_path) if log_path.exists() else None
    accounting = dr.exposure_accounting(selection, committed or [])
    assert not accounting["conflicts"], (
        f"outputs/{dr.EXPOSURE_LOG.name} records different events under the same id "
        f"{accounting['conflicts']}. Restore it from version control rather than editing "
        f"it.")
    undeclared = accounting["undeclared_damage"]
    assert not undeclared, (
        f"outputs/{dr.EXPOSURE_LOG.name} is damaged, so the number of holdout "
        f"exposures it records cannot be read:\n  "
        + "\n  ".join(f"line {d.get('line_number')}: {d.get('problem')}"
                      for d in undeclared)
        + ("\nA lost-log declaration covers other lines of this file; these are not "
           "among them." if selection.get("log_loss") else "")
        + "\nRestore it from version control rather than editing it. If it is genuinely "
          "unrecoverable, tell the course staff and declare it with --declare-lost-log.")

    if any(e.get("type") == "open" for e in embedded_events):
        assert committed is not None, (
            f"replay.json reports recorded holdout exposures, but "
            f"outputs/{dr.EXPOSURE_LOG.name} is missing. That file is the append-only "
            f"authority the count rests on; the copy inside the artefact is not evidence "
            f"of itself. Commit the log.")
    # WHENEVER THE LOG EXISTS, IT IS THE HISTORY - not only when the report admits to
    # having one. Guarding this comparison on the EMBEDDED list having an open event let
    # a report replace its history with [] and, under a prior-exposure declaration, pass
    # while the committed log held two real exposures. The report has to agree with the
    # file whether it claims exposures or claims none.
    if committed is not None:
        assert committed == embedded_events, (
            f"the exposure history embedded in replay.json is not the history in "
            f"outputs/{dr.EXPOSURE_LOG.name}: the artefact reports "
            f"{len([e for e in embedded_events if e.get('type') == 'open'])} exposure(s) "
            f"and the log records "
            f"{len([e for e in committed if e.get('type') == 'open'])}. One of them has "
            f"been edited; regenerate the Replay stage from the committed log.")

    # THE AUTHORITATIVE HISTORY, from here on. `committed` is the file; the embedded copy
    # has just been required to equal it, so anything derived below rests on the file.
    authoritative = committed if committed is not None else []
    # DISTINCT readable exposures: the same event restored twice is one exposure, not two.
    opened = accounting["readable"]
    # A DECLARED LOSS OF THE LOG is the third way a holdout result can stand: the exposures
    # happened - the freeze stamped them when they did - and their record cannot be
    # produced. It is read from the SELECTION, where it is declared, never from the report.
    log_loss = selection.get("log_loss") or {}
    # EITHER the log records the exposure, OR the team declares that it happened before
    # the log existed. An empty log means one of two opposite things - untouched, or
    # exposed by a process that never recorded it - and the difference has to be stated
    # rather than inferred. What is NOT acceptable is holdout results with neither.
    assert opened or declared_prior or log_loss, (
        "the holdout results are reported, but no holdout exposure was recorded and none "
        "was declared. Either regenerate the stage so the log records what was asked, or "
        "declare an exposure that predates the log:\n"
        '    python src/decision_replay.py --dev --prior-exposure --note="..."')

    # ---- the REPORTED benchmark is bound to ONE exposure: the latest this freeze made --
    known = accounting["known_ids"]
    if known:
        event_id = exposure.get("holdout_event_id")
        assert event_id, (
            "replay.json reports holdout numbers without naming the exposure they came "
            "from. Regenerate the Replay stage so the benchmark carries its event id.")
        current_freeze = selection.get("freeze_id")
        latest = dr.latest_exposure_for(accounting, current_freeze)
        readable = {e.get("event_id"): e for e in opened}
        if event_id in readable:
            event = readable[event_id]
            assert event.get("selection_config_hash") == selection.get("config_hash"), (
                f"the reported holdout benchmark belongs to exposure {event_id} of "
                f"selection {event.get('selection_config_hash')!r}, but the frozen "
                f"selection is {selection.get('config_hash')!r}. Older exposures of older "
                f"configurations are legitimate history; the one this report rests on must "
                f"be the current freeze. Re-run the holdout under the frozen selection.")
            assert event.get("freeze_id") == current_freeze, (
                f"exposure {event_id} was recorded against freeze "
                f"{event.get('freeze_id')!r}, not the current freeze {current_freeze!r}")
            closed = {e.get("event_id") for e in authoritative if e.get("type") == "close"}
            assert event_id in closed, (
                f"exposure {event_id} was never closed, so the benchmark it reports did not "
                f"finish. Re-run the stage to completion before submitting.")
        else:
            # AN EXPOSURE WHOSE RECORD WAS DECLARED LOST AND IS STILL MISSING. This branch
            # used to be reachable only when NO readable exposure remained, so a benchmark
            # honestly bound to a lost exposure was refused whenever any older one survived.
            # Its open and close events cannot be produced; what can be checked is that the
            # declaration covers it and the freeze stamped it.
            assert event_id in accounting["still_missing_declared"], (
                f"holdout_event_id {event_id!r} names no event in the exposure log and is "
                f"not one of the exposures the lost-log declaration covers "
                f"({accounting['still_missing_declared']})")
            stamped_freeze = accounting["freeze_of"].get(event_id)
            assert stamped_freeze in (None, current_freeze), (
                f"exposure {event_id} was stamped against freeze {stamped_freeze!r}, not "
                f"the current freeze {current_freeze!r}")
        # NOT AN OLDER EXPOSURE THAT HAPPENED TO SURVIVE. After a partial loss the producer
        # attached the benchmark to the last exposure still readable, and this check
        # accepted it. A benchmark belongs to the latest exposure its freeze made.
        assert event_id == latest, (
            f"replay.json binds the holdout benchmark to exposure {event_id!r}, but the "
            f"latest exposure this freeze made is {latest!r}. Regenerate the Replay stage: "
            f"a benchmark is bound to the exposure that produced it.")
        # Every exposure must have been authorised - counted as DISTINCT exposures, so a
        # restored record is not a second look.
        authorised = int(selection.get("authorised_exposures", 1))
        assert len(known) <= authorised, (
            f"{len(known)} distinct holdout exposure(s) are known and the freeze "
            f"authorises {authorised}. A further look has to be declared with "
            f"--revalidate.")

    # ---- the status comes from ONE classifier, and this check reads it -------------
    # It used to re-derive the rule here, and the two derivations disagreed: a team that
    # declared a genuine pre-log exposure AND then ran a declared second one was required
    # to report `previously_exposed` (for the declaration) and `retrospective` (for the
    # two logged exposures) at the same time. No value satisfied both, so a legitimate
    # submission could not pass. The producer computes it; this check confirms the
    # artefact agrees with the module rather than inventing a second opinion.
    import decision_replay as dr
    status = exposure.get("evidence_status")
    assert status in dr.EVIDENCE_STATUSES, (
        f"replay.json classifies its holdout evidence as {status!r}, which is not one of "
        f"{list(dr.EVIDENCE_STATUSES)}")
    flags = exposure.get("evidence_flags")
    assert isinstance(flags, dict) and flags.get("status"), (
        "replay.json carries no evidence_flags - regenerate the Replay stage")
    assert flags["status"] == status, (
        f"replay.json's evidence_status ({status!r}) disagrees with its own flags "
        f"({flags['status']!r})")

    # ---- THE FLAGS ARE CHECKED AGAINST THE FILES, NOT AGAINST THEMSELVES ------------
    # This used to run the shared classifier on the numbers the REPORT supplied, so it
    # only ever asked "is this report internally consistent?". It was: a report whose
    # flags claimed one exposure, no prior declaration and no revalidation classified
    # as `prospective`, agreed with itself, and passed - while the frozen selection
    # carried a prior-exposure declaration and the committed log held two exposures it
    # had said nothing about. A summary that contradicts its own evidence is the one
    # thing this check exists to catch, so the facts are now DERIVED from the frozen
    # selection and the committed log and the report is compared against them.
    freeze_id = selection.get("freeze_id")
    mine = [e for e in opened if e.get("freeze_id") == freeze_id] if freeze_id else []
    derived = {
        "logged_exposures": len(opened),
        "exposures_under_this_freeze": len(mine),
        "exposures_under_earlier_freezes": len(opened) - len(mine),
        "prior_exposure_declared": bool(selection.get("prior_exposure_declared")),
        "revalidated": bool(selection.get("revalidated")),
        "exposures_declared_lost": len(accounting["lost_declared"]),
        "log_loss_declared": bool(selection.get("log_loss")),
        "exposures_still_missing": len(accounting["still_missing_declared"]),
        "log_loss_in_effect": accounting["loss_in_effect"],
    }
    # ONE CLASSIFIER STILL - fed verified facts rather than reported ones.
    derived["status"] = dr._classify_evidence(
        derived["logged_exposures"], derived["exposures_under_this_freeze"],
        derived["prior_exposure_declared"], derived["revalidated"], True,
        derived["log_loss_in_effect"])
    # FLAGS ADDED AFTER A REPORT COULD ALREADY EXIST. A replay.json written before lost-log
    # declarations were possible cannot have recorded one, so for these fields - and only
    # these - absence reads as "nothing declared", and is accepted only when the frozen
    # selection and the log agree that nothing was declared. A report that omits them while
    # the evidence holds a declaration still disagrees, and is rejected. Every other flag
    # existed when replay.json gained evidence_flags: its absence is a disagreement like any
    # other, and the fix is to regenerate the report, never to assume a value for it.
    reported = dict(flags)
    for field, not_declared in (("exposures_declared_lost", 0),
                                ("log_loss_declared", False),
                                ("exposures_still_missing", 0),
                                ("log_loss_in_effect", False)):
        if field not in reported and derived[field] == not_declared:
            reported[field] = not_declared
    disagreements = [
        f"{field}: replay.json says {reported.get(field)!r}, the evidence says {value!r}"
        for field, value in derived.items()
        if reported.get(field) != value
    ]
    assert not disagreements, (
        "replay.json's exposure summary contradicts the evidence it summarises "
        f"(outputs/{dr.SELECTION_STAMP.name} and outputs/{dr.EXPOSURE_LOG.name}):\n  "
        + "\n  ".join(disagreements)
        + "\nThe files are the record; the summary is a copy of it. Regenerate the "
          "Replay stage rather than editing the report.")
    assert status == derived["status"], (
        f"replay.json calls its holdout evidence {status!r}, but the frozen selection "
        f"and the committed log make it {derived['status']!r}")

    # ---- and the DECLARATIONS each qualification requires must be present -----------
    # Read from the SELECTION, which is where a declaration is made, rather than from
    # the report's copy of it.
    if derived["prior_exposure_declared"]:
        assert str(selection.get("note", "")).strip(), (
            "a declared prior exposure must say what happened, in words a marker reads")
    if derived["log_loss_declared"]:
        assert str((selection.get("log_loss") or {}).get("note", "")).strip(), (
            "a declared loss of the exposure log must say what happened to it")
    if len(accounting["known_ids"]) > 1 or derived["revalidated"]:
        assert str(selection.get("authorisation", "")).strip(), (
            "a second exposure must carry the declared reason it was authorised for")


def test_the_causal_ab_comparison_used_two_genuinely_different_prompts():
    """
    Guidance that is blank, or identical in both arms, makes the A/B comparison a
    comparison of one thing with itself. The runner rejects that; nothing verified it at
    SUBMISSION time, so an artefact with the prompt identities stripped out passed.
    """
    rep = _artefact("replay.json")
    plain = rep.get("recommendation_plain") or {}
    guided = rep.get("recommendation_guided") or {}
    for name, arm in (("recommendation_plain", plain), ("recommendation_guided", guided)):
        assert arm.get("prompt_sha"), (
            f"replay.json[{name!r}] carries no prompt_sha - the A/B arms cannot be shown "
            f"to have asked different questions. Rerun the Replay stage.")
    assert plain["prompt_sha"] != guided["prompt_sha"], (
        "the guided and unguided recommendations were produced from the SAME prompt, so "
        "the reported comparison is between a thing and itself. Write real causal "
        "guidance and rerun.")


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


#: Evidence the submission matrix REQUIRES, which `.gitignore` must therefore let out.
#: outputs/ is ignored by default with a per-file allowlist, and three required files were
#: missing from it - a team following the instructions would have had them silently
#: dropped by git, and the marker would have read their absence as work not done.
REQUIRED_COMMITTED_EVIDENCE = (
    "outputs/cycle.json",
    "outputs/words_audit.json",
    "outputs/words_validation.json",
    "outputs/words_pilot/0123abcd-4567ef01.json",
    "outputs/replay.json",
    "outputs/replay_statement.md",
    "outputs/replay_selection.json",
    "outputs/replay_exposures.jsonl",
    "outputs/shock.json",
    "outputs/tot_trees.json",
    "outputs/reaction_profiles.json",
    "data/processed/construct_scores.parquet",
)


@pytest.mark.submission
def test_gitignore_lets_every_required_artefact_be_committed(tmp_path):
    """
    Asked of REAL git, not of a regex over the file: the allowlist uses negated patterns
    whose interaction with directory ignores is exactly the part that is easy to get
    wrong, and a hand-rolled matcher would agree with the bug.
    """
    import shutil
    import subprocess

    if shutil.which("git") is None:                      # pragma: no cover
        pytest.skip("git is not on PATH")
    ignore = ROOT / ".gitignore"
    if not ignore.exists():
        pytest.skip("no .gitignore in this repository")

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True,
                   capture_output=True)
    shutil.copyfile(ignore, tmp_path / ".gitignore")
    for rel in REQUIRED_COMMITTED_EVIDENCE:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")

    dropped = [rel for rel in REQUIRED_COMMITTED_EVIDENCE
               if subprocess.run(["git", "check-ignore", "-q", rel], cwd=tmp_path,
                                 capture_output=True).returncode == 0]
    assert not dropped, (
        f".gitignore would silently drop required submission evidence: {dropped}. "
        f"Add a negated pattern for each - the submission check reads these files, and a "
        f"team cannot be marked on evidence git refused to commit.")
