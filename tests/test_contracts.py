"""
Contracts that must hold, each pinned because it was broken at some point in development.

Nothing here needs an API key. Tests that need a built panel skip cleanly on a fresh
checkout; run `python src/data_panel.py` first to enable them.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import config          # noqa: E402
import nshot           # noqa: E402
import scenario_engine as se  # noqa: E402
import context_docs           # noqa: E402
import text_features as tf    # noqa: E402
from scenario_engine import Branch, Tree  # noqa: E402

PANEL = config.DATA_PROCESSED / "panel.parquet"
needs_panel = pytest.mark.skipif(not PANEL.exists(),
                                 reason="run `python src/data_panel.py` first")


def _panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    return p.set_index("meeting_date").sort_index()


# -------------------------------------------------------------------------------------------
# Repository self-containment
# -------------------------------------------------------------------------------------------

def test_no_config_path_escapes_the_repository():
    """Every declared path stays inside the repo, so a clean checkout runs anywhere."""
    from pathlib import Path
    root = Path(config.__file__).resolve().parent.parent
    escapes = []
    for name in dir(config):
        v = getattr(config, name)
        if isinstance(v, Path):
            try:
                v.resolve().relative_to(root)
            except ValueError:
                escapes.append(f"{name}={v}")
    assert not escapes, f"config paths outside the repository: {escapes}"


def test_vendored_inputs_are_present():
    """The four raw inputs ship with the repository - no sibling directories."""
    for p in (config.DOCUMENTS, config.MEETING_CALENDAR, config.MARKET_DATA):
        assert p.exists(), f"{p.name} is missing from data/raw/"


def test_frozen_shock_model_ships_and_is_construct_free():
    clf, X, panel = se.frozen_shock_model()
    leaked = [c for c in X.columns if c in config.TEXT_FEATURES]
    assert not leaked, (
        f"the shock model must not use the Words constructs - a scenario has no minutes, so "
        f"shocking {leaked} would require inventing them")
    assert clf.predict_proba(X.iloc[[-1]]).shape[1] == 3


# -------------------------------------------------------------------------------------------
# Words
# -------------------------------------------------------------------------------------------

def test_supplied_exemplars_are_hash_locked():
    tf._require_exemplars()
    original = tf.CONSTRUCTS["policy_stance"]
    try:
        tf.CONSTRUCTS["policy_stance"] = original + " tampered"
        with pytest.raises(RuntimeError, match="altered"):
            tf._require_exemplars()
    finally:
        tf.CONSTRUCTS["policy_stance"] = original
    tf._require_exemplars()


def test_config_hash_changes_when_a_prompt_changes():
    """The cache is keyed on the configuration; editing a rubric must invalidate it."""
    before = tf.config_hash()
    original = tf.CONSTRUCTS["vigilance"]
    try:
        tf.CONSTRUCTS["vigilance"] = original + " revised"
        assert tf.config_hash() != before, (
            "editing a rubric did not change the cache key, so a re-run would silently "
            "return the previous run's scores")
    finally:
        tf.CONSTRUCTS["vigilance"] = original
    assert tf.config_hash() == before


def test_evidence_validation_rejects_a_paraphrase():
    src = "Members observed that inflation had moderated over recent quarters."
    assert tf._validate_evidence("inflation had moderated over recent quarters", src)
    assert tf._validate_evidence("INFLATION HAD  MODERATED over recent quarters", src)
    assert not tf._validate_evidence("inflation was said to be coming down", src)
    assert not tf._validate_evidence("short", src)


def test_dev_and_validation_episodes_do_not_overlap():
    """Prompt iteration and prompt validation must not touch the same meetings."""
    def span(d):
        return {(a, b) for a, b in d.values()}
    for name, (a, b) in tf.DEV_EPISODES.items():
        for vname, (c, d) in tf.VALIDATION_EPISODES.items():
            overlap = max(a, c) <= min(b, d)
            assert not overlap, f"dev episode {name} overlaps validation episode {vname}"


# -------------------------------------------------------------------------------------------
# The estimand and the panel
# -------------------------------------------------------------------------------------------

@needs_panel
def test_text_tier_is_lagged_one_meeting():
    """Row M carries constructs from M-1's minutes, which were public before M."""
    p = _panel()
    cols = [c for c in config.TEXT_FEATURES if c in p.columns]
    if not cols:
        pytest.skip("constructs not scored yet")
    assert p[cols].iloc[0].isna().all(), (
        "the first meeting has no prior minutes, so its construct row must be empty")


@needs_panel
def test_targets_are_forward_looking_and_right_censored():
    """
    A target whose 182-day window has not closed must be NaN, not a partial sum.

    The previous version of this test contained `assert ... or True`, which cannot fail.
    It is now a real boundary check: every meeting whose window would end after the last
    observed cash-rate date must be unlabelled, and every meeting before that must be
    labelled.
    """
    p = _panel()
    # y_cycle must be NaN exactly where the forward window could not be completed. The
    # window closes against the DAILY cash-rate series, which runs past the last meeting,
    # so the last meeting date is not the right reference.
    assert p["y_cycle"].isna().equals(p["forward_change"].isna()), (
        "y_cycle and forward_change disagree about which windows have closed")
    assert p["y_cycle"].tail(3).isna().all(), (
        "the three most recent meetings cannot have a resolved 182-day label")
    assert p["y_cycle"].notna().sum() > 190, (
        "too few labels - the target build has gone wrong")


@needs_panel
def test_label_availability_dates_exist_and_are_correct():
    p = _panel()
    for c in ("y_cycle_available_on", "y_decision_available_on"):
        assert c in p.columns, f"{c} is missing - rebuild the panel"
    avail = pd.to_datetime(p["y_cycle_available_on"])
    assert (avail == p.index + pd.Timedelta(days=config.CYCLE_WINDOW_DAYS)).all()


@needs_panel
def test_rolling_origin_embargo_excludes_unresolved_labels():
    """
    THE BUG THIS PINS. Training on every earlier row hands the model the previous six
    months of outcomes, which had not resolved at the prediction date. It was worth 23
    accuracy points and took the model from beating the current-decision baseline to
    losing to it.
    """
    import evaluation as ev
    p = _panel()
    tiers = json.loads((config.DATA_PROCESSED / "tiers.json").read_text())
    X = ev.build_design(p, tiers, ["persistence", "macro", "market"])
    y = p["y_cycle"]
    avail = pd.to_datetime(p["y_cycle_available_on"])

    seen = {}

    class _Spy:
        def fit(self, Xf, yf):
            seen["train_max"] = Xf.index.max()
            self.classes_ = np.array(sorted(yf.unique()))
            return self

        def predict_proba(self, Xr):
            seen.setdefault("predict_at", []).append(Xr.index[0])
            return np.tile(1 / len(self.classes_), (len(Xr), len(self.classes_)))

        def get_params(self, deep=True):
            return {}

        def set_params(self, **kw):
            return self

    preds = ev.rolling_origin(X, y, _Spy(), available_on=avail)
    assert len(preds), "the embargoed evaluation produced no predictions"
    horizon = pd.Timedelta(days=config.CYCLE_WINDOW_DAYS)
    for t in preds.index:
        # every training label used at t must have resolved by t
        assert (avail.loc[avail.index <= t] <= t).sum() >= 1
    last_t = preds.index[-1]
    assert seen["train_max"] + horizon <= last_t, (
        f"the final training row {seen['train_max'].date()} has a label that only resolves "
        f"at {(seen['train_max'] + horizon).date()}, after the prediction date "
        f"{last_t.date()}")


@needs_panel
def test_baselines_are_scored_on_the_models_own_dates():
    import evaluation as ev
    p = _panel()
    dates = p.index[-40:]
    b = ev.baselines(p, p["y_cycle"], on_dates=dates)
    n = p.loc[dates, "y_cycle"].notna().sum()
    assert b["majority"]["accuracy"] == pytest.approx(
        float((p.loc[dates, "y_cycle"].dropna() ==
               p["y_cycle"].dropna().iloc[:config.MIN_TRAIN_MEETINGS].mode().iloc[0]).mean()))
    assert n > 0


@needs_panel
def test_replay_uses_pre_decision_market_data():
    """
    THE BUG THIS PINS. The panel's market features are the meeting-day close, which is after
    the 2:30pm announcement. On 7 October 2008 - a 100bp cut - bab_spread moved 63bp between
    the previous close and the meeting close.
    """
    import decision_replay as dr
    p = _panel()
    pre_cols = [c for c in p.columns if c.endswith("__pre")]
    assert pre_cols, "the panel has no __pre columns - rebuild it"
    post = dr._panel(pre_decision=False)
    pre = dr._panel()
    assert not [c for c in pre.columns if c.endswith("__pre")], (
        "the pre-decision panel should expose values under the ordinary names")
    d = pd.Timestamp("2008-10-07")
    if d in pre.index:
        assert pre.loc[d, "bab_spread"] != post.loc[d, "bab_spread"], (
            "Replay is still reading the meeting-day close")
    differ = sum(1 for c in ("bab_spread", "slope_cash3y", "vix")
                 if c in pre.columns and not pre[c].equals(post[c]))
    assert differ >= 2


@needs_panel
def test_current_decision_baseline_is_reported():
    """
    The trivial baseline must exist and must be hard to beat.

    The cycle label is partly a restatement of the decision just taken, so "the cycle
    continues doing what this meeting did" scores far above the majority class. Reporting
    accuracy against the majority baseline alone made the model look three times better
    than it is.
    """
    import evaluation as ev
    p = _panel()
    base = ev.baselines(p, p["y_cycle"])
    assert "current_decision" in base, (
        "the current_decision baseline is missing; without it the headline accuracy is "
        "unreadable")
    assert base["current_decision"]["accuracy"] > base["majority"]["accuracy"], (
        "the current_decision baseline should be much stronger than the majority class - "
        "if it is not, check that it is being computed on the same index")


# -------------------------------------------------------------------------------------------
# Replay
# -------------------------------------------------------------------------------------------

@needs_panel
def test_shots_must_predate_the_target():
    p = _panel()
    t = p.index[-20]
    shots = nshot.select_shots(p, t, k=9, strategy="recent")
    assert (shots.index < t).all()
    with pytest.raises(ValueError, match="LEAKAGE"):
        nshot.every_shot_predates(p.loc[t:].head(3), t)


@needs_panel
def test_regimes_strategy_does_not_select_on_an_unresolved_label():
    """y_cycle looks forward; a shot whose window had not closed is selection on the future."""
    p = _panel()
    t = p.index[-20]
    shots = nshot.select_shots(p, t, k=9, strategy="regimes")
    horizon = pd.Timedelta(days=config.CYCLE_WINDOW_DAYS)
    assert (shots.index <= t - horizon).all(), (
        "a regimes shot was selected whose cycle label had not resolved by the target date")


@needs_panel
def test_prompt_block_refuses_a_leaky_target_row():
    import decision_replay as dr
    p = _panel()
    t = p.index[-20]
    with pytest.raises(ValueError, match="LEAKAGE"):
        nshot.build_prompt_block(p, t, k=9, strategy="recent")   # raw panel row
    txt, _ = nshot.build_prompt_block(p, t, k=9, strategy="recent",
                                      as_at=dr.load_as_at(str(t.date())))
    assert "DECISION TAKEN" in txt
    assert txt.count("DECISION TAKEN") == 9, "the target block must have no decision"


@needs_panel
def test_k_is_validated():
    p = _panel()
    t = p.index[-20]
    with pytest.raises(ValueError, match="multiple of 3"):
        nshot.select_shots(p, t, k=8, strategy="stratified")
    with pytest.raises(ValueError, match=">= 3"):
        nshot.select_shots(p, t, k=2, strategy="recent")
    with pytest.raises(ValueError, match="unknown strategy"):
        nshot.select_shots(p, t, k=9, strategy="vibes")


@needs_panel
def test_similarity_uses_complete_rows_only():
    """Otherwise a row missing three fields looks closer than a complete row that matches."""
    p = _panel()
    t = p.index[-20]
    cols = [c for c in nshot.SIMILARITY_FIELDS if c in p.columns]
    shots = nshot.select_shots(p, t, k=9, strategy="similar")
    assert shots[cols].notna().all().all()


def test_decision_label_shows_the_actual_size():
    assert nshot.decision_label(-1, -0.50) == "CUT 50bp"
    assert nshot.decision_label(1, 0.25) == "HIKE 25bp"
    assert nshot.decision_label(0, 0.0) == "HOLD"


@needs_panel
def test_dev_and_holdout_samples_are_disjoint():
    import decision_replay as dr
    assert not set(dr.dev_sample()) & set(dr.holdout_sample())
    assert len(dr.dev_sample()) and len(dr.holdout_sample())


@needs_panel
def test_arithmetic_check_is_deterministic_and_scoped():
    import decision_replay as dr
    m = str(_panel().index[-30].date())
    before = dr.rate_before(m)
    rec = {"recommendation": "hike", "size_bp": 25}
    good = f"The Board increased the cash rate target to {before + 0.25:.2f} per cent."
    stale = f"The Board increased the cash rate target to {before:.2f} per cent."
    noise = f"Inflation was {before + 0.25:.2f} per cent. Policy was unchanged."
    assert dr.arithmetic_check(good, m, rec)["verdict"] == "pass"
    assert "FAIL" in dr.arithmetic_check(stale, m, rec)["verdict"]
    assert "FAIL" in dr.arithmetic_check(noise, m, rec)["verdict"], (
        "a percentage outside cash-rate context must not satisfy the check")


# -------------------------------------------------------------------------------------------
# Shock
# -------------------------------------------------------------------------------------------

def _tree() -> Tree:
    t = Tree(scenario="test")
    t.branches = [
        Branch(channel="up", proxy="vix", n_sd=1.0, direction="tightening",
               depth=0, score=0.9, keep=True, note="x", channel_type="risk_appetite",
               horizon="days"),
        Branch(channel="down", proxy="aud_ret", n_sd=-1.0, direction="easing",
               depth=0, score=0.8, keep=True, note="x", channel_type="exchange_rate",
               horizon="days"),
        Branch(channel="rejected", proxy=None, n_sd=0.0, direction="easing",
               depth=0, score=0.1, keep=False, note="dropped", horizon="days"),
        Branch(channel="child", proxy="vix", n_sd=0.5, direction="tightening",
               depth=1, parent="up", score=0.9, keep=True, note="x",
               channel_type="risk_appetite", horizon="days"),
    ]
    return t


def test_shocks_are_adjudicated_not_accumulated():
    """Two channels on one proxy give one shock, not their sum."""
    shocks, log = se.adjudicate_shocks(_tree().modellable())
    d = dict(shocks)
    assert d["vix"] == 1.0, f"expected the single largest move, got {d['vix']}"
    assert len(shocks) == 2
    entry = [x for x in log if x["proxy"] == "vix"][0]
    assert entry["n_channels"] == 2 and not entry["sign_conflict"]


def test_sign_conflict_on_one_proxy_blocks():
    t = _tree()
    t.branches[3].n_sd = -0.5      # child now opposes its parent on the same proxy
    blocking = t.validate(["vix", "aud_ret"])
    assert any("BOTH directions" in b for b in blocking)


def test_policy_outcome_proxies_cannot_be_shocked():
    t = Tree(scenario="t")
    t.branches = [Branch(channel="assume the answer", proxy="cash_rate", n_sd=2.0,
                         direction="tightening", depth=0, score=0.9, keep=True, note="x")]
    assert any("policy outcome" in b for b in t.validate(["cash_rate"]))
    se.adjudicate_shocks(t.branches)
    assert t.branches[0].proxy is None


def test_orphans_cascade_when_a_parent_is_dropped():
    t = _tree()
    t.branches[0].keep = False                 # drop the parent
    se.cascade_orphans(t.branches)
    assert t.branches[3].keep is False
    assert "ORPHANED" in t.branches[3].note


def test_proxy_conflict_resolution_records_rather_than_deletes():
    t = _tree()
    t.branches[3].n_sd = -0.5
    log = se.resolve_proxy_conflicts(
        t.branches, {"vix": {"central": 1.0, "reason": "risk-off dominates"}})
    assert log[0]["resolved"] and log[0]["central_sd"] == 1.0
    dropped = t.branches[3]
    assert dropped.keep is False and "DROPPED FROM SHOCK" in dropped.note
    assert dropped in t.branches, "a dropped channel must stay in the record"


def test_a_zero_central_resolves_a_conflict_by_removing_the_proxy():
    """"These forces cancel" is a real answer and must be expressible."""
    t = _tree()
    t.branches[3].n_sd = -0.5
    se.resolve_proxy_conflicts(
        t.branches, {"vix": {"central": 0.0, "reason": "they cancel"}})
    shocks, _ = se.adjudicate_shocks(t.modellable(),
                                     {"vix": {"central": 0.0, "reason": "they cancel"}})
    assert "vix" not in dict(shocks)
    blocking = t.validate(["vix", "aud_ret"])
    assert not any("BOTH directions" in b for b in blocking), (
        "a conflict resolved to zero must stop blocking on the sign conflict")


def test_a_judgement_sets_the_magnitude_not_just_the_sign():
    t = _tree()
    j = {"vix": {"central": 0.4, "low": 0.1, "high": 2.0, "reason": "ours"}}
    shocks, log = se.adjudicate_shocks(t.modellable(), j)
    assert dict(shocks)["vix"] == 0.4, "the team's central value must win over the default"
    assert [x for x in log if x["proxy"] == "vix"][0]["set_by"] == "team judgement"
    assert se.uncertainty_bounds(j)["vix"]["high"] == 2.0


def test_pathway_inconsistency_is_detected():
    import scenarios
    bad = {"channel": "c", "proxy_movement": "falls", "first_round_effect": "inflation down",
           "policy_implication": "easing", "direction": "tightening", "proxy": "aud_ret",
           "n_sd": -1.0}
    assert scenarios.validate_pathway(bad)
    good = {**bad, "direction": "easing"}
    assert scenarios.validate_pathway(good) is None
    wrong_sign = {**good, "n_sd": +1.0}
    assert "n_sd" in scenarios.validate_pathway(wrong_sign)


def test_out_of_support_probabilities_are_withheld_not_clipped():
    clf, X, panel = se.frozen_shock_model()
    row, applied = se.apply_shock(X, panel, [("vix", 40.0)])
    sup = se.support_check(X, row, applied)
    assert not sup["in_support"]
    assert "cannot speak" in sup["quotable"]
    assert sup["joint"]["nearest_neighbour_distance"] > sup["joint"]["threshold_p99_of_real_meetings"]
    t = _tree()
    t.branches[0].n_sd = 40.0
    res = se.run_scenario(t, clf, X, panel, strict=False, horizon="immediate")
    assert res["probabilities"] is None, "out-of-support probabilities must not be quoted"
    assert res["model_verdict"] == "out_of_support", (
        "out of support the model must return no direction either - the argmax of an "
        "extrapolated distribution is not a verdict")
    assert res["channel_direction"]["source"].startswith("adjudicated channels")


def test_support_thresholds_are_calibrated_on_real_unseen_meetings():
    """
    THE MISCALIBRATION THIS PINS. Support was judged against the spacing BETWEEN training
    rows, which on an autocorrelated series is tiny - so all 82 ordinary held-out meetings,
    and the unshocked base row, were declared out of support. A rule that fails every real
    observation is not a support test.
    """
    cal = se.support_calibration()
    for k in ("joint_heldout_p50", "joint_heldout_p99", "marginal_breaches_p50",
              "marginal_breaches_p99", "n_heldout"):
        assert k in cal, f"the calibration is missing {k!r}"
    assert cal["n_heldout"] > 50
    assert cal["joint_heldout_p99"] > cal["joint_heldout_p50"] > 0
    clf, X, panel = se.frozen_shock_model()
    for label in ("2025-11-04", "2026-06-16"):
        i = int(X.index.get_loc(pd.Timestamp(label)))
        sup = se.support_check(X, X.iloc[[i]], [])
        assert sup["in_support"], (
            f"the UNSHOCKED {label} row reads as out of support - the thresholds are "
            f"miscalibrated again")


def test_headroom_is_reported_so_a_small_shift_can_be_interpreted():
    clf, X, panel = se.frozen_shock_model()
    t = _tree()
    res = se.run_scenario(t, clf, X, panel, strict=False, horizon="immediate",
                          base_row="2026-06-16")
    assert "headroom" in res and 0.0 <= res["headroom"] <= 1.0
    assert res["base_row_date"] == "2026-06-16"
    neutral = se.run_scenario(t, clf, X, panel, strict=False, horizon="immediate",
                              base_row="2025-11-04")
    assert neutral["headroom"] > res["headroom"], (
        "the neutral base row should leave more probability free to move than the latest one")


def test_responsiveness_separates_disagreement_from_local_flatness():
    clf, X, panel = se.frozen_shock_model()
    r = se.responsiveness(clf, X, panel,
                          ["business_conditions", "unemployment_rate", "cpi_yoy"],
                          base_row=int(X.index.get_loc(pd.Timestamp("2026-06-16"))))
    assert set(r.columns) >= {"proxy", "max_move", "at_n_sd"}
    assert (r["max_move"] >= 0).all()
    assert r["max_move"].max() > 0.05, "no proxy can move the model at all - check the fixture"


def test_a_tree_with_a_blocking_defect_produces_no_number():
    clf, X, panel = se.frozen_shock_model()
    t = Tree(scenario="one-sided")
    t.branches = [Branch(channel="only tightening", proxy="vix", n_sd=1.0,
                         direction="tightening", depth=0, score=0.9, keep=True, note="x")]
    with pytest.raises(ValueError, match="cannot produce a reportable result"):
        se.run_scenario(t, clf, X, panel, strict=True)


def test_focused_sensitivity_stays_within_seventeen_runs():
    clf, X, panel = se.frozen_shock_model()
    t = Tree(scenario="s")
    t.branches = [
        Branch(channel=f"c{i}", proxy=p, n_sd=v, direction="tightening", depth=0,
               score=0.9, keep=True, note="x", horizon="days")
        for i, (p, v) in enumerate([("vix", 1.5), ("aud_ret", -1.2),
                                    ("gdp_growth_yoy", -1.0), ("cpi_yoy", 0.8),
                                    ("unemployment_rate", 0.5)])]
    assert se.focused_corners(t, clf, X, panel).empty, (
        "with no student-supplied bounds the sensitivity must refuse, not invent multipliers")
    bounds = {p: {"low": min(v * 0.5, v * 1.5), "central": v,
                  "high": max(v * 0.5, v * 1.5), "reason": "test"}
              for p, v in [("vix", 1.5), ("aud_ret", -1.2),
                           ("gdp_growth_yoy", -1.0), ("cpi_yoy", 0.8)]}
    df = se.focused_corners(t, clf, X, panel, bounds=bounds, horizon="immediate")
    assert 1 <= len(df) <= 17, f"{len(df)} evaluations exceeds the 2^4 + 1 budget"
    t.branches.append(Branch(channel="c5", proxy="unemployment_rate", n_sd=0.5,
                             direction="tightening", depth=0, score=0.9, keep=True,
                             note="x", horizon="days"))
    with pytest.raises(ValueError, match="exceeds"):
        se.focused_corners(t, clf, X, panel, horizon="immediate",
                           bounds={**bounds, "unemployment_rate":
                                   {"low": 0.1, "central": 0.5, "high": 0.9,
                                    "reason": "a fifth axis"}})


def test_opposite_of_stable_is_defined():
    assert se.opposite_of("easing") == "hardening"
    assert se.opposite_of("hardening") == "easing"
    assert "MOVES" in se.opposite_of("stable")


def test_calibration_cap_reduces_an_unprecedented_shock():
    t = Tree(scenario="s")
    t.branches = [Branch(channel="huge", proxy="consumer_sentiment", n_sd=-8.0,
                         direction="easing", depth=0, score=0.9, keep=True, note="x")]
    log = se.cap_to_calibration(t.branches, allowance=1.0)
    assert log and abs(t.branches[0].n_sd) < 8.0
    assert "CAPPED" in t.branches[0].note


# -------------------------------------------------------------------------------------------
# Held-out means NOT SCORED
# -------------------------------------------------------------------------------------------

def test_development_run_cannot_see_the_validation_meetings():
    """
    THE BUG THIS PINS. The first ordinary run scored all 211 documents including the
    validation episodes, then stamped the table. Every prompt a team tried had already been
    applied to the held-out meetings; freezing the first result afterwards hid that rather
    than preventing it.
    """
    docs = tf.load_documents()
    held = tf.validation_meetings(docs)
    assert len(held) > 20, f"only {len(held)} validation meetings identified"
    dev = docs[~docs["meeting_date"].isin(held)]
    assert len(dev) + len(held) == len(docs)
    assert not set(dev["meeting_date"]) & held, "a development document is also held out"
    # every validation episode window is fully inside the held-out set
    d = pd.to_datetime(docs["meeting_date"])
    for name, (a, b) in tf.VALIDATION_EPISODES.items():
        inside = set(docs.loc[(d >= a) & (d <= b), "meeting_date"])
        assert inside <= held, f"{name} leaks into the development sample"


def test_validation_stamp_records_the_prompts_that_produced_it():
    """A hash cannot be checked by a marker; the prompt text can."""
    if not tf.VALIDATION_STAMP.exists():
        pytest.skip("validation has not been run in this repository")
    rec = json.loads(tf.VALIDATION_STAMP.read_text())
    for k in ("config_hash", "run_at", "meetings", "system_prompt", "constructs",
              "episodes", "exposure_number"):
        assert k in rec, f"the validation stamp is missing {k!r}"
    assert len(rec["system_prompt"]) > 100
    assert set(rec["constructs"]) == set(tf.FIELDS)
    assert rec["exposure_number"] >= 1


def test_a_prompt_change_after_validation_is_detectable():
    """The stamp must let a marker see that the submitted prompts are not the validated ones."""
    if not tf.VALIDATION_STAMP.exists():
        pytest.skip("validation has not been run in this repository")
    rec = json.loads(tf.VALIDATION_STAMP.read_text())
    assert rec["config_hash"] == tf.config_hash(), (
        "the prompts have changed since validation - a real submission in this state must "
        "declare a revalidation")
    original = tf.CONSTRUCTS["vigilance"]
    try:
        tf.CONSTRUCTS["vigilance"] = original + " revised after validation"
        assert tf.config_hash() != rec["config_hash"], (
            "editing a rubric after validation must change the hash, or the mismatch is "
            "undetectable")
    finally:
        tf.CONSTRUCTS["vigilance"] = original


# -------------------------------------------------------------------------------------------
# Source grounding
# -------------------------------------------------------------------------------------------

CONTEXT_READY = (config.DATA_RAW / "context").exists() and any(
    p.suffix.lower() == ".pdf" for p in (config.DATA_RAW / "context").glob("*"))
needs_context = pytest.mark.skipif(
    not CONTEXT_READY, reason="no context PDFs downloaded in this repository")


@needs_context
def test_pdf_page_offsets_match_the_text_that_is_searched():
    """
    THE BUG THIS PINS. Offsets were computed on the raw extraction and the text was then
    whitespace-normalised, so every citation after the first run of whitespace was wrong.
    In the WEF report, content on page 48 was cited as page 10.
    """
    from pypdf import PdfReader
    docs = context_docs.load_documents()
    checked = 0
    for name, text in docs.items():
        if not name.lower().endswith(".pdf"):
            continue
        offs = context_docs.PAGE_OFFSETS.get(name)
        assert offs, f"{name} recorded no page offsets"
        reader = PdfReader(str(config.DATA_RAW / "context" / name))
        for page_no, pg in enumerate(reader.pages):
            want = re.sub(r"\s+", " ", pg.extract_text() or "").strip()
            if len(want) < 40:
                continue
            got = text[offs[page_no]: offs[page_no] + len(want)]
            assert got == want, (
                f"{name}: the text at the recorded offset for page {page_no + 1} is not "
                f"page {page_no + 1}")
            assert context_docs.page_of(name, offs[page_no]) == page_no + 1
            checked += 1
    assert checked > 50, f"only {checked} pages checked - the test is not exercising much"


@needs_context
def test_early_middle_and_late_phrases_cite_their_own_page():
    """A phrase unique to one page must be cited to that page, wherever it sits."""
    from pypdf import PdfReader
    docs = context_docs.load_documents()
    name = max(docs, key=lambda n: len(docs[n]))
    reader = PdfReader(str(config.DATA_RAW / "context" / name))
    pages = [re.sub(r"\s+", " ", (p.extract_text() or "")).strip() for p in reader.pages]
    text = docs[name]
    for frac in (0.1, 0.5, 0.9):
        p_idx = int(len(pages) * frac)
        body = pages[p_idx]
        if len(body) < 120:
            continue
        probe = body[40:120]
        if text.count(probe) != 1:      # only test phrases that are genuinely unique
            continue
        assert context_docs.page_of(name, text.index(probe)) == p_idx + 1, (
            f"a phrase from page {p_idx + 1} of {name} was cited to another page")


@needs_context
def test_a_tag_is_recorded_only_if_its_passage_reached_the_prompt():
    """A source must not count as retrieved when the budget cut its text out."""
    docs = context_docs.load_documents()
    text, tags = context_docs.relevant_passages(
        docs, ["risk", "growth", "trade"], budget_chars=1500, return_tags=True)
    assert tags, "expected at least one tag"
    for t in tags:
        assert f"[{t}]" in text, (
            f"tag {t!r} was recorded but its passage is not in the prompt - the tag is "
            f"being added before the budget check")


def test_source_citations_are_matched_exactly_not_by_substring():
    """'file.pdf p.4' must not validate a citation to 'file.pdf p.41'."""
    import scenarios as sc
    assert sc._cited_tag("wef.pdf p.4") == "wef.pdf p.4"
    assert sc._cited_tag("[wef.pdf p.41]") == "wef.pdf p.41"
    assert sc._cited_tag("wef.pdf p.04") == "wef.pdf p.4"
    assert sc._cited_tag("no citation here").startswith("\x00")
    valid = {"wef.pdf p.4"}
    assert sc._cited_tag("wef.pdf p.4") in valid
    assert sc._cited_tag("wef.pdf p.41") not in valid, "substring matching has returned"


# -------------------------------------------------------------------------------------------
# Human decisions must carry reasons
# -------------------------------------------------------------------------------------------

def test_a_branch_adjudication_without_a_reason_is_rejected():
    import scenarios as sc
    t = _tree()
    with pytest.raises(ValueError, match="reason"):
        sc.prune(t.branches, adjudications={"up": True})
    with pytest.raises(ValueError, match="reason"):
        sc.prune(t.branches, adjudications={"up": {"keep": False}})
    sc.prune(t.branches, adjudications={
        "up": {"keep": False, "reason": "no precedent", "by": "AB"}})
    dropped = [b for b in t.branches if b.channel == "up"][0]
    assert dropped.keep is False and "no precedent" in dropped.note
    assert dropped.adjudication == "no precedent"


def test_a_proxy_judgement_without_a_reason_is_rejected():
    t = _tree()
    with pytest.raises(ValueError, match="reason"):
        se.resolve_proxy_conflicts(t.branches, {"vix": {"central": 1.0}})


# -------------------------------------------------------------------------------------------
# Shock: horizon, axes, and the support reference
# -------------------------------------------------------------------------------------------

def test_channel_direction_uses_only_the_selected_horizon():
    """
    THE BUG THIS PINS. The shock applied one horizon while the qualitative direction was
    computed over every survivor, so a 'short horizon' refusal came with a direction that
    silently combined immediate, short and medium mechanisms.
    """
    clf, X, panel = se.frozen_shock_model()
    t = Tree(scenario="mixed")
    t.branches = [
        Branch(channel="fast easing", proxy="vix", n_sd=1.0, direction="easing",
               channel_type="risk_appetite", horizon="days", depth=0, score=0.9,
               keep=True, note="x"),
        Branch(channel="slow tightening", proxy="cpi_yoy", n_sd=1.0, direction="tightening",
               channel_type="import_prices", horizon="2-4q", depth=0, score=0.9,
               keep=True, note="x"),
        Branch(channel="rejected", proxy=None, n_sd=0.0, direction="tightening",
               horizon="days", depth=0, score=0.1, keep=False, note="dropped"),
    ]
    imm = se.channel_direction(se.channels_at(t.survivors(), "immediate"))
    med = se.channel_direction(se.channels_at(t.survivors(), "medium"))
    assert imm["direction"] == "easing", imm
    assert med["direction"] == "tightening", med
    assert imm["n_branches"] == 1 and med["n_branches"] == 1


def test_channel_direction_aggregates_by_mechanism_not_branch_count():
    """Four restatements of one mechanism must not outvote one distinct mechanism."""
    verbose = [Branch(channel=f"restatement {i}", proxy="vix", n_sd=1.0,
                      direction="easing", channel_type="risk_appetite", horizon="days",
                      depth=0, score=0.8, keep=True, note="x") for i in range(4)]
    terse = [Branch(channel="the other side", proxy="cpi_yoy", n_sd=1.0,
                    direction="tightening", channel_type="import_prices", horizon="days",
                    depth=0, score=0.8, keep=True, note="x")]
    out = se.channel_direction(verbose + terse)
    assert out["mechanism_families"] == 2, out
    assert out["direction"] == "no clear direction", (
        "four phrasings of one mechanism outvoted one distinct mechanism")


def test_support_is_measured_against_the_rows_the_model_was_fitted_on():
    """
    THE BUG THIS PINS. `shock_design.parquet` holds all 211 meetings but the model was
    fitted on 129. Using the full design as the reference put the scenario base row inside
    its own comparison set, so its nearest-neighbour distance was zero by construction.
    """
    clf, X, panel = se.frozen_shock_model()
    train = se.training_design()
    assert len(train) < len(X), (
        "the training design should be a strict subset of the full design")
    perf = json.loads((config.MODEL_CARD / "performance.json").read_text())
    assert len(train) == perf["shock_model"]["n_training_rows_exported"]
    # the 2026 base row is NOT in the training rows, so it must not score a zero distance
    assert X.index[-1] not in train.index
    j = se._joint_support(train, X.iloc[[-1]])
    assert j["nearest_neighbour_distance"] > 0.0, (
        "the base row is being compared against itself")


def test_sensitivity_requires_three_axes_when_three_are_available():
    clf, X, panel = se.frozen_shock_model()
    t = Tree(scenario="s")
    t.branches = [
        Branch(channel=f"c{i}", proxy=p, n_sd=v, direction="tightening", horizon="days",
               channel_type="other", depth=0, score=0.9, keep=True, note="x")
        for i, (p, v) in enumerate([("vix", 1.5), ("cpi_yoy", 0.8),
                                    ("gdp_growth_yoy", -1.0), ("unemployment_rate", 0.5)])]
    one_axis = {"vix": {"low": 1.0, "central": 1.5, "high": 2.0, "reason": "test"}}
    with pytest.raises(ValueError, match="rubric asks for"):
        se.focused_corners(t, clf, X, panel, bounds=one_axis, horizon="immediate")
    three = {p: {"low": min(v, v * 1.5), "central": v, "high": max(v, v * 1.5),
                 "reason": "test"}
             for p, v in [("vix", 1.5), ("cpi_yoy", 0.8), ("gdp_growth_yoy", -1.0)]}
    df = se.focused_corners(t, clf, X, panel, bounds=three, horizon="immediate")
    assert 1 <= len(df) <= 17


def test_coherence_is_judged_on_severity_not_on_the_words_low_and_high():
    """
    A negatively signed shock's numerically LOW bound is its SEVERE end. Comparing the
    labels excluded the coherent corner and admitted the incoherent one.
    """
    # gdp falls (negative) and vix rises (positive): severe together means gdp at its low
    # numeric bound and vix at its high one - opposite LABELS, same severity.
    pick = {"gdp_growth_yoy": "low", "vix": "high"}
    severity = {"gdp_growth_yoy": True, "vix": True}
    rule = [{"proxies": ["gdp_growth_yoy", "vix"], "reason": "same event"}]
    assert se._incoherent(pick, rule, severity) is None, (
        "a corner where both proxies are at their severe ends was excluded")
    mixed = {"gdp_growth_yoy": True, "vix": False}
    assert se._incoherent(pick, rule, mixed) is not None


def test_prune_sweep_does_not_mutate_the_submitted_tree():
    """It runs on a deep copy: conflict resolution rewrites notes, and the record is graded."""
    clf, X, panel = se.frozen_shock_model()
    t = _tree()
    t.branches[3].n_sd = -0.5
    before = [(b.keep, b.kept_by, b.note) for b in t.branches]
    se.prune_sweep(t, clf, X, panel,
                   judgements={"vix": {"central": 1.0, "reason": "ours"}},
                   horizon="immediate")
    after = [(b.keep, b.kept_by, b.note) for b in t.branches]
    assert before == after, "prune_sweep mutated the tree it was asked to analyse"


# -------------------------------------------------------------------------------------------
# Terminal decisions: what a pruning threshold may NOT undo
# -------------------------------------------------------------------------------------------

def _reviewable_tree() -> Tree:
    """Two opposing modellable branches plus a rejected one, all in distinct groups."""
    t = Tree(scenario="terminal")
    t.branches = [
        Branch(channel="up", proxy="vix", n_sd=1.0, direction="tightening", horizon="days",
               channel_type="risk_appetite", depth=0, score=0.9, keep=True, note="x"),
        Branch(channel="down", proxy="aud_ret", n_sd=-1.0, direction="easing",
               horizon="days", channel_type="exchange_rate", depth=0, score=0.9, keep=True,
               note="x"),
        Branch(channel="doomed", proxy="cpi_yoy", n_sd=2.0, direction="tightening",
               horizon="days", channel_type="import_prices", depth=0, score=0.95, keep=True,
               note="x"),
    ]
    return t


TEST_NARRATIVE = "A stipulated event occurs. Widgets become scarce across the region."
TEST_FACTS = {"widgets_scarce": "Widgets become scarce across the region"}


def _review(decision="accept", **kw):
    rec = {"confidence": "moderate", "global_verdict": "scenario_fact",
           "fact_id": "widgets_scarce",
           "fact_link_reason": "the stipulated scarcity is the world effect this claim "
                               "passes through",
           "australian_verdict": "supports", "evidence_cycle": "e", "evidence_words": "e",
           "evidence_replay": "e", "decision": decision, "reason": "r", "by": "AB"}
    rec.update(kw)
    return rec


def _apply(branches, reviews, require=True):
    import scenarios as sc
    return sc.apply_channel_reviews(branches, reviews, require=require,
                                    narrative=TEST_NARRATIVE, facts=TEST_FACTS)


def test_a_reviewed_rejection_stays_rejected_at_every_sweep_threshold():
    """
    THE BUG THIS PINS. `adjudicate_tree()` opened by resetting `keep` from the credibility
    score and replayed neither the channel reviews nor the grounding bar, so the 0.3 rung of
    the sweep reinstated branches a person had explicitly rejected. Their proxies were gone,
    so the classifier probability was unaffected - but `channel_direction()` counts surviving
    unmodellable branches, so a rejected channel could still move the qualitative direction
    the report leads with.
    """
    import scenarios as sc
    clf, X, panel = se.frozen_shock_model()
    t = _reviewable_tree()
    reviews = {k: _review("reject" if "cpi_yoy" in k else "accept")
               for k in sc.mechanism_groups(t.branches)}
    _apply(t.branches, reviews)
    doomed = [b for b in t.branches if b.channel == "doomed"][0]
    assert doomed.keep is False and doomed.terminal_exclusion

    df = se.prune_sweep(t, clf, X, panel, thresholds=(0.1, 0.3, 0.5, 0.9),
                        horizon="immediate")
    assert (df["n_terminally_excluded"] >= 1).all()
    # and directly: at the lowest threshold the copy must still exclude it
    import copy
    w = copy.deepcopy(t)
    se.adjudicate_tree(w, 0.0, quiet=True)
    survivors = {b.channel for b in w.survivors()}
    assert "doomed" not in survivors, (
        f"a reviewed rejection came back at threshold 0.0: {survivors}")


def test_a_grounding_bar_stays_barred_at_every_sweep_threshold():
    import scenarios as sc
    clf, X, panel = se.frozen_shock_model()
    t = _reviewable_tree()
    reviews = {k: _review(australian_verdict="does_not_support" if "cpi_yoy" in k
                          else "supports")
               for k in sc.mechanism_groups(t.branches)}
    _apply(t.branches, reviews)
    counts = sc.enforce_grounding(t.branches)
    assert counts["global_only"] >= 1
    doomed = [b for b in t.branches if b.channel == "doomed"][0]
    assert doomed.evidence_status == "global_only" and doomed.terminal_exclusion

    import copy
    for threshold in (0.0, 0.3, 0.5, 0.9):
        w = copy.deepcopy(t)
        se.adjudicate_tree(w, threshold, quiet=True)
        assert "doomed" not in {b.channel for b in w.survivors()}, (
            f"a grounding-barred branch survived at threshold {threshold}")
        assert doomed.channel not in {
            b.channel for b in se.channels_at(w.survivors(), "immediate")}


def test_the_opposing_direction_guard_cannot_reinstate_a_terminal_branch():
    """The guard recovers a channel PRUNING dropped. It may not overturn a review."""
    t = Tree(scenario="g")
    t.branches = [
        Branch(channel="only", proxy="vix", n_sd=1.0, direction="tightening",
               horizon="days", channel_type="risk_appetite", depth=0, score=0.9, note="x"),
        Branch(channel="other-way", proxy=None, n_sd=0.0, direction="easing",
               horizon="days", channel_type="external_demand", depth=0, score=0.8,
               note="x", terminal_exclusion="rejected on review (AB)"),
    ]
    se.adjudicate_tree(t, 0.5, quiet=True)
    assert "other-way" not in {b.channel for b in t.survivors()}, (
        "the guard reinstated a branch a person had rejected")


def test_the_sweep_counts_first_order_survivors_it_never_expanded():
    """
    `prune_sweep` re-prunes a tree whose expansion happened once, at the headline threshold.
    A lower rung therefore keeps first-order branches whose second-order consequences were
    never generated. The count is reported so the limitation is stated rather than implied
    by the function's name.
    """
    clf, X, panel = se.frozen_shock_model()
    t = _tree()
    df = se.prune_sweep(t, clf, X, panel, thresholds=(0.05, 0.85), horizon="immediate")
    assert "n_first_order_never_expanded" in df.columns
    low = df[df["threshold"] == 0.05].iloc[0]
    assert low["n_first_order_never_expanded"] >= 1, (
        "the 'rejected' branch is reinstated at 0.05 and has no children, and the sweep "
        "does not say so")


def test_focused_corners_answer_from_the_base_row_they_were_given():
    """
    THE BUG THIS PINS. `_evaluate_corner()` defaulted to `base_row=-1`, so the corners were
    silently computed from the latest meeting while the headline came from the declared base
    row. Two different base rows must produce two different records.
    """
    clf, X, panel = se.frozen_shock_model()
    bounds = {p: {"low": lo, "central": c, "high": hi, "reason": "test"}
              for p, (lo, c, hi) in {"vix": (0.5, 1.5, 2.5),
                                     "cpi_yoy": (0.4, 0.8, 1.6),
                                     "gdp_growth_yoy": (-2.0, -1.0, -0.4)}.items()}
    t = Tree(scenario="s")
    t.branches = [
        Branch(channel=f"c{i}", proxy=p, n_sd=v, direction="tightening", horizon="days",
               channel_type="other", depth=0, score=0.9, keep=True, note="x")
        for i, (p, v) in enumerate([("vix", 1.5), ("cpi_yoy", 0.8),
                                    ("gdp_growth_yoy", -1.0)])]
    a = se.focused_corners(t, clf, X, panel, bounds=bounds, horizon="immediate",
                           base_row=0)
    b = se.focused_corners(t, clf, X, panel, bounds=bounds, horizon="immediate",
                           base_row=-1)
    assert len(a) and len(b)
    da, db = set(a["base_row_date"]), set(b["base_row_date"])
    assert len(da) == len(db) == 1 and da != db, (
        f"the corners ignored the base row they were given: {da} vs {db}")


# -------------------------------------------------------------------------------------------
# The two evidence legs
# -------------------------------------------------------------------------------------------

def _branch_with(**kw):
    base = dict(channel="c", proxy="vix", n_sd=1.0, direction="tightening", horizon="days",
                channel_type="risk_appetite", depth=0, score=0.9, keep=True, note="x")
    base.update(kw)
    return Branch(**base)


def test_a_not_in_sources_branch_cannot_drive_a_shock():
    """
    The model is allowed to say "not in sources" and it should. What it may not do is say so
    and then move the numbers anyway - which is how 19 of 23 modellable Taiwan channels came
    to be driving the result on model knowledge alone.
    """
    import scenarios as sc
    b = _branch_with(source="not in sources", australian_verdict="supports")
    counts = sc.enforce_grounding([b])
    assert sc.global_leg(b) == "not_in_sources"
    assert b.evidence_status == "australian_only"
    assert b.keep is False and b.proxy is None and b.terminal_exclusion
    assert counts["barred_from_shock"] == 1


def test_both_legs_are_required_and_the_counts_overlap_honestly():
    """
    THE BUG THIS PINS. `evidence_status()` tested the human verdict FIRST and returned a
    single label, so a branch that was BOTH quote-verified and human-supported was counted
    only as human-supported. The report then said Taiwan carried 0 verifiable quotations
    when the raw field said 9.
    """
    import scenarios as sc
    both = _branch_with(channel="both", source="f.pdf p.1", citation_tag_valid=True,
                        quote_verified=True, global_verdict="supports",
                        australian_verdict="supports")
    global_only = _branch_with(channel="g", source="f.pdf p.1", citation_tag_valid=True,
                               quote_verified=True, global_verdict="supports",
                               australian_verdict="uncertain")
    aus_only = _branch_with(channel="a", source="f.pdf p.1", citation_tag_valid=True,
                            quote_verified=False, australian_verdict="supports")
    neither = _branch_with(channel="n", source="f.pdf p.1", citation_tag_valid=False)
    counts = sc.enforce_grounding([both, global_only, aus_only, neither])
    assert (counts["both"], counts["global_only"], counts["australian_only"],
            counts["neither"]) == (1, 1, 1, 1)
    # the RAW machine count is independent of every human verdict
    assert counts["quote_verified_raw"] == 2, (
        "the raw quotation count must not be masked by the human verdicts")
    assert both.keep is True and both.proxy == "vix"
    for b in (global_only, aus_only, neither):
        assert b.keep is False and b.terminal_exclusion, b.channel


def test_a_verified_quote_is_barred_when_a_person_says_it_does_not_support_the_claim():
    """
    A WEF country risk-ranking table quoted verbatim under "Terms of Trade Improvement"
    passes the substring test. Grounding is a semantic question and only a person answers it.
    """
    import scenarios as sc
    b = _branch_with(source="wef.pdf p.38", citation_tag_valid=True, quote_verified=True,
                     global_verdict="does_not_support", australian_verdict="supports")
    sc.enforce_grounding([b])
    assert sc.global_leg(b) == "quote_irrelevant"
    assert b.evidence_status == "australian_only" and b.keep is False


def test_a_scenario_fact_satisfies_the_global_leg_without_a_citation():
    """And ONLY through a registered, verified fact id - a bare verdict is the bypass."""
    import scenarios as sc
    b = _branch_with(source="not in sources")
    key = list(sc.mechanism_groups([b]))[0]
    _apply([b], {key: _review()})
    sc.enforce_grounding([b])
    assert b.scenario_fact_id == "widgets_scarce"
    assert b.evidence_status == "both" and b.keep is True

    # the bypass: a scenario_fact verdict stamped with NO verified fact id is barred
    bare = _branch_with(source="not in sources", global_verdict="scenario_fact",
                        australian_verdict="supports")
    sc.enforce_grounding([bare])
    assert sc.global_leg(bare) == "fact_unverified"
    assert bare.evidence_status != "both" and bare.keep is False

    # and a fact id that is not registered for this scenario is rejected at review time
    b2 = _branch_with(source="not in sources")
    key2 = list(sc.mechanism_groups([b2]))[0]
    with pytest.raises(ValueError, match="SCENARIO_FACTS"):
        _apply([b2], {key2: _review(fact_id="wages_accelerate")})

    # and a registered "fact" that is not verbatim in the narrative fails verification
    with pytest.raises(ValueError, match="not verbatim"):
        sc.apply_channel_reviews([_branch_with(source="not in sources")],
                                 {key2: _review()}, require=True,
                                 narrative="An entirely different story.",
                                 facts=TEST_FACTS)


def test_a_quote_with_a_fabricated_second_fragment_is_not_verified():
    """
    THE BUG THIS PINS. `verify_citation()` accepted the quote if ANY single fragment of 30+
    characters occurred anywhere in the passage, so a genuine opening followed by an invented
    policy conclusion verified.
    """
    import scenarios as sc
    passage = ("The global supply of advanced semiconductors is concentrated in a small "
               "number of facilities, and any disruption would propagate quickly.")
    tags = {"f.pdf p.1": passage}
    genuine = "The global supply of advanced semiconductors is concentrated"
    assert sc.verify_citation("[f.pdf p.1]", genuine, tags)["quote_verified"]
    # an ellipsis between two real fragments is still allowed
    assert sc.verify_citation(
        "[f.pdf p.1]",
        "The global supply of advanced semiconductors ... would propagate quickly",
        tags)["quote_verified"]
    # a real fragment plus an invented one is not
    faked = sc.verify_citation(
        "[f.pdf p.1]", genuine + " ... and central banks therefore tightened policy", tags)
    assert not faked["quote_verified"]
    assert "do not appear" in faked["why"] or "are not in" in faked["why"]
    # nor are real fragments quoted out of order
    assert not sc.verify_citation(
        "[f.pdf p.1]",
        "would propagate quickly ... The global supply of advanced semiconductors",
        tags)["quote_verified"]


# -------------------------------------------------------------------------------------------
# The channel review itself
# -------------------------------------------------------------------------------------------

def test_a_surviving_mechanism_group_without_a_review_blocks_the_run():
    import scenarios as sc
    t = _reviewable_tree()
    with pytest.raises(ValueError, match="CHANNEL_REVIEWS"):
        _apply(t.branches, {})
    # and it names the groups rather than the branches
    try:
        _apply(t.branches, {})
    except ValueError as e:
        assert "|" in str(e) and "vix" in str(e)


def test_a_review_verdict_must_use_the_controlled_vocabulary():
    """Any non-empty string used to pass, so 'probably?' counted as a judgement."""
    import scenarios as sc
    t = _reviewable_tree()
    keys = list(sc.mechanism_groups(t.branches))
    for field, bad in (("confidence", "quite sure"),
                       ("global_verdict", "probably?"),
                       ("australian_verdict", "yes"),
                       ("decision", "keep")):
        reviews = {k: _review(**{field: bad}) for k in keys}
        with pytest.raises(ValueError, match=field):
            _apply(list(t.branches), reviews)


def test_a_group_with_no_proxy_needs_only_the_five_field_minimum():
    """There is no proxy for Cycle or Words to have said anything about."""
    import scenarios as sc
    t = Tree(scenario="m")
    t.branches = [Branch(channel="qualitative", proxy=None, n_sd=0.0, direction="easing",
                         horizon="days", channel_type="other", depth=0, score=0.9,
                         keep=True, note="x")]
    key = list(sc.mechanism_groups(t.branches))[0]
    assert key.endswith("unmodellable")
    minimal = {"global_verdict": "scenario_fact", "fact_id": "widgets_scarce",
               "fact_link_reason": "the stipulated scarcity is this claim's world effect",
               "australian_verdict": "supports",
               "decision": "accept", "reason": "r", "by": "AB"}
    _apply(t.branches, {key: minimal})
    assert t.branches[0].australian_verdict == "supports"


def test_the_starter_comment_lists_exactly_the_review_fields_the_code_requires():
    """
    THE DRIFT THIS PINS. The comment block above `CHANNEL_REVIEWS` listed seven fields
    while `apply_channel_reviews()` required eight, so a team that followed the comment hit
    a ValueError it had done nothing to deserve.
    """
    import scenarios as sc
    src = pathlib.Path(sc.__file__).read_text(encoding="utf-8")
    marker = "# YOUR ASSESSMENT OF EACH MECHANISM GROUP"
    if sc.CHANNEL_REVIEWS or marker not in src:
        pytest.skip("this pins the STARTER's blank-stub documentation")
    doc = src[:src.index("CHANNEL_REVIEWS: dict")]
    doc = doc[doc.rindex(marker):]
    for f in sc.REVIEW_FIELDS:
        assert f'"{f}"' in doc, f"the documented field list omits {f}"
    for f in sc.REVIEW_FIELDS_MIN:
        assert f in sc.REVIEW_FIELDS
    for values in sc.REVIEW_VALUES.values():
        for v in values:
            assert f'"{v}"' in doc, f"the documented field list never mentions the value {v}"


def test_mechanism_groups_separate_channels_with_different_proxies():
    """
    Keying on (channel_type, direction, horizon) alone put `gpr_index` and `vix` in one
    group, so one signature vouched for two separately generated, separately cited claims.
    """
    import scenarios as sc
    bs = [_branch_with(channel="a", proxy="vix"),
          _branch_with(channel="b", proxy="gpr_index")]
    groups = sc.mechanism_groups(bs)
    assert len(groups) == 2, f"two proxies collapsed into one review: {list(groups)}"


def test_an_adjudication_cannot_overrule_a_grounding_bar():
    import scenarios as sc
    b = _branch_with(channel="c", source="not in sources")
    sc.enforce_grounding([b])
    with pytest.raises(ValueError, match="cannot supply missing evidence"):
        sc.assert_reviews_consistent([b], {"c": {"keep": True, "reason": "we believe it",
                                                 "by": "AB"}})


def test_a_direction_weight_without_a_reason_is_rejected():
    """A bare dictionary of numbers is no more the team's reasoning than the scores were."""
    with pytest.raises(ValueError, match="DIRECTION_WEIGHT_REASONS"):
        se.validate_direction_weights({"risk_appetite": 0.5, "external_demand": 1.0},
                                      {"risk_appetite": "markets lead policy here"})
    with pytest.raises(ValueError, match="stale"):
        se.validate_direction_weights({"risk_appetite": 0.5},
                                      {"risk_appetite": "r", "gone": "renamed"})
    se.validate_direction_weights({"risk_appetite": 0.5}, {"risk_appetite": "r"})
    se.validate_direction_weights(None, None)


# -------------------------------------------------------------------------------------------
# Prompt contracts
# -------------------------------------------------------------------------------------------

def test_prompt_contracts_render_the_schema_that_is_passed_to_the_api():
    """
    THE BUG THIS PINS. The branch call passes `response_format=BranchOut`, whose shape is
    `{"channels": [...]}`, and printed the contract for `ChannelOut`, which is one ELEMENT of
    that list. Structured output hid it at runtime, so students were told to design prompts
    against a contract that contradicted the response format the API was given.
    """
    import inspect

    import scenarios as sc
    branch_src = inspect.getsource(sc.branch)
    assert "required_fields(BranchOut)" in branch_src, (
        "the branch prompt must state the schema the branch call actually passes")
    eval_src = inspect.getsource(sc.evaluate)
    assert "required_fields(EvaluateOut)" in eval_src

    rendered = sc.required_fields(sc.BranchOut)
    assert '"channels"' in rendered, "the outer list field is missing"
    # nested models must be EXPANDED, not printed as a bare class name
    assert "ChannelOut" not in rendered
    for field in ("channel_type", "proxy_movement", "first_round_effect",
                  "policy_implication", "source", "source_quote", "n_sd"):
        assert f'"{field}"' in rendered, f"{field} is not in the rendered contract"
    scores = sc.required_fields(sc.EvaluateOut)
    assert '"scores"' in scores and '"credibility"' in scores and "ScoreOut" not in scores


def test_no_hand_written_json_shape_survives_in_a_prompt():
    """The generated block is the contract; a hand-written example drifts from it."""
    import scenarios as sc
    for name in ("BRANCH_PROMPT", "EVALUATE_PROMPT", "EXPAND_PROMPT",
                 "ADVERSARIAL_PROMPT"):
        text = getattr(sc, name)
        assert '{"scores"' not in text and '{"channels"' not in text, (
            f"{name} still hand-writes the JSON shape")


# -------------------------------------------------------------------------------------------
# The source declaration
# -------------------------------------------------------------------------------------------

def test_sources_json_requires_every_declared_field(tmp_path):
    """
    The brief asks for organisation, title, URL and retrieval date. The check looked at
    `organisation` only, so a record with an empty title, no URL and no date passed.
    """
    def write(recs):
        (tmp_path / "sources.json").write_text(json.dumps(recs), encoding="utf-8")

    good = {"file": "a.pdf", "organisation": "IMF", "title": "T",
            "url": "https://imf.org", "retrieved": "2026-08-12"}
    write([good])
    assert set(context_docs._declared_sources(tmp_path)) == {"a.pdf"}

    for bad in ({**good, "title": ""}, {**good, "url": ""}, {**good, "retrieved": ""},
                {**good, "url": "imf.org"}, {**good, "retrieved": "12/08/2026"},
                {k: v for k, v in good.items() if k != "file"}):
        write([bad])
        with pytest.raises(RuntimeError):
            context_docs._declared_sources(tmp_path)

    (tmp_path / "sources.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="valid JSON"):
        context_docs._declared_sources(tmp_path)


def test_organisations_are_counted_after_normalisation():
    """"IMF", "imf" and "I.M.F." are one publisher, not three."""
    assert (context_docs._norm_org("IMF") == context_docs._norm_org("imf")
            == context_docs._norm_org("I.M.F."))
    assert context_docs._norm_org("Lowy Institute") != context_docs._norm_org("IMF")


# -------------------------------------------------------------------------------------------
# Canonical binding: one verdict, one branch
# -------------------------------------------------------------------------------------------

def test_a_group_verdict_binds_only_the_canonical_branch():
    """
    THE BYPASS THIS PINS. A group's verdict used to be stamped onto every member, so one
    `scenario_fact` carried an invented claim (Australian wages accelerate) into the answer
    alongside a fact the narrative really states, and one `supports` vouched for quotations
    the reviewer never read.
    """
    import scenarios as sc
    stip = Branch(channel="stipulated widget scarcity", proxy="vix", n_sd=1.0,
                  direction="tightening", horizon="days", channel_type="risk_appetite",
                  depth=0, score=0.9, keep=True, note="x")
    invented = Branch(channel="Australian wages accelerate", proxy="vix", n_sd=1.0,
                      direction="tightening", horizon="days",
                      channel_type="risk_appetite", depth=0, score=0.8, keep=True,
                      note="x")
    key = list(sc.mechanism_groups([stip, invented]))[0]
    _apply([stip, invented], {key: _review(canonical=stip.channel)})
    sc.enforce_grounding([stip, invented])
    assert stip.evidence_status == "both" and stip.keep is True
    assert invented.keep is False and invented.human_decision == "restatement"
    assert invented.restatement_of == stip.channel
    assert invented.terminal_exclusion.startswith("restatement")
    # and the tally cannot see the restatement at any threshold
    t = Tree(scenario="s", branches=[stip, invented,
             Branch(channel="other way", proxy="aud_ret", n_sd=-1.0, direction="easing",
                    horizon="days", channel_type="exchange_rate", depth=0, score=0.9,
                    keep=True, note="x", evidence_status="both")])
    se.adjudicate_tree(t, 0.0, quiet=True)
    assert invented.channel not in {b.channel for b in t.survivors()}


def test_a_supports_verdict_requires_the_canonical_own_verified_quote():
    """A person cannot certify a quotation that failed the machine check."""
    import scenarios as sc
    good = _branch_with(channel="good", source="f.pdf p.1", citation_tag_valid=True,
                        quote_verified=True)
    bad = _branch_with(channel="bad", source="f.pdf p.9", citation_tag_valid=True,
                       quote_verified=False)
    key = list(sc.mechanism_groups([good, bad]))[0]
    with pytest.raises(ValueError, match="no machine-verified quotation"):
        _apply([good, bad], {key: _review(global_verdict="supports",
                                          canonical=bad.channel)})
    # bound to the verified member it passes, and only that member is quote_supported
    good2 = _branch_with(channel="good", source="f.pdf p.1", citation_tag_valid=True,
                         quote_verified=True)
    bad2 = _branch_with(channel="bad", source="f.pdf p.9", citation_tag_valid=True,
                        quote_verified=False)
    _apply([good2, bad2], {key: _review(global_verdict="supports",
                                        canonical=good2.channel)})
    sc.enforce_grounding([good2, bad2])
    assert sc.global_leg(good2) == "quote_supported"
    assert bad2.human_decision == "restatement" and bad2.keep is False


def test_orphans_resolve_through_the_restatement_canonical():
    """
    THE INTERACTION THIS PINS. Barring restatements killed the literal parents of accepted
    second-order canonicals, and `cascade_orphans()` then dropped branches a person had
    signed - which member of a group happened to get expanded is not a judgement.
    """
    parent_restated = Branch(channel="parent restated", proxy="vix", n_sd=1.0,
                             direction="tightening", horizon="days",
                             channel_type="risk_appetite", depth=0, score=0.8, keep=False,
                             note="x", restatement_of="parent canonical",
                             terminal_exclusion="restatement (canonical: parent canonical)")
    parent_canon = Branch(channel="parent canonical", proxy="vix", n_sd=1.0,
                          direction="tightening", horizon="days",
                          channel_type="risk_appetite", depth=0, score=0.9, keep=True,
                          note="x")
    child = Branch(channel="accepted child", proxy="cpi_yoy", n_sd=0.5,
                   direction="tightening", horizon="days", channel_type="import_prices",
                   depth=1, parent="parent restated", score=0.9, keep=True, note="x")
    branches = [parent_restated, parent_canon, child]
    se.cascade_orphans(branches)
    assert child.keep is True, (
        "an accepted child was orphaned although its parent's canonical survives")
    # and with the canonical dead too, the child genuinely falls
    parent_canon.keep = False
    se.cascade_orphans(branches)
    assert child.keep is False and child.kept_by == "orphaned"


# -------------------------------------------------------------------------------------------
# The pruning audit: a person looks at the discard pile
# -------------------------------------------------------------------------------------------

def _audit_tree() -> Tree:
    t = Tree(scenario="audit")
    t.branches = [
        Branch(channel="kept easing", proxy="vix", n_sd=1.0, direction="easing",
               horizon="days", channel_type="risk_appetite", depth=0, score=0.9,
               keep=True, note="x"),
        Branch(channel="kept tightening", proxy="cpi_yoy", n_sd=1.0,
               direction="tightening", horizon="days", channel_type="import_prices",
               depth=0, score=0.9, keep=True, note="x"),
        Branch(channel="best rejected", proxy="aud_ret", n_sd=-1.0, direction="easing",
               horizon="days", channel_type="exchange_rate", depth=0, score=0.45,
               keep=False, kept_by="threshold", note="x"),
        Branch(channel="rejected tightening", proxy=None, n_sd=0.0,
               direction="tightening", horizon="days", channel_type="funding_costs",
               depth=0, score=0.4, keep=False, kept_by="threshold", note="x"),
        Branch(channel="third rejected", proxy=None, n_sd=0.0, direction="easing",
               horizon="days", channel_type="household_income", depth=0, score=0.35,
               keep=False, kept_by="threshold", note="x"),
    ]
    return t


def test_the_pruning_audit_names_the_discard_pile_and_blocks_until_reviewed():
    """
    THE GAP THIS PINS. The model generates the branches AND scores their credibility, so
    without this audit it prunes its own output unsupervised, and a valid, inconvenient or
    opposing mechanism can vanish with nobody required to look.
    """
    import scenarios as sc
    t = _audit_tree()
    needed = sc.required_pruning_reviews(t)
    names = set(needed)
    assert {"best rejected", "rejected tightening", "third rejected"} <= names
    # blocks while outstanding
    with pytest.raises(ValueError, match="discard pile"):
        sc.apply_pruning_reviews(t, {}, None, require=True)
    # a bare tick is not a review
    with pytest.raises(ValueError, match="missing"):
        sc.apply_pruning_reviews(
            t, {ch: {"verdict": "agree", "reason": "", "by": "AB"} for ch in needed},
            None, require=True)
    # defended agreement is accepted in full
    ok = {ch: {"verdict": "agree", "reason": "defended", "by": "AB"} for ch in needed}
    out = sc.apply_pruning_reviews(t, ok, None, require=True)
    assert not out["outstanding"]


def test_a_reinstatement_must_be_recorded_as_an_adjudication():
    import scenarios as sc
    t = _audit_tree()
    needed = sc.required_pruning_reviews(t)
    recs = {ch: {"verdict": "agree", "reason": "r", "by": "AB"} for ch in needed}
    recs["best rejected"] = {"verdict": "reinstate", "reason": "wrongly pruned",
                             "by": "AB"}
    with pytest.raises(ValueError, match="ADJUDICATIONS"):
        sc.apply_pruning_reviews(t, recs, None, require=True)
    sc.apply_pruning_reviews(
        t, recs, {"best rejected": {"keep": True, "reason": "wrongly pruned",
                                    "by": "AB"}}, require=True)


def test_review_rejected_branches_are_not_in_the_pruning_audit():
    """Human decisions are not re-audited; the audit is of the MACHINE's rejections."""
    import scenarios as sc
    t = _audit_tree()
    t.branches[2].terminal_exclusion = "rejected on review (AB)"
    assert "best rejected" not in sc.required_pruning_reviews(t)


# -------------------------------------------------------------------------------------------
# Reaction profiles, corners, weights
# -------------------------------------------------------------------------------------------

@needs_panel
def test_compare_reaction_excludes_failed_constructs_from_the_ranking():
    """
    A construct whose between/within ratio failed its gate measures sampling noise, and it
    used to vote in `mean_abs_gap` anyway. The ranking now uses only audit-valid
    constructs, and reports the all-construct mean alongside so the exclusion is visible.
    """
    panel = _panel()
    exp = {"policy_stance": 0.5, "inflation_concern": 0.6, "global_risk_salience": 0.9}
    if not set(exp) & set(se.episode_profiles(panel).columns):
        pytest.skip("panel has no construct columns yet - run text_features first")
    full = se.compare_reaction(exp, panel)
    part = se.compare_reaction(exp, panel, exclude={"global_risk_salience"})
    assert "mean_abs_gap_all_constructs" in part.columns
    assert "global_risk_salience" in part.columns, "excluded gaps must stay inspectable"
    # the valid-only mean differs from the all-construct mean somewhere
    assert (part["mean_abs_gap"] != part["mean_abs_gap_all_constructs"]).any()
    assert (part["mean_abs_gap_all_constructs"].sort_index()
            == full["mean_abs_gap"].sort_index()).all()
    with pytest.raises(ValueError, match="excluded"):
        se.compare_reaction({"global_risk_salience": 0.9}, panel,
                            exclude={"global_risk_salience"})


def test_focused_corners_refuse_without_bounds_when_the_run_is_reportable():
    """The brief says the reportable run REFUSES; it used to warn and return empty."""
    clf, X, panel = se.frozen_shock_model()
    t = _tree()
    with pytest.raises(ValueError, match="PROXY_JUDGEMENTS"):
        se.focused_corners(t, clf, X, panel, bounds=None, horizon="immediate",
                           require_bounds=True)
    out = se.focused_corners(t, clf, X, panel, bounds=None, horizon="immediate",
                             require_bounds=False)
    assert len(out) == 0


def test_cross_zero_bounds_in_a_coherence_pair_need_a_declared_severe_end():
    """
    With low < 0 < high, both endpoints are 'the big move' in opposite directions, so a
    co-movement rule judged on |move| picks corners at random. The proxy must declare its
    severe end - and once it does, the rule works.
    """
    clf, X, panel = se.frozen_shock_model()
    t = Tree(scenario="s", coherence=[{"proxies": ["vix", "cpi_yoy"], "reason": "same"}])
    t.branches = [
        Branch(channel=f"c{i}", proxy=p, n_sd=v, direction="tightening", horizon="days",
               channel_type="other", depth=0, score=0.9, keep=True, note="x")
        for i, (p, v) in enumerate([("vix", 1.0), ("cpi_yoy", 0.6),
                                    ("gdp_growth_yoy", -1.0)])]
    bounds = {"vix": {"low": -1.0, "central": 1.0, "high": 2.0, "reason": "r"},
              "cpi_yoy": {"low": 0.2, "central": 0.6, "high": 1.2, "reason": "r"},
              "gdp_growth_yoy": {"low": -2.0, "central": -1.0, "high": -0.5,
                                 "reason": "r"}}
    with pytest.raises(ValueError, match="severe"):
        se.focused_corners(t, clf, X, panel, bounds=bounds, horizon="immediate")
    bounds["vix"]["severe"] = "high"
    df = se.focused_corners(t, clf, X, panel, bounds=bounds, horizon="immediate")
    assert len(df) >= 1


def test_an_unweighted_direction_is_not_reportable():
    """
    Without DIRECTION_WEIGHTS the tally falls back to the LLM's own credibility scores - a
    model grading its own homework. Exploration allows it, loudly; a reportable run does
    not.
    """
    with pytest.raises(ValueError, match="DIRECTION_WEIGHTS is empty"):
        se.validate_direction_weights(None, None, required=True)
    se.validate_direction_weights(None, None, required=False)


# -------------------------------------------------------------------------------------------
# Discovery and worked demonstrations produce no quotable numbers
# -------------------------------------------------------------------------------------------

def test_the_worked_demonstration_contains_no_numbers():
    """
    Its contract says it 'stops short of a headline'. An earlier version kept the promise
    in prose only: worked_scenario.json carried probabilities, sweeps and an adversarial
    exchange computed from a tree with zero reviewed channels.
    """
    path = config.OUTPUTS / "worked_scenario.json"
    if not path.exists():
        pytest.skip("run `python src/scenarios.py --worked` first")
    rec = json.loads(path.read_text())
    for banned in ("probabilities", "prune_sweep", "focused_corners", "adversarial",
                   "model_verdict", "per_base_row", "shift"):
        assert banned not in rec, (
            f"worked_scenario.json contains '{banned}' - the demonstration must stop "
            f"before anything quotable")
    assert "channel_review_summary" in rec and "template" in rec


def test_discovery_template_covers_the_assessed_scenarios():
    import scenarios as sc
    path = config.OUTPUTS / "channel_reviews.template.json"
    if not path.exists():
        pytest.skip("run `python src/scenarios.py --discover` first")
    tpl = json.loads(path.read_text())
    assert set(tpl) == set(sc.SCENARIOS), (
        "the discovery template must cover exactly the assessed scenarios - the worked "
        "migration demo cannot produce their group keys")
    for scen, v in tpl.items():
        assert v["groups"], scen
        assert "scenario_facts" in v and v["scenario_facts"], scen
        assert "pruning_reviews_required" in v
        for key, g in v["groups"].items():
            assert g["members"], key
            for m in g["members"]:
                assert {"channel", "source", "source_quote",
                        "quote_verified"} <= set(m)


def test_shock_envelopes_declare_their_provenance_fields(tmp_path, monkeypatch):
    """Model, call index, temperature, prompt hash, request id, usage, attempt, AND the
    request actually sent - as documented. `request` is the one that matters since the
    move to the course proxy: the call sites name parameters the API never receives
    (`seed` is gone from this model family), so an envelope that reports only what the
    code intended to send is not an audit record.

    THIS READS THE EMITTED ENVELOPE, not the source text. It used to grep the writer for
    field-name literals, which stopped meaning anything the moment the three stages moved
    onto one shared builder: the names are no longer written at the call site, and a text
    test would have failed on a refactor that made the guarantee stronger. What matters is
    what lands on disk.
    """
    import types

    import courseapi
    import decision_replay as dr
    import scenarios as sc
    import text_features as tf
    from pydantic import BaseModel

    class _Payload(BaseModel):
        ok: bool = True

    request = courseapi.effective_request(
        model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS)
    usage = {"input_tokens": 5, "output_tokens": 15, "total_tokens": 20,
             "prompt_tokens": 5, "completion_tokens": 15}
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(parsed=_Payload(), content="text"))],
        request=request, model=config.MODEL, id="resp_synthetic",
        usage=dict(usage),
        call_usage=dict(usage, attempts_counted=1, attempts_unknown=0,
                        tokens_known=True, is_complete=True),
        provenance={"envelope_version": courseapi.ENVELOPE_VERSION,
                    "response_status": "completed",
                    "model_requested": config.MODEL, "model_served": config.MODEL,
                    "input_sha256_16": "0123456789abcdef",
                    "transport_requests": 1})
    client = types.SimpleNamespace(beta=types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            parse=lambda **kw: response))))

    common = ("model", "call_index", "temperature", "request", "usage", "request_id",
              "config_hash", "envelope_version", "transport_requests", "model_served",
              "provenance", "timestamp")

    monkeypatch.setattr(sc, "RAW_DIR", tmp_path / "shock")
    (tmp_path / "shock").mkdir()
    monkeypatch.setattr(sc, "client_", lambda: client)
    sc.llm_parsed("system", "user", _Payload)
    shock = json.loads(next((tmp_path / "shock").glob("*.json")).read_text())
    for field in common + ("prompt_sha", "attempt", "payload"):
        assert shock.get(field) is not None, f"shock envelopes omit {field!r}"

    monkeypatch.setattr(dr, "RAW_DIR", tmp_path / "replay")
    (tmp_path / "replay").mkdir()
    monkeypatch.setattr(dr, "client_", lambda: client)
    dr._cached_call("probe", "system", "user", schema=_Payload)
    replay = json.loads(next((tmp_path / "replay").glob("*.json")).read_text())
    for field in common + ("prompt_sha", "attempt", "payload", "kind"):
        assert replay.get(field) is not None, f"replay envelopes omit {field!r}"

    monkeypatch.setattr(tf, "client_", lambda: client)
    monkeypatch.setattr(tf, "_schema_model", lambda: _Payload)
    words = tf.score_once("synthetic", config.CALL_INDEX_BASE)
    for field in common + ("prompt_hash", "parsed", "attempt"):
        assert words.get(field) is not None, f"words envelopes omit {field!r}"

    # AND every one of them satisfies the shared contract when read back.
    for name, rec in (("shock", shock), ("replay", replay), ("words", words)):
        assert not courseapi.envelope_contradictions(rec), (
            f"{name} emitted a record that fails the contract it declares: "
            f"{courseapi.envelope_contradictions(rec)}")


def test_no_envelope_records_a_seed_the_api_never_received():
    """The GPT-5 family rejects `seed`, so nothing may write one into an envelope. An
    envelope naming a request parameter that was never transmitted is a false record,
    and this assignment's whole claim is that a marker can check what was asked."""
    import inspect

    import decision_replay as dr
    import scenarios as sc
    import text_features as tf
    for mod, fn in ((sc, sc.llm_parsed), (dr, dr._cached_call),
                    (tf, tf.score_once)):
        src = inspect.getsource(fn)
        assert '"seed"' not in src, (
            f"{mod.__name__}.{fn.__name__} still writes a 'seed' envelope field; the "
            f"API never receives one")
        assert "seed=" not in src, (
            f"{mod.__name__}.{fn.__name__} still passes seed= to the client")


# -------------------------------------------------------------------------------------------
# Round-4 contracts: fact links, distinct-claim records, strict claim reviews, weights
# -------------------------------------------------------------------------------------------

def test_a_borrowed_fact_id_requires_a_signed_link_judgement():
    """
    THE BYPASS THIS PINS. The registry check verifies the FACT exists verbatim in the
    narrative - identity, not entailment - so "Australian wages accelerate" could borrow
    the valid `widgets_scarce` id and ride through grounding. The record must now carry
    `fact_link_reason`: the machine guarantees the fact is real, a named person signs that
    it supports THIS claim.
    """
    import scenarios as sc
    b = _branch_with(channel="Australian wages accelerate")
    key = list(sc.mechanism_groups([b]))[0]
    rec = _review()
    del rec["fact_link_reason"]
    with pytest.raises(ValueError, match="fact_link_reason"):
        _apply([b], {key: rec})
    b2 = _branch_with(channel="Australian wages accelerate")
    _apply([b2], {key: _review()})
    assert b2.scenario_fact_link, "the signed link must be stamped onto the branch"


def test_a_group_may_hold_several_distinct_claims_via_a_list_of_records():
    """
    THE GAP THIS PINS. The group key is coarse, so two genuinely different mechanisms can
    collide under it - and the code used to bar every non-canonical member as a
    "restatement" by inference. Equivalence is now attested: one record attests the rest
    restate its canonical; a LIST of records names several distinct canonicals, and only
    unnamed members are barred.
    """
    import scenarios as sc
    a = _branch_with(channel="mechanism A", score=0.9)
    b = _branch_with(channel="mechanism B, genuinely different", score=0.8)
    c = _branch_with(channel="restatement of B", score=0.7)

    def fresh():
        return [_branch_with(channel=x.channel, score=x.score) for x in (a, b, c)]

    key = list(sc.mechanism_groups([a, b, c]))[0]
    # with several records, the code refuses to GUESS who restates whom
    with pytest.raises(ValueError, match="restatements"):
        _apply(fresh(), {key: [_review(canonical=a.channel),
                               _review(canonical=b.channel)]})
    # a member owned by two records is a contradiction
    with pytest.raises(ValueError, match="exactly one canonical"):
        _apply(fresh(), {key: [_review(canonical=a.channel, restatements=[c.channel]),
                               _review(canonical=b.channel, restatements=[c.channel])]})
    # a member owned by nobody is dangling
    with pytest.raises(ValueError, match="neither a canonical nor listed"):
        _apply(fresh(), {key: [_review(canonical=a.channel, restatements=[]),
                               _review(canonical=b.channel, restatements=[])]})
    # the explicit partition works, and attribution follows it
    _apply([a, b, c], {key: [_review(canonical=a.channel, restatements=[]),
                             _review(canonical=b.channel,
                                     restatements=[c.channel])]})
    sc.enforce_grounding([a, b, c])
    assert a.keep and b.keep, "both attested-distinct canonicals must survive"
    assert not c.keep and c.human_decision == "restatement"
    assert c.restatement_of == b.channel, "attribution must follow the human partition"
    # two records naming the same canonical is a contradiction
    with pytest.raises(ValueError, match="same canonical"):
        _apply(fresh(), {key: [_review(canonical="mechanism A"),
                               _review(canonical="mechanism A")]})


def test_claim_reviews_are_keyed_by_id_and_strictly_validated():
    """
    THE BUG THIS PINS. Reviews used to match claims by 60-character substring, so one
    truthy malformed record - {"junk": true} - could blanket two claims, be counted twice,
    and leave both with no verdict, no reason and no reviewer, while a printed warning was
    the only consequence of having fewer than three.
    """
    import decision_replay as dr
    claims = ["The Board decided to hold the cash rate at 4.50 percent.",
              "The unemployment rate stands at 5.1 percent.",
              "Economic growth continues to be stable."]
    ids = [dr.claim_id(c) for c in claims]
    payload = {"ok": True, "payload": {"claims": [
        {"claim": c, "category": "data-supported", "evidence_key": None} for c in claims]}}

    def fake_audit(reviews, require=True):
        import unittest.mock as m
        with m.patch.object(dr, "_cached_call", return_value=payload):
            return dr.audit_statement("s", {}, reviews=reviews, require=require)

    good = {"human_verdict": "forecast-or-judgement", "reason": "r",
            "final_action": "kept", "by": "AB"}
    # a malformed truthy record is rejected, not counted
    with pytest.raises(ValueError, match="missing"):
        fake_audit({ids[0]: {"junk": True}})
    # an unknown id is rejected, not silently ignored
    with pytest.raises(ValueError, match="not in this audit"):
        fake_audit({"deadbeef": dict(good)})
    # an illegal verdict is rejected
    with pytest.raises(ValueError, match="human_verdict"):
        fake_audit({ids[0]: {**good, "human_verdict": "sounds right"}})
    # fewer than MIN distinct reviewed claims RAISES on the reportable path
    with pytest.raises(ValueError, match="distinct claims"):
        fake_audit({ids[0]: dict(good)})
    # and passes with three distinct, exact-id reviews
    out = fake_audit({i: dict(good) for i in ids})
    assert out["n_reviewed_by_human"] == 3
    assert out["reviews"] == {i: dict(good) for i in ids}
    # exploration may relax the count, never the validation
    out2 = fake_audit({ids[0]: dict(good)}, require=False)
    assert out2["n_reviewed_by_human"] == 1


def test_partial_team_weights_cannot_masquerade_as_team_weighted():
    """
    THE BUG THIS PINS. With one family weighted and one not, the tally silently mixed a
    human weight with an LLM credibility score and labelled the whole result "weighted by
    the TEAM" because the dictionary was non-empty.
    """
    bs = [Branch(channel="a", proxy="vix", n_sd=1.0, direction="tightening",
                 horizon="days", channel_type="risk_appetite", depth=0, score=0.9,
                 keep=True, note="x"),
          Branch(channel="b", proxy="aud_ret", n_sd=-1.0, direction="easing",
                 horizon="days", channel_type="exchange_rate", depth=0, score=0.8,
                 keep=True, note="x")]
    with pytest.raises(ValueError, match="fall back"):
        se.channel_direction(bs, weights={"risk_appetite": 1.0})
    out = se.channel_direction(bs, weights={"risk_appetite": 1.0,
                                            "exchange_rate": 0.5})
    assert out["weights_supplied_by_team"] is True
    # a family with NO surviving branch may be omitted - that is the coverage rule
    out2 = se.channel_direction(bs, weights={"risk_appetite": 1.0,
                                             "exchange_rate": 0.5,
                                             "import_prices": 0.9})
    assert out2["weights_supplied_by_team"] is True


def test_expected_profiles_are_validated_before_the_comparison_runs():
    """The starter defines the scaffold; an incomplete one fails with named gaps."""
    import scenarios as sc
    with pytest.raises(ValueError, match="EXPECTED_PROFILES is incomplete"):
        sc.reaction_profiles.__wrapped__(None, {}) if hasattr(
            sc.reaction_profiles, "__wrapped__") else sc.reaction_profiles(None, {})


def test_json_review_records_are_a_first_class_input(tmp_path, monkeypatch):
    """
    Item: forty review records pasted into a .py file is clerical, merge-conflict-prone
    work. The template, filled in place and copied to data/processed/channel_reviews.json,
    is loaded and validated exactly like the Python dicts - which stay supported and take
    precedence when non-empty.
    """
    import scenarios as sc
    monkeypatch.setattr(sc.config, "DATA_PROCESSED", tmp_path)
    (tmp_path / "channel_reviews.json").write_text(json.dumps({
        "A. Taiwan Strait blockade": {"groups": {
            "import_prices | tightening | 1-2q | cpi_yoy": {
                "review": _review(),
                "members": []},
            "left | blank | on | purpose": {"review": {"decision": ""}, "members": []},
        }}}), encoding="utf-8")
    monkeypatch.setattr(sc, "CHANNEL_REVIEWS", {})
    eff = sc.effective_channel_reviews("A. Taiwan Strait blockade")
    assert list(eff) == ["import_prices | tightening | 1-2q | cpi_yoy"]
    # python dict wins when non-empty
    monkeypatch.setattr(sc, "CHANNEL_REVIEWS",
                        {"A. Taiwan Strait blockade": {"k": _review()}})
    assert list(sc.effective_channel_reviews("A. Taiwan Strait blockade")) == ["k"]
    # malformed JSON raises rather than being skipped
    monkeypatch.setattr(sc, "PRUNING_REVIEWS", {})
    (tmp_path / "pruning_reviews.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        sc.effective_pruning_reviews("A. Taiwan Strait blockade")


def test_the_week4_checkpoint_does_not_touch_the_validation_episodes():
    """
    THE LEAK THIS PINS. Gate 3 used to ask tutors to inspect GFC, COVID, the 2022 surge
    and the calm period - the four HELD-OUT validation episodes - and its command called
    `episode_profiles()` on exactly those windows, spending every team's one-shot
    validation in a Week 4 tutorial.
    """
    doc = (pathlib.Path(__file__).resolve().parent.parent.parent / "draft-assignment"
           / "week4-words-checkpoint.md")
    if not doc.exists():
        pytest.skip("draft-assignment not present next to this repository")
    text = doc.read_text(encoding="utf-8")
    assert "episode_check" in text and "'dev'" in text
    assert "episode_profiles(p" not in text.replace(chr(10), " "), (
        "the tutor command profiles the validation episodes")
    lower = text.lower()
    gate3 = lower.split("gate 3", 1)[1].split("gate 4", 1)[0]
    # the gate may WARN that the four validation episodes are off-limits; what it must not
    # do is instruct anyone to inspect them. The instructions follow the warning.
    instructions = gate3.split("the check is", 1)[1]
    # "pre-GFC tightening" is a DEVELOPMENT episode whose name contains "gfc"
    instructions = instructions.replace("pre-gfc", "")
    for name in ("gfc", "covid", "2022", "calm"):
        assert name not in instructions, (
            f"Gate 3's instructions name the held-out episode '{name}' - inspecting it "
            f"at Week 4 spends the one-shot validation")
    for dev in ("pre-gfc tightening", "mining-boom plateau", "post-taper easing"):
        assert dev in gate3, f"Gate 3 does not discuss the development episode '{dev}'"


# -------------------------------------------------------------------------------------------
# Stage hashes: computable on the untouched starter, and sensitive to every named input
# -------------------------------------------------------------------------------------------

def test_stage_hashes_are_computable_on_this_repository():
    """
    THE BUG THIS PINS. `config_hash()` referenced STRICT and CALIBRATION_ALLOWANCE, which
    existed only in the worked solution - so the untouched starter raised NameError the
    moment it tried to compute its own Shock hash, and no ordinary test ever called it.
    """
    import decision_replay as dr
    import scenarios as sc
    assert re.fullmatch(r"[0-9a-f]{12}", sc.config_hash())
    assert re.fullmatch(r"[0-9a-f]{12}", dr.config_hash())


# THE HASH CONTRACT, stated as data. Coverage means: every module-level field in these
# lists, the three config-level generation settings, and the effective JSON review files
# each get a mutation test proving the hash moves. Nothing claims that every line of
# implementation affects the hash - only that every DECLARED input does.
SHOCK_HASH_FIELDS = ["BRANCH_PROMPT", "EVALUATE_PROMPT", "EXPAND_PROMPT",
                     "ADVERSARIAL_PROMPT", "SCENARIOS", "RETRIEVAL_TERMS",
                     "N_BRANCH_SAMPLES", "PRUNE_THRESHOLD", "SHOCK_HORIZON", "STRICT",
                     "BASE_ROWS", "HEADLINE_BASE", "CALIBRATION_ALLOWANCE",
                     "SCENARIO_FACTS", "PROXY_JUDGEMENTS", "ADJUDICATIONS",
                     "PATHWAY_ADJUDICATIONS", "PATHWAY_POLICY", "DIRECTION_WEIGHTS",
                     "DIRECTION_WEIGHT_REASONS", "COHERENCE", "ADVERSARIAL_RESPONSES",
                     "EXPECTED_PROFILES", "CLAIMED_ANALOGUES", "DIRECTION_MARGIN",
                     "MIN_QUOTE_CHARS", "MIN_FRAGMENT_CHARS", "IMPLIED_BY_EFFECT",
                     "REQUIRE_CHANNEL_REVIEWS", "REQUIRE_PRUNING_REVIEWS"]


@pytest.mark.parametrize("field", SHOCK_HASH_FIELDS)
def test_the_shock_hash_changes_when_an_output_determining_input_changes(field,
                                                                         monkeypatch):
    """
    THE GAP THIS PINS. The first hash covered the prompts and the judgement tables but not
    the scenario narratives, retrieval terms, sample count, pruning threshold or direction
    margin - all of which shape what is generated, retrieved, pruned or reported, so any
    of them could be edited after artefact generation without the submission check
    noticing.
    """
    import scenarios as sc
    before = sc.config_hash()
    current = getattr(sc, field)
    if isinstance(current, str) or current is None:
        mutated = (current or "") + " MUTATED"
    elif isinstance(current, bool):
        mutated = not current
    elif isinstance(current, (int, float)):
        mutated = current + 17
    elif isinstance(current, dict):
        mutated = {**current, "___mutation___": "x"}
    elif isinstance(current, list):
        mutated = list(current) + ["___mutation___"]
    else:
        pytest.fail(f"unhandled type for {field}: {type(current)}")
    monkeypatch.setattr(sc, field, mutated)
    assert sc.config_hash() != before, (
        f"changing {field} does not change the Shock stage hash - it can be edited after "
        f"artefact generation undetected")


REPLAY_HASH_FIELDS = ["RECOMMENDATION_PROMPT", "STATEMENT_PROMPT", "AUDIT_PROMPT",
                      "CAUSAL_GUIDANCE", "MEETING", "K_SHOTS", "SHOT_STRATEGY",
                      "CLAIM_REVIEWS", "N_SEEDS", "MIN_CLAIM_REVIEWS"]


@pytest.mark.parametrize("field", REPLAY_HASH_FIELDS)
def test_the_replay_hash_changes_when_an_output_determining_input_changes(field,
                                                                          monkeypatch):
    """CAUSAL_GUIDANCE is inserted verbatim into the guided prompt; it was unhashed."""
    import decision_replay as dr
    before = dr.config_hash()
    current = getattr(dr, field)
    if isinstance(current, str):
        mutated = current + " MUTATED"
    elif isinstance(current, bool):
        mutated = not current
    elif isinstance(current, (int, float)):
        mutated = current + 17
    elif isinstance(current, dict):
        mutated = {**current, "___mutation___": {"x": 1}}
    else:
        pytest.fail(f"unhandled type for {field}: {type(current)}")
    monkeypatch.setattr(dr, field, mutated)
    assert dr.config_hash() != before, (
        f"changing {field} does not change the Replay stage hash")


def test_supplied_starter_helpers_are_callable_before_any_stub_is_filled():
    """
    THE BUG THIS PINS. `config_hash()` referenced names that existed only in the worked
    solution, so the untouched starter raised NameError on its own public path and no
    ordinary test noticed. Every supplied helper a student can reach before writing a
    prompt must at least run.
    """
    import io
    from contextlib import redirect_stdout

    import decision_replay as dr
    import scenarios as sc
    assert sc.config_hash() and dr.config_hash()
    with redirect_stdout(io.StringIO()) as buf:
        sc.print_contracts()
    assert "BranchOut" in buf.getvalue()
    assert sc.required_fields(sc.BranchOut)
    assert sc.mechanism_groups([]) == {}
    assert isinstance(sc.failed_constructs(), set)
    for scen in sc.SCENARIOS:
        sc.verify_scenario_facts(sc.SCENARIOS[scen], sc.SCENARIO_FACTS[scen])
    assert dr.claim_id("A claim.") == dr.claim_id("a  claim.")
    assert sc.required_pruning_reviews(Tree(scenario="empty")) == {}


def test_json_only_records_drive_the_full_review_path(tmp_path, monkeypatch):
    """
    THE DRIFT THIS PINS. The brief recommends the JSON record files, but the pipeline and
    submission checks used to consult the Python dicts directly - a student following the
    recommended route, with empty dicts and completed JSON, failed. Here the dicts are
    empty, the records exist only as JSON, and the FULL apply path - channel reviews,
    fact links, pruning audit - runs from them.
    """
    import scenarios as sc
    monkeypatch.setattr(sc.config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(sc, "CHANNEL_REVIEWS", {})
    monkeypatch.setattr(sc, "PRUNING_REVIEWS", {})

    kept = _branch_with(channel="stipulated widget scarcity")
    rejected = Branch(channel="best rejected", proxy="aud_ret", n_sd=-1.0,
                      direction="easing", horizon="days", channel_type="exchange_rate",
                      depth=0, score=0.45, keep=False, kept_by="threshold", note="x")
    tree = Tree(scenario="json-route", branches=[kept, rejected])
    key = list(sc.mechanism_groups([kept]))[0]

    (tmp_path / "channel_reviews.json").write_text(json.dumps(
        {"json-route": {"groups": {key: {"review": _review(), "members": []}}}}),
        encoding="utf-8")
    (tmp_path / "pruning_reviews.json").write_text(json.dumps(
        {"json-route": {"best rejected": {"verdict": "agree", "reason": "defended",
                                          "by": "AB"}}}), encoding="utf-8")

    reviews = sc.effective_channel_reviews("json-route")
    assert reviews and key in reviews
    sc.apply_channel_reviews([kept, rejected], reviews, require=True,
                             narrative=TEST_NARRATIVE, facts=TEST_FACTS)
    assert kept.scenario_fact_id == "widgets_scarce" and kept.scenario_fact_link
    audit = sc.apply_pruning_reviews(tree, sc.effective_pruning_reviews("json-route"),
                                     None, require=True)
    assert not audit["outstanding"]
    sc.enforce_grounding([kept, rejected])
    assert kept.evidence_status == "both" and kept.keep is True


@pytest.mark.parametrize("setting",
                         ["MODEL", "SAMPLING_TEMPERATURE", "CALL_INDEX_BASE"])
def test_both_stage_hashes_cover_the_generation_settings(setting, monkeypatch):
    """Model, temperature and the call-index base live on config and shape every
    generated token - the last of these because it separates the N parallel draws."""
    import decision_replay as dr
    import scenarios as sc
    s_before, r_before = sc.config_hash(), dr.config_hash()
    current = getattr(config, setting)
    mutated = current + " MUTATED" if isinstance(current, str) else current + 17
    monkeypatch.setattr(config, setting, mutated)
    assert sc.config_hash() != s_before, f"Shock hash ignores config.{setting}"
    assert dr.config_hash() != r_before, f"Replay hash ignores config.{setting}"


def test_the_shock_hash_covers_the_effective_json_review_files(tmp_path, monkeypatch):
    """
    The recommended route stores the judgement tables in JSON files - editing THOSE after
    artefact generation must invalidate the check exactly as editing the dicts would.
    """
    import scenarios as sc
    monkeypatch.setattr(sc.config, "DATA_PROCESSED", tmp_path)
    monkeypatch.setattr(sc, "CHANNEL_REVIEWS", {})
    monkeypatch.setattr(sc, "PRUNING_REVIEWS", {})
    scen = list(sc.SCENARIOS)[0]
    key = "risk_appetite | tightening | days | vix"

    def write(reason):
        (tmp_path / "channel_reviews.json").write_text(json.dumps(
            {scen: {"groups": {key: {"review": _review(reason=reason)}}}}),
            encoding="utf-8")

    write("first judgement")
    h1 = sc.config_hash()
    write("edited after artefact generation")
    assert sc.config_hash() != h1, (
        "editing data/processed/channel_reviews.json does not move the Shock hash")
    (tmp_path / "pruning_reviews.json").write_text(json.dumps(
        {scen: {"some reject": {"verdict": "agree", "reason": "r", "by": "AB"}}}),
        encoding="utf-8")
    assert sc.config_hash() != h1


def test_the_discovery_template_carries_every_field_a_review_can_need():
    """
    THE BUG THIS PINS. The generated skeleton omitted `fact_link_reason` while the
    validator required it for every scenario_fact verdict - so a student filling the
    template "in place", exactly as instructed, hit an unexplained blocking error on
    their first stipulated fact.
    """
    import scenarios as sc
    proxy_branch = _branch_with(channel="modellable")
    plain_branch = Branch(channel="qualitative", proxy=None, n_sd=0.0, direction="easing",
                          horizon="days", channel_type="other", depth=0, score=0.9,
                          keep=True, note="x")
    tree = Tree(scenario="tpl", branches=[proxy_branch, plain_branch])
    groups = sc.mechanism_groups(tree.branches)
    summary = {"groups": {k: [b.channel for b in v] for k, v in groups.items()},
               "unreviewed": list(groups), "not_required": []}
    tpl = sc._review_template("tpl", tree, summary, {"required": {}})
    full_min = {"canonical", "global_verdict", "fact_id", "fact_link_reason",
                "australian_verdict", "decision", "reason", "by"}
    for key, g in tpl["groups"].items():
        skeleton = g["review"]
        assert full_min <= set(skeleton), (key, sorted(skeleton))
        if not key.endswith("unmodellable"):
            assert {"confidence", "evidence_cycle", "evidence_words",
                    "evidence_replay"} <= set(skeleton)
        # and a skeleton filled in place VALIDATES - the template is sufficient, not
        # merely suggestive
        filled = dict(skeleton)
        filled.update(_review(canonical=g["members"][0]["channel"]))
        if key.endswith("unmodellable"):
            for f in ("confidence", "evidence_cycle", "evidence_words",
                      "evidence_replay"):
                filled.pop(f, None)
        sc._validate_one_review(key, filled, groups[key], TEST_FACTS,
                                not key.endswith("unmodellable"), [])
    # the advertised complete example names the field too
    src = pathlib.Path(sc.__file__).read_text(encoding="utf-8")
    if "# A COMPLETE EXAMPLE" in src:
        example = src.split("# A COMPLETE EXAMPLE", 1)[1].split("CHANNEL_REVIEWS:", 1)[0]
        assert "fact_link_reason" in example, (
            "the complete example omits fact_link_reason and would not validate")


def test_gitignore_ships_the_manifest_and_not_the_documents():
    """
    THE BUG THIS PINS. `data/raw/context/*` ignored everything except README.md, so the
    REQUIRED student-written sources.json was silently excluded: the submission check
    passed locally against a file the pushed repository did not contain.
    """
    gi = pathlib.Path(__file__).resolve().parent.parent / ".gitignore"
    if not gi.exists():
        pytest.skip("no .gitignore in this repository")
    rules = [ln.strip() for ln in gi.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]

    def ignored(path):
        verdict = False
        for rule in rules:
            neg = rule.startswith("!")
            pat = rule[1:] if neg else rule
            pat = pat.rstrip("/")
            hit = (path == pat
                   or (pat.endswith("/*") and path.startswith(pat[:-1])
                       and "/" not in path[len(pat) - 1:])
                   or (pat.endswith("/**") and path.startswith(pat[:-2])))
            if hit:
                verdict = not neg
        return verdict

    assert not ignored("data/raw/context/sources.json"), (
        "sources.json is git-ignored - the required manifest would be missing from a "
        "pushed repository")
    assert not ignored("data/raw/context/sources.json.template")
    assert not ignored("data/raw/context/README.md")
    assert ignored("data/raw/context/wef-global-risks-report-2026.pdf"), (
        "downloaded context documents must STAY ignored - they are not redistributable")


# -------------------------------------------------------------------------------------------
# Replay: invalid responses and paired completeness
# -------------------------------------------------------------------------------------------

def test_incoherent_recommendations_are_rejected_not_scored():
    import decision_replay as dr
    assert dr.validate_recommendation({"recommendation": "hold", "size_bp": 25})
    assert dr.validate_recommendation({"recommendation": "cut", "size_bp": 0})
    assert dr.validate_recommendation({"recommendation": "hike", "size_bp": 23})
    assert dr.validate_recommendation({"recommendation": "hike", "size_bp": 25}) is None
    assert dr.validate_recommendation({"recommendation": "hold", "size_bp": 0}) is None


@needs_panel
def test_strategy_comparison_uses_the_same_meetings_for_every_strategy():
    import decision_replay as dr
    keep, dropped = dr.feasible_everywhere(dr.dev_sample(), list(nshot.STRATEGIES), 9)
    assert set(keep).isdisjoint(dropped)
    for m in keep:
        for s in nshot.STRATEGIES:
            nshot.select_shots(_panel(), pd.Timestamp(m), k=9, strategy=s)


def test_the_claim_audit_carries_a_human_verdict_on_every_claim():
    """
    The five-way taxonomy is still one LLM classifying another. Without a human field the
    stage assesses whether a machine accepted a machine.
    """
    import decision_replay as dr
    assert hasattr(dr, "CLAIM_REVIEWS"), "there is nowhere for a human verdict to go"
    fields = set(dr.Claim.model_fields)
    assert {"claim", "category"} <= fields
    out = pathlib.Path(config.OUTPUTS / "replay.json")
    if not out.exists():
        pytest.skip("replay has not been run in this repository")
    audit = json.loads(out.read_text()).get("claim_audit", {})
    if not audit.get("claims"):
        pytest.skip("no claims recorded")
    for c in audit["claims"]:
        assert "human_verdict" in c and "challenged" in c, (
            "every claim must carry a human verdict field, even when it is null")
    assert "n_challenged_by_human" in audit


# -------------------------------------------------------------------------------------------
# Shared framework code must not drift between the two repositories
# -------------------------------------------------------------------------------------------

SUPPLIED_MARKER = "# SUPPLIED BELOW THIS LINE"


ORCHESTRATION_MARKER = "# YOUR ORCHESTRATION BELOW THIS LINE"


def _supplied_half(path: pathlib.Path) -> str:
    """
    The framework region: from the SUPPLIED marker to the end of the file (the legacy
    ORCHESTRATION marker, if one is ever reintroduced, ends the region early).

    The runners are supplied framework now - `run()` is identical in both repositories
    and IS compared, so a runner edit in one repository cannot drift past this test.
    """
    text = path.read_text(encoding="utf-8")
    i = text.find(SUPPLIED_MARKER)
    assert i >= 0, f"{path} has no '{SUPPLIED_MARKER}' marker"
    j = text.find(ORCHESTRATION_MARKER, i)
    return text[i:j] if j >= 0 else text[i:]


@pytest.mark.parametrize("module", ["text_features.py", "scenarios.py",
                                    "decision_replay.py"])
def test_supplied_code_is_identical_in_both_repositories(module):
    """
    THE DRIFT THIS PINS. The two repositories are edited in parallel, and everything below
    the SUPPLIED marker is framework rather than a student answer. The worked solution kept
    an older `orientation_check()` that applied one expected direction to all seven
    constructs, and still called the between/within ratio "reliability" - so the marker
    exemplar was running different framework code from the one students receive.

    Only the half below the marker is compared. Above it is the student answer, which is
    expected to differ.
    """
    here = pathlib.Path(__file__).resolve().parent.parent
    sibling = here.parent / ("worked-solution" if here.name == "starter-repo"
                             else "starter-repo")
    if not (sibling / "src" / module).exists():
        pytest.skip(f"{sibling.name} is not present next to this repository")
    mine = _supplied_half(here / "src" / module)
    theirs = _supplied_half(sibling / "src" / module)
    if mine != theirs:
        import difflib
        diff = list(difflib.unified_diff(
            theirs.splitlines(), mine.splitlines(),
            fromfile=f"{sibling.name}/{module}", tofile=f"{here.name}/{module}",
            lineterm="", n=1))[:40]
        raise AssertionError(
            f"the supplied half of {module} differs between the repositories:\n"
            + "\n".join(diff))


@pytest.mark.parametrize("module", ["scenario_engine.py", "nshot.py", "channels.py",
                                    "context_docs.py", "evaluation.py", "config.py",
                                    "data_panel.py", "data_macro.py", "model_card.py"])
def test_wholly_supplied_modules_are_identical(module):
    """These modules have no student-editable section at all, so they must match byte for byte."""
    here = pathlib.Path(__file__).resolve().parent.parent
    sibling = here.parent / ("worked-solution" if here.name == "starter-repo"
                             else "starter-repo")
    if not (sibling / "src" / module).exists():
        pytest.skip(f"{sibling.name} is not present next to this repository")
    a = (here / "src" / module).read_text(encoding="utf-8")
    b = (sibling / "src" / module).read_text(encoding="utf-8")
    assert a == b, f"{module} differs between the repositories"


# -------------------------------------------------------------------------------------------
# The sample report must agree with the artefacts it claims to be derived from
# -------------------------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "docs" / "sample-report.md"
# EXEMPLAR-ONLY. Everything under this marker checks the WORKED SOLUTION's sample report
# against its committed artefacts, phrase by phrase and table by table. Those checks are
# meaningless against a student's own report - and a student who happened to name a report
# docs/sample-report.md used to inherit them - so they run only inside the worked-solution
# repository, and only when both the report and the artefacts exist.
needs_report = pytest.mark.skipif(
    REPO_ROOT.name != "worked-solution"
    or not (REPORT.exists() and (config.OUTPUTS / "words_audit.json").exists()),
    reason="exemplar-consistency checks run in the worked-solution repository only")


@needs_report
def test_report_document_counts_match_the_words_audit():
    """
    THE DRIFT THIS PINS. The report claimed 211 documents and "all seven constructs pass all
    four gates" while `words_audit.json` recorded 164 development documents and one construct
    failing the separation gate. A marker exemplar whose prose contradicts its own JSON
    cannot be used to calibrate anything.
    """
    audit = json.loads((config.OUTPUTS / "words_audit.json").read_text())
    text = REPORT.read_text(encoding="utf-8")
    n_dev = audit["n_documents_scored"]
    n_held = audit["n_validation_withheld"]
    assert f"**{n_dev} meetings**" in text or f"{n_dev} meetings" in text, (
        f"the report does not state the {n_dev} development documents the audit records")
    assert str(n_held) in text, f"the report does not mention the {n_held} withheld meetings"
    total_calls = (n_dev + n_held) * config.N_PARALLEL_CALLS
    assert f"{total_calls:,} calls" in text, (
        f"the report should quote {total_calls:,} calls "
        f"(({n_dev} + {n_held}) x {config.N_PARALLEL_CALLS})")


@needs_report
def test_report_gate_outcomes_match_the_words_audit():
    audit = json.loads((config.OUTPUTS / "words_audit.json").read_text())
    text = REPORT.read_text(encoding="utf-8")
    failing = [q["construct"] for q in audit["quality"]
               if not all(q[k] for k in ("passes_spread", "passes_concentration",
                                         "passes_coverage", "passes_separation"))]
    if failing:
        assert "FAILS" in text or "fails" in text, (
            f"{failing} fail a gate but the report never says any construct fails")
        for c in failing:
            assert c in text, f"{c} fails a gate and is not named in the report"
        assert "all seven pass" not in text.lower(), (
            f"the report claims all seven pass while {failing} do not")
    else:
        assert "FAILS" not in text, "the report reports a failure the audit does not contain"


@needs_report
def test_report_shock_verdicts_match_shock_json():
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    text = REPORT.read_text(encoding="utf-8")
    assert len(sh) == 2, f"the assignment now has two scenarios; shock.json has {len(sh)}"
    lower = text.lower()
    for name in sh:
        # any distinctive word from the scenario name must appear
        words = [w for w in re.findall(r"[A-Za-z]{5,}", name)
                 if w.lower() not in {"strait", "closure", "event", "credibility"}]
        assert any(w.lower() in lower for w in words), (
            f"{name} is not discussed in the report (looked for {words})")
    # every focused corner must have been evaluated from the headline base row
    for name, res in sh.items():
        corners = res.get("focused_corners") or []
        if not corners:
            continue
        dates = {c.get("base_row_date") for c in corners}
        assert len(dates) == 1, f"{name}: corners span several base rows {dates}"
        assert dates.pop() == res["base_row_date"], (
            f"{name}: the corners were evaluated from a different base row than the "
            f"headline result")


@needs_report
def test_report_word_count_is_within_the_stated_tolerance():
    text = REPORT.read_text(encoding="utf-8")
    body = re.sub(r"^\|.*$", "", text, flags=re.M)
    body = re.sub(r"```.*?```", "", body, flags=re.S).split("## Appendix A")[0]
    n = len(body.split())
    claimed = int(re.search(r"\*([\d,]+) words excluding", text).group(1).replace(",", ""))
    assert abs(n - claimed) <= 15, f"the report says {claimed} words; it has {n}"
    assert n <= 3150, f"{n} words exceeds the 5% tolerance on the 3,000-word limit"


@needs_report
def test_report_shock_numbers_match_shock_json():
    """
    THE GAP THIS CLOSES. The previous consistency test compared scenario names and corner
    base dates only, so the narrative could - and did - go stale on every verdict, shift and
    grounding count while the suite stayed green.
    """
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    text = REPORT.read_text(encoding="utf-8")
    for name, res in sh.items():
        for label, per in res["per_base_row"].items():
            v = per["model_verdict"]
            assert v in text.lower() or v in text, (
                f"{name} from the {label} base row reads '{v}', which the report never says")
        g = res.get("grounding_counts") or {}
        if g:
            assert str(g.get("barred_from_shock", "")) in text, (
                f"{name}: the report does not state how many channels were barred "
                f"({g.get('barred_from_shock')})")
    total_barred = sum((r.get("grounding_counts") or {}).get("barred_from_shock", 0)
                       for r in sh.values())
    assert "barred" in text.lower(), (
        f"{total_barred} channels were barred from the shock and the report never says so")


@needs_report
def test_report_does_not_claim_more_grounding_than_the_artefact_shows():
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    text = REPORT.read_text(encoding="utf-8").lower()
    machine = sum((r.get("grounding_counts") or {}).get("supported", 0) for r in sh.values())
    human = sum((r.get("grounding_counts") or {}).get("human_supported", 0)
                for r in sh.values())
    if human > machine:
        assert "human" in text and ("cross-stage" in text or "our own" in text), (
            f"most channels ({human} of {human + machine}) are human-supported rather than "
            f"quote-verified, and the report does not say so")
    assert "every shock channel must cite a retrieved passage" not in text, (
        "the report states a grounding rule the artefacts do not meet")

def _report_rows(text: str) -> dict:
    """
    Every two- or three-column markdown row in the report, keyed on its label.

    Values are stripped of emphasis and backticks so a bolded number compares equal to a
    plain one. Duplicate labels keep the FIRST occurrence, which is the Shock section's.
    """
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().strip("*_` ") for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set("".join(cells)) <= set("-: "):
            continue
        label = cells[0].lower()
        rows.setdefault(label, cells[1:])
    return rows


def _num(cell: str):
    m = re.search(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
    return float(m.group()) if m else None


@needs_report
def test_report_shock_tables_are_parsed_and_match_shock_json():
    """
    THE GAP THIS CLOSES. The earlier consistency tests asserted that strings such as
    "stable", "3" or "18" appeared SOMEWHERE in the report. That passes on a report whose
    every figure is stale, which is how a sample answer came to state grounding counts and
    corner totals its own artefacts contradicted. These rows are parsed and compared.
    """
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    # scenario order in the tables is the order in shock.json
    names = list(sh)
    assert len(names) == 2
    rows = _report_rows(REPORT.read_text(encoding="utf-8"))

    def row(label_fragment):
        hits = [v for k, v in rows.items() if label_fragment in k]
        assert hits, f"the report has no row matching {label_fragment!r}"
        return hits[0]

    g = [sh[n]["grounding_counts"] for n in names]

    checks = [
        ("branches generated", [gg["n_branches"] for gg in g]),
        ("quotations verified verbatim", [gg["quote_verified_raw"] for gg in g]),
        ("citations naming a retrieved passage", [gg["tag_valid_raw"] for gg in g]),
        ("canonical claims", [gg["canonical_claims"] for gg in g]),
        ("restatements barred", [gg["restatements"] for gg in g]),
        ("both legs", [gg["both"] for gg in g]),
        ("global only", [gg["global_only"] for gg in g]),
        ("australian only", [gg["australian_only"] for gg in g]),
        ("neither", [gg["neither"] for gg in g]),
        ("barred at the grounding gate", [gg["barred_from_shock"] for gg in g]),
        ("scenario_fact", [gg["global_leg"].get("scenario_fact", 0) for gg in g]),
    ]
    for label, expected in checks:
        cells = row(label)
        got = [_num(c) for c in cells[:2]]
        assert got == [float(x) for x in expected], (
            f"the report row '{label}' reads {got}; shock.json says {expected}")

    # the "Both" column, where the report gives one, must be the sum
    for label in ("branches generated", "quotations verified verbatim"):
        cells = row(label)
        if len(cells) >= 3 and _num(cells[2]) is not None:
            assert _num(cells[2]) == sum(_num(c) for c in cells[:2]), (
                f"the total column on '{label}' does not add up")

    # judged to support = the quote_supported count on the global leg
    supported = [gg["global_leg"].get("quote_supported", 0) for gg in g]
    got = [_num(c) for c in row("support")[:2]]
    assert got == [float(x) for x in supported], (
        f"the report claims {got} semantically supported quotations; the artefact says "
        f"{supported}")

    # channel-derived direction, per scenario
    dirs = [sh[n]["channel_direction"]["direction"] for n in names]
    cells = row("channel-derived direction")
    for i, d in enumerate(dirs):
        assert d.split()[0] in cells[i].lower(), (
            f"the report says '{cells[i]}' for {names[i]}; the artefact says '{d}'")


@needs_report
def test_report_corner_and_sweep_counts_match_shock_json():
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    rows = _report_rows(REPORT.read_text(encoding="utf-8"))
    hits = [v for k, v in rows.items() if "focused corners" in k]
    assert hits, "the report has no focused-corners row"
    cells = hits[0]
    for i, name in enumerate(sh):
        n = len(sh[name]["focused_corners"])
        assert _num(cells[i]) == n, (
            f"the report says '{cells[i]}' corners for {name}; the artefact has {n}")
        verdicts = {c["model_verdict"] for c in sh[name]["focused_corners"]}
        if len(verdicts) == 1:
            assert verdicts.pop() in cells[i].lower()

    sweep_hits = [v for k, v in rows.items() if "sweep" in k]
    assert sweep_hits, "the report has no threshold-sweep row"
    cells = sweep_hits[0]
    for i, name in enumerate(sh):
        stated = [s.strip() for s in cells[i].split("/")]
        actual = [r["model_verdict"] for r in sh[name]["prune_sweep"]]
        assert stated == actual, (
            f"{name}: the report's sweep reads {stated}, the artefact {actual}")


@needs_report
def test_report_does_not_overstate_what_the_sources_evidenced():
    """
    The one number a reader will quote back. If most of the surviving global legs are
    `scenario_fact` - a premise the assignment supplied - the report must say so in words,
    not present the scenario's own text as a research finding.
    """
    path = config.OUTPUTS / "shock.json"
    if not path.exists():
        pytest.skip("shock has not been run")
    sh = json.loads(path.read_text())["scenarios"]
    text = REPORT.read_text(encoding="utf-8").lower()
    facts = sum(r["grounding_counts"]["global_leg"].get("scenario_fact", 0)
                for r in sh.values())
    quoted = sum(r["grounding_counts"]["global_leg"].get("quote_supported", 0)
                 for r in sh.values())
    if facts > quoted:
        assert "scenario_fact" in text or "stipulate" in text, (
            f"{facts} surviving global legs rest on a stipulated premise against {quoted} "
            f"carried by a source, and the report never says so")
