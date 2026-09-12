"""
Shock scenario machinery. SUPPLIED - do not modify.

The tree-of-thought session runs IN THIS PIPELINE, against the course model, with your four
prompts and the context passages your retrieval terms pull from the documents you
downloaded. Nothing happens in a chat window and nothing is transcribed by hand:
`scenarios.py` records every branch as it is generated, and this module is the machinery
underneath it.

    Branch / Tree       the record structure - every branch generated, scored, kept, barred
    apply_shock()       shocks panel variables by N historical standard deviations
    run_scenario()      collapses a pruned tree into shocks and reports the state shift
    prune_sweep()       conditional re-pruning sensitivity across thresholds
    focused_corners()   the corners of YOUR stated uncertainty bounds
    reaction_profile()  the seven Words constructs as the vocabulary for a reaction

WHY THE RECORD STRUCTURE MATTERS. The marks in Shock are not for the net direction you
arrive at. They are for the channels you considered and rejected, and why - and for the
evidence legs a named person signed. A Tree with three branches and no pruning decisions
recorded is worth very little even if its answer is right.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import channels
import config
import evaluation as ev


# -------------------------------------------------------------------------------------------
# The record structure - transcribe your ChatGPT session into these
# -------------------------------------------------------------------------------------------

@dataclass
class Branch:
    """
    One candidate transmission channel, as it appeared in your tree-of-thought session.

    Fill every field. `keep` and `score` record the PRUNING DECISION you made in the chat -
    a branch you generated and rejected must still appear here, with keep=False and a note
    saying why. Rejected branches are worth marks; missing ones are not.
    """
    channel: str
    mechanism: str = ""            # the causal path, one sentence
    direction: str = "ambiguous"   # "tightening" | "easing" | "ambiguous"
    horizon: str = "unknown"       # "days" | "weeks" | "1-2q" | "2-4q" | "years"
    channel_type: str = "other"    # one of channels.CHANNEL_TAXONOMY, or "other"
    proxy: str | None = None       # a column in panel.parquet, or None if unmodellable
    n_sd: float = 0.0              # shock size in historical standard deviations
    depth: int = 0                 # 0 = first-order, 1 = second-order
    parent: str | None = None      # which first-order channel this expanded from
    score: float | None = None     # credibility the evaluate step assigned, 0-1
    keep: bool | None = None       # the pruning decision
    kept_by: str = ""              # "threshold" | "student adjudication" | guard
    pathway_ok: bool | None = None # False if validate_pathway() found a contradiction
    pathway_note: str = ""         # what the contradiction was
    adjudication: str = ""         # YOUR recorded decision on a flagged branch
    source: str = ""               # the [file p.N] citation, or "not in sources"
    source_tag: str = ""           # the parsed canonical tag
    source_quote: str = ""         # the verbatim words from the cited passage
    citation_tag_valid: bool = False  # the tag names a passage that reached the prompt
    quote_verified: bool = False   # the quote appears verbatim in that passage
    # THE TWO LEGS, RECORDED SEPARATELY. A channel claims two things and they are
    # evidenced differently: scenario -> global effect (a retrieved passage can settle
    # it) and global effect -> Australian proxy -> RBA (these documents cannot).
    # Collapsing them into one `support_verdict` meant a human "supports" was read as
    # covering both, and the reported quote counts were wrong because the human answer
    # was tested first and masked the machine one.
    global_verdict: str = ""       # HUMAN, leg 1: supports | does_not_support |
                                   #               uncertain | scenario_fact
    scenario_fact_id: str = ""     # WHICH stipulated fact carries the global leg - set by
                                   # apply_channel_reviews() after verifying the id exists
    scenario_fact_quote: str = ""  # the registered fact text, verified verbatim against
                                   # the scenario narrative (an IDENTITY check only)
    scenario_fact_link: str = ""   # the reviewer's signed sentence saying why that fact
                                   # supports THIS claim - the entailment judgement
    australian_verdict: str = ""   # HUMAN, leg 2: supports | does_not_support | uncertain
    global_leg: str = ""           # the resolved leg-1 status, set by enforce_grounding()
    australian_leg: str = ""       # the resolved leg-2 status
    evidence_status: str = ""      # both | global_only | australian_only | neither
    # TERMINAL. A human rejection and a grounding bar are properties of the BRANCH, not
    # of the pruning threshold, so re-pruning at another threshold may not undo them.
    # See `_apply_terminal_exclusions()`.
    terminal_exclusion: str = ""
    restatement_of: str = ""       # the canonical member this branch restates, if barred
                                   # as a restatement by apply_channel_reviews()
    human_confidence: str = ""     # YOUR confidence in this channel, from CHANNEL_REVIEWS
    human_decision: str = ""       # accept | modify | reject
    human_reason: str = ""         # why
    reviewed_by: str = ""          # initials
    review_group: str = ""         # the mechanism group this was reviewed as part of
    pathway: str = ""              # proxy movement -> first-round effect -> policy implication
    note: str = ""                 # WHY it was kept or dropped


@dataclass
class Tree:
    """The full record: every branch generated, scored, kept or pruned."""
    scenario: str
    branches: list[Branch] = field(default_factory=list)
    # Scenario-specific co-movement rules used by focused_corners(). Each is
    # {"proxies": [a, b], "reason": "..."} - two proxies that must be sized in the
    # same direction FOR THIS SCENARIO.
    coherence: list = field(default_factory=list)
    # Per-phase citation accounting, persisted into tot_trees.json. The rubric requires an
    # ungrounded-citation count; printing it per call and discarding it does not supply one.
    grounding: dict = field(default_factory=dict)

    def at_depth(self, d: int) -> list[Branch]:
        return [b for b in self.branches if b.depth == d]

    def survivors(self, d: int | None = None) -> list[Branch]:
        return [b for b in self.branches if b.keep and (d is None or b.depth == d)]

    def modellable(self) -> list[Branch]:
        return [b for b in self.survivors() if b.proxy and b.n_sd != 0]

    def unmodellable(self) -> list[Branch]:
        return [b for b in self.survivors() if not b.proxy]

    def validate(self, panel_columns: list[str]) -> list[str]:
        """
        Returns BLOCKING defects - the ones that make a headline number meaningless.

        Advisories are printed but not returned, so `run_scenario(strict=True)` refuses to
        produce a figure only when the tree is actually broken. The earlier version returned
        everything as one undifferentiated warning list, printed nine lines, and returned a
        number anyway; a warning above a result reads as decoration.

        BLOCKING:
          - the tree is empty, or nothing was rejected (there is no tree-of-thought record)
          - survivors point in only one direction (a real opposing channel was almost
            certainly pruned)
          - two surviving channels shock the same proxy in OPPOSITE directions and you have
            not resolved which is right
          - a surviving branch names a proxy that is not a panel column
          - a second-order branch survives whose parent did not

        ADVISORY:
          - thin channel-type coverage, missing second-order channels, untagged branches,
            shock sizes that look large against the calibration table, kept-without-a-reason
        """
        blocking, advisory = [], []
        if not self.branches:
            blocking.append("the tree is empty")
        if not any(b.keep is False for b in self.branches):
            blocking.append("no branch was rejected - a tree with no pruning decisions is "
                            "not a tree-of-thought record")
        dirs = {b.direction for b in self.survivors()}
        if self.branches and not {"tightening", "easing"} <= dirs:
            blocking.append(f"surviving channels point only {sorted(dirs)} - a genuine "
                            f"opposing channel was almost certainly pruned")
        # A branch whose stated pathway contradicts itself may not enter the shock until a
        # person has said what to do about it. Flagging it in a note - which is all an
        # earlier version did - let 22 self-contradicting channels through into the numbers.
        for b in self.modellable():
            if b.evidence_status != "both":
                blocking.append(
                    f"'{b.channel[:40]}' drives a shock but only "
                    f"'{b.evidence_status or 'unchecked'}' of its two evidence legs "
                    f"holds (global: {b.global_leg or 'unchecked'}; australian: "
                    f"{b.australian_leg or 'unchecked'}). Both legs are required: a "
                    f"verified, relevant quotation for the global claim - or "
                    f"global_verdict='scenario_fact' where the scenario itself "
                    f"stipulates it - and a recorded australian_verdict of 'supports'")
            if b.pathway_ok is False and not b.adjudication:
                blocking.append(
                    f"'{b.channel[:40]}': pathway contradicts itself ({b.pathway_note[:70]}) "
                    f"and is still being modelled. Correct the chain, set proxy=None to make "
                    f"it unmodellable, or record an override in PATHWAY_ADJUDICATIONS with a "
                    f"reason")
        # The opposing-direction guard re-enables a branch AFTER your pruning. That is a
        # decision made by the code on your behalf, so it needs your sign-off too.
        for b in self.survivors():
            if b.kept_by == "opposing-direction guard" and not b.adjudication:
                blocking.append(
                    f"'{b.channel[:40]}': reinstated automatically by the opposing-direction "
                    f"guard and never reviewed. Confirm or drop it in PATHWAY_ADJUDICATIONS")
        kept_first = {b.channel for b in self.survivors(0)}
        alias = {b.channel: b.restatement_of for b in self.branches if b.restatement_of}
        for b in self.survivors():
            if (b.depth > 0 and b.parent not in kept_first
                    and alias.get(b.parent) not in kept_first):
                blocking.append(f"'{b.channel[:40]}' survives but its parent "
                                f"'{str(b.parent)[:40]}' was pruned")
            if b.proxy in FORBIDDEN_PROXIES:
                blocking.append(f"'{b.channel[:40]}': proxy '{b.proxy}' is the "
                                f"policy outcome - shocking it assumes the answer")
            if b.proxy and b.proxy not in panel_columns:
                blocking.append(f"'{b.channel[:40]}': proxy '{b.proxy}' is not a panel "
                                f"column")
        by_proxy: dict = {}
        for b in self.modellable():
            by_proxy.setdefault(b.proxy, []).append(b)
        for proxy, bs in by_proxy.items():
            if len({1 if b.n_sd > 0 else -1 for b in bs}) > 1:
                blocking.append(
                    f"proxy '{proxy}' is shocked in BOTH directions by "
                    f"{[b.channel[:30] for b in bs]} - decide which is right and prune the "
                    f"other, or say why the net is zero")

        if not any(b.depth > 0 for b in self.branches):
            advisory.append("no second-order channels - the expand step appears to be "
                            "missing")
        cov = channels.coverage(self.survivors())
        if cov["n_channel_types"] < 3:
            advisory.append(f"surviving branches touch only {cov['n_channel_types']} of the "
                            f"nine canonical channels - missing {cov['missing']}")
        if cov["untagged"]:
            advisory.append(f"{cov['untagged']} surviving branch(es) have no channel_type")
        for b in self.survivors():
            if b.proxy and b.n_sd:
                msg = channels.sanity_check_shock(b.proxy, b.n_sd)
                if msg:
                    advisory.append(msg)
            if not b.note:
                advisory.append(f"'{b.channel[:40]}': kept with no reason recorded")
        for x in advisory:
            print(f"    advisory: {x}")
        for x in blocking:
            print(f"    BLOCKING: {x}")
        return blocking

    def to_dict(self) -> dict:
        tot = {"generated": 0, "grounded": 0, "ungrounded": 0, "not_in_sources": 0}
        for phase in self.grounding.values():
            for k in tot:
                tot[k] += phase.get(k, 0)
        return {"scenario": self.scenario,
                "grounding_by_phase": self.grounding,
                "grounding_total": tot,
                "channel_coverage": channels.coverage(self.survivors()),
                "branches": [asdict(b) for b in self.branches],
                "n_generated": len(self.branches),
                "n_kept": len(self.survivors()),
                "n_rejected": sum(1 for b in self.branches if b.keep is False),
                "n_modellable": len(self.modellable()),
                "n_unmodellable": len(self.unmodellable()),
                "directions_kept": sorted({b.direction for b in self.survivors()})}


# -------------------------------------------------------------------------------------------
# The shock model: FROZEN, and CONSTRUCT-FREE
# -------------------------------------------------------------------------------------------
# Two properties that are not optional.
#
# CONSTRUCT-FREE. The Cycle model uses four tiers, one of which is the seven Words
# constructs. A scenario has no minutes. To shock a construct-using model you would have to
# invent the construct values the Board WOULD have written, which is the thing you are trying
# to work out - the scenario would supply its own answer. So the shock model is fitted on
# persistence + macro + market only. The constructs are still used in Shock, but through
# `reaction_profile()`, where they are compared against real episodes rather than fabricated.
#
# FROZEN. The model is fitted once, on the frozen panel, and cached. Every team shocks the
# same coefficients, so a difference between two teams' answers is a difference in their
# reasoning rather than a difference in whose panel happened to be rebuilt when.

SHOCK_TIERS = ["persistence", "macro", "market"]
_FROZEN_CACHE: dict = {}


def fit_reference(panel: pd.DataFrame, tiers: dict, target: str = "y_cycle",
                  upto: list[str] | None = None, clf=None):
    """
    Fit the reference model. Defaults to the CONSTRUCT-FREE tiers - see the note above.

    Passing `upto=["persistence", "macro", "market", "text"]` is possible and is what the
    Cycle stage does, but a tree of channels cannot shock a text feature honestly, so do not
    use it here.
    """
    upto = upto or SHOCK_TIERS
    X = ev.build_design(panel, tiers, upto)
    y = panel[target]
    ok = y.notna()
    return (clf or ev.default_classifier()).fit(X[ok], y[ok]), X


def frozen_shock_model():
    """
    LOADS the exported construct-free Shock model. Returns (clf, X, panel).

    It does not fit anything. An earlier version called `fit_reference()` here, which
    silently trained a multinomial logistic regression at runtime while the brief explained
    out-of-support behaviour in terms of a gradient-boosted tree holding its edge value
    constant. Those are different models with opposite extrapolation behaviour - a logit's
    logits keep moving linearly forever - so the documentation described something nobody
    was running, and the support argument did not apply to the model in use.

    The artefact is built by `model_card.run()`: same estimator family as the Cycle card,
    same training meetings, no text tier. It ships with the repository, so this works on a
    clean checkout with no API key.
    """
    if "m" in _FROZEN_CACHE:
        return _FROZEN_CACHE["m"]
    mpath = config.MODEL_CARD / "shock_model.joblib"
    dpath = config.MODEL_CARD / "shock_design.parquet"
    if not (mpath.exists() and dpath.exists()):
        raise FileNotFoundError(
            f"{mpath.name} / {dpath.name} are missing from the model card. They ship with "
            f"the repository; restore them, or rebuild with `python src/model_card.py`. Do "
            f"not substitute a model of your own - every team shocks the same one, or the "
            f"Shock answers are not comparable.")
    import joblib
    clf = joblib.load(mpath)
    X = pd.read_parquet(dpath)
    if not isinstance(X.index, pd.DatetimeIndex):
        X.index = pd.to_datetime(X.index)
    # The rows the model was fitted on, and the calibration that says how unusual an
    # ORDINARY unseen meeting is. Support is judged against both - see support_check().
    tpath = config.MODEL_CARD / "shock_train_design.parquet"
    if tpath.exists():
        T = pd.read_parquet(tpath)
        if not isinstance(T.index, pd.DatetimeIndex):
            T.index = pd.to_datetime(T.index)
        _FROZEN_CACHE["train"] = T
        cpath = config.MODEL_CARD / "support_calibration.json"
        if cpath.exists():
            _FROZEN_CACHE["calib"] = json.loads(cpath.read_text())
    else:
        raise FileNotFoundError(
            f"{tpath.name} is missing. Rebuild the card with `python src/model_card.py`; "
            f"support cannot be judged against rows the model never saw.")
    panel = pd.read_parquet(config.FROZEN_PANEL)
    panel["meeting_date"] = pd.to_datetime(panel["meeting_date"])
    panel = panel.set_index("meeting_date").sort_index()
    leaked = [c for c in X.columns if c in config.TEXT_FEATURES]
    if leaked:
        raise RuntimeError(f"the shock design carries text constructs {leaked}")
    _FROZEN_CACHE["m"] = (clf, X, panel)
    return clf, X, panel


def shock_model_family() -> str:
    clf, _, _ = frozen_shock_model()
    return type(clf).__name__


# -------------------------------------------------------------------------------------------
# Turning a tree into shocks
# -------------------------------------------------------------------------------------------

# Variables that must never be shocked, because shocking them assumes the answer.
#
# The cash rate and its derivatives ARE the policy stance. A channel that says "the RBA
# raises the cash rate, therefore the model predicts hardening" has assumed its conclusion,
# and the model will faithfully agree. The scenario shocks the ECONOMY; the model says what
# policy does about it. A channel naming one of these is treated as unmodellable and its
# reasoning is kept as text.
FORBIDDEN_PROXIES = {"cash_rate", "decision", "change_pct", "trailing_change_182d",
                     "trailing_change_364d", "meetings_since_change", "rate_after_decision"}


def _drop_forbidden(branches: list["Branch"]) -> int:
    n = 0
    for b in branches:
        if b.proxy in FORBIDDEN_PROXIES:
            b.note = (b.note + f" | PROXY '{b.proxy}' IS THE POLICY OUTCOME - shocking it "
                               f"would assume the conclusion. Channel kept as reasoning, "
                               f"removed from the shock.")[:900]
            b.proxy = None
            n += 1
    if n:
        print(f"    {n} channel(s) named a policy-outcome proxy and were made unmodellable")
    return n


def cap_to_calibration(branches: list["Branch"], allowance: float = 1.0) -> list[dict]:
    """
    Reduce any shock larger than the biggest CHANGE that variable has actually made.

    `channels.CALIBRATION` records the peak change each variable made in real episodes,
    measured from a pre-event baseline. A model asked to size a dramatic scenario routinely
    proposes moves with no referent, which push the row so far outside the training range
    that the resulting probabilities are manufactured.

    `allowance` scales the cap: 1.0 caps at the historical peak change, 1.5 permits half
    again for a scenario you argue is genuinely worse than anything on record. Whatever you
    choose, SAY SO - and report the UNCAPPED support status too, because capping until a run
    fits inside the training range is how you would hide the extrapolation rather than
    handle it.

    Variables with no calibration episode, and those `channels.UNCALIBRATABLE` lists, are
    left alone; `sanity_check_shock()` has already flagged them for you to size by argument.
    """
    log = []
    for b in branches:
        if not (b.keep and b.proxy and b.n_sd):
            continue
        if b.proxy in channels.UNCALIBRATABLE:
            continue
        cal = channels.calibration_for(b.proxy)
        if not len(cal):
            continue
        cap = float(cal["change_sd"].abs().max()) * allowance
        if abs(b.n_sd) > cap:
            before = b.n_sd
            b.n_sd = float(np.sign(b.n_sd) * cap)
            b.note = (b.note + f" | CAPPED {before:+.2f}sd -> {b.n_sd:+.2f}sd at "
                               f"{allowance:g}x the largest change on record")[:900]
            log.append({"channel": b.channel[:60], "proxy": b.proxy,
                        "proposed_sd": round(before, 2), "capped_sd": round(b.n_sd, 2),
                        "cap": round(cap, 2)})
    if log:
        print(f"    calibration cap ({allowance:g}x): {len(log)} shock(s) reduced")
        for r in log:
            print(f"      {r['proxy']:22s} {r['proposed_sd']:+.2f} -> {r['capped_sd']:+.2f}")
    return log


def cascade_orphans(branches: list["Branch"]) -> int:
    """
    Drop any surviving second-order branch whose parent no longer survives.

    A second-order channel exists only as a consequence of its parent. Once the parent is
    gone - pruned on score, or dropped by a proxy judgement - the child is not a separate
    decision to make, it is arithmetic. Applied repeatedly so a chain collapses in one pass.
    """
    # A parent barred as a RESTATEMENT is not a dead mechanism - its canonical carries the
    # claim. Resolving through `restatement_of` stops the review from orphaning an accepted
    # second-order canonical merely because its literal parent lost the coin-toss of which
    # restatement got expanded.
    alias = {b.channel: b.restatement_of for b in branches if b.restatement_of}

    def parent_alive(parent, kept):
        return parent in kept or alias.get(parent) in kept

    n = 0
    for _ in range(8):
        kept = {b.channel for b in branches if b.keep}
        orphans = [b for b in branches
                   if b.keep and b.depth > 0 and not parent_alive(b.parent, kept)]
        if not orphans:
            break
        for b in orphans:
            b.keep, b.kept_by = False, "orphaned"
            b.note = (b.note + f" | ORPHANED: its parent '{str(b.parent)[:40]}' did not "
                               f"survive, so this second-order channel cannot stand")[:900]
            n += 1
    if n:
        print(f"    cascade: {n} second-order channel(s) dropped as orphans")
    return n


def resolve_proxy_conflicts(branches: list["Branch"],
                            judgements: dict[str, dict] | None) -> list[dict]:
    """
    Where several surviving channels shock ONE proxy, YOUR judgement replaces the default.

    `judgements` is {proxy: {"central": float, "low": float, "high": float, "reason": str}},
    all in standard deviations. It does three jobs at once, because they are one decision:

      - it RESOLVES SIGN CONFLICTS. Channels arguing the other way are dropped from the
        shock and marked, but stay in the tree with your reason attached.
      - it SETS THE MAGNITUDE. The default when you say nothing is the largest absolute
        value proposed, which systematically favours whichever channel was most dramatic.
        An earlier version could only record a sign, so magnitude was never yours to choose.
      - it SUPPLIES THE UNCERTAINTY BOUNDS that `focused_corners()` varies.

    A proxy with a genuine net effect of zero is expressible: set central to 0.0.

    An unresolved sign conflict is left in place and `Tree.validate()` blocks on it. That is
    intended: it is the question the stage is asking you.
    """
    by_proxy: dict[str, list] = {}
    for b in branches:
        if b.keep and b.proxy and b.n_sd:
            by_proxy.setdefault(b.proxy, []).append(b)
    log = []
    for proxy, bs in sorted(by_proxy.items()):
        j = (judgements or {}).get(proxy)
        signs = {1 if b.n_sd > 0 else -1 for b in bs}
        if j is None:
            if len(signs) > 1:
                log.append({"proxy": proxy, "resolved": False, "n_conflicting": len(bs),
                            "note": "no team judgement recorded - this will block"})
            continue
        for k in ("central", "reason"):
            if k not in j:
                raise ValueError(f"judgement for '{proxy}' is missing '{k}'")
        want = j["central"]
        dropped = []
        for b in bs:
            # A central of exactly 0.0 is a real answer - "these forces cancel, or this
            # proxy should not carry the shock at all" - so it removes EVERY channel on the
            # proxy from the shock, not just one side. An earlier version skipped the zero
            # case entirely, so declaring a net of zero left the sign conflict in place and
            # the run stayed blocked with no way to express the judgement.
            drop = (b.n_sd != 0) if want == 0 else (
                (1 if b.n_sd > 0 else -1) != (1 if want > 0 else -1))
            if drop:
                b.keep, b.kept_by = False, "team proxy judgement"
                verdict = ("nets to zero" if want == 0
                           else f"is {want:+.2f}sd")
                b.note = (b.note + f" | DROPPED FROM SHOCK: the team judged the net effect "
                                   f"on {proxy} {verdict} ({j['reason'][:80]}); this channel "
                                   f"is recorded, not deleted")[:900]
                dropped.append(b.channel[:45])
        log.append({"proxy": proxy, "resolved": True,
                    "central_sd": want, "low_sd": j.get("low"), "high_sd": j.get("high"),
                    "reason": j["reason"], "n_conflicting": len(bs),
                    "had_sign_conflict": len(signs) > 1,
                    "dropped_from_shock": dropped})
        if len(signs) > 1:
            print(f"    proxy conflict on {proxy}: team set {want:+.2f}sd, "
                  f"{len(dropped)} channel(s) dropped from the shock")
    cascade_orphans(branches)
    return log


def adjudicate_shocks(branches: list["Branch"],
                      judgements: dict[str, dict] | None = None
                      ) -> tuple[list[tuple[str, float]], list[dict]]:
    """
    ONE shock per proxy. Your stated central value where you gave one; otherwise a default.

    Conflicts are surfaced, never summed. An earlier version accumulated, so a scenario with
    three channels naming `aud_ret` produced a shock three times the size purely because the
    model had restated one mechanism three times.

    THE DEFAULT IS DELIBERATELY UNATTRACTIVE. With no judgement recorded the largest absolute
    proposal wins, which biases every unadjudicated proxy towards the most dramatic channel.
    That is a reason to record a judgement, not a reasonable fallback.
    """
    _drop_forbidden(branches)
    by_proxy: dict[str, list] = {}
    for b in branches:
        if b.proxy and b.n_sd:
            by_proxy.setdefault(b.proxy, []).append(b)
    shocks, log = [], []
    for proxy, bs in sorted(by_proxy.items()):
        vals = [b.n_sd for b in bs]
        j = (judgements or {}).get(proxy)
        if j is not None and "central" in j:
            value, source = float(j["central"]), "team judgement"
        else:
            chosen = max(bs, key=lambda b: abs(b.n_sd))
            value, source = chosen.n_sd, "default: largest absolute proposal"
        log.append({"proxy": proxy, "n_channels": len(bs),
                    "values_proposed": [round(v, 3) for v in vals],
                    "adjudicated": round(value, 3), "set_by": source,
                    "sign_conflict": len({np.sign(v) for v in vals if v}) > 1,
                    "channels": [b.channel for b in bs]})
        if value:
            shocks.append((proxy, value))
    return shocks, log


def uncertainty_bounds(judgements: dict[str, dict] | None) -> dict[str, dict]:
    """The proxies you nominated as uncertain, in the shape `focused_corners()` wants."""
    out = {}
    for proxy, j in (judgements or {}).items():
        if j.get("low") is not None and j.get("high") is not None:
            out[proxy] = {"low": float(j["low"]), "central": float(j["central"]),
                          "high": float(j["high"]), "reason": j["reason"]}
    return out


def apply_shock(X: pd.DataFrame, panel: pd.DataFrame,
                shocks: list[tuple[str, float]],
                row_index: int = -1) -> tuple[pd.DataFrame, list[dict]]:
    """
    Shock one row by N historical standard deviations per variable.

    Standard deviations come from the panel's own history, so a shock is in units the model
    has actually seen: +2.5sd on the VIX is roughly the GFC, and any shock size can be
    checked against an episode that really happened.

    Repeated entries for one variable are applied in order and DO accumulate. Pass shocks
    through `adjudicate_shocks()` first if they came from a tree - see the note there.
    """
    row = X.iloc[[row_index]].copy()
    applied = []
    for var, n_sd in shocks:
        if var not in X.columns or n_sd == 0:
            continue
        sd = float(panel[var].std())
        base = float(row[var].iloc[0])
        row[var] = base + n_sd * sd
        applied.append({"variable": var, "n_sd": n_sd, "sd": round(sd, 4),
                        "from": round(base, 4),
                        "shocked_to": round(float(row[var].iloc[0]), 4)})
    return row, applied


# -------------------------------------------------------------------------------------------
# Support: when the model may speak at all
# -------------------------------------------------------------------------------------------
# Support is the OBSERVED RANGE of each shocked variable, plus a joint nearest-neighbour
# check. Inside it the tree interpolates between real data; outside it every split has
# already fired and the prediction is frozen at the edge value however much further the input
# moves.
#
# WHEN A RUN IS OUT OF SUPPORT THE MODEL RETURNS NOTHING. Not a clipped probability, and not
# a direction either. An earlier version withheld the probability but still reported
# `net_direction`, which was the argmax of the same extrapolated distribution that had just
# been declared meaningless - the number was suppressed and its rank order published. If the
# probabilities cannot be quoted then neither can the ordering derived from them.
#
# A qualitative direction is still available, from `channel_direction()`, but it is computed
# from the adjudicated channels rather than from the classifier, and it is labelled as such
# everywhere it appears.

# Below this share gap no direction is declared. A reporting convention, chosen
# for legibility - not a significance threshold, and it has no test behind it.
DIRECTION_MARGIN = 0.15

SUPPORT_Q = 0.0            # marginal support is the observed min/max


def training_design() -> pd.DataFrame:
    """The rows the frozen Shock model was fitted on. The reference for every support test."""
    if "train" not in _FROZEN_CACHE:
        frozen_shock_model()
    return _FROZEN_CACHE["train"]


def support_calibration() -> dict:
    """How unusual an ORDINARY unseen meeting is. Shipped with the model card."""
    if "calib" not in _FROZEN_CACHE:
        frozen_shock_model()
    if "calib" not in _FROZEN_CACHE:
        raise FileNotFoundError(
            "support_calibration.json is missing from the model card. Rebuild it with "
            "`python src/model_card.py`; without it the support test has no scale.")
    return _FROZEN_CACHE["calib"]


def support_check(X: pd.DataFrame, row: pd.DataFrame,
                  applied: list[dict], reference: pd.DataFrame | None = None) -> dict:
    """
    Is the shocked row further from the model's training data than REAL unseen data ever is?

    That is the question, and the reference population is the second half of it. Two checks,
    both calibrated on the 82 meetings the model was not fitted on:

    MARGINAL - how many shocked variables sit outside their training range. Ordinary unseen
        meetings breach a median of 6 of 38, and not one of the 82 is clean, so "any
        variable outside the range" is not a usable rule. The threshold is the 99th
        percentile of what real meetings do.

    JOINT - the standardised nearest-neighbour distance to the training rows. Ordinary unseen
        meetings sit at a median of 8.8; the threshold is their 99th percentile. Comparing
        instead against the spacing BETWEEN training rows, as an earlier version did, used a
        scale of 2.8 - which measures how alike consecutive RBA meetings are, not how spread
        the data is, and duly declared every real meeting out of support.

    A run is in support only if BOTH pass. Where they do, probabilities may be quoted; where
    they do not, `run_scenario()` returns no probability and no direction.
    """
    cal = support_calibration()
    ref = training_design() if reference is None else reference
    out = []
    for a in applied:
        v = a["shocked_to"]
        col = ref[a["variable"]].dropna() if a["variable"] in ref.columns else pd.Series(dtype=float)
        if col.empty:
            out.append({"variable": a["variable"], "shocked_to": round(v, 4),
                        "in_range": False, "sd_beyond": None,
                        "note": "not present in the training design"})
            continue
        lo, hi = float(col.min()), float(col.max())
        inside = lo <= v <= hi
        out.append({"variable": a["variable"], "shocked_to": round(v, 4),
                    "training_lo": round(lo, 4), "training_hi": round(hi, 4),
                    "in_range": inside,
                    "sd_beyond": (0.0 if inside else
                                  round(abs(v - (hi if v > hi else lo)) /
                                        (float(col.std()) or 1), 2))})
    n_out = sum(1 for o in out if not o["in_range"])
    marginal_ok = n_out <= cal["marginal_breaches_p99"] if "marginal_breaches_p99" in cal         else n_out <= cal["marginal_breaches_p95"]
    joint = _joint_support(ref, row, cal)
    ok = marginal_ok and joint["in_support"]
    return {"in_support": bool(ok),
            "marginal_in_support": bool(marginal_ok),
            "n_outside_marginal": n_out,
            "marginal_threshold": cal.get("marginal_breaches_p99",
                                          cal.get("marginal_breaches_p95")),
            "marginal_typical_for_a_real_meeting": cal["marginal_breaches_p50"],
            "variables": out,
            "joint": joint,
            "quotable": "probabilities" if ok else "nothing - the model cannot speak here"}


def _joint_support(X: pd.DataFrame, row: pd.DataFrame, cal: dict | None = None) -> dict:
    """
    Standardised nearest-neighbour distance from the row to the TRAINING rows, expressed as
    a percentile of how far ORDINARY unseen meetings sit from them.
    """
    cal = cal or support_calibration()
    cols = [c for c in cal["columns"] if c in X.columns and c in row.columns]
    A = X[cols].astype(float)
    mu, sd = A.mean(), A.std().replace(0, 1)
    Z = ((A - mu) / sd).fillna(0.0).to_numpy()
    z = ((row[cols].astype(float) - mu) / sd).fillna(0.0).to_numpy()
    from scipy.spatial import cKDTree
    d = float(cKDTree(Z).query(z, k=1)[0][0])
    thresh = float(cal["joint_heldout_p99"])
    return {"nearest_neighbour_distance": round(d, 3),
            "typical_real_unseen_meeting": cal["joint_heldout_p50"],
            "threshold_p99_of_real_meetings": round(thresh, 3),
            "in_support": bool(d <= thresh),
            "multiple_of_typical_unseen": round(d / (cal["joint_heldout_p50"] or 1), 2)}


def predict_state(clf, row: pd.DataFrame) -> dict:
    p = clf.predict_proba(row)[0]
    return {config.CYCLE_STATES[int(c)]: float(v) for c, v in zip(clf.classes_, p)}


def validate_direction_weights(weights: dict | None, reasons: dict | None,
                               required: bool = False) -> None:
    """
    Every weight needs a written reason. The number is not the judgement.

    `DIRECTION_WEIGHTS` replaces the LLM's own credibility scores with the team's, which is
    the point of it - but a bare dictionary of numbers is no more the team's reasoning than
    the scores were. The rubric marks the reason, so the code requires one, exactly as it
    does for `ADJUDICATIONS` and `PROXY_JUDGEMENTS`.
    """
    if not weights:
        if required:
            raise ValueError(
                "DIRECTION_WEIGHTS is empty. A reportable run needs your weight and your "
                "reason for every mechanism family in the tally - without them, "
                "channel_direction() falls back to the LLM's own credibility scores, and "
                "a tally of one model's self-assessment is not the team's judgement.")
        return
    missing = [k for k in weights if not str((reasons or {}).get(k, "")).strip()]
    if missing:
        raise ValueError(
            f"{len(missing)} direction weight(s) have no entry in DIRECTION_WEIGHT_REASONS: "
            f"{sorted(missing)}. A weight without a reason is a number you cannot defend, "
            f"and the qualitative direction is built out of these.")
    orphan = [k for k in (reasons or {}) if k not in weights]
    if orphan:
        raise ValueError(
            f"DIRECTION_WEIGHT_REASONS names {sorted(orphan)}, which have no weight. Either "
            f"the family was renamed or the reason is stale.")


def channel_direction(branches: list["Branch"],
                      weights: dict[str, float] | None = None) -> dict:
    """
    A qualitative direction from the adjudicated channels. NOT a model output.

    AGGREGATED BY (MECHANISM FAMILY, DIRECTION) CELL, NOT BY BRANCH COUNT.

    The unit is the CELL, not the family: one family can contribute once to tightening and
    once to easing, which is correct when a mechanism genuinely cuts both ways, and the
    reported `family_direction_cells` counts cells while `mechanism_families` counts distinct
    families. Calling the cell count a family count overstated how many independent
    mechanisms were behind a direction. The tree generates several
    branches for one underlying mechanism whenever the model restates it, so counting
    branches let verbosity decide the answer: three phrasings of "external demand falls"
    outvoted one "terms of trade rises" that was equally credible. Branches are grouped by
    `channel_type` and each mechanism family contributes once, at its best score.

    WEIGHTS ARE YOURS IF YOU SUPPLY THEM. `weights` is {channel_type: weight}; without it
    the LLM's own credibility scores are used and the result is labelled accordingly,
    because a tally of one model's self-assessment is not the team's reasoning.

    The margin below which no direction is declared is `DIRECTION_MARGIN`, and it is
    arbitrary - it is a reporting convention, not a test.
    """
    for k, v in (weights or {}).items():
        if not isinstance(v, (int, float)) or v < 0 or v != v:
            raise ValueError(f"direction weight for '{k}' must be a finite value >= 0, "
                             f"got {v!r}")
    kept = [b for b in branches if b.keep]
    # PARTIAL weights are worse than none: a tally mixing one human weight with the LLM's
    # credibility scores for the families the dictionary missed used to be labelled
    # "weighted by the TEAM" whenever the dictionary was non-empty. If weights are
    # supplied, every family actually IN the tally must have one - families with no
    # surviving branch may be omitted, and that is the coverage rule, stated.
    if weights:
        active = sorted({b.channel_type or "other" for b in kept})
        uncovered = [f for f in active if f not in weights]
        if uncovered:
            raise ValueError(
                f"DIRECTION_WEIGHTS covers {sorted(weights)} but the surviving branches "
                f"span {active}: {uncovered} would silently fall back to the LLM's own "
                f"credibility scores while the result was labelled team-weighted. Add a "
                f"weight and a reason for every family with a surviving branch (families "
                f"with no survivor may be omitted).")
    families: dict[tuple, float] = {}
    for b in kept:
        key = (b.channel_type or "other", b.direction if b.direction in
               ("tightening", "easing") else "ambiguous")
        w = (weights or {}).get(b.channel_type, None)
        score = w if w is not None else (b.score or 0.5)
        families[key] = max(families.get(key, 0.0), score)

    tally = {"tightening": 0.0, "easing": 0.0, "ambiguous": 0.0}
    for (_ctype, direction), score in families.items():
        tally[direction] += score
    total = sum(tally.values()) or 1.0
    share = {k: round(v / total, 3) for k, v in tally.items()}
    lead = max(("tightening", "easing"), key=lambda k: tally[k])
    other = "easing" if lead == "tightening" else "tightening"
    margin = share[lead] - share[other]
    verdict = lead if margin >= DIRECTION_MARGIN else "no clear direction"
    return {"source": ("adjudicated channels weighted by the TEAM" if weights else
                       "adjudicated channels weighted by the LLM's own credibility scores "
                       "- not a model prediction, and not independent of the model either"),
            "weights_supplied_by_team": bool(weights),
            "family_direction_cells": len(families),
            "mechanism_families": len({k[0] for k in families}),
            "n_branches": len(kept),
            "weights": {k: round(v, 2) for k, v in tally.items()},
            "shares": share,
            "margin": round(margin, 3),
            "margin_threshold": DIRECTION_MARGIN,
            "direction": verdict}


def opposite_of(state: str) -> str:
    """
    What arguing against a conclusion means, including when the conclusion is 'stable' or
    when the model declined to speak.
    """
    return {
        "easing": "hardening",
        "hardening": "easing",
        "stable": ("that policy MOVES - in either direction. The claim you are attacking is "
                   "that this shock is absorbed without a change in the policy stance, so "
                   "the counter-case is that it forces one, and you should say which way "
                   "and why"),
        "out_of_support": ("that the shock is in fact SMALL ENOUGH for the historical record "
                           "to speak to it. The claim you are attacking is that this "
                           "scenario has no precedent, so the counter-case is that a "
                           "precedent exists - name it and say what happened"),
        "no clear direction": ("that the channels do NOT cancel. The claim you are attacking "
                               "is that the opposing forces are balanced, so the counter-case "
                               "is that one side dominates - say which and why"),
    }.get(state, f"that '{state}' is wrong - argue the strongest available alternative")


# -------------------------------------------------------------------------------------------
# Horizons
# -------------------------------------------------------------------------------------------
# Branches carry a horizon, and an earlier version ignored it: a day-one VIX spike and a
# 2-4-quarter GDP contraction were added to the same static row and fed to the model as
# simultaneous inputs. That is not a scenario, it is a superposition of three different
# moments.
#
# Each run now names ONE horizon and uses only the channels that operate at it. The horizons
# are evaluated separately rather than accumulated, because the design matrix is a snapshot:
# it describes the economy at a single meeting, so the honest reading of a shocked row is
# "the state of the world at horizon h", not "everything that ever happens".

HORIZONS = {
    "immediate": ["days", "weeks"],
    "short": ["1-2q"],
    "medium": ["2-4q", "years"],
}
DEFAULT_HORIZON = "short"


def channels_at(branches: list["Branch"], horizon: str) -> list["Branch"]:
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}; choose from {list(HORIZONS)}")
    buckets = HORIZONS[horizon]
    return [b for b in branches if b.horizon in buckets]


# -------------------------------------------------------------------------------------------
# Running a tree
# -------------------------------------------------------------------------------------------

def responsiveness(clf, X: pd.DataFrame, panel: pd.DataFrame,
                   proxies: list[str], base_row: int = -1,
                   probe=(-3, -2, -1, 1, 2, 3)) -> pd.DataFrame:
    """
    How much can the model move at all, if each proxy is varied on its own?

    WHY THIS EXISTS. A gradient-boosted tree is piecewise constant: between two splits it
    does not respond at all. So "we shocked the economy and the forecast barely moved" has
    two completely different explanations, and you cannot tell them apart from the shift:

      the model disagrees that this shock matters      - a finding
      the model cannot respond to this proxy here      - an artefact of where the splits are

    This separates them. `max_move` is the largest probability change ANY shock to that
    proxy alone could produce from your base row; `your_move` is what YOUR adjudicated size
    actually produced. A channel with a large `max_move` and a near-zero `your_move` is a
    real channel that your sizing did not push past a threshold - say so, rather than
    reporting that the model was unmoved.
    """
    base = clf.predict_proba(X.iloc[[base_row]])[0]
    rows = []
    for p in proxies:
        if p not in X.columns:
            continue
        best, at = 0.0, None
        for n in probe:
            r, _ = apply_shock(X, panel, [(p, n)], row_index=base_row)
            d = float(np.abs(clf.predict_proba(r)[0] - base).max())
            if d > best:
                best, at = d, n
        rows.append({"proxy": p, "max_move": round(best, 3), "at_n_sd": at})
    # No proxies is a real state, not an error: it is what a tree looks like while the
    # channel reviews are still being written, and every one of its channels is barred.
    # `pd.DataFrame([])` has no columns, so sorting on one raised KeyError and took the
    # whole run down at the exact moment a team most needs to see the group keys.
    if not rows:
        return pd.DataFrame(columns=["proxy", "max_move", "at_n_sd"])
    out = pd.DataFrame(rows).sort_values("max_move", ascending=False)
    flat = out[out["max_move"] < 0.02]
    if len(flat):
        print(f"    the model is LOCALLY FLAT to {len(flat)} of {len(out)} shocked proxies "
              f"at this base row: {list(flat['proxy'])}")
        print(f"    that is a property of where the tree's splits are, not evidence that "
              f"those channels do not matter")
    return out


def run_scenario(tree: Tree, clf, X: pd.DataFrame, panel: pd.DataFrame,
                 strict: bool = True, horizon: str = DEFAULT_HORIZON,
                 base_row: "int | str" = -1,
                 judgements: dict[str, dict] | None = None,
                 weights: dict[str, float] | None = None) -> dict:
    """
    Collapse a pruned tree into shocks at ONE horizon, apply them, and report what may be
    reported - which is sometimes nothing.

    `strict=True` makes `Tree.validate()` blocking: a tree with a blocking defect raises
    rather than returning a headline. `base_row` is the meeting the shock is applied to; it
    is recorded in the result, because every answer is conditional on that starting state.
    """
    blocking = tree.validate(list(X.columns))
    if strict and blocking:
        raise ValueError(
            "This tree cannot produce a reportable result until these are fixed:\n  - "
            + "\n  - ".join(blocking)
            + "\n(run with strict=False to inspect the numbers anyway - but they are not "
              "reportable)")

    at_h = [b for b in channels_at(tree.modellable(), horizon)]
    excluded = [b for b in tree.modellable() if b not in at_h]
    # `base_row` may be a positional index or a meeting date.
    if isinstance(base_row, str):
        base_row = int(X.index.get_loc(pd.Timestamp(base_row)))
    base_date = str(X.index[base_row].date()) if isinstance(X.index, pd.DatetimeIndex) else \
        str(base_row)

    # BASELINE SUPPORT, BEFORE ANY SHOCK. If the starting row is already outside the data
    # the model was fitted on, nothing downstream can rescue it - the shock is being applied
    # to a state the model has never seen. This is checked first and reported first, because
    # discovering it after the shock invites the reading that the shock caused it.
    base_support = support_check(X, X.iloc[[base_row]], [])
    base = predict_state(clf, X.iloc[[base_row]])
    if not base_support["joint"]["in_support"]:
        j = base_support["joint"]
        print(f"    BASELINE ALREADY OUT OF SUPPORT: the unshocked {base_date} row sits "
              f"{j['nearest_neighbour_distance']} from the nearest of the "
              f"{len(training_design())} training rows, against {j['threshold_p99_of_real_meetings']} "
              f"for the 99th-percentile real unseen meeting.")
        print(f"    No shock result from this base row can be quotable. Choose a starting "
              f"state the model has seen something like.")
    shocks, adjudication = adjudicate_shocks(at_h, judgements)
    row, applied = apply_shock(X, panel, shocks, row_index=base_row)
    after = predict_state(clf, row)
    support = support_check(X, row, applied)
    # The SAME horizon-filtered set the shock used. Calling this on every survivor let a
    # "short horizon" refusal be accompanied by a qualitative direction that silently
    # combined immediate, short and medium mechanisms.
    at_h_all = channels_at(tree.survivors(), horizon)
    chan = channel_direction(at_h_all, weights=weights)
    chan["horizon"] = horizon
    chan["n_survivors_other_horizons"] = len(tree.survivors()) - len(at_h_all)

    print(f"    horizon '{horizon}' ({'/'.join(HORIZONS[horizon])}): "
          f"{len(at_h)} of {len(tree.modellable())} modellable channels apply, "
          f"{len(excluded)} operate at other horizons")
    print(f"    base row: {base_date} (every answer is conditional on this starting state)")
    print(f"    channels: {len(tree.branches)} generated, {len(tree.survivors())} kept, "
          f"{len(shocks)} distinct proxies shocked, {len(tree.unmodellable())} unmodellable")
    for a in adjudication:
        if a["n_channels"] > 1:
            print(f"    adjudicated {a['proxy']}: {a['values_proposed']} -> "
                  f"{a['adjudicated']}" + ("  SIGN CONFLICT" if a["sign_conflict"] else ""))

    # HEADROOM. A shock cannot show in a model that is already certain. If the unshocked
    # row sits at 0.93 on one state, the largest honest move left is 0.07, and a small
    # shift then means "there was nowhere to go", not "the shock does not matter".
    headroom = round(1.0 - float(max(base.values())), 3)
    if headroom < 0.15:
        print(f"    LITTLE HEADROOM: the unshocked {base_date} row is already "
              f"{max(base.values()):.2f} on '{max(base, key=base.get)}'. At most "
              f"{headroom:.2f} of probability can move, so read the SHIFT, not its size.")
    resp = responsiveness(clf, X, panel, [p for p, _ in shocks], base_row=base_row)
    reportable = support["in_support"]
    if reportable:
        shift = {k: round(after[k] - base[k], 3) for k in after}
        print(f"    baseline {({k: round(v, 3) for k, v in base.items()})}")
        print(f"    shocked  {({k: round(v, 3) for k, v in after.items()})}")
        print(f"    shift    {shift}  ->  {max(after, key=after.get).upper()}")
        verdict = max(after, key=after.get)
    else:
        shift = None
        why = []
        if not support["marginal_in_support"]:
            why.append(f"{support['n_outside_marginal']} variable(s) outside their observed "
                       f"range: "
                       f"{[o['variable'] for o in support['variables'] if not o['in_range']]}")
        if not support["joint"]["in_support"]:
            jj = support["joint"]
            why.append(f"the shocked row sits {jj['nearest_neighbour_distance']} from the "
                       f"nearest training row - further than the 99th-percentile real "
                       f"unseen meeting, at {jj['threshold_p99_of_real_meetings']}")
        verdict = "out_of_support"
        print(f"    MODEL VERDICT: OUT OF SUPPORT - no probability and no direction.")
        for w in why:
            print(f"      {w}")
        print(f"    channel-derived direction (NOT the model): {chan['direction']} "
              f"(shares {chan['shares']})")

    return {**tree.to_dict(),
            "baseline_support": base_support,
            "baseline_in_support": bool(base_support["joint"]["in_support"]),
            "horizon": horizon,
            "horizon_buckets": HORIZONS[horizon],
            "base_row_date": base_date,
            "headroom": headroom,
            "n_channels_at_horizon": len(at_h),
            "n_channels_other_horizons": len(excluded),
            "channels_excluded_by_horizon": [
                {"channel": b.channel[:60], "horizon": b.horizon} for b in excluded],
            "model_verdict": verdict,
            "baseline": base if reportable else None,
            "probabilities": after if reportable else None,
            "shift": shift,
            "applied_shocks": applied,
            "adjudication": adjudication,
            "support": support,
            "responsiveness": resp.to_dict("records"),
            "reportable_as": support["quotable"],
            "channel_direction": chan,
            "counter_case_is": opposite_of(
                verdict if verdict != "out_of_support" else "out_of_support")}


def apply_adjudications(branches: list["Branch"],
                        adjudications: dict[str, dict] | None) -> int:
    """
    Apply the team's branch-level keep/drop overrides. Returns how many were applied.

    THE BUG THIS REPLACES. The sensitivity sweep assigned the whole record to `keep`:

        b.keep = {"keep": False, "reason": "...", "by": "AB"}

    A dict is truthy, so an explicit student REJECTION was read as a keep, and the sweep
    silently reinstated exactly the channels a team had ruled out. Extracting
    `record["keep"]` is the whole fix, and this function exists so there is only one place
    it can be got wrong.
    """
    if not adjudications:
        return 0
    by = {k.strip().lower()[:60]: v for k, v in adjudications.items()}
    n = 0
    for b in branches:
        rec = by.get(b.channel.strip().lower()[:60])
        if rec is None:
            continue
        if not isinstance(rec, dict) or "keep" not in rec or not rec.get("reason"):
            raise ValueError(
                f"adjudication for '{b.channel[:45]}' must be a dict with 'keep', 'reason' "
                f"and 'by'. A bare boolean records a decision with no reason, and the "
                f"rubric marks the reason.")
        if bool(rec["keep"]) != bool(b.keep):
            b.keep = bool(rec["keep"])
            b.kept_by = f"student adjudication ({rec.get('by', '??')})"
            b.adjudication = rec["reason"]
            b.note = (b.note + f" | ADJUDICATED BY {rec.get('by', 'the team')}: "
                               f"{'kept' if b.keep else 'dropped'} against a score of "
                               f"{b.score if b.score is None else round(b.score, 2)} - "
                               f"{rec['reason']}")[:900]
            n += 1
    return n


def opposing_direction_guard(branches: list["Branch"], enabled: bool = True) -> list[str]:
    """
    Reinstate the best-scoring branch of a direction that pruning removed entirely.

    Every scenario here has genuinely opposing channels, so a surviving set that points one
    way usually means a real channel was pruned rather than a bad one. The reinstated branch
    is flagged and `Tree.validate()` blocks until a person signs it off.
    """
    if not enabled:
        return []
    kept = [b for b in branches if b.keep]
    if not kept:
        return []
    dirs = {b.direction for b in kept}
    out = []
    for want in ("tightening", "easing"):
        if want in dirs:
            continue
        # Never reinstate a branch a person rejected or the grounding bar removed. The
        # guard exists to recover a channel PRUNING dropped, not to overturn a review.
        cands = [b for b in branches
                 if b.direction == want and not b.keep and not b.terminal_exclusion]
        if not cands:
            continue
        best = max(cands, key=lambda b: b.score or 0)
        best.keep, best.kept_by = True, "opposing-direction guard"
        best.note = (best.note + " | REINSTATED BY OPPOSING-DIRECTION GUARD - defend or "
                                 "drop this explicitly")[:900]
        out.append(best.channel)
    return out


def _apply_terminal_exclusions(branches: list["Branch"]) -> int:
    """
    Re-impose the decisions a pruning threshold is not allowed to undo.

    THE BUG THIS REPLACES. `adjudicate_tree()` opened by resetting `keep` from the
    credibility score, and replayed neither the channel reviews nor the grounding bar.
    A minimal probe made the point: a branch with keep=False, a recorded human
    rejection and evidence_status="neither" came back with keep=True after
    `adjudicate_tree(..., threshold=0.5)`. On real trees the lower rungs of a
    sensitivity sweep reinstated a batch of unsupported branches in both scenarios.
    Their proxies had been erased, so the classifier probability was unaffected - but
    `channel_direction()` tallies surviving UNMODELLABLE branches too, so they moved
    the qualitative direction, which is the output the report leads with.

    A human rejection and a grounding bar are properties of the branch. Re-pruning
    cannot reach them, at any threshold, in any sweep.
    """
    n = 0
    for b in branches:
        if not b.terminal_exclusion:
            continue
        if b.keep or b.proxy:
            n += 1
        b.keep, b.proxy = False, None
        b.kept_by = b.terminal_exclusion
    return n


def assert_no_terminal_survivors(tree: "Tree") -> None:
    """Belt and braces: nothing terminally excluded may appear among the survivors."""
    bad = [b.channel[:40] for b in tree.survivors() if b.terminal_exclusion]
    if bad:
        raise AssertionError(
            f"{len(bad)} terminally excluded branch(es) survived pruning: {bad}. This "
            f"is the sweep-reinstatement defect; _apply_terminal_exclusions() should "
            f"have removed them.")


def adjudicate_tree(tree: "Tree", threshold: float,
                    adjudications: dict[str, dict] | None = None,
                    pathway_adjudications: dict[str, str] | None = None,
                    judgements: dict[str, dict] | None = None,
                    guard: bool = True, quiet: bool = False) -> dict:
    """
    THE ONE adjudication pipeline. Used by the headline run and by every sweep threshold.

    Order matters and is fixed:

      1. prune on the credibility score
      2. RE-IMPOSE the terminal exclusions - human rejections and grounding bars
      3. apply the team's branch-level overrides
      4. re-attach recorded pathway/guard sign-offs
      5. run the opposing-direction guard, which may not reinstate a terminal branch
      6. resolve proxy conflicts from the team's judgements
      7. re-impose the terminal exclusions again, and cascade orphaned branches

    Step 2 is what makes a `reject` mean reject at EVERY threshold. Without it the
    sweep silently re-ran the analysis with the reviewed-out channels put back.

    Having two implementations of this was how the sweep came to reinstate rejected channels
    and skip the guard while its docstring claimed it replayed everything.
    """
    for b in tree.branches:
        b.keep = (b.score or 0) >= threshold
        b.kept_by = "threshold"
    _apply_terminal_exclusions(tree.branches)
    n_adj = apply_adjudications(tree.branches, adjudications)
    if pathway_adjudications:
        byp = {k.strip().lower()[:60]: v for k, v in pathway_adjudications.items()}
        for b in tree.branches:
            rec = byp.get(b.channel.strip().lower()[:60])
            if rec:
                b.adjudication = rec
    reinstated = opposing_direction_guard(tree.branches, enabled=guard)
    conflicts = resolve_proxy_conflicts(tree.branches, judgements)
    _apply_terminal_exclusions(tree.branches)
    n_terminal = sum(1 for b in tree.branches if b.terminal_exclusion)
    before = sum(1 for b in tree.branches if b.keep)
    cascade_orphans(tree.branches)
    orphans = before - sum(1 for b in tree.branches if b.keep)
    if not quiet:
        if n_adj:
            print(f"    {n_adj} branch(es) overridden by the team")
        if reinstated:
            print(f"    opposing-direction guard reinstated {len(reinstated)}: "
                  f"{[c[:40] for c in reinstated]}")
    assert_no_terminal_survivors(tree)
    return {"threshold": threshold, "n_adjudicated": n_adj,
            "n_terminally_excluded": n_terminal,
            "reinstated": reinstated, "orphans_dropped": orphans,
            "proxy_conflicts": conflicts,
            "n_kept": len(tree.survivors())}


def prune_sweep(tree: Tree, clf, X: pd.DataFrame, panel: pd.DataFrame,
                thresholds=(0.3, 0.5, 0.7),
                judgements: dict[str, dict] | None = None,
                horizon: str = DEFAULT_HORIZON,
                adjudications: dict[str, dict] | None = None,
                pathway_adjudications: dict[str, str] | None = None,
                base_row: "int | str" = -1,
                weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    CONDITIONAL RE-PRUNING SENSITIVITY. Not a full re-run of the tree.

    WHAT IT DOES NOT DO, stated plainly because the name invites the wrong reading. Expansion
    happened once, on the branches that survived the HEADLINE threshold. This re-prunes the
    tree that produced, so at a lower threshold it can reinstate a first-order branch whose
    second-order consequences were never generated. Your sweep will report how many; the
    point is that they are branches whose downstream consequences do not exist. It answers
    "what would this tree have concluded under a different threshold", not "what would the
    analysis have concluded". Report it as the former.

    RUNS ON A DEEP COPY, AND REPLAYS THE WHOLE ADJUDICATION PIPELINE. Two separate faults
    made the previous version untrustworthy:

      - it re-pruned on the credibility score alone, so branch-level `ADJUDICATIONS` and
        pathway decisions were not reapplied and a branch the team had explicitly rejected
        could be reinstated at a different threshold;
      - it restored `keep`/`kept_by` afterwards but conflict resolution also mutates the
        notes, so running a sensitivity sweep permanently altered the submitted tree record.

    A deep copy fixes the second completely and lets the first be done properly: at every
    threshold the copy is re-pruned, re-adjudicated, conflict-resolved and cascaded exactly
    as the headline run was.

    WHAT A LOWER THRESHOLD MAY NOT DO. It may not reinstate a branch a person rejected
    in CHANNEL_REVIEWS, and it may not reinstate one the grounding bar removed. Those
    are terminal - see `_apply_terminal_exclusions()`. Only branches that passed both
    gates and lost on their credibility score are in play here.

    Out-of-support rows report `out_of_support` and NO probabilities.
    """
    import copy as _copy

    if isinstance(base_row, str):
        base_row = int(X.index.get_loc(pd.Timestamp(base_row)))
    rows = []
    for t in thresholds:
        w = _copy.deepcopy(tree)
        info = adjudicate_tree(w, t, adjudications=adjudications,
                               pathway_adjudications=pathway_adjudications,
                               judgements=judgements, quiet=True)
        at_h = channels_at(w.modellable(), horizon)
        shocks, _ = adjudicate_shocks(at_h, judgements)
        row, applied = apply_shock(X, panel, shocks, row_index=base_row)
        sup = support_check(X, row, applied)
        # Which first-order survivors at THIS threshold were never expanded. Expansion ran
        # once, on the headline threshold's survivors, so a lower rung reinstates
        # first-order branches whose second-order consequences do not exist and were never
        # generated. Counting them is the difference between a stated limitation and a
        # hidden one.
        parents = {b.channel for b in w.branches if b.depth > 0 and b.parent}
        unexpanded = sum(1 for b in w.survivors(0) if b.channel not in parents)
        rec = {"threshold": t, "n_kept": info["n_kept"],
               "n_adjudicated": info["n_adjudicated"],
               "n_terminally_excluded": info["n_terminally_excluded"],
               "n_first_order_never_expanded": unexpanded,
               "n_reinstated": len(info["reinstated"]),
               "orphans_dropped": info["orphans_dropped"],
               "n_proxies": len(shocks),
               "in_support": sup["in_support"],
               "channel_direction": channel_direction(
                   channels_at(w.survivors(), horizon), weights=weights)["direction"]}
        if sup["in_support"]:
            after = predict_state(clf, row)
            rec.update({k: round(v, 3) for k, v in after.items()})
            rec["model_verdict"] = max(after, key=after.get)
        else:
            rec.update({s: None for s in config.CYCLE_STATES.values()})
            rec["model_verdict"] = "out_of_support"
        rows.append(rec)

    df = pd.DataFrame(rows)
    print("    CONDITIONAL RE-PRUNING (expansion is fixed at the headline threshold):")
    print(df.to_string(index=False))
    worst = int(df["n_first_order_never_expanded"].max())
    if worst:
        print(f"    ^ up to {worst} first-order survivor(s) at some threshold have NO "
              f"second-order branches, because expansion ran once at the headline "
              f"threshold. Those rungs answer 'what would THIS TREE have concluded', not "
              f"'what would the analysis have concluded'.")
    if df["model_verdict"].nunique() > 1:
        print("    ^ the verdict CHANGES with the threshold - say so in your report")
    if (df["model_verdict"] == "out_of_support").all():
        print("    ^ the model cannot speak at ANY threshold; only the channel-derived "
              "direction is available")
    return df


# -------------------------------------------------------------------------------------------
# Focused sensitivity: the bounds YOU state, on the proxies YOU nominate
# -------------------------------------------------------------------------------------------
# This used to pick the four largest shocks automatically and multiply them by 0.5 and 1.5.
# Both halves of that were wrong. "Largest" is not the same as "most uncertain" - a shock you
# are confident about can be large, and the one you are least sure of can be small - and a
# fixed multiplier is a made-up interval wearing the costume of an analysis.
#
# You now nominate three or four genuinely uncertain proxies and state low, central and high
# for each, with a reason. The reason is marked; the arithmetic is not.

MIN_UNCERTAIN_AXES = 3
MAX_UNCERTAIN_AXES = 4


def focused_corners(tree: Tree, clf, X: pd.DataFrame, panel: pd.DataFrame,
                    bounds: dict[str, dict] | None = None,
                    horizon: str = DEFAULT_HORIZON,
                    judgements: dict[str, dict] | None = None,
                    base_row: "int | str" = -1,
                    weights: dict[str, float] | None = None,
                    require_bounds: bool = False) -> pd.DataFrame:
    """
    Evaluate the corners of YOUR stated uncertainty. At most 2^4 + 1 = 17 runs.

    `bounds` is {proxy: {"low": float, "central": float, "high": float, "reason": str}} in
    standard deviations - the same units as `n_sd`. Proxies you do not nominate are held at
    their adjudicated value.

    Corners whose combination is incoherent for THIS scenario are excluded, using the
    `coherence` rules you supply on the tree rather than a hard-coded global list - two
    proxies that must move together in one scenario need not in another.

    Every corner is support-checked independently. A corner out of support reports
    `out_of_support` and no probabilities.
    """
    from itertools import product

    if isinstance(base_row, str):
        base_row = int(X.index.get_loc(pd.Timestamp(base_row)))
    at_h = channels_at(tree.modellable(), horizon)
    adjudicated, _ = adjudicate_shocks(at_h, judgements)
    if not adjudicated:
        return pd.DataFrame()
    if not bounds:
        msg = ("no uncertainty bounds supplied. Nominate three or four uncertain proxies "
               "in PROXY_JUDGEMENTS with low/central/high and a reason - the sensitivity "
               "analysis is a judgement you make, not one the code makes for you.")
        if require_bounds:
            # The brief says a reportable run REFUSES without bounds. This used to print a
            # warning and return an empty table, so a run with no sensitivity at all still
            # produced a headline.
            raise ValueError("focused_corners: " + msg)
        print("    NO UNCERTAINTY BOUNDS - exploratory run only. " + msg)
        return pd.DataFrame()

    named = [p for p in bounds if p in dict(adjudicated)]
    missing = [p for p in bounds if p not in dict(adjudicated)]
    if missing:
        print(f"    bounds given for proxies not shocked at this horizon: {missing} "
              f"(ignored)")
    if len(named) > MAX_UNCERTAIN_AXES:
        raise ValueError(
            f"{len(named)} uncertainty axes exceeds the {MAX_UNCERTAIN_AXES} the budget "
            f"allows (2^{MAX_UNCERTAIN_AXES} + 1 = 17 runs). Choose the ones you are least "
            f"sure of.")
    n_available = len(adjudicated)
    if len(named) < MIN_UNCERTAIN_AXES:
        if n_available >= MIN_UNCERTAIN_AXES:
            raise ValueError(
                f"only {len(named)} of the {n_available} proxies shocked at horizon "
                f"'{horizon}' have low/high bounds. The rubric asks for "
                f"{MIN_UNCERTAIN_AXES}-{MAX_UNCERTAIN_AXES}. Nominate more, or move the "
                f"bounds you did give onto proxies that are actually shocked here.")
        print(f"    only {n_available} proxies are shocked at this horizon, so "
              f"{len(named)} axes is all that is available - stated, not hidden")
    for p in named:
        b = bounds[p]
        for k in ("low", "central", "high", "reason"):
            if k not in b:
                raise ValueError(f"bounds for '{p}' is missing '{k}'")
        if not (b["low"] <= b["central"] <= b["high"]):
            raise ValueError(f"bounds for '{p}' are not ordered: {b['low']} / "
                             f"{b['central']} / {b['high']}")

    fixed = [(v, n) for v, n in adjudicated if v not in named]
    central = fixed + [(p, bounds[p]["central"]) for p in named]
    rows = [{"corner": "central", "n_high": 0,
             **_evaluate_corner(clf, X, panel, central, base_row)}]

    coherence = getattr(tree, "coherence", None) or []
    # Severity is derived from the larger absolute move - which is ill-defined when a
    # proxy's bounds CROSS ZERO: both endpoints are then "the big move", in opposite
    # economic directions, and a co-movement rule can no longer tell them apart. A proxy
    # in a coherence pair with cross-zero bounds must declare its severe endpoint.
    in_rules = {p for r in coherence for p in r.get("proxies", [])}
    for p in named:
        bb = bounds[p]
        if p in in_rules and bb["low"] < 0 < bb["high"] and bb.get("severe") not in (
                "low", "high"):
            raise ValueError(
                f"bounds for '{p}' cross zero ({bb['low']} to {bb['high']}) and '{p}' is "
                f"in a coherence rule. Add 'severe': 'low' or 'high' to its "
                f"PROXY_JUDGEMENTS entry - once the sign changes inside the interval, the "
                f"larger |move| no longer identifies the severe end, and the co-movement "
                f"rule would judge corners at random.")
    excluded = []
    for combo in product(["low", "high"], repeat=len(named)):
        pick = dict(zip(named, combo))
        # The SEVERE endpoint per proxy: the larger absolute move, or the declared one
        # where the bounds cross zero.
        severity = {p: (pick[p] == bounds[p]["severe"] if bounds[p].get("severe") in
                        ("low", "high")
                        else abs(bounds[p][pick[p]]) >= abs(bounds[p]["central"]))
                    for p in named}
        why = _incoherent(pick, coherence, severity)
        label = " ".join(f"{p}:{pick[p]}" for p in named)
        if why:
            excluded.append((label, why))
            continue
        scaled = fixed + [(p, bounds[p][pick[p]]) for p in named]
        rows.append({"corner": label, "n_high": sum(1 for v in combo if v == "high"),
                     **_evaluate_corner(clf, X, panel, scaled, base_row)})

    df = pd.DataFrame(rows)
    print(f"    focused sensitivity at horizon '{horizon}': {len(named)} student-nominated "
          f"axes ({', '.join(named)}), {len(df)} evaluations, {len(excluded)} incoherent "
          f"corners excluded")
    for lab, why in excluded:
        print(f"      excluded {lab}: {why}")
    print(df[["corner", "model_verdict", "in_support"]].to_string(index=False))
    n_speak = int(df["in_support"].sum())
    if n_speak == 0:
        print("    ^ the model cannot speak at ANY corner - report the channel-derived "
              "direction and say why the model is silent")
    elif df.loc[df["in_support"], "model_verdict"].nunique() > 1:
        print("    ^ among the corners where the model CAN speak, the verdict is not robust")
    return df


def _incoherent(pick: dict, coherence: list[dict], severity: dict) -> str | None:
    """
    Scenario-specific coherence, judged on SCENARIO SEVERITY rather than on the words
    "low" and "high".

    Each rule is {"proxies": [a, b], "reason": "..."}: those two move together in THIS
    scenario, so a corner where one is at its severe end and the other at its mild end is
    not a smaller or larger version of the same event.

    WHY SEVERITY AND NOT THE LABELS. For a negatively signed shock the numerically LOW
    bound is the more severe one - a GDP shock of -3.5 is worse than -0.8. Comparing the
    labels excluded corners where a negative proxy and a positive proxy were both at their
    severe ends, which is precisely the coherent corner, and admitted the incoherent one.
    `severity` maps each proxy to True when the picked endpoint is the more extreme.

    A hard-coded global list used to assert that, for example, the AUD and the terms of
    trade always co-move - true in a commodity shock, false in a pure risk-off event, and
    the code applied it to both.
    """
    for rule in coherence:
        ps = [p for p in rule.get("proxies", []) if p in pick]
        if len(ps) < 2:
            continue
        if len({severity[p] for p in ps}) > 1:
            return rule.get("reason", f"{ps} move together in this scenario, so they cannot "
                                      f"be at opposite ends of their ranges")
    return None


def _evaluate_corner(clf, X, panel, shocks, base_row: int = -1) -> dict:
    """
    One corner, evaluated FROM THE SELECTED BASE ROW.

    An earlier version resolved `base_row` in `focused_corners()` and then never passed it
    here, so every corner silently used the default of -1 - the latest meeting. A headline
    computed from the neutral row was therefore reported next to corners computed from a
    different starting state, and the two disagreed. Both numbers were correct; they were
    answers to different questions, and nothing said so.
    """
    row, applied = apply_shock(X, panel, shocks, row_index=base_row)
    base = predict_state(clf, X.iloc[[base_row]])
    when = (str(X.index[base_row].date())
            if isinstance(X.index, pd.DatetimeIndex) else str(base_row))
    sup = support_check(X, row, applied)
    if not sup["in_support"]:
        return {**{s: None for s in config.CYCLE_STATES.values()},
                "model_verdict": "out_of_support", "in_support": False,
                "base_row_date": when,
                "baseline": {k: round(v, 3) for k, v in base.items()}, "shift": None}
    p = predict_state(clf, row)
    return {**{k: round(v, 3) for k, v in p.items()},
            "model_verdict": max(p, key=p.get), "in_support": True,
            "base_row_date": when,
            "baseline": {k: round(v, 3) for k, v in base.items()},
            "shift": {k: round(p[k] - base[k], 3) for k in p}}


# -------------------------------------------------------------------------------------------
# The reaction profile - the construct vocabulary applied to a scenario
# -------------------------------------------------------------------------------------------
# The seven constructs from Words are the DIMENSIONS along which the Board's reaction can be
# described. Shocking economic proxies tells you where the cycle model thinks policy goes; it
# does not tell you what the Board's REACTION would look like. This does.
#
# The workflow:
#   1. state the reaction you expect, as a percentile on each construct dimension
#   2. compare it against a historical episode whose reaction you can actually observe
#   3. where the two disagree, either your scenario is genuinely unlike that episode, or
#      your expectation is wrong - and you have to say which

EPISODES = {
    "GFC": ("2008-10-01", "2009-02-28"),
    "COVID": ("2020-03-01", "2020-06-30"),
    "inflation surge": ("2022-05-01", "2023-06-30"),
    "calm": ("2015-01-01", "2016-12-31"),
}


def reaction_profile(panel: pd.DataFrame, start: str | None = None,
                     end: str | None = None) -> dict:
    """
    Percentile rank of each construct over a window, against the whole corpus.

    Percentiles, not raw scores, because a construct's absolute level depends on how the
    team wrote its rubric. Rank against your own corpus is comparable; 0.61 is not.
    """
    cols = [c for c in config.TEXT_FEATURES if c in panel.columns]
    pct = panel[cols].rank(pct=True)
    w = pct.loc[start:end]
    return {c: round(float(w[c].mean()), 3) for c in cols}


def episode_profiles(panel: pd.DataFrame) -> pd.DataFrame:
    """The reaction profile of each named historical episode. Your reference points."""
    rows = {name: reaction_profile(panel, a, b) for name, (a, b) in EPISODES.items()}
    return pd.DataFrame(rows).T


def compare_reaction(expected: dict, panel: pd.DataFrame,
                     analogue: str | None = None,
                     exclude: "set[str] | frozenset" = frozenset()) -> pd.DataFrame:
    """
    Your expected reaction against every historical episode.

    `expected` is {construct: percentile}, 0-1. Returns the gap per construct and the mean
    absolute gap per episode, so you can see which episode your scenario most resembles -
    and whether that is the one you claimed in CLAIMED_ANALOGUES.

    `exclude` names constructs the ranking must NOT use - the ones that failed their audit
    gates. A construct whose between/within ratio is below 1 measures sampling noise more
    than it measures documents, and an earlier version averaged it into `mean_abs_gap`
    anyway, so a failed instrument was quietly voting on which history the scenario
    resembles. Excluded constructs still appear as gap columns for inspection, and the
    all-construct mean is reported alongside so the exclusion is visible, not silent.
    """
    prof = episode_profiles(panel)
    cols = [c for c in expected if c in prof.columns]
    valid = [c for c in cols if c not in (exclude or ())]
    if not valid:
        raise ValueError("every expected construct is excluded - nothing to compare on")
    gaps = prof[cols].sub(pd.Series(expected)[cols], axis=1)
    out = gaps.copy()
    out["mean_abs_gap"] = gaps[valid].abs().mean(axis=1).round(3)
    out["mean_abs_gap_all_constructs"] = gaps.abs().mean(axis=1).round(3)
    out = out.sort_values("mean_abs_gap")
    dropped = sorted(set(cols) - set(valid))
    if dropped:
        print(f"    excluded from the ranking (failed audit gates): {dropped} - the "
              f"all-construct mean is reported alongside")
    best = out["mean_abs_gap"].iloc[0]
    tied = out.index[(out["mean_abs_gap"] - best).abs() < 0.005].tolist()
    if len(tied) > 1:
        print(f"    closest historical analogue: TIE between {tied} at a mean gap of "
              f"{best:.2f} - do not pick one silently, say the comparison cannot separate "
              f"them and argue from the components")
    else:
        print(f"    closest historical analogue: {out.index[0]} (mean gap {best:.2f})")
    if analogue and out.index[0] != analogue and analogue not in tied:
        print(f"    NOTE: you claimed {analogue}; the profile is closer to {out.index[0]}. "
              f"Defend the claim or change it.")
    return out.round(3)
