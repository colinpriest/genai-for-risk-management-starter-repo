"""
REPLAY - reconstruct a historical RBA decision by n-shot prompting.

YOU SUPPLY THE PROMPT AND THE SHOT STRATEGY. EVERYTHING ELSE IS BUILT.

WHAT YOU DO

  1. Choose a meeting from draft-assignment/replay-meeting-shortlist.md and set MEETING.

  2. Write RECOMMENDATION_PROMPT. It receives k worked historical meetings - conditions in,
     decision out - and then the target meeting's conditions, and must return a
     recommendation with reasoning and transmission mechanisms.

  3. Choose the shot strategy and k, and JUSTIFY THEM. This matters more than the prompt
     wording. See nshot.py: `recent`, `similar`, `stratified`, `regimes`.

  4. Run the A/B: the same prompt with and without your Cycle causal findings included as
     guidance. Does telling the model which associations are causal improve the answer?

  5. Have the model write the public statement, then audit every claim in it.

WHAT IS SUPPLIED
    load_as_at()          the panel truncated to the target, with that meeting's own
                          decision and targets blanked. The lagged constructs are KEPT -
                          the previous minutes were public before this meeting
    feasible_strategies() which strategies can actually be built for your meeting at your k
    nshot.select_shots()  the four strategies, with three leakage checks that raise
    dev_sample()          meetings you may iterate on
    holdout_sample()      meetings you run once and report as they fall
    evaluate_all()        every strategy across a sample, concurrently and cached
    audit_statement()     claim-by-claim check against evidence built from the PANEL
    arithmetic_check()    deterministic check that the statement quotes the right new rate

HOW WORDS REACHES REPLAY
    Every shot carries the seven construct scores from the PREVIOUS meeting's minutes, so
    the model sees not only what the economy was doing but how the Board had been writing
    about it. If your Words constructs are poor, Replay inherits the problem.

THE TRAP YOU WILL HIT
    74% of meetings are holds. `recent` and `similar` will both hand the model examples that
    are almost entirely holds, and it will learn to say hold. `stratified` fixes the balance
    but shows a history that never happened. Print `shot_mix()` before you spend credit.

WRITES outputs/replay.json, outputs/replay_statement.md
       data/processed/replay_raw/*.json  (every raw call, for reproduction without a key)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field as PField

sys.path.insert(0, os.path.dirname(__file__))
import config      # noqa: E402
import courseapi  # noqa: E402
import nshot       # noqa: E402

load_dotenv()
_client = None


def client_():
    """The course API client (see courseapi): same shape, UNSW proxy underneath."""
    return courseapi.client_()


# ###########################################################################################
# YOUR WORK STARTS HERE
# ###########################################################################################

MEETING = "TODO: choose from replay-meeting-shortlist.md, e.g. 2010-11-02"

K_SHOTS = 9
SHOT_STRATEGY = "stratified"     # recent | similar | stratified | regimes

# YOUR Cycle verdicts, in your words, for the compulsory A/B in step 4.
#
# THIS IS ASSESSED AND IT IS NOT OPTIONAL. The runner calls the model twice either way -
# once plain, once with this text - so leaving it empty does not skip anything: it spends
# the same credit to run the A/B against nothing and reports an A/B whose treatment arm is
# blank. The rubric marks the comparison AND what you concluded from it.
#
# Write one line per feature you tested, naming YOUR verdict from your own Cycle results
# and what the model should do with it. The verdicts are the four the Cycle stage uses:
# `reverse`, `confounded`, `untestable` and `not_ruled_out`. "Causal" is not among them -
# correlational diagnostics on overlapping windows cannot establish it, and a guidance
# block that claims it is asserting something your own evidence does not support.
#
# The SHAPE, with the features and verdicts left for you to supply:
#
#     CAUSAL_GUIDANCE = """Our causal testing found:
#      - <feature> is <verdict>: <what that means mechanically, in one line>.
#        <what the model should therefore do with it when recommending>.
#      - <feature> is <verdict>: <...>"""
CAUSAL_GUIDANCE = ""

RECOMMENDATION_PROMPT = """
TODO: WRITE THE RECOMMENDATION PROMPT.

The model receives k worked historical meetings with the decision taken at each, then the
target meeting's conditions with no decision. It must recommend hike / hold / cut.

It must return:
  {"recommendation": "hike"|"hold"|"cut",
   "size_bp": int,
   "confidence": "low"|"moderate"|"high",
   "key_risks": [str],
   "reasoning": str,
   "transmission_mechanisms": [{"channel": str, "horizon": str, "mechanism": str}],
   "what_would_change_our_mind": str}

Things worth deciding explicitly:
  - what you tell it about the RBA's mandate and reaction function
  - whether you tell it the base rate of holds, or let the examples imply it
  - how you stop it simply pattern-matching the most recent example
  - how you make it commit to a size, not just a direction
"""

STATEMENT_PROMPT = """
TODO: WRITE THE STATEMENT PROMPT.

The model receives your recommendation and the evidence behind it, and must write a 250-350
word public statement in the RBA's register.

The constraint that carries the marks: it may only assert what the evidence supports. Write
that constraint so it actually binds - and then audit whether it did.
"""

# YOUR verdict on the machine's classifications, keyed by CLAIM ID.
#
# The audit assigns every claim a stable id - the first 8 hex characters of the SHA-256 of
# its normalised text - printed in the audit output and stored in replay.json. Reviews are
# keyed on that id EXACTLY: an earlier version matched 60-character substrings, so one
# malformed record could blanket several claims, be counted once per match, and satisfy
# the requirement while carrying no verdict at all.
#
#     "a1b2c3d4": {
#         "human_verdict": "forecast-or-judgement",   # one of the five audit categories
#         "reason": "the evidence has one unemployment print, not a trend",
#         "final_action": "kept the number, flagged the inference",
#         "by": "AB"},
#
# All four fields are required; human_verdict must be one of the five categories. Review
# at least MIN_CLAIM_REVIEWS distinct claims - the highest-risk classifications, typically
# anything data-supported that rests on a number and anything unverifiable a person could
# in fact verify. A DISAGREEMENT is recorded where one exists; where none does, the review
# is a DEFENDED agreement. The reportable run raises, not warns, until this holds.
CLAIM_REVIEWS: dict[str, dict] = {}

# ###########################################################################################
# SUPPLIED BELOW THIS LINE - DO NOT MODIFY
# ###########################################################################################

# How many distinct claims must carry a human verdict before a run is reportable. The
# reportable run RAISES below this number rather than warning: an audit that nobody checked
# is precisely the failure this stage exists to make visible. Agreements count towards it -
# what is required is that each reviewed classification was defended rather than assumed.
MIN_CLAIM_REVIEWS = 3

# Seeds per meeting in the strategy comparison. Three is what turns one lucky call into a
# majority vote; the stage hash covers it because changing it changes every reported
# accuracy.
N_SEEDS = 3


def _require_causal_guidance() -> None:
    """The A/B is compulsory, so its treatment arm has to exist.

    THE GAP THIS CLOSES. Removing the old "leave empty to skip the A/B" comment did not
    make the A/B happen. With `CAUSAL_GUIDANCE = ""` the guided call builds the identical
    prompt, is served from the plain call's cache, and both recorded recommendations come
    back with `with_causal_guidance: False` - a stage that looks complete and contains no
    comparison. A team that genuinely has nothing to say here should say so in the report
    and lose the A/B marks, not pass silently.
    """
    text = (CAUSAL_GUIDANCE or "").strip()
    if len(text) < 40:
        raise RuntimeError(
            "CAUSAL_GUIDANCE is empty or too short to be a treatment arm. The A/B in "
            "step 4 is assessed and the runner calls the model twice either way, so an "
            "empty guidance block spends the same credit to compare a prompt with "
            "itself. Write your Cycle verdicts there - one line per feature you tested, "
            "in your words - and re-run. See the Replay stage of the brief.")


def _require_complete_scores() -> None:
    """
    A REPORTABLE run needs the full corpus scored. Partial construct scores are a
    legitimate mid-development state - the panel builder tolerates them so Words can be
    iterated - but a Replay or Shock artefact produced from a partially scored corpus
    would silently carry empty text features into its shots and profiles. Delegates to
    THE shared contract (exact authoritative meeting set, values, sd, n_calls_valid),
    so Replay, Shock and the panel cannot drift apart on what "valid scores" means.
    """
    config.validate_construct_scores(reportable=True)


def panel_fingerprint() -> str:
    """
    sha256 of panel.parquet's bytes - the evidence base every shot, evidence block and
    audit was built from. Part of the stage hash, so regenerating or editing the panel
    after replay.json was written invalidates the artefact exactly as editing a prompt
    would. Rebuilding the panel from the vendored inputs under the pinned library
    versions reproduces the same bytes, so an honest rebuild does not trip it.
    """
    p = config.DATA_PROCESSED / "panel.parquet"
    if not p.exists():
        return "absent"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def config_hash() -> str:
    """
    Identifies THIS Replay configuration - prompts, settings, the human judgement tables,
    AND the panel the run read. Stored in replay.json and compared by the submission
    check, so an artefact generated under different prompts or data than the repository
    holds cannot pass as current. Words has carried the same guarantee since its cache
    was rebuilt; Replay and Shock used to carry none, which in an assignment about GenAI
    meant the marker could not establish that the assessed prompts produced the submitted
    results.
    """
    payload = json.dumps({
        "recommendation_prompt": RECOMMENDATION_PROMPT,
        "statement_prompt": STATEMENT_PROMPT,
        "audit_prompt": AUDIT_PROMPT,
        "panel": panel_fingerprint(),
        "min_usable_call_share": MIN_USABLE_CALL_SHARE,
        "response_schemas": {m.__name__: hashlib.sha256(json.dumps(
            m.model_json_schema(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12] for m in (Recommendation, ClaimAudit)},
        # CAUSAL_GUIDANCE is inserted verbatim into the guided recommendation prompt - an
        # earlier version of this hash omitted it, so the entire guidance block could be
        # rewritten after artefact generation without the submission check noticing.
        "causal_guidance": CAUSAL_GUIDANCE,
        "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
        # the output ceiling changes the answer: a lower one truncates
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "call_index_base": config.CALL_INDEX_BASE, "n_seeds": N_SEEDS,
        "meeting": MEETING, "k": K_SHOTS, "strategy": SHOT_STRATEGY,
        "claim_reviews": CLAIM_REVIEWS,
        "min_claim_reviews": MIN_CLAIM_REVIEWS,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Feasibility note: the two earliest meetings have short histories. Before 2007-11-06 there
# are not three cuts on record, and the forward-looking cycle label has not resolved for the
# recent past, so `stratified` and `regimes` cannot be built there at any k. Those meetings
# support `recent` and `similar` only. `feasible_strategies()` reports this before a run
# rather than letting select_shots raise halfway through one.
SHORTLIST = ["2007-11-06", "2008-10-07", "2010-11-02", "2013-08-06", "2022-05-03"]

RAW_DIR = config.DATA_PROCESSED / "replay_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


class TransmissionMechanism(BaseModel):
    channel: str
    horizon: str
    mechanism: str


class Recommendation(BaseModel):
    """
    Schema-validated, so a malformed reply is a retry rather than a silent empty dict.

    `size_bp` is cross-checked against `recommendation` after parsing - the schema alone
    permits "hold, 25bp" and "cut, 0bp", both of which are incoherent and both of which the
    model produced during development.
    """
    recommendation: Literal["hike", "hold", "cut"]
    size_bp: int = PField(ge=0, le=200)
    confidence: Literal["low", "moderate", "high"]
    key_risks: list[str]
    reasoning: str
    transmission_mechanisms: list[TransmissionMechanism]
    what_would_change_our_mind: str


def validate_recommendation(rec: dict) -> str | None:
    """Deterministic coherence check the schema cannot express. Returns a problem or None."""
    d, bp = rec.get("recommendation"), rec.get("size_bp")
    if d is None or bp is None:
        return "missing recommendation or size_bp"
    if d == "hold" and bp != 0:
        return f"'hold' with a size of {bp}bp is not a coherent recommendation"
    if d in ("hike", "cut") and bp == 0:
        return f"'{d}' with a size of 0bp is not a coherent recommendation"
    if d in ("hike", "cut") and bp % 5:
        return f"{bp}bp is not a multiple of 5 - the RBA does not move in these increments"
    return None


def _prompts_written() -> bool:
    return ("TODO" not in RECOMMENDATION_PROMPT and "TODO" not in STATEMENT_PROMPT
            and "TODO" not in MEETING)


def _panel(pre_decision: bool = True) -> pd.DataFrame:
    """
    The meeting panel. For Replay, on the PRE-DECISION market state by default.

    WHY. The panel's market and rate features are the last close on or before the meeting,
    which on a meeting day is the meeting-day close. The RBA announces at 2:30pm Sydney and
    the ASX closes at 4pm, with bonds and FX trading through, so that close already contains
    the decision. On 7 October 2008 - a 100bp cut - `bab_spread` moved 63 basis points
    between the previous close and the meeting close. Handing Replay that number is handing
    it the answer.

    `data_panel` therefore also stores every market and rate feature as at the previous
    business day, suffixed `__pre`. With `pre_decision=True` those values replace the
    same-day ones under their ordinary names, so nothing downstream has to know.

    This applies to the SHOTS as well as the target. A shot that pairs post-announcement
    conditions with the decision teaches the model to read the announcement.

    Cycle does NOT do this, and should not: its estimand is explicitly "predict at the end
    of meeting T, knowing T's decision".
    """
    p = pd.read_parquet(config.DATA_PROCESSED / "panel.parquet")
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    p = p.set_index("meeting_date").sort_index()
    if pre_decision:
        pre = [c for c in p.columns if c.endswith("__pre")]
        if not pre:
            raise RuntimeError(
                "the panel has no __pre columns - rebuild it with `python src/data_panel.py`. "
                "Replay must not run on meeting-day closes.")
        for c in pre:
            p[c[:-5]] = p[c]
        p = p.drop(columns=pre)
    return p


def feasible_strategies(meeting: str, k: int | None = None) -> list[str]:
    """
    Which shot strategies can actually be built for this meeting at this k.

    Only ValueError counts as infeasibility - that is what nshot raises when a strategy
    genuinely cannot be built here. Anything else (a missing column, a corrupt panel, a
    programming error) is a real defect and propagates: an earlier version caught every
    exception, so a broken panel looked like a legitimate strategy limitation.
    """
    k = K_SHOTS if k is None else k
    panel = _panel()
    out = []
    for s in nshot.STRATEGIES:
        try:
            nshot.select_shots(panel, pd.Timestamp(meeting), k=k, strategy=s)
            out.append(s)
        except ValueError:
            pass
    return out


def load_as_at(meeting_date: str) -> pd.DataFrame:
    """
    The panel as it stood immediately before a meeting.

    Two cuts:
      1. every row after the meeting is removed
      2. the meeting's own decision, rate change and both targets are blanked

    WHY THE CONSTRUCTS ARE NOT BLANKED. `data_panel` lags the whole text tier by one
    meeting, so the construct values on row M are scored from the minutes of M-1. Those
    minutes published about 14 days after M-1 and the meetings are roughly five weeks apart,
    so they were public well before M. `data_panel._assert_text_available()` fails the build
    if any meeting gap is short enough to break that.

    Blanking them - which an earlier version did - would have shown the model seven construct
    values for every historical example and none for the meeting it is being asked to judge.
    That is not caution, it is an asymmetric prompt.
    """
    p = _panel()
    d = pd.Timestamp(meeting_date)
    if d not in p.index:
        raise ValueError(f"{meeting_date} is not a meeting date")
    hist = p.loc[:d].copy()
    for c in ("decision", "change_pct", "y_cycle", "y_decision",
              "y_decision_change", "forward_change"):
        if c in hist.columns:
            hist.loc[d, c] = np.nan
    return hist


def actual_decision(meeting_date: str) -> dict:
    """What the RBA actually did. For the comparison step ONLY - not for the prompt."""
    r = _panel().loc[pd.Timestamp(meeting_date)]
    d = int(r["decision"])
    return {"decision": d, "change_pct": float(r["change_pct"]),
            "size_bp": int(round(abs(float(r["change_pct"])) * 100)),
            "word": nshot.DECISION_WORD[d],
            "label": nshot.decision_label(d, float(r["change_pct"]))}


def rate_before(meeting_date: str) -> float:
    """The cash rate going INTO the meeting. Used for the deterministic arithmetic check."""
    return float(_panel().loc[pd.Timestamp(meeting_date), "cash_rate"])


# -------------------------------------------------------------------------------------------
# Calling, cached and retried
# -------------------------------------------------------------------------------------------

class LLMCallError(RuntimeError):
    """
    An LLM call this stage NEEDS did not produce a valid response.

    Raised where a failure must not be papered over: a missing recommendation, an empty
    statement or a failed claim audit cannot flow into replay.json as though the work
    happened. Benchmark calls in `evaluate_all()` keep their bounded tolerance - failures
    there are counted, reported and capped instead.
    """


# Errors that retrying cannot fix: bad credentials, malformed requests (including a schema
# the endpoint rejects), and plain programming errors. Matched by name so no extra imports
# are needed; anything else (rate limits, timeouts, connection drops, 5xx) is transient and
# retried with backoff.
_NON_TRANSIENT = ("AuthenticationError", "PermissionDeniedError", "BadRequestError",
                  "NotFoundError", "UnprocessableEntityError",
                  "TypeError", "KeyError", "AttributeError", "ValidationError"
                  ) + courseapi.NON_TRANSIENT_ERRORS


def _quarantine(path, why: str) -> None:
    """Move an unusable cache file aside and say exactly what to do about it."""
    bad = path.with_suffix(path.suffix + ".invalid")
    try:
        path.replace(bad)
    except OSError:
        bad = None
    print(f"    CACHE QUARANTINED: {path.name} - {why}."
          + (f" Moved to {bad.name}; " if bad else " ")
          + "re-run to make a fresh call (an unchanged prompt re-bills nothing else).")


def _cache_key(kind: str, system: str, user: str, seed_offset: int = 0,
               schema: type[BaseModel] | None = None) -> str:
    # The key covers the RESPONSE SCHEMA where one is used, not just the request: a
    # schema edit used to leave old envelopes valid-looking, so a cached payload could
    # bypass the validation a fresh response would have faced.
    schema_fp = "" if schema is None else hashlib.sha256(json.dumps(
        schema.model_json_schema(), sort_keys=True, default=str
    ).encode("utf-8")).hexdigest()[:12]
    return hashlib.sha256(
        json.dumps([kind, system, user, config.MODEL, config.SAMPLING_TEMPERATURE,
                    config.CALL_INDEX_BASE + seed_offset, schema_fp], sort_keys=True
                   ).encode("utf-8")).hexdigest()[:16]


def reusable_under_current_ceiling(rec: dict) -> bool:
    """
    THE OUTPUT-CEILING COMPATIBILITY RULE. Same rule as Words - see
    `text_features.reusable_under_current_ceiling` for the full reasoning.

    A cached SUCCESS is reusable only when the ceiling now in force is at least the one
    it was produced under (it finished inside that allowance, so more room cannot cut it
    short; less room might). A cached TRUNCATION stops being settled as soon as the
    ceiling RISES, which is what makes the brief's advice - raise
    config.MAX_OUTPUT_TOKENS and re-run - actually do something.
    """
    current = int(config.MAX_OUTPUT_TOKENS)
    v = (rec.get("request") or {}).get("max_output_tokens")
    recorded = int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    if rec.get("ok"):
        return recorded is not None and current >= recorded
    if "IncompleteResponseError" in str(rec.get("error", "")):
        return recorded is not None and current <= recorded
    return True


def _cached_call(kind: str, system: str, user: str,
                 schema: type[BaseModel] | None = None, seed_offset: int = 0) -> dict:
    """
    One call, cached on the full request and retried with bounded exponential backoff.

    Every raw reply is kept under data/processed/replay_raw/ so a marker can reproduce the
    run without an API key, and so re-running does not re-bill work already done. The key
    covers both prompts, the model, the temperature and the seed, so editing a prompt
    produces a genuinely fresh call rather than the previous answer.

    Cached payloads are RE-validated on load - against the current schema where one is
    given, and for a non-empty text otherwise - and an invalid cache is quarantined, not
    trusted because it once said "ok". Failures are persisted as envelopes and returned as
    {"ok": False, ...}; a failure envelope never short-circuits a retry on the next run.
    Non-transient errors (authentication, malformed request/schema, programming errors)
    raise immediately - exponential backoff cannot fix a bad key.
    """
    key = _cache_key(kind, system, user, seed_offset, schema)
    path = RAW_DIR / f"{kind}-{key}.json"
    if path.exists():
        try:
            rec = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            _quarantine(path, f"unreadable ({type(e).__name__})")
            rec = {}
        if rec.get("ok"):
            payload = rec.get("payload") or {}
            # metadata must match the request being served: a valid envelope renamed or
            # copied over another cache path must not be accepted for it
            expected = {"kind": kind, "model": config.MODEL,
                        "temperature": config.SAMPLING_TEMPERATURE,
                        "call_index": config.CALL_INDEX_BASE + seed_offset,
                        "prompt_sha": key}
            wrong = {k: (rec.get(k), v) for k, v in expected.items()
                     if rec.get(k) != v}
            # THE SAME CONTRACT WORDS APPLIES. This loader compared the request-side
            # fields and the payload schema, and nothing else - so a cache edited to name
            # a different served model, with `provenance.response_status` set to
            # `incomplete`, was reused: an answer from another deployment that the
            # service had not finished, scored as a completed result.
            contradictions = courseapi.envelope_contradictions(
                rec, prompt_hash=key, prompt_field="prompt_sha")
            if contradictions:
                _quarantine(path, "envelope contradicts itself or this run: "
                                  + "; ".join(contradictions))
            elif wrong:
                _quarantine(path, f"envelope metadata does not match this request "
                                  f"({list(wrong)})")
            elif not reusable_under_current_ceiling(rec):
                print(f"    re-asking {path.name}: produced under a different output "
                      f"ceiling than the one now in force")
            else:
                try:
                    if schema is not None:
                        rec["payload"] = schema.model_validate(payload).model_dump()
                        config.ledger_add("replay", path)
                        return rec
                    if str(payload.get("text", "")).strip():
                        config.ledger_add("replay", path)
                        return rec
                    raise ValueError("empty text payload")
                except Exception as e:  # noqa: BLE001 - invalid cache is quarantined
                    _quarantine(path, f"payload no longer validates ({type(e).__name__})")
    # THE EXPOSURE IS RECORDED HERE, before the request leaves - not after the benchmark
    # finishes, and not at freeze time. Reaching this point means the cache could not
    # answer, so the held-out meetings are about to be asked something new.
    global _EXPOSURE_RECORDED, _EXPOSURE_EVENT
    if _CURRENT_SAMPLE == "holdout":
        # SERIALISED, AND DURABLE BEFORE ANY WORKER GOES ON. The flag used to be set
        # first, so the other benchmark threads saw "already recorded" and began asking
        # the holdout while the record of that exposure was still being written.
        with _EXPOSURE_LOCK:
            if not _EXPOSURE_RECORDED:
                _EXPOSURE_EVENT = record_holdout_exposure(
                    f"fresh {kind} call on the holdout sample")
                _EXPOSURE_RECORDED = True
    last = None
    failure = None
    request = courseapi.effective_request(
        model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS)
    for attempt in range(config.MAX_RETRIES):
        try:
            if schema is not None:
                r = client_().beta.chat.completions.parse(
                    model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                    call_index=config.CALL_INDEX_BASE + seed_offset,
                    max_tokens=config.MAX_OUTPUT_TOKENS, response_format=schema,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                payload = r.choices[0].message.parsed.model_dump()
            else:
                r = client_().chat.completions.create(
                    model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                    call_index=config.CALL_INDEX_BASE + seed_offset,
                    max_tokens=config.MAX_OUTPUT_TOKENS,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                payload = {"text": r.choices[0].message.content.strip()}
            rec = {"ok": True, "kind": kind, "payload": payload,
                   "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
                   "call_index": config.CALL_INDEX_BASE + seed_offset,
                   "request": getattr(r, "request", None),
                   "provenance": getattr(r, "provenance", None),
                   "model_served": getattr(r, "model", None), "attempt": attempt + 1,
                   "prompt_sha": key, "request_id": getattr(r, "id", None),
                   # THE WHOLE CALL'S BILL, not just the last response's - and the
                   # version marker, absent from this writer while Words carried one, so
                   # evidence written a minute ago classified itself as "legacy".
                   "usage": getattr(r, "call_usage", None) or getattr(r, "usage", None),
                   "usage_final_response": getattr(r, "usage", None),
                   "transport_requests": (getattr(r, "provenance", None) or {}
                                          ).get("transport_requests"),
                   "config_hash": config_hash(),
                   "envelope_version": courseapi.ENVELOPE_VERSION,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
            path.write_text(json.dumps(rec, indent=1))
            config.ledger_add("replay", path)
            return rec
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:140]}"
            failure = courseapi.describe_failure(e, request=request)
            if type(e).__name__ in _NON_TRANSIENT:
                path.write_text(json.dumps(
                    {"ok": False, "kind": kind, "error": last, "failure": failure,
                     "prompt_sha": key, "request": dict(request),
                     "envelope_version": courseapi.ENVELOPE_VERSION,
                     "timestamp": datetime.now(timezone.utc).isoformat()}, indent=1))
                raise LLMCallError(
                    f"{kind} call failed with a non-transient error ({last}); retrying "
                    f"cannot fix this, so the run stops here. The failure envelope is "
                    f"{path.name}.")
            # The adapter has already spent whatever retry budget this failure has;
            # opening a second one here multiplied a persistent 429 by four.
            if courseapi.transport_budget_spent(e):
                break
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BASE_SECONDS * (2 ** attempt) + random.uniform(0, .5))
    rec = {"ok": False, "kind": kind, "error": last, "failure": failure,
           "prompt_sha": key, "request": dict(request),
           "envelope_version": courseapi.ENVELOPE_VERSION,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(rec, indent=1))
    # a TOLERATED failure is still part of what the artefact rests on: it shaped the
    # usable-call denominator, the reported reliability, and possibly the chosen
    # strategy - so it is ledgered with its expected (failed) state
    config.ledger_add("replay", path, ok=False)
    return rec


def recommend(meeting: str, k: int | None = None, strategy: str | None = None,
              causal_guidance: str = "", seed_offset: int = 0) -> dict:
    """One n-shot recommendation. Returns the parsed result plus the shot mix."""
    k = K_SHOTS if k is None else k
    strategy = SHOT_STRATEGY if strategy is None else strategy
    target = pd.Timestamp(meeting)
    # Shots come from the full panel - past decisions are public - but the TARGET row is read
    # from the as-at frame, so the meeting's own decision cannot reach the prompt.
    block, shots = nshot.build_prompt_block(
        _panel(), target, k=k, strategy=strategy, as_at=load_as_at(meeting))
    sep = "=" * 60
    user = block if not causal_guidance else (
        f"{block}\n\n{sep}\nCAUSAL GUIDANCE FROM OUR OWN ANALYSIS:\n{causal_guidance}")
    rec = _cached_call("rec", RECOMMENDATION_PROMPT, user, schema=Recommendation,
                       seed_offset=seed_offset)
    return {"meeting": meeting, "k": k, "strategy": strategy,
            "with_causal_guidance": bool(causal_guidance),
            "shot_mix": nshot.shot_mix(shots),
            "ok": rec.get("ok", False), "error": rec.get("error"),
            "prompt_chars": len(user),
            # the identity of what was actually SENT, so the A/B arms can be shown to
            # have differed rather than assumed to have
            "prompt_sha": hashlib.sha256(user.encode("utf-8")).hexdigest()[:16],
            "result": rec.get("payload", {})}


# -------------------------------------------------------------------------------------------
# Choosing a strategy without choosing it on your answer
# -------------------------------------------------------------------------------------------

def benchmark_sample(n_per_class: int = 12, min_history: int = 60) -> list[str]:
    """
    Meetings for comparing shot strategies. Stratified by ACTUAL decision.

    WHY NOT JUST THE FIVE SHORTLISTED MEETINGS. Five is far too few to separate four
    strategies - one lucky call moves a strategy by 20 percentage points and you would be
    reporting noise.

    WHY STRATIFIED. On a random sample 74% of meetings are holds, "always hold" scores 74%,
    and no strategy can be distinguished from any other. Stratifying the EVALUATION set is
    not cheating - it is how you get a readable signal on the minority classes. Report
    balanced accuracy alongside raw, and say what you did.
    """
    p = _panel().iloc[min_history:]
    out = []
    for d in (-1, 0, 1):
        sub = p[p["decision"] == d]
        idx = np.linspace(0, len(sub) - 1, min(n_per_class, len(sub))).astype(int)
        out += [str(x.date()) for x in sub.index[idx]]
    return sorted(out)


def dev_sample() -> list[str]:
    """
    Meetings you may iterate on: try strategies, k values and prompt wordings here.

    Alternate meetings of the stratified sample, so dev and holdout are balanced the same way
    and span the same period.
    """
    return benchmark_sample()[0::2]


def holdout_sample() -> list[str]:
    """
    Meetings you run ONCE, after the strategy and k are fixed, and report as they fall.

    Choosing the strategy on the same meetings you then quote the accuracy from is selection
    on the outcome: the best of four strategies looks several points better than it is. Pick
    on `dev_sample()`, report on this. If you go back and change the strategy after seeing
    these numbers, say so in the report - that is a defensible choice, but only if declared.
    """
    return benchmark_sample()[1::2]


# The floor on usable benchmark calls. Individual failures at temperature 1 are tolerated,
# counted and reported; below this share of usable calls the comparison is not reportable.
MIN_USABLE_CALL_SHARE = 0.9


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    The two strategies answered the SAME meetings, so the meetings where they agree carry
    no information about which is better: only the discordant ones do. Under "no
    difference" each discordant meeting is a fair coin, and this is the exact binomial
    two-sided probability of a split at least this lopsided.

    scipy is already a dependency; statsmodels' `contingency_tables.mcnemar` computes the
    same quantity if you prefer to cite it.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(a_only, b_only), n, 0.5).pvalue)


def _paired_comparison(voted: "pd.DataFrame", strategies: list[str]) -> list[dict]:
    """Every strategy pair, compared on the meetings where exactly one was right."""
    import itertools
    by = {s: g.set_index("meeting")["correct"]
          for s, g in voted.groupby("strategy")}
    out = []
    for a, b in itertools.combinations(sorted(strategies), 2):
        if a not in by or b not in by:
            continue
        both = by[a].index.intersection(by[b].index)
        ra, rb = by[a].loc[both], by[b].loc[both]
        a_only = int((ra & ~rb).sum())
        b_only = int((~ra & rb).sum())
        out.append({"a": a, "b": b, "n_meetings": int(len(both)),
                    "a_only": a_only, "b_only": b_only,
                    "accuracy_gap": round(float(ra.mean() - rb.mean()), 3),
                    "p": round(mcnemar_exact(a_only, b_only), 4),
                    "test": "exact McNemar, two-sided"})
    return out


def evaluate_all(k: int | None = None, strategies: list[str] | None = None,
                 causal_guidance: str = "", meetings: list[str] | None = None,
                 sample: str = "dev", max_workers: int = 8,
                 n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """
    Run every strategy across a PAIRED sample of meetings, repeated over several seeds.

    THREE THINGS THIS FIXES, ALL OF WHICH MADE THE OLD COMPARISON UNREADABLE.

    PAIRING. Meetings where any strategy cannot be built are dropped for ALL strategies and
    reported. Previously each strategy was scored on whatever it could manage, so the ones
    with the strictest selection rules skipped the hardest early meetings and were rewarded
    for it.

    REPETITION. The model is sampled at temperature 1, so a single call per meeting is one
    draw. With 18 meetings a strategy can move five points on noise alone. Each meeting is
    now run `n_seeds` times and the per-meeting majority vote is scored, with the unanimity
    rate reported so you can see how settled each call was.

    SIZE SEPARATELY. `size_exact` used to include holds, where the size is trivially 0, so
    a strategy predicting more holds scored better on sizing without being better at it.
    Size accuracy is now conditional on a move having been correctly called.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sklearn.metrics import balanced_accuracy_score

    k = K_SHOTS if k is None else k
    strategies = strategies or list(nshot.STRATEGIES)
    if meetings is None:
        meetings = dev_sample() if sample == "dev" else holdout_sample()
    global _CURRENT_SAMPLE, _EXPOSURE_RECORDED, _EXPOSURE_EVENT
    _CURRENT_SAMPLE, _EXPOSURE_RECORDED, _EXPOSURE_EVENT = sample, False, None
    meetings, dropped = feasible_everywhere(meetings, strategies, k)
    if dropped:
        print(f"  {len(dropped)} meeting(s) dropped so every strategy is scored on the same "
              f"set:")
        for m, bad in list(dropped.items())[:5]:
            print(f"    {m}: infeasible for {bad}")
    jobs = [(s, m, i) for s in strategies for m in meetings for i in range(n_seeds)]

    def _one(job):
        strat, m, call_i = job
        try:
            r = recommend(m, k=k, strategy=strat, causal_guidance=causal_guidance,
                          seed_offset=call_i)
        except LLMCallError:
            raise  # non-transient: every remaining call would fail the same way
        except ValueError as e:
            # nshot's genuine infeasibility, surviving the feasible_everywhere prefilter
            # on an edge case. Anything else - KeyError, NameError, a corrupt panel - is
            # a DEFECT and propagates; an earlier version recorded those as ordinary
            # call attrition, which presented a programming fault as model unreliability.
            return {"strategy": strat, "meeting": m, "call_index": call_i, "skipped": True,
                    "invalid": False, "reason": str(e)[:80]}
        if not r["ok"]:
            return {"strategy": strat, "meeting": m, "call_index": call_i, "skipped": True,
                    "invalid": False, "reason": f"api: {r['error']}"}
        problem = validate_recommendation(r["result"])
        if problem:
            # An incoherent answer is a FAILED response, not a wrong one. Scoring "hold,
            # 25bp" as an incorrect prediction credited the model with an attempt it did
            # not coherently make, and quietly penalised whichever strategy produced them.
            return {"strategy": strat, "meeting": m, "call_index": call_i, "skipped": True,
                    "invalid": True, "reason": f"incoherent: {problem[:60]}"}
        got = str(r["result"].get("recommendation", "")).lower()
        act = actual_decision(m)
        return {"strategy": strat, "meeting": m, "call_index": call_i, "skipped": False,
                "invalid": False,
                "recommended": got, "actual": act["word"].lower(),
                "correct": got == act["word"].lower(),
                "confidence": r["result"].get("confidence"),
                "size_bp": r["result"].get("size_bp"), "actual_size_bp": act["size_bp"],
                "actual_is_move": act["decision"] != 0,
                "hold_share_of_shots": r["shot_mix"]["hold_share"]}

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        try:
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 60 == 0:
                    print(f"    {i}/{len(jobs)}")
        except BaseException:
            # A run-wide failure (exhausted daily budget, bad credentials) makes every
            # QUEUED call pointless. Cancel what has not started rather than letting the
            # executor drain hundreds of doomed requests; each call that DID complete has
            # already written its own envelope, so the next run resumes from them.
            for pending in futs:
                pending.cancel()
            raise

    df = pd.DataFrame(rows)
    if "invalid" not in df.columns:
        df["invalid"] = False
    df["invalid"] = df["invalid"].fillna(False)
    ok = df[~df["skipped"]]
    n_fail = int((df["skipped"] & ~df["invalid"]).sum())
    n_bad = int(df["invalid"].sum())
    print(f"  GenAI reliability: {n_fail}/{len(df)} calls failed after retries, "
          f"{n_bad}/{len(df)} returned an incoherent direction/size pair "
          f"({(n_fail + n_bad) / max(1, len(df)):.1%} unusable)")
    # BOUNDED tolerance, not unbounded. Individual benchmark failures are normal at
    # temperature 1 over hundreds of calls; a benchmark quietly built from a fraction of
    # its intended calls is not a benchmark. The expected and achieved coverage are in the
    # returned rows (and hence in replay.json), and below the floor the run stops.
    usable_share = 1.0 - (n_fail + n_bad) / max(1, len(df))
    if usable_share < MIN_USABLE_CALL_SHARE:
        raise LLMCallError(
            f"only {usable_share:.0%} of {len(df)} benchmark calls were usable "
            f"(floor {MIN_USABLE_CALL_SHARE:.0%}). A strategy comparison built on this "
            f"few calls is not reportable; fix the failures and re-run - successful "
            f"calls are cached.")
    if not len(ok):
        raise LLMCallError("no usable benchmark call - nothing to compare")

    # PAIRED COMPLETENESS. A strategy/meeting pair missing any seed is dropped for EVERY
    # strategy, so the comparison is over identical work. Comparing a strategy with three
    # seeds against one with two is comparing different estimators.
    counts = ok.groupby(["strategy", "meeting"]).size().unstack(fill_value=0)
    complete = [m for m in counts.columns if (counts[m] == n_seeds).all()]
    dropped_incomplete = [m for m in counts.columns if m not in complete]
    if dropped_incomplete:
        print(f"  {len(dropped_incomplete)} meeting(s) dropped for incomplete seed coverage "
              f"across strategies: {dropped_incomplete[:4]}")
    ok = ok[ok["meeting"].isin(complete)]
    if not len(ok):
        raise LLMCallError(
            "no meeting has complete seed coverage across all strategies - the paired "
            "comparison cannot be built, so nothing reportable exists")

    def _vote(g):
        return pd.Series({
            "recommended": g["recommended"].mode().iloc[0],
            "actual": g["actual"].iloc[0],
            "actual_is_move": bool(g["actual_is_move"].iloc[0]),
            "actual_size_bp": g["actual_size_bp"].iloc[0],
            "size_bp": g["size_bp"].mode().iloc[0],
            "unanimous": g["recommended"].nunique() == 1})

    voted = (ok.groupby(["strategy", "meeting"])
               .apply(_vote, include_groups=False).reset_index())
    voted["correct"] = voted["recommended"] == voted["actual"]

    def _summary(g):
        moves = g[g["actual_is_move"] & g["correct"]]
        return pd.Series({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "balanced_accuracy": balanced_accuracy_score(g["actual"], g["recommended"]),
            "share_predicted_hold": (g["recommended"] == "hold").mean(),
            "seed_unanimous": g["unanimous"].mean(),
            "n_moves_called": len(moves),
            "size_exact_given_move": ((moves["size_bp"] == moves["actual_size_bp"]).mean()
                                      if len(moves) else float("nan"))})

    summary = voted.groupby("strategy").apply(_summary, include_groups=False).round(3)

    # TWO ESTIMATORS, BOTH NAMED AND BOTH SAVED.
    #
    # The assessed one is the MEETING-LEVEL MAJORITY VOTE above: the three draws for a
    # strategy/meeting vote, and the vote is scored once. Its denominator is the number of
    # retained meetings.
    #
    # The PER-DRAW estimator scores every draw separately, so its denominator is
    # meetings x seeds. It answers a different question - how often does a single call get
    # it right - and it is reported here because a team that averages raw rows without
    # noticing will otherwise compare their number with a summary that was computed a
    # different way. They do not agree, and neither is wrong; they are different estimands.
    def _per_draw(g):
        return pd.Series({
            "n_draws": len(g),
            "accuracy": g["correct"].mean(),
            "balanced_accuracy": balanced_accuracy_score(g["actual"], g["recommended"])})

    per_draw = ok.groupby("strategy").apply(_per_draw, include_groups=False).round(3)

    n_meetings = len(complete)          # RETAINED, not proposed: incomplete ones are gone
    print(f"  sample: {sample} | {n_meetings} retained paired meetings x {n_seeds} seeds, "
          f"k={k}")
    if len(meetings) != n_meetings:
        print(f"    ({len(meetings) - n_meetings} of {len(meetings)} proposed meetings "
              f"dropped for incomplete coverage)")
    print("  ASSESSED ESTIMATOR - meeting-level majority vote, "
          f"n={n_meetings} meetings:")
    print(summary.to_string())
    print(f"  per-draw estimator (n={n_meetings * n_seeds} draws), reported so the two "
          f"are not confused:")
    print(per_draw.to_string())
    base = (voted[voted["strategy"] == strategies[0]]["actual"] == "hold").mean()
    print(f"  'always hold' would score {base:.3f} here")

    # THE COMPARISON IS PAIRED, because every strategy answers the SAME meetings.
    #
    # The old guidance printed 1.96*sqrt(.25/n) - the widest normal-approximation
    # half-width for ONE accuracy - and told you to call strategies indistinguishable
    # when their gap fell inside it. That rule is not valid for a difference. It throws
    # away the pairing, and it is not even conservative in the right direction: five
    # discordant meetings all favouring one strategy is a gap of 0.278 on 18 meetings -
    # wider than the 0.231 "width" - while exact McNemar gives p = 0.0625.
    #
    # So the paired test is what is reported. `mcnemar_exact()` counts the meetings where
    # exactly one of the two strategies was right and asks how surprising that split is
    # under "no difference". The descriptive width is still printed, clearly labelled as
    # a picture of how coarse an 18-meeting sample is, and NOT as a decision rule.
    pairs = _paired_comparison(voted, strategies)
    half = 1.96 * (0.25 / max(1, n_meetings)) ** 0.5
    print(f"  paired comparison on the same meetings (exact McNemar):")
    for row in pairs:
        print(f"    {row['a']:11s} vs {row['b']:11s}  "
              f"{row['a_only']:2d}-{row['b_only']:<2d} discordant  p={row['p']:.3f}")
    print(f"  for scale, one accuracy on {n_meetings} meetings has a descriptive width of "
          f"about +/-{half:.3f}")
    print("  THESE ARE DESCRIPTIVE RESULTS, AND THERE IS NO WINNER RULE HERE. Report the")
    print("  discordant counts and the p; do not turn either into a decision. What the")
    print("  exact McNemar test handles is the WITHIN-MEETING PAIRING - the same meetings")
    print("  scored by both strategies. What it does NOT handle, and what your report has")
    print("  to name rather than assume away:")
    print("    - the meetings are serially dependent; the test assumes the discordant")
    print("      pairs are independent Bernoulli trials, and they are not;")
    print("    - the samples are drawn stratified, which the test does not model;")
    print(f"    - {len(pairs)} pairwise comparisons are being read at once, and nothing")
    print("      here corrects for looking at all of them.")
    print("  NEITHER number licenses 'these strategies perform equally'. A large p means")
    print("  INSUFFICIENT EVIDENCE TO DISTINGUISH them on this sample - which is what to")
    print("  write, and what the rubric credits.")

    # The benchmark finished, so this exposure is closed: a later resume is a NEW
    # question and needs its own authority, while an interrupted run can be resumed
    # inside the exposure it already recorded.
    if _EXPOSURE_EVENT is not None:
        close_holdout_exposure(_EXPOSURE_EVENT.get("event_id"),
                               f"{sample} benchmark completed on {n_meetings} meetings")
    _CURRENT_SAMPLE = None
    # A CACHE-ONLY REPLAY ASKS THE HOLDOUT NOTHING, and therefore records no new event -
    # but the result it reports still BELONGS to the exposure that produced the
    # envelopes, and writing `null` there severed the report from its own evidence.
    # Re-running the stage from committed caches must keep naming that exposure.
    event_id = (_EXPOSURE_EVENT or {}).get("event_id")
    if event_id is None and sample == "holdout":
        event_id = originating_exposure_id()
    df.attrs["exposure_event_id"] = event_id
    df.attrs["exposure_was_fresh"] = _EXPOSURE_EVENT is not None
    df.attrs["evidence_status"] = evidence_status()
    df.attrs["paired_comparison"] = pairs
    df.attrs["summary_majority_vote"] = summary.reset_index().to_dict("records")
    df.attrs["summary_per_draw"] = per_draw.reset_index().to_dict("records")
    df.attrs["estimator_meta"] = {
        "assessed_estimator": "meeting_level_majority_vote",
        "n_meetings_proposed": len(meetings),
        "n_meetings_retained": n_meetings,
        "n_seeds": n_seeds,
        "n_draws": n_meetings * n_seeds,
        "rough_half_width_one_accuracy": round(half, 4),
        "sample": sample}
    return df


# -------------------------------------------------------------------------------------------
# The statement, and two independent checks on it
# -------------------------------------------------------------------------------------------

def evidence_block(meeting: str, rec: dict) -> dict:
    """
    The evidence the statement is allowed to rest on, built from the PANEL.

    Deliberately independent of the model's own reasoning. Auditing a statement against the
    reasoning that produced it asks whether the model contradicted itself, which it rarely
    does. Auditing it against the data asks whether the claims are true, which is the
    question. `rec` supplies only the decision and its size.
    """
    row = load_as_at(meeting).loc[pd.Timestamp(meeting)]
    facts = {label: (fmt % row[col])
             for col, label, fmt in nshot.SHOT_FIELDS + nshot.CONSTRUCT_SHOT_FIELDS
             if col in row.index and pd.notna(row[col])}
    return {"meeting_date": meeting,
            "cash_rate_before_decision_pct": round(rate_before(meeting), 2),
            "decision_recommended": rec.get("recommendation"),
            "size_bp": rec.get("size_bp"),
            "observed_conditions": facts}


CLAIM_CATEGORIES = ["data-supported", "model-derived", "forecast-or-judgement",
                    "contradicted", "unverifiable"]


class Claim(BaseModel):
    claim: str
    category: Literal["data-supported", "model-derived", "forecast-or-judgement",
                      "contradicted", "unverifiable"]
    evidence_key: str | None = None
    note: str = ""


class ClaimAudit(BaseModel):
    claims: list[Claim]


AUDIT_PROMPT = """
You are auditing a draft central bank statement against the evidence used to write it.

Classify EVERY factual or forward-looking claim into exactly one category:

  data-supported        traceable to a specific item in the evidence block. Give the key.
  model-derived         follows from the recommendation and its stated reasoning rather
                        than from an observation - for example a statement about what the
                        decision is intended to achieve.
  forecast-or-judgement a claim about the future or an assessment. NOT a defect: a central
                        bank statement is largely forward-looking. Flag it as such so a
                        human can judge whether it is reasonable.
  contradicted          the evidence says otherwise. This is the serious one.
  unverifiable          too vague to check, or about something the evidence does not cover
                        at all.

A blanket "unsupported" category was tried and was useless: an ordinary RBA statement is
mostly forecasts, so everything came back unsupported and the audit carried no information.
The distinction that matters is between a claim the evidence CONTRADICTS and one it merely
does not contain.

Do not check arithmetic - that is checked separately and deterministically.
"""





def claim_id(claim: str) -> str:
    """First 8 hex chars of SHA-256 over the normalised claim text - stable across runs."""
    normalised = " ".join(str(claim).split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:8]


def audit_statement(statement: str, evidence: dict,
                    reviews: dict[str, dict] | None = None,
                    require: bool = True) -> dict:
    """
    Structured claim audit against PANEL-DERIVED evidence. Returns counts plus the records.

    The evidence block is built from the data, not from the model's own reasoning: auditing
    a statement against the reasoning that produced it asks whether the model contradicted
    itself, which it rarely does.
    """
    rec = _cached_call(
        "audit", AUDIT_PROMPT,
        f"EVIDENCE BLOCK:\n{json.dumps(evidence, indent=1, default=str)}\n\n"
        f"STATEMENT:\n{statement}", schema=ClaimAudit)
    if not rec.get("ok"):
        raise LLMCallError(
            f"the claim-audit call failed ({rec.get('error')}); an unaudited statement "
            f"must not reach replay.json, so the run stops here")
    claims = rec["payload"]["claims"]
    reviews = CLAIM_REVIEWS if reviews is None else reviews
    for c in claims:
        c["claim_id"] = claim_id(c["claim"])
    ids = {c["claim_id"] for c in claims}

    # STRICT, BY ID. The old matcher accepted any truthy record whose 60-character key was
    # a substring of a claim, so {"junk": True} could blanket two claims, count twice, and
    # leave both with no verdict, no reason and no reviewer. Every review must now name an
    # existing claim id and carry all four fields with a legal verdict.
    unknown = sorted(set(reviews) - ids)
    if unknown:
        raise ValueError(
            f"CLAIM_REVIEWS names claim id(s) {unknown} that are not in this audit. Ids "
            f"are printed with each claim and stored in replay.json; re-run the audit and "
            f"key your reviews on what it actually produced.")
    for cid, r in reviews.items():
        gaps = [f for f in ("human_verdict", "reason", "final_action", "by")
                if not str((r or {}).get(f, "") if isinstance(r, dict) else "").strip()]
        if gaps:
            raise ValueError(
                f"the claim review for id {cid} is missing {gaps}. Every review needs "
                f"your verdict, a reason, what you did about it, and initials.")
        if r["human_verdict"] not in CLAIM_CATEGORIES:
            raise ValueError(
                f"claim review {cid}: human_verdict {r['human_verdict']!r} must be one "
                f"of {sorted(CLAIM_CATEGORIES)}.")

    n_reviewed = n_challenged = 0
    for c in claims:
        match = reviews.get(c["claim_id"])
        if match:
            n_reviewed += 1
            c["human_verdict"] = match["human_verdict"]
            c["human_reason"] = match["reason"]
            c["final_action"] = match["final_action"]
            c["reviewed_by"] = match["by"]
            c["challenged"] = bool(match["human_verdict"] != c["category"])
            n_challenged += int(c["challenged"])
        else:
            c["human_verdict"] = None
            c["challenged"] = None
    counts = {c: sum(1 for x in claims if x["category"] == c) for c in CLAIM_CATEGORIES}
    if n_reviewed < MIN_CLAIM_REVIEWS and require:
        listing = chr(10).join(f"      {c['claim_id']}  [{c['category']:22s}] "
                               f"{c['claim'][:70]}" for c in claims)
        raise ValueError(
            f"only {n_reviewed} of the required {MIN_CLAIM_REVIEWS} distinct claims carry "
            f"a human review. Review the highest-risk classifications - typically "
            f"anything data-supported that rests on a number, and anything unverifiable "
            f"a person could in fact verify. The claims and their ids:" + chr(10)
            + listing)
    if n_challenged == 0:
        print(f"    no classification challenged: all {n_reviewed} reviewed claims agree "
              f"with the machine. A defended agreement is a legitimate review - say in "
              f"the report what each was checked against and what would have changed "
              f"your mind.")
    else:
        print(f"    {n_challenged} of {n_reviewed} reviewed classifications challenged by "
              f"the team")
    return {"ok": True, "claims": claims, "counts": counts,
            "n_claims": len(claims),
            "n_reviewed_by_human": n_reviewed,
            "n_challenged_by_human": n_challenged,
            "reviews": reviews,
            "n_contradicted": counts.get("contradicted", 0)}


def feasible_everywhere(meetings: list[str], strategies: list[str],
                        k: int) -> tuple[list[str], dict]:
    """
    The meetings on which EVERY strategy can be built, plus what was dropped.

    Comparing strategies across different meeting sets compares different questions: the
    strategies with the strictest selection rules skip the hardest early meetings and score
    higher for it. Accuracies must be paired.
    """
    panel = _panel()
    keep, dropped = [], {}
    for m in meetings:
        bad = []
        for s in strategies:
            try:
                nshot.select_shots(panel, pd.Timestamp(m), k=k, strategy=s)
            except ValueError:
                # nshot's genuine "cannot build this here"; anything else is a real
                # defect and propagates rather than masquerading as infeasibility
                bad.append(s)
        if bad:
            dropped[m] = bad
        else:
            keep.append(m)
    return keep, dropped


def generate_statement(rec: dict, evidence: dict) -> str:
    """
    Produce the public statement from your `STATEMENT_PROMPT`. Supplied, cached, retried.

    The README says the machinery is built, so this is built. What is yours is the PROMPT and
    the orchestration in `run()` - which of the pieces below you call, in what order, and what
    you conclude from them.
    """
    out = _cached_call(
        "stmt", STATEMENT_PROMPT,
        f"DECISION AND REASONING:{chr(10)}{json.dumps(rec, indent=1, default=str)}"
        f"{chr(10)}{chr(10)}EVIDENCE BLOCK - you may assert nothing beyond this:{chr(10)}"
        f"{json.dumps(evidence, indent=1, default=str)}")
    if not out.get("ok"):
        raise LLMCallError(
            f"the statement call failed ({out.get('error')}); there is no statement to "
            f"audit or to write into replay.json, so the run stops here")
    text = str(out.get("payload", {}).get("text", "")).strip()
    if not text:
        raise LLMCallError("the statement call returned empty text; an empty statement "
                           "is not auditable and must not reach replay.json")
    return text


def arithmetic_check(statement: str, meeting: str, rec: dict) -> dict:
    """
    Deterministic. Does the statement quote the correct post-decision cash rate?

    Asking the auditing model to check arithmetic asks a language model to do the one thing
    it is worst at, and in testing it passed a statement asserting that a 25 basis point
    increase took 4.50 per cent to 4.50 per cent. This computes the expected rate and looks
    for it in the text.
    """
    before = rate_before(meeting)
    sign = {"hike": 1, "cut": -1, "hold": 0}.get(str(rec.get("recommendation", "")).lower())
    if sign is None:
        return {"checked": False, "reason": "no parseable recommendation"}
    after = round(before + sign * (rec.get("size_bp", 0) or 0) / 100.0, 2)
    # Only rates stated in CASH RATE context. An earlier version matched every percentage
    # in the statement, so inflation and unemployment figures landed in `rates_quoted` and
    # a hold passed on any 4.5 anywhere in the text.
    # "per cent", "percent" and "%" are the same unit. Matching only "per cent|%" gave a
    # FALSE FAIL on a statement that opened "the cash rate at 4.50 percent" - the check
    # reported that the post-decision rate was never stated when it was in the first
    # sentence. A deterministic check that is wrong is worse than no check, because the
    # whole point of it is that its verdict is not a matter of opinion.
    num = r"(\d+(?:\.\d+)?)\s*(?:per\s*cent|%)"
    ctx = r"cash rate[^.]{0,120}?" + num + r"|" + num + r"[^.]{0,60}?cash rate"
    hits = [g for m in re.finditer(ctx, statement, re.I) for g in m.groups() if g]
    quoted = sorted({float(h) for h in hits})
    ok = any(abs(q - after) < 0.005 for q in quoted)
    stale = [q for q in quoted if abs(q - before) < 0.005] if sign != 0 else []
    return {"checked": True, "rate_before": before, "expected_after": after,
            "rates_quoted": quoted, "correct_rate_stated": ok,
            "stale_rate_stated": bool(stale),
            "verdict": ("pass" if ok and not stale else
                        "FAIL - quotes the pre-decision rate as the new rate" if stale and not ok
                        else "pass with a stale rate also mentioned" if ok and stale
                        else "FAIL - the post-decision rate is never stated")}


# -------------------------------------------------------------------------------------------
# The SUPPLIED runner. You do not write control flow in this assignment.
# -------------------------------------------------------------------------------------------

#: Where the frozen strategy selection lives. Written by `--dev`, read by `--holdout`.
SELECTION_STAMP = config.OUTPUTS / "replay_selection.json"


def selection_config_hash() -> str:
    """EVERYTHING that must not move between choosing on development and reporting on
    the holdout.

    Not just the prompts: anything that changes what a holdout draw would return, or
    which meetings it would run on, lets a team keep asking the holdout a slightly
    different question until they like the answer. An earlier version covered only the
    prompts, the strategy, k and the seed count, so shifting `CALL_INDEX_BASE` produced a
    completely fresh set of holdout draws while the freeze still verified.
    """
    payload = json.dumps({
        "recommendation_prompt": RECOMMENDATION_PROMPT,
        "statement_prompt": STATEMENT_PROMPT,
        "audit_prompt": AUDIT_PROMPT,
        "strategy": SHOT_STRATEGY, "k": K_SHOTS, "n_seeds": N_SEEDS,
        "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
        # which DRAWS: moving the base re-rolls every call in the benchmark
        "call_index_base": config.CALL_INDEX_BASE,
        # what the model was allowed to produce
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        # which MEETINGS, and the evidence underneath them
        "dev_sample": dev_sample(), "holdout_sample": holdout_sample(),
        "panel": panel_fingerprint(),
        "min_usable_call_share": MIN_USABLE_CALL_SHARE,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _read_selection() -> dict:
    if not SELECTION_STAMP.exists():
        return {}
    try:
        return json.loads(SELECTION_STAMP.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


#: THE EXPOSURE LOG IS APPEND-ONLY AND LIVES IN ITS OWN FILE.
#:
#: It used to be a list and a counter inside the freeze record, which meant a freeze
#: could erase it - and did: `freeze_selection()` reset `exposure_number` to 0 and
#: `exposures` to [] on EVERY freeze, so after two real holdout exposures an ordinary
#: same-configuration `--dev` freeze wiped the evidence and the next changed-configuration
#: freeze sailed through because the count it consulted was zero. A record that the thing
#: it constrains can rewrite is not a record. Nothing in this module ever rewrites a line
#: of this file; new facts are appended, including the fact that an exposure finished.
EXPOSURE_LOG = config.OUTPUTS / "replay_exposures.jsonl"

#: One writer at a time, and the EVENT IS DURABLE BEFORE ANY WORKER PROCEEDS. The flag
#: used to be set before the write, so the other benchmark workers - up to N_PARALLEL
#: threads - could start requesting holdout data while the record of that exposure was
#: still in flight, and a crash in between left holdout draws with no exposure recorded.
_EXPOSURE_LOCK = threading.Lock()


def _freeze_id(config_hash: str, frozen_at: str) -> str:
    """A stable identity for one freeze, so events can name the freeze they belong to."""
    return hashlib.sha256(f"{config_hash}|{frozen_at}".encode("utf-8")).hexdigest()[:12]


def _event_id(freeze_id: str, sequence: int, at: str) -> str:
    return hashlib.sha256(
        f"{freeze_id}|{sequence}|{at}".encode("utf-8")).hexdigest()[:12]


#: Fields an `open` event must carry to count as a record of anything.
_OPEN_EVENT_FIELDS = ("type", "event_id", "sequence", "at", "freeze_id",
                      "selection_config_hash")


def exposure_events() -> list[dict]:
    """Every event ever appended, oldest first. Damaged lines are kept as damage.

    A LINE THAT WILL NOT PARSE IS NOT AN ABSENT LINE. Skipping it, or turning it into a
    harmless marker the counters ignore, makes a corrupted log read as a SHORTER history -
    so damaging one line was a way to reduce the recorded exposure count and buy another
    look at the held-out sample. `log_damage()` is what callers must consult before
    trusting any count derived from this list.
    """
    if not EXPOSURE_LOG.exists():
        return []
    events = []
    for n, line in enumerate(
            EXPOSURE_LOG.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            events.append({"type": "damaged", "line_number": n,
                           "problem": f"not JSON ({exc.msg})"})
            continue
        if not isinstance(parsed, dict):
            events.append({"type": "damaged", "line_number": n,
                           "problem": f"a {type(parsed).__name__}, not an event object"})
            continue
        if parsed.get("type") == "open":
            missing = [f for f in _OPEN_EVENT_FIELDS if parsed.get(f) is None]
            if missing:
                parsed = dict(parsed, type="damaged", line_number=n,
                              problem=f"an exposure event missing {missing}")
        elif parsed.get("type") not in ("close",):
            parsed = dict(parsed, type="damaged", line_number=n,
                          problem=f"unknown event type {parsed.get('type')!r}")
        events.append(parsed)
    return events


def log_damage(events: list[dict] | None = None) -> list[str]:
    """Every damaged line in the log, described. Empty means the history is readable."""
    events = exposure_events() if events is None else events
    return [f"line {e.get('line_number', '?')}: {e.get('problem', 'unreadable')}"
            for e in events if e.get("type") == "damaged"]


def require_readable_log() -> None:
    """Refuse to derive an exposure COUNT from a history that does not parse.

    The count is what authorises another look at the holdout. A damaged log whose
    damage is ignored answers "zero exposures so far" to a question it cannot answer.
    """
    damage = log_damage()
    if damage:
        raise RuntimeError(
            f"{EXPOSURE_LOG.name} has {len(damage)} damaged line(s), so the number of "
            f"holdout exposures it records cannot be read:\n  "
            + "\n  ".join(damage)
            + f"\nThis file is append-only evidence and the count in it decides whether "
              f"the held-out sample may be asked anything more. Restore it from version "
              f"control rather than editing or deleting it; if it is genuinely "
              f"unrecoverable, say so in the report and re-freeze with "
              f'--revalidate --note="...".')


def _append_event(event: dict) -> dict:
    EXPOSURE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXPOSURE_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return event


def exposures_recorded() -> list[dict]:
    """The `open` events - one per time the holdout was actually asked something new."""
    return [e for e in exposure_events() if e.get("type") == "open"]


def open_exposure(freeze_id: str) -> dict | None:
    """An exposure of this freeze that was started and never closed.

    RESUMING ONE IS NOT A NEW EXPOSURE. A run that crashes, or that is stopped and
    restarted to fill cache misses, is still the same question asked once; charging it a
    second exposure would either block a legitimate resume or make the count meaningless.
    A new exposure begins only after the previous one is closed by a completed benchmark.
    """
    state: dict[str, dict] = {}
    for e in exposure_events():
        if e.get("type") == "open":
            state[e.get("event_id")] = e
        elif e.get("type") == "close":
            state.pop(e.get("event_id"), None)
    for event in reversed(list(state.values())):
        if event.get("freeze_id") == freeze_id:
            return event
    return None


def originating_exposure_id(freeze_id: str | None = None) -> str | None:
    """The exposure whose requests produced the evidence this freeze reports on.

    The LAST event recorded against this freeze - open or closed. A run served entirely
    from cache creates no event, and must still be able to say which exposure its numbers
    came from; `None` means this freeze has never asked the holdout anything, which is a
    different statement and is left as one.
    """
    if freeze_id is None:
        freeze_id = (_read_selection() or {}).get("freeze_id")
    if not freeze_id:
        return None
    mine = [e for e in exposures_recorded() if e.get("freeze_id") == freeze_id]
    return mine[-1].get("event_id") if mine else None


def freeze_selection(note: str = "", revalidate: bool = False,
                     prior_exposure: bool = False) -> dict:
    """Record the strategy chosen on development, BEFORE the holdout is touched.

    A FREEZE IS NOT AN EXPOSURE. Freezing costs nothing and changes nothing about the
    held-out meetings; what costs is asking them a question. Exposures are counted in the
    append-only `EXPOSURE_LOG`, written by `record_holdout_exposure()` at the moment a
    fresh holdout draw is about to be requested. An earlier version incremented a counter
    on every development freeze, so a team that iterated twice and never touched the
    holdout reported "exposure 2" and looked as though they had.

    RE-FREEZING THE SAME CONFIGURATION IS IDEMPOTENT. It keeps the freeze identity, the
    authorisation and every recorded event: an ordinary `--dev` re-run while iterating on
    something that does not enter `selection_config_hash()` is not an event at all.

    OVERWRITING A FREEZE THE HOLDOUT HAS ALREADY SEEN NEEDS `revalidate=True` AND A NOTE.
    Otherwise re-freezing after a disappointing holdout result would be the whole problem,
    silently.

    `prior_exposure=True` DECLARES that the holdout was run before this log existed - the
    honest position for evidence produced under an older workflow. It cannot be inferred:
    an empty log looks identical whether the holdout is untouched or was exposed by a
    process that never recorded it, and those are opposite claims. Declaring it makes the
    evidence `previously_exposed` rather than `prospective`, which is what a marker needs
    to read. It also requires a note saying so.
    """
    old = _read_selection()
    require_readable_log()
    recorded = len(exposures_recorded())
    same_configuration = bool(old) and old.get("config_hash") == selection_config_hash()

    # NOTE THE ABSENCE OF `old` FROM THIS CONDITION. It used to read `if old and ...`, so
    # DELETING outputs/replay_selection.json restored a clean slate: with no freeze record
    # to compare against, a changed configuration froze without complaint and authorised a
    # further exposure. The log is the authority on what the holdout has seen, and it
    # answers that question whether or not a freeze record still exists.
    if recorded > 0 and not same_configuration and not revalidate:
        frozen_on = str(old.get("frozen_at"))[:10] if old else (
            "a freeze record that is no longer present")
        raise RuntimeError(
            f"the holdout has already been run {recorded} time(s) against the selection "
            f"frozen on {frozen_on}, and this configuration differs "
            f"from it. Re-freezing now would mean choosing on the held-out sample.\n"
            f"If you genuinely need a second exposure, ask for it and declare it in the "
            f"report:\n"
            f'    python src/decision_replay.py --dev --revalidate --note="why"')
    if revalidate and not str(note).strip():
        raise RuntimeError(
            "a revalidation authorises a SECOND look at the held-out sample, so it has to "
            "say why in words a marker can read:\n"
            '    python src/decision_replay.py --dev --revalidate --note="..."')
    if prior_exposure and not str(note).strip():
        raise RuntimeError(
            "declaring a PRIOR exposure is a statement about your own evidence, so it has "
            "to say what happened:\n"
            '    python src/decision_replay.py --dev --prior-exposure --note="the holdout '
            'was run before the freeze workflow existed"')

    if same_configuration:
        # Nothing about the selection moved. Keep the identity and the authorisation; do
        # not add a history entry for a freeze that changed nothing.
        rec = dict(old)
        rec["refrozen_at"] = datetime.now(timezone.utc).isoformat()
        # A freeze record written before identities existed gets one now, derived from
        # what it already carries, so it keeps the same identity on every later freeze.
        rec.setdefault("freeze_id", _freeze_id(str(old.get("config_hash")),
                                               str(old.get("frozen_at"))))
        rec.setdefault("authorised_exposures", max(recorded, 1))
        if note:
            rec["note"] = note
        if revalidate:
            rec["authorised_exposures"] = recorded + 1
            rec["authorisation"] = note
            rec["revalidated"] = True
        if prior_exposure:
            rec["prior_exposure_declared"] = True
    else:
        frozen_at = datetime.now(timezone.utc).isoformat()
        config_hash = selection_config_hash()
        history = list(old.get("history", []))
        if old:
            history.append({k: old.get(k) for k in
                            ("freeze_id", "config_hash", "strategy", "k", "frozen_at",
                             "authorised_exposures")}
                           | {"exposures_at_supersession": recorded})
        rec = {"freeze_id": _freeze_id(config_hash, frozen_at),
               "config_hash": config_hash, "strategy": SHOT_STRATEGY,
               "k": K_SHOTS, "n_seeds": N_SEEDS,
               "call_index_base": config.CALL_INDEX_BASE,
               "max_output_tokens": config.MAX_OUTPUT_TOKENS,
               "frozen_at": frozen_at,
               # HOW MANY EXPOSURES THIS SELECTION IS ALLOWED, in total, ever. A first
               # freeze authorises the one prospective look the design is built around.
               # A second needs `--revalidate` and a stated reason.
               "authorised_exposures": recorded + 1,
               "authorisation": note if revalidate else "prospective selection freeze",
               "note": note,
               "history": history,
               "revalidated": bool(revalidate),
               # DECLARED, never inferred: an empty log cannot distinguish "the holdout is
               # untouched" from "the holdout was run by a process that never recorded it".
               "prior_exposure_declared": bool(prior_exposure)}
    # DERIVED, never stored as something a freeze can reset.
    rec["exposure_number"] = recorded
    rec["exposure_log"] = EXPOSURE_LOG.name
    # Both spellings, and BOTH read from the append-only log rather than from anything a
    # freeze can set. The old `exposures` list lived inside this record and was reset to
    # [] on every freeze; it is now a view of the log, so a freeze cannot shorten it.
    rec["exposures"] = exposures_recorded()
    rec["exposure_event_ids"] = [e.get("event_id") for e in rec["exposures"]]
    config.atomic_write_text(SELECTION_STAMP, json.dumps(rec, indent=1))
    return rec


def evidence_status() -> str:
    """How the reported holdout evidence stands in relation to the selection.

    prospective          - frozen first, exposed once, and not re-frozen since.
    previously_exposed   - the holdout had been run before the freeze that governs it.
    retrospective        - the selection was re-frozen after the holdout was exposed.
    unexposed            - nothing has been asked of the holdout yet.
    """
    rec = _read_selection()
    events = exposures_recorded()
    if not rec:
        return "unexposed" if not events else "previously_exposed"
    mine = [e for e in events if e.get("freeze_id") == rec.get("freeze_id")]
    if not events:
        # An empty log is only "unexposed" if nobody has declared otherwise.
        return "previously_exposed" if rec.get("prior_exposure_declared") else "unexposed"
    if not mine or rec.get("prior_exposure_declared"):
        return "previously_exposed"
    if len(events) > len(mine) or rec.get("revalidated"):
        return "retrospective"
    return "prospective" if len(mine) == 1 else "retrospective"


def record_holdout_exposure(reason: str = "fresh holdout draw") -> dict:
    """Persist an exposure BEFORE the first new holdout call of this freeze.

    Appended and fsynced before the request leaves, so a run that crashes mid-benchmark
    still leaves the evidence that the holdout was asked. Cached replay of an exposure
    already recorded is free and does not count again - re-running the stage from
    committed envelopes asks the held-out meetings nothing new.

    A NEW EXPOSURE NEEDS DECLARED AUTHORITY. The freeze says how many looks at the
    held-out sample this selection is allowed; the first is the prospective one the design
    exists to protect, and any further look has to be asked for with `--revalidate` and a
    stated reason. Previously this function incremented a counter and never asked.
    """
    rec = _read_selection()
    if not rec:
        raise RuntimeError("no frozen selection to record an exposure against")
    require_readable_log()
    freeze_id = rec.get("freeze_id") or _freeze_id(
        str(rec.get("config_hash")), str(rec.get("frozen_at")))
    if rec.get("config_hash") != selection_config_hash():
        raise RuntimeError(
            "the configuration has moved since the selection was frozen; freeze again "
            "before asking the holdout anything.")

    resumed = open_exposure(freeze_id)
    if resumed is not None:
        print(f"    continuing holdout exposure {resumed['sequence']} "
              f"({resumed['event_id']}) - a resumed run is not a new exposure")
        # `exposure_number` is carried alongside `sequence` so a caller written against
        # the old counter-in-the-freeze-record still reads the right number.
        return dict(resumed, resumed=True,
                    exposure_number=resumed["sequence"],
                    exposures=exposures_recorded())

    recorded = len(exposures_recorded())
    authorised = int(rec.get("authorised_exposures", 1))
    if recorded >= authorised:
        raise RuntimeError(
            f"the held-out sample has already been exposed {recorded} time(s) and this "
            f"freeze authorises {authorised}. Asking it again is a second experiment on "
            f"the same data, and it has to be declared rather than taken:\n"
            f'    python src/decision_replay.py --dev --revalidate --note="why a second '
            f'exposure is justified"\n'
            f"The report must then say so; the rubric credits the declaration, not the "
            f"number of attempts.")

    at = datetime.now(timezone.utc).isoformat()
    sequence = recorded + 1
    event = _append_event({
        "type": "open", "event_id": _event_id(freeze_id, sequence, at),
        "sequence": sequence, "at": at, "freeze_id": freeze_id,
        "selection_config_hash": selection_config_hash(),
        "strategy": SHOT_STRATEGY, "k": K_SHOTS, "n_seeds": N_SEEDS,
        "call_index_base": config.CALL_INDEX_BASE,
        "authorisation": rec.get("authorisation", ""),
        "reason": reason})
    print(f"    HOLDOUT EXPOSURE {sequence} recorded as {event['event_id']} ({reason})")
    return dict(event, resumed=False, exposure_number=sequence,
                exposures=exposures_recorded())


def close_holdout_exposure(event_id: str, outcome: str = "benchmark completed") -> None:
    """Append the fact that an exposure finished. Never edits the `open` line."""
    if not event_id:
        return
    _append_event({"type": "close", "event_id": event_id, "outcome": outcome,
                   "at": datetime.now(timezone.utc).isoformat()})


#: Which sample the benchmark is currently running, so `_cached_call` can record an
#: exposure at the moment a fresh HOLDOUT request is about to be made.
_CURRENT_SAMPLE: str | None = None
_EXPOSURE_RECORDED = False
_EXPOSURE_EVENT: dict | None = None


def _require_frozen_selection() -> None:
    """The holdout may only be run against a selection frozen on development.

    THE GAP THIS CLOSES. `run()` used to call development and holdout back to back, with
    the strategy already chosen in the file. Editing a prompt or k and re-running spent the
    holdout again - every time - so "report on the holdout once" was a convention the code
    did nothing to keep. Words has always had this workflow; Replay now does too.
    """
    if not SELECTION_STAMP.exists():
        raise RuntimeError(
            "the holdout has no frozen selection to report on. Run the development "
            "sample and freeze what you chose first:\n"
            "    python src/decision_replay.py --dev\n"
            "then re-run. The freeze records the prompts, strategy, k and seed count that "
            "the holdout result will belong to.")
    rec = json.loads(SELECTION_STAMP.read_text(encoding="utf-8"))
    if rec.get("config_hash") != selection_config_hash():
        raise RuntimeError(
            f"the frozen selection no longer matches this configuration - the prompts, "
            f"strategy, k, seed count, draw indices, output ceiling, benchmark samples or "
            f"panel have moved since it was frozen on "
            f"{str(rec.get('frozen_at'))[:10]} (frozen {rec.get('strategy')!r} at "
            f"k={rec.get('k')}, hash {rec.get('config_hash')}; now "
            f"{selection_config_hash()}).\n"
            f"Running the holdout now would be choosing on it. Either restore the frozen "
            f"configuration, or re-select on development and declare the second exposure:\n"
            f"    python src/decision_replay.py --dev --revalidate\n"
            f"A second exposure is a defensible choice you must state in the report, not a "
            f"silent one.")
    require_readable_log()
    recorded = len(exposures_recorded())
    status = evidence_status()
    if recorded > 1 or status in ("retrospective", "previously_exposed"):
        print(f"    NOTE: the held-out sample has been exposed {recorded} time(s) and "
              f"this evidence is {status.replace('_', ' ')}. Your report must say so, "
              f"and why. See {EXPOSURE_LOG.name} for the events.")


def run_dev(revalidate: bool = False, note: str = "",
            prior_exposure: bool = False) -> pd.DataFrame:
    """Development sample only, then freeze the selection. Costs no holdout exposure.

    `note` is recorded verbatim in the freeze. Use it when the record needs a caveat a
    marker should read - for instance that the holdout had already been run before the
    freeze existed, which no amount of re-running can undo.
    """
    if not _prompts_written():
        raise NotImplementedError(
            "Set MEETING and write RECOMMENDATION_PROMPT and STATEMENT_PROMPT first.")
    print("  CHOOSING THE STRATEGY - development sample (the holdout is NOT touched)")
    dev = evaluate_all(sample="dev")
    rec = freeze_selection(note=note, revalidate=revalidate,
                           prior_exposure=prior_exposure)
    seen = len(rec.get("history", []))
    print(f"\n  FROZEN: {rec['strategy']!r} at k={rec['k']}, config {rec['config_hash']}")
    print(f"  holdout exposures under this freeze: {rec['exposure_number']}"
          + (f" (superseding {seen} earlier freeze(s))" if seen else ""))
    print(f"  written to {SELECTION_STAMP.name}. Now run the full stage:")
    print("      python src/decision_replay.py")
    return dev


def run() -> dict:
    """
    The SUPPLIED Replay runner. You do not assemble control flow: you set MEETING, K_SHOTS,
    SHOT_STRATEGY and CAUSAL_GUIDANCE, write the two prompts, and fill CLAIM_REVIEWS after
    the audit prints its claims. The runner validates those inputs, refuses to write
    replay.json from any failed or unreviewed state, and records the statement's hash so
    replay_statement.md cannot be swapped afterwards.

    ORDER OF WORK. Choose on development first and freeze it:

        python src/decision_replay.py --dev      development only, then freeze
        python src/decision_replay.py            the full stage, including the holdout

    The holdout refuses to run against a selection that has changed since the freeze.
    """
    if not _prompts_written():
        raise NotImplementedError(
            "Set MEETING and write RECOMMENDATION_PROMPT and STATEMENT_PROMPT first. "
            "See the brief, Replay stage.")
    _require_causal_guidance()
    _require_complete_scores()

    t0 = time.time()
    config.stage_begin("replay", config_hash())
    config.ledger_reset("replay")
    feasible = feasible_strategies(MEETING)
    print(f"  meeting {MEETING}, k={K_SHOTS}")
    print(f"  strategies feasible here: {feasible}")
    if SHOT_STRATEGY not in feasible:
        raise RuntimeError(f"{SHOT_STRATEGY!r} cannot be built for {MEETING} at k={K_SHOTS}; "
                           f"feasible: {feasible}")

    print("\n  CHOOSING THE STRATEGY - development sample")
    dev = evaluate_all(sample="dev")

    print("\n  REPORTING THE STRATEGY - holdout sample, run once")
    _require_frozen_selection()
    hold = evaluate_all(sample="holdout")

    print(f"\n  DEEP REPLAY of {MEETING} with strategy {SHOT_STRATEGY!r}")
    plain = recommend(MEETING)
    guided = recommend(MEETING, causal_guidance=CAUSAL_GUIDANCE)
    # The two arms must actually have differed. With empty guidance `recommend()` builds
    # an identical prompt, the second call is served from the first one's cache, and the
    # stage used to report an A/B whose treatment arm never existed.
    if plain.get("prompt_sha") and plain.get("prompt_sha") == guided.get("prompt_sha"):
        raise RuntimeError(
            "the two A/B arms sent the SAME prompt, so there is no comparison to report. "
            "CAUSAL_GUIDANCE is what distinguishes them; see the Replay stage of the "
            "brief.")
    actual = actual_decision(MEETING)
    print(f"    shot mix: {plain['shot_mix']}")
    for tag, r in (("without causal guidance", plain), ("with causal guidance", guided)):
        got = r["result"].get("recommendation")
        print(f"    {tag:24s} -> {got} {r['result'].get('size_bp')}bp "
              f"({r['result'].get('confidence')})  "
              f"{'CORRECT' if got == actual['word'].lower() else 'wrong'}")
    print(f"    actual: {actual['label']}")

    # NOTHING REPORTABLE FROM A FAILED CALL. Both deep recommendations must have come
    # back valid before anything downstream is built - a failed call is not a "hold with
    # no reasoning", and replay.json must not be written from one.
    for tag, r in (("plain", plain), ("guided", guided)):
        if not r["ok"]:
            raise LLMCallError(
                f"the {tag} deep-replay recommendation failed ({r['error']}); "
                f"replay.json is not written from a failed call - re-run to retry it")

    chosen = guided if CAUSAL_GUIDANCE else plain
    evidence = evidence_block(MEETING, chosen["result"])
    statement = generate_statement(chosen["result"], evidence)  # raises if failed or empty
    coherent = validate_recommendation(chosen["result"])
    audit = audit_statement(statement, evidence)                # raises if the call failed
    if audit.get("n_reviewed_by_human", 0) < MIN_CLAIM_REVIEWS:
        raise RuntimeError(
            f"only {audit.get('n_reviewed_by_human', 0)} audited claims carry a human "
            f"review; {MIN_CLAIM_REVIEWS} are required before the artefact is written - "
            f"fill CLAIM_REVIEWS for the three highest-risk classifications first")
    arith = arithmetic_check(statement, MEETING, chosen["result"])
    if not arith.get("checked"):
        raise RuntimeError(f"the deterministic arithmetic check could not run "
                           f"({arith.get('reason')}); an unchecked statement must not "
                           f"reach replay.json")
    print(f"\n    statement: {len(statement.split())} words")
    print(f"    arithmetic: {arith['verdict']} "
          f"(before {arith.get('rate_before')}, expected {arith.get('expected_after')}, "
          f"quoted {arith.get('rates_quoted')})")
    print(f"    coherence: {coherent or 'direction and size agree'}")
    print(f"    claim audit: {audit.get('n_claims', 0)} claims - {audit.get('counts', {})}")
    print(f"    human review: {audit.get('n_reviewed_by_human', 0)} reviewed, "
          f"{audit.get('n_challenged_by_human', 0)} challenged")

    out = {
        "meeting": MEETING, "k": K_SHOTS, "strategy": SHOT_STRATEGY,
        "feasible_strategies": feasible,
        "dev_sample": dev.to_dict("records"),
        "holdout_sample": hold.to_dict("records"),
        # BOTH estimators, committed rather than printed. The raw rows above are per
        # DRAW; a table that averages them is answering a different question from the
        # assessed meeting-level vote, and the two were silently conflated once. Each
        # block names its estimator, its denominator and its seed count.
        # the freeze/exposure record, bound into the artefact so a marker reads the two
        # together rather than having to trust a separate file
        "selection": _read_selection(),
        # THE EXPOSURE EVIDENCE, bound to the benchmark it governs. A marker can check
        # that the holdout numbers below belong to a recorded exposure of the frozen
        # selection, rather than taking the freeze file's word for it.
        "exposure": {
            "log": EXPOSURE_LOG.name,
            "events": exposure_events(),
            "recorded": len(exposures_recorded()),
            "evidence_status": evidence_status(),
            "holdout_event_id": hold.attrs.get("exposure_event_id"),
            "selection_config_hash": selection_config_hash(),
        },
        "benchmark_summary": {
            "dev": {
                "assessed": dev.attrs.get("summary_majority_vote"),
                "per_draw": dev.attrs.get("summary_per_draw"),
                "paired_comparison": dev.attrs.get("paired_comparison"),
                **(dev.attrs.get("estimator_meta") or {})},
            "holdout": {
                "assessed": hold.attrs.get("summary_majority_vote"),
                "per_draw": hold.attrs.get("summary_per_draw"),
                "paired_comparison": hold.attrs.get("paired_comparison"),
                **(hold.attrs.get("estimator_meta") or {})}},
        "recommendation_plain": plain,
        "recommendation_guided": guided,
        "actual": actual,
        "evidence_block": evidence,
        "statement_words": len(statement.split()),
        # the statement's identity, whitespace-normalised: replay_statement.md must
        # contain THIS statement, not one with the same word count
        "statement_sha256": hashlib.sha256(
            " ".join(statement.split()).encode("utf-8")).hexdigest(),
        "claim_audit": audit,
        "recommendation_coherent": coherent is None,
        "coherence_problem": coherent,
        "arithmetic_check": arith,
        "claim_reviews": CLAIM_REVIEWS,
        "config_hash": config_hash(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    config.atomic_write_text(config.OUTPUTS / "replay.json",
                             json.dumps(out, indent=2, default=str))
    config.atomic_write_text(
        config.OUTPUTS / "replay_statement.md",
        f"# Replay statement - {MEETING}\n\n"
        f"Generated by {config.MODEL} from a {K_SHOTS}-shot '{SHOT_STRATEGY}' prompt.\n"
        f"Actual decision: **{actual['label']}**. "
        f"Recommended: **{chosen['result'].get('recommendation')} "
        f"{chosen['result'].get('size_bp')}bp**.\n\n"
        f"---\n\n{statement}\n\n---\n\n"
        f"## Deterministic arithmetic check\n\n```\n"
        f"{json.dumps(arith, indent=1)}\n```\n\n"
        f"## Claim audit against panel evidence\n\n{audit}\n")
    config.ledger_commit("replay")
    config.stage_complete("replay", config_hash())
    return out


if __name__ == "__main__":
    if "--dev" in sys.argv:
        # Development sample only, then freeze the selection. The holdout is untouched,
        # so iterate here as often as the budget allows.
        _note = ""
        for _a in sys.argv:
            if _a.startswith("--note="):
                _note = _a.split("=", 1)[1]
        run_dev(revalidate="--revalidate" in sys.argv, note=_note,
                prior_exposure="--prior-exposure" in sys.argv)
    else:
        run()
