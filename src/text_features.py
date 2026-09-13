"""
WORDS - seven quantitative dimensions of the Board's reaction, extracted from the minutes.

YOU SUPPLY FOUR CONSTRUCT RUBRICS AND THE SYSTEM PROMPT. EVERYTHING ELSE IS BUILT.

Three of the seven rubrics are supplied as fixed exemplars, one per scale type. They are
hash-checked and the run ABORTS if they have been edited - see `_require_exemplars()`.

-------------------------------------------------------------------------------------------
WHAT THESE MEASURES ARE FOR
-------------------------------------------------------------------------------------------
They are the AXES along which the Board's reaction to economic circumstances is described.
They are not predictors, and the Cycle stage shows they add nothing to forecasting the cycle.
Their job is to give Shock a vocabulary: a stress scenario can then be positioned against
the reaction profiles of episodes that actually happened.

So what they must be is DISCRIMINATING, RELIABLE and CORRECTLY ORIENTED. The audit below
tests all three, and the third is the one that catches real failures - a dimension can pass
every spread and stability statistic while pointing the wrong way.

-------------------------------------------------------------------------------------------
THE DECISION IS LEFT IN THE TEXT, DELIBERATELY
-------------------------------------------------------------------------------------------
The minutes state the decision taken at that meeting. That is not leakage here: the panel
carries `decision` as a numeric column anyway, both targets are forward-looking, and for a
measure of the Board's reaction the decision is part of what is being measured.

Where it WOULD leak is using a meeting's own minutes at that meeting, and `data_panel`
prevents that by lagging the whole text tier one meeting.

HOW TO RUN IT, IN ORDER:
    python src/text_features.py --pilot      iterate here: 25 fixed development documents,
                                             writes nothing, prints the full audit
    python src/text_features.py              development pass - validation withheld
    python src/text_features.py --validate   one shot, held-out meetings only

    add --offline to any of them to replay committed envelopes without calling the API
    add --dry-run to see the cost and stop before any call

WRITES data/processed/construct_scores.parquet
       data/processed/llm_raw/<config-hash>/<date>.json  (reproducibility
           envelopes - the parsed result plus the request settings, usage and
           request id. NOT the complete API response object)
       outputs/words_audit.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import courseapi  # noqa: E402

load_dotenv()

# ###########################################################################################
# YOUR WORK STARTS HERE
# ###########################################################################################

SYSTEM_PROMPT = """
TODO: WRITE THE SYSTEM PROMPT.

Who is the model? What is it reading? What is it judging?

Things worth deciding explicitly, because the default behaviour is poor:
  - what the midpoint means, and whether it is a legitimate answer or a hedge
  - what it should do when it is unsure (hint: its uncertainty is measured by the spread
    ACROSS calls, so hedging inside a single call destroys that measurement)
  - whether you want it to quote its evidence

Two things NOT to do, both of which were tried and failed:

  Do not claim the model has read the whole corpus. It sees one document per request. A
  persona that says otherwise does not create calibration, it just licenses invention.
  If you want cross-corpus calibration, describe the range in the rubric itself.

  Do not tell it to avoid round numbers to make the scores look spread out. That defeats
  the concentration diagnostic without improving the measurement, and the audit will still
  report the binned concentration.
"""

# -------------------------------------------------------------------------------------------
# THREE OF SEVEN ARE WRITTEN FOR YOU. FOUR ARE YOURS.
# -------------------------------------------------------------------------------------------
# The exemplars demonstrate the three SCALE TYPES and three techniques:
#
#   policy_stance          SIGNED. How to defeat a midpoint pile-up, and how to keep an axis
#                          pointing at DIRECTION when the language is about severity.
#   uncertainty_language   INTENSITY. How to score linguistic FORM rather than content.
#   global_risk_salience   PROPORTIONAL. How to score a SHARE of the discussion.
#
# Yours: inflation_concern, downside_risk_emphasis, financial_conditions_concern, vigilance.
# `downside_risk_emphasis` is the hard one - it is signed, and "risks are broadly balanced"
# is boilerplate that will concentrate the scores if you let it.

CONSTRUCTS: dict[str, str] = {

    # ---- SUPPLIED EXEMPLAR 1 of 3 - SIGNED - DO NOT ALTER --------------------------------
    "policy_stance": (
        "How hawkish is the Board's stance? SIGNED, 50 = exactly neutral. "
        "THIS AXIS IS DIRECTION, NOT FORCE. A Board cutting aggressively in a crisis is at "
        "the DOVISH extreme however decisive, urgent or grave its language. Do not let the "
        "severity of the discussion pull the score upward - severity belongs to the other "
        "constructs. Ask only: which way is policy leaning? "
        "MOST MEETINGS ARE HOLDS, AND A HOLD IS ALMOST NEVER EXACTLY NEUTRAL. The "
        "concluding 'Considerations for Monetary Policy' section nearly always leans one "
        "way - through what the Board says it would need to see, which risk it names "
        "first, and whether it calls the current setting accommodative or restrictive. "
        "Find the lean and score it. Reserve 50 for the rare document with no detectable "
        "lean. "
        "0 = maximally dovish: a large cut delivered with more foreshadowed. "
        "15 = a cut delivered, or an explicit statement that further easing is likely. "
        "30 = no cut this meeting, but a clear easing bias: conditions named that would "
        "prompt one, or policy called too restrictive. "
        "45 = hold leaning slightly dovish: downside risks named before upside ones. "
        "50 = no detectable lean. Rare. "
        "55 = hold leaning slightly hawkish: inflation risk named before activity risk. "
        "70 = no hike this meeting, but a clear tightening bias: conditions named that "
        "would prompt one, or policy called still accommodative. "
        "85 = a hike delivered, or an explicit statement that further tightening is likely. "
        "100 = maximally hawkish: a large hike delivered with more foreshadowed and policy "
        "described as needing to become restrictive."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "inflation_concern": (
        "TODO: how concerned is the Board about inflation? An INTENSITY construct - study "
        "the uncertainty_language exemplar for the pattern. Note that being BELOW target is "
        "not the same as being comfortably within it, and your rubric should separate them."
    ),

    # ---- YOURS - the hard one -----------------------------------------------------------
    "downside_risk_emphasis": (
        "TODO: are the risks the Board discusses skewed to the downside? SIGNED, like "
        "policy_stance - study that exemplar and apply the same landmark technique. "
        "State DOWNSIDE TO WHAT. And note the trap: 'upside risk to INFLATION' is not "
        "upside in the sense of activity or employment - it is adverse, and it usually "
        "implies tighter policy. Decide how you treat it and say so in the rubric."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "financial_conditions_concern": (
        "TODO: credit availability, funding costs, bank lending, housing finance, market "
        "functioning. An INTENSITY construct."
    ),

    # ---- SUPPLIED EXEMPLAR 2 of 3 - INTENSITY - DO NOT ALTER ----------------------------
    # RECALIBRATED 2026-09-11 for gpt-5.4-mini. The previous landmarks anchored the scale to
    # plain declarative English ("0 = no hedging at all", "25 = routine caveats only"), which
    # the RBA never writes. The whole corpus therefore compressed into the top half, and 65%
    # of calls came back as exactly 75 - the construct failed both the concentration and the
    # coverage gate and could not discriminate between documents at all. The axis is now
    # anchored to the BOARD'S OWN usual level at 50 and graded finely either side of it,
    # which is the same technique `policy_stance` uses to defeat its midpoint pile-up.
    "uncertainty_language": (
        "How heavily does the Board hedge? INTENSITY, 50 = the Board's OWN usual level. "
        "THE RBA ALWAYS HEDGES. Conditional forecasts, 'a range of outcomes' and 'depends "
        "on' appear in every set of minutes, so the PRESENCE of hedging tells you nothing "
        "and a document is not high on this axis merely for reading like central-bank "
        "prose. Score against how this Board writes ACROSS ITS 2006-2026 RANGE: measured "
        "against plain declarative English every document would sit near the top and the "
        "axis would measure nothing at all. "
        "Judge the FORM of the language, not whether the Board is right to be uncertain. "
        "WHAT SEPARATES DOCUMENTS is whether the hedging is BOILERPLATE or LOAD-BEARING. "
        "Boilerplate sits in the standard forecast caveats and the closing paragraph and "
        "could be deleted without changing the argument. Load-bearing hedging drives the "
        "discussion: scenarios set out and weighed, a decision explicitly deferred for more "
        "information, the Board naming what would change its mind. Apply that test before "
        "you score. "
        "50 IS THE BASELINE, NOT A REFUGE. Most documents sit somewhat above or below it, "
        "and the gradations between 35 and 80 exist to be used. "
        "CALIBRATION CHECK BEFORE YOU COMMIT: by construction the MEDIAN meeting of "
        "2006-2026 scores 50 here, because 50 is defined as this Board's usual level. So a "
        "routine meeting - no crisis, no turning point, the standard caveats in the standard "
        "places - is a 50, not a 70. If you are about to score an unremarkable document in "
        "the 70s, you are measuring hedging against ordinary English instead of against this "
        "Board. Re-read the 50 landmark and come down. Reserve the 70s and 80s for documents "
        "where uncertainty is visibly doing work the routine ones do not ask of it. "
        "0 = no hedging whatever: every forecast stated flat, no conditionals, no caveats. "
        "Not observed in this corpus - it anchors the scale rather than describing a "
        "document. "
        "20 = markedly more direct than this Board's usual: forecasts stated with unusual "
        "firmness, caveats confined to a single sentence. "
        "35 = slightly more direct than usual: the standard caveats are present but brief "
        "and undeveloped. "
        "50 = the Board's usual level: conditional forecast language throughout and "
        "uncertainty acknowledged in the usual places, none of it shaping the argument. "
        "60 = one source of uncertainty is named repeatedly and developed beyond the "
        "standard caveat. "
        "70 = two or more sources are argued rather than noted, and they shape how the "
        "outlook is described. "
        "80 = uncertainty shapes the DECISION: the Board says it is waiting for further "
        "information, or names what would change its mind. "
        "90 = multiple explicit scenarios are set out and weighed, with no single central "
        "path committed to. "
        "100 = the Board repeatedly says it does not know: forecasting explicitly "
        "suspended, uncertainty named as the reason for the decision."
    ),

    # ---- YOURS --------------------------------------------------------------------------
    "vigilance": (
        "TODO: how strongly does the Board commit to watching and responding? An INTENSITY "
        "construct. Judge the COMMITMENT LANGUAGE, not the sentiment - a worried Board that "
        "promises nothing scores lower than a calm Board that names what it will do. Almost "
        "every document contains some monitoring boilerplate, so decide what the floor is."
    ),

    # ---- SUPPLIED EXEMPLAR 3 of 3 - PROPORTIONAL - DO NOT ALTER -------------------------
    "global_risk_salience": (
        "How much of the Board's risk discussion is OFFSHORE rather than domestic? "
        "PROPORTIONAL, 50 = evenly split. "
        "AN EVEN SPLIT IS UNCOMMON. Every set of minutes opens with an international "
        "section, so the presence of offshore material tells you nothing. The question is "
        "how much of it reaches the RISK discussion and the policy conclusion. Most "
        "documents lean domestic; find the lean rather than settling on 50. "
        "0 = wholly domestic: the international section is perfunctory and no offshore "
        "risk enters the policy discussion. "
        "15 = offshore conditions summarised as benign background, then not referred to "
        "again. "
        "25 = mostly domestic; one offshore risk named but not developed. "
        "40 = domestic risks lead, offshore risks argued rather than merely listed. "
        "50 = offshore and domestic risks get comparable weight and comparable "
        "development. Rare. "
        "60 = offshore risks lead, but the policy conclusion turns on domestic conditions. "
        "75 = offshore risk is the dominant theme in the risk discussion. "
        "85 = an offshore development is named as a reason for the policy setting. "
        "100 = the decision is framed primarily around international developments - "
        "global financial stress, a major trading partner's downturn, or a global shock."
    ),
}

STUDENT_CONSTRUCTS = ["inflation_concern", "downside_risk_emphasis",
                      "financial_conditions_concern", "vigilance"]
SUPPLIED_CONSTRUCTS = ["policy_stance", "uncertainty_language", "global_risk_salience"]

# Whether the model must quote the phrase that drove each score. Quotes are validated as
# verbatim substrings of the source document, so this is a real check rather than decoration.
REQUIRE_EVIDENCE = True

# ###########################################################################################
# SUPPLIED BELOW THIS LINE - DO NOT MODIFY
# ###########################################################################################

# THE SCALE IS FIXED AT 0-100 INTEGERS. It is not a student choice, because the three
# supplied exemplars are written with 0/15/25/.../100 landmarks and rescaling them would
# make the locked rubrics incoherent. Scores are divided by 100 on aggregation, so every
# downstream contract sees 0-1.
SCORE_MIN, SCORE_MAX = 0, 100

FIELDS = list(CONSTRUCTS)
_client = None

EXEMPLAR_HASHES = {
    "policy_stance": "c1d9f9d7e666adfa",
    # recalibrated 2026-09-11 for gpt-5.4-mini - see the rubric comment above
    "uncertainty_language": "5bd868014bd1b1a2",
    "global_risk_salience": "20dbb4c6db17f337",
}


def client_():
    """The course API client (see courseapi): same shape, UNSW proxy underneath."""
    return courseapi.client_()


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _require_exemplars() -> None:
    """
    ABORT if a supplied exemplar has been edited. Not a warning.

    They are marked separately and every team must score against identical locked rubrics,
    or the audits are not comparable. If an exemplar genuinely fails on your data, that is a
    staff matter - report it, do not repair it yourself.
    """
    changed = [c for c in SUPPLIED_CONSTRUCTS
               if _sha(CONSTRUCTS[c]) != EXEMPLAR_HASHES.get(c)]
    if changed:
        raise RuntimeError(
            f"Supplied exemplar rubric(s) altered: {changed}. Restore them from the "
            f"repository. If you believe an exemplar is genuinely defective, report it to "
            f"the course staff - a versioned replacement will be issued.")


def _prompts_written() -> bool:
    return ("TODO" not in SYSTEM_PROMPT
            and not any("TODO" in CONSTRUCTS[c] for c in STUDENT_CONSTRUCTS))


def call_config_hash() -> str:
    """
    Identifies the CALL configuration - everything that changes what the API returns.
    The cache directory is keyed on this: revising a rubric and re-running makes fresh
    calls rather than silently returning the previous run's answers.
    """
    payload = json.dumps({
        "system_prompt": SYSTEM_PROMPT,
        "constructs": CONSTRUCTS,
        "scale": [SCORE_MIN, SCORE_MAX],
        "evidence": REQUIRE_EVIDENCE,
        "model": config.MODEL,
        "temperature": config.SAMPLING_TEMPERATURE,
        "call_index_base": config.CALL_INDEX_BASE,
        "n_calls": config.N_PARALLEL_CALLS,
        # The corpus the scores were extracted FROM. Swapping or editing the vendored
        # minutes after scoring must invalidate the run exactly as editing a rubric would.
        "corpus": _corpus_fingerprint(),
    }, sort_keys=True)
    return _sha(payload, 12)


def config_hash() -> str:
    """
    Identifies the STAGE configuration - the call configuration PLUS everything that
    shapes what is audited, gated, oriented or validated without touching the calls.
    Stored in words_audit.json and words_validation.json and compared by the submission
    check: loosening a gate or moving a validation episode after the artefacts were
    written invalidates them exactly as editing a rubric would.
    """
    payload = json.dumps({
        "calls": call_config_hash(),
        # The output ceiling CHANGES THE ANSWER: the same prompt under a lower ceiling
        # truncates. It belongs HERE, in the stage hash, and NOT in call_config_hash()
        # above - that one keys the cache directory, so putting it there would discard
        # every committed envelope and re-call the whole corpus the first time anyone
        # touched the ceiling. Lowering it now invalidates the ARTEFACT, which is what a
        # marker needs to see, while the calls made at the ceiling then in force survive.
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "gates": {"min_spread": config.MIN_CONSTRUCT_SPREAD,
                  "max_binned_concentration": config.MAX_BINNED_CONCENTRATION,
                  "bin_width": config.BIN_WIDTH,
                  "min_effective_bins": getattr(config, "MIN_EFFECTIVE_BINS", None),
                  "min_signal_to_noise": config.MIN_SIGNAL_TO_NOISE},
        "min_valid_calls": config.MIN_VALID_CALLS,
        "validation_episodes": VALIDATION_EPISODES,
        "expected_orientation": EXPECTED_ORIENTATION,
    }, sort_keys=True)
    return _sha(payload, 12)


_CORPUS_FP: str | None = None


def _corpus_fingerprint() -> str:
    """sha256 of the vendored minutes file, memoised - the corpus never changes within a
    process, and config_hash() is called per document during scoring."""
    global _CORPUS_FP
    if _CORPUS_FP is None:
        p = config.DOCUMENTS
        _CORPUS_FP = (hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                      if p.exists() else "absent")
    return _CORPUS_FP


def run_dir() -> "os.PathLike":
    d = config.LLM_RAW / call_config_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


# -------------------------------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------------------------------

def _schema_model() -> type[BaseModel]:
    fields: dict = {}
    for name, rubric in CONSTRUCTS.items():
        fields[name] = (int, Field(ge=SCORE_MIN, le=SCORE_MAX, description=rubric))
        if REQUIRE_EVIDENCE:
            fields[f"{name}_evidence"] = (
                str, Field(description=f"The exact phrase from the document that drove the "
                                       f"{name} score. Quote it verbatim; do not paraphrase."))
    return create_model("ConstructScores", **fields)


def score_once(text: str, call_index: int) -> dict:
    """
    One call, with bounded exponential backoff.

    Returns a REPRODUCIBILITY ENVELOPE: the parsed result plus the request settings, token
    usage, attempt count and request id. It is deliberately not called a raw response
    archive, because it is not one - the full API response object, including the unparsed
    message content and response headers, is not retained. What is here is enough to
    reproduce the run and to audit what was asked and answered.

    `request` is recorded from what the API client ACTUALLY sent, not from what this
    function meant to send. The two used to be assumed identical; since the move to the
    course proxy they are not - `seed` is gone from this model family - and an envelope
    that reports a parameter the API never received is a false audit record.

    `call_index` says WHICH of the N parallel draws this is. It is not a random seed and
    nothing here can pin the model's sampling; see config.CALL_INDEX_BASE.
    """
    schema = _schema_model()
    # THE EFFECTIVE REQUEST, COMPUTED BEFORE THE CALL, so a failure can record it.
    # A failed envelope used to carry only ok/error/call_index/config_hash/timestamp.
    # Without the ceiling it ran under, the compatibility rule below could not tell a
    # truncation worth re-asking at a higher ceiling from one already settled - so
    # re-running five real truncations at an UNCHANGED ceiling made five fresh calls.
    request = courseapi.effective_request(
        model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
        max_tokens=config.MAX_OUTPUT_TOKENS)
    failure = None
    spent_usage = None
    transport = 0
    _last_exception = None
    for attempt in range(config.MAX_RETRIES):
        try:
            r = client_().beta.chat.completions.parse(
                model=config.MODEL, temperature=config.SAMPLING_TEMPERATURE,
                call_index=call_index, max_tokens=config.MAX_OUTPUT_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content":
                           "RBA minutes of the monetary policy meeting:\n\n" + text}],
                response_format=schema)
            return {
                "ok": True,
                "parsed": r.choices[0].message.parsed.model_dump(),
                "model": config.MODEL,
                "temperature": config.SAMPLING_TEMPERATURE,
                "call_index": call_index,
                "request": getattr(r, "request", None),
                "provenance": getattr(r, "provenance", None),
                "model_served": getattr(r, "model", None),
                "config_hash": call_config_hash(),
                "prompt_hash": _prompt_hash(),
                "request_id": getattr(r, "id", None),
                # THE BILL for this draw, across every attempt it took - including the
                # ones instructor repaired away. `usage_final_response` is the last
                # response's own figure, kept separately so the two are never confused.
                "usage": courseapi.merge_usage(spent_usage,
                                               getattr(r, "call_usage", None)),
                "usage_final_response": getattr(r, "usage", None),
                "attempt": attempt + 1,
                "transport_requests": transport + ((getattr(r, "provenance", None) or {}
                                                   ).get("transport_requests") or 0),
                "envelope_version": courseapi.ENVELOPE_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:  # noqa: BLE001
            failure = courseapi.describe_failure(e, request=request)
            _last_exception = e
            # ACCUMULATED ACROSS OUTER ATTEMPTS, and `0` is a real answer: a failure
            # raised before any request left cost no requests, and `or 1` said it cost one.
            attempted = courseapi.transport_attempts(e)
            # AN UNSTAMPED FAILURE IS NOT EVIDENCE OF A REQUEST. The adapter stamps every
            # failure that passed through its transport path with the real count; one with
            # no stamp was not counted there, and treating it as one request wrote a
            # request into the record of a draw that never contacted the service.
            transport += attempted if isinstance(attempted, int) else 0
            spent_usage = courseapi.merge_usage(
                spent_usage, courseapi.failed_attempt_usage(e))
            # A RUN-LEVEL failure is not this call's failure, it is the end of the run:
            # every remaining call will fail the same way, and the fix is outside the
            # repository. Recording it as a failed call instead let one run mark 66 calls
            # "failed", score twelve documents with zero valid calls, and only stop later
            # in aggregate() - by which point the cache held a corpus that looked scored
            # and was not. Worse, a mistyped access code wrote settled failures that
            # survived correcting it, and an adapter TypeError cached five settled
            # failures per document that survived FIXING THE BUG. Credentials, deployment,
            # daily budget and plain programming faults are all in this class: stop here,
            # keep what is committed, and resume once the fault is repaired.
            if failure["run_level"]:
                raise
            # THE TRANSPORT BUDGET IS NOT THIS LOOP'S TO SPEND TWICE. The adapter says
            # whether anything below it already retried this failure. One persistent 429
            # used to cost 5 wrapper requests x 4 attempts here = 20 HTTP requests for a
            # single logical draw, against a 60/minute budget shared by the whole class.
            if courseapi.transport_budget_spent(e):
                break
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_BASE_SECONDS * (2 ** attempt)
                           + random.uniform(0, 0.5))
    # THE SHARED BUILDER, so this stage cannot drift from the other two. It records the
    # classified reason - `truncated`, `refusal` and `nonterminal` all used to arrive as
    # "IncompleteResponseError" and were treated identically - the ceiling and model the
    # attempts actually ran under, and the accumulated bill: a truncated answer is billed,
    # and dropping its usage understates the run against a shared allowance.
    return courseapi.failure_envelope(
        _last_exception, request=request, config_hash=call_config_hash(),
        call_index=call_index, transport=transport, usage=spent_usage,
        prompt_hash=_prompt_hash())


# Errors that retrying cannot fix: bad credentials, malformed requests (including a schema
# the endpoint rejects), and plain programming errors. Matched by name so no extra imports
# are needed; anything else (rate limits, timeouts, connection drops, 5xx) is transient.
_NON_TRANSIENT = ("AuthenticationError", "PermissionDeniedError", "BadRequestError",
                  "NotFoundError", "UnprocessableEntityError",
                  "TypeError", "KeyError", "AttributeError", "ValidationError"
                  ) + courseapi.NON_TRANSIENT_ERRORS


#: Set when a run-wide failure (exhausted daily budget, bad credentials) makes every
#: remaining call pointless. Workers check it before starting, so a document already in
#: flight finishes and nothing new is scheduled.
_RUN_ABORTED = threading.Event()

#: One writer at a time per process. Each completed draw is flushed to disk as it lands,
#: so a quota failure on the fifth call cannot discard the four that succeeded.
_CACHE_LOCK = threading.Lock()


#: Failures of the RUN, not of the document. Bad or missing credentials, an undeployed
#: model and an exhausted daily budget are properties of how the run was configured, and
#: every one of them is fixed OUTSIDE the repository - by editing .env, or by waiting.
#:
#: THE DISTINCTION MATTERS BECAUSE SETTLING THEM POISONS THE CACHE. An earlier version
#: treated every non-transient error as a settled draw, so a run started with a mistyped
#: access code wrote five "settled" failures per document; correcting the code and
#: re-running then made ZERO fresh calls and produced ZERO valid draws, because the
#: cache configuration had not changed and every index looked resolved. The first
#: keystroke error silently destroyed the corpus.
#: A PROGRAMMING FAULT IS IN THIS LIST TOO, and for the same reason. A synthetic
#: adapter TypeError used to be cached as five settled failures per document; repairing
#: the bug and re-running then made ZERO calls and kept ZERO valid draws, because the
#: configuration had not changed and every index looked resolved. A bug in this
#: repository is not a property of the document, it is a property of the run - and it is
#: fixed in the same place a mistyped access code is fixed.
_RUN_LEVEL_ERRORS = (
    "MissingCredentialsError", "InvalidAccessCodeError", "MissingHeaderError",
    "AuthenticationError", "PermissionDeniedError",
    "ModelNotAvailableError", "DailyQuotaExceededError",
    "ProxyUnreachableError",
    "TypeError", "KeyError", "AttributeError", "IndexError", "NameError",
    "ImportError", "ModuleNotFoundError", "ZeroDivisionError",
)


#: THE SHARED CONTRACT, not a Words-local copy. Replay and Shock validate against the
#: same definition, because three stages with three private ideas of what an envelope
#: guarantees is how two of them ended up writing no version at all and reusing a cache
#: that named a different deployment.
envelope_generation = courseapi.envelope_generation


def failure_category(call: dict) -> str | None:
    """The classified reason a draw failed, for records that carry one.

    `None` means a legacy failure envelope that predates classification. Callers fall
    back to the exception name for those and must not pretend to more.
    """
    if call.get("ok"):
        return None
    cat = (call.get("failure") or {}).get("category")
    return cat if cat in courseapi.FAILURE_CATEGORIES else None


def _error_name(call: dict) -> str:
    return str(call.get("error", "")).split(":", 1)[0].strip()


def _is_run_level(call: dict) -> bool:
    """Did this draw fail because the RUN is misconfigured, rather than the document?"""
    cat = failure_category(call)
    if cat is not None:
        return cat in courseapi.RUN_LEVEL_CATEGORIES
    return _error_name(call) in _RUN_LEVEL_ERRORS      # legacy envelope


def recorded_ceiling(call: dict) -> int | None:
    """The output ceiling this draw was actually produced under, if it recorded one."""
    v = (call.get("request") or {}).get("max_output_tokens")
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _truncated(call: dict) -> bool:
    """Did this draw stop at the output ceiling - as opposed to being refused, filtered
    or left unfinished? All three used to arrive as `IncompleteResponseError` and a
    substring test could not tell them apart, so a refusal was re-asked whenever the
    ceiling moved and a truncation was settled whether or not it could still be fixed."""
    cat = failure_category(call)
    if cat is not None:
        return cat == "truncated"
    return "IncompleteResponseError" in str(call.get("error", ""))


def reusable_under_current_ceiling(call: dict) -> bool:
    """
    THE OUTPUT-CEILING COMPATIBILITY RULE, applied per draw.

    The ceiling is deliberately NOT part of the cache key: putting it there would throw
    away every committed envelope the first time anyone touched it, and the committed
    evidence for this assignment was in fact produced under three different ceilings
    (1500, 4000 and 8000 tokens) as the pipeline was developed. Discarding all of it
    would be a large, pointless bill. Instead each draw records the ceiling it ran under
    and this rule decides, honestly, whether it still stands:

      SUCCESS at ceiling X, current ceiling C
        C >= X  reusable. The answer finished inside X tokens, so a larger allowance
                could not have cut it short.
        C <  X  NOT reusable. That answer was allowed more room than the current
                configuration gives, and may not fit; re-ask at C.

      TRUNCATION at ceiling X
        C >  X  NOT settled. More room may finish it - this is the case the brief tells
                students to fix by RAISING config.MAX_OUTPUT_TOKENS, and it silently did
                nothing before.
        C <= X  still settled: the same wall, in the same place.

      NO RECORDED CEILING
        Successes are not reused (nothing establishes what they were allowed), and
        truncations are retried. Content filtering does not depend on the ceiling, so a
        filtered draw stays settled either way.
    """
    current = int(config.MAX_OUTPUT_TOKENS)
    recorded = recorded_ceiling(call)
    if call.get("ok"):
        return recorded is not None and current >= recorded
    if _truncated(call):
        return recorded is not None and current <= recorded
    return True                      # filtering and schema refusals: ceiling-independent


def _settled_failure(call: dict) -> bool:
    """A failed draw that asking again cannot fix, and that is THIS DOCUMENT's problem.

    A content filter, a rejected schema or a truncated answer is deterministic: the same
    prompt at the same settings fails the same way on this document. Re-running it on
    every pass spends the shared token budget to re-learn what the envelope already
    records.

    NOT settled: transient failures (timeouts, 429s, dropped connections), and run-level
    failures (credentials, deployment, daily budget). The second group is deterministic
    too - but it is fixed by changing the run, and a fixed run must be able to fill the
    gap. See `_RUN_LEVEL_ERRORS`.
    """
    if call.get("ok"):
        return False
    if _is_run_level(call):
        return False
    cat = failure_category(call)
    if cat is not None:
        return cat in courseapi.SETTLED_CATEGORIES
    return _error_name(call) in _NON_TRANSIENT         # legacy envelope


def _prompt_hash() -> str:
    """Identifies the rubric this draw was scored against."""
    return _sha(SYSTEM_PROMPT + json.dumps(CONSTRUCTS, sort_keys=True))


def _envelope_contradictions(call: dict) -> list[str]:
    """Words' view of the shared contract: adds the rubric identity to the common checks."""
    return courseapi.envelope_contradictions(
        call, config_hash=call_config_hash(), prompt_hash=_prompt_hash())


def _archive_superseded(out, dropped: list[tuple[dict, str]]) -> None:
    """Keep an envelope that a settings change has replaced, instead of deleting it.

    A draw re-asked under a raised ceiling overwrites its predecessor in the document
    file. The previous version treated the committed git history as the archive, which
    only works for teams that commit between every run - and the record most worth
    keeping is the truncation that motivated raising the ceiling in the first place.
    """
    if not dropped:
        return
    archive = out.parent / "superseded" / f"{out.stem}.jsonl"
    archive.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with _CACHE_LOCK, archive.open("a", encoding="utf-8") as fh:
        for call, why in dropped:
            fh.write(json.dumps({"superseded_at": stamp, "superseded_because": why,
                                 "envelope": call}) + "\n")


def _load_cached_calls(out, schema) -> dict[int, dict]:
    """Valid draws already on disk, by call index. Quarantines an unusable file.

    Returns BOTH successful draws and settled failures: a document with three good draws
    and two filtered ones is complete under MIN_VALID_CALLS, and re-calling the two
    filtered indices would just reproduce the filter.
    """
    if not out.exists():
        return {}
    expected = {config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)}
    try:
        calls = json.loads(out.read_text())
        keep: dict[int, dict] = {}
        superseded: list[tuple[dict, str]] = []
        for c in calls:
            idx = c.get("call_index")
            # DRAW IDS ARE PART OF THE EXPERIMENT, not a label. An envelope numbered
            # outside the current draw set was produced by a different configuration
            # (or by hand), and counting it towards this run's coverage would let a
            # document pass the minimum on evidence this run never requested. A
            # duplicate id is worse: a dict silently collapses it, so five copies of
            # one draw would have read as five independent draws.
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise ValueError(f"call_index {idx!r} is not an integer")
            if idx not in expected:
                raise ValueError(
                    f"call_index {idx} is outside the current draw set "
                    f"{sorted(expected)}")
            if idx in keep:
                raise ValueError(f"duplicated call_index {idx}")
            if c.get("config_hash") != call_config_hash():
                continue
            contradictions = _envelope_contradictions(c)
            if contradictions:
                raise ValueError(
                    f"draw {idx} contradicts itself or this run: "
                    + "; ".join(contradictions))
            if not reusable_under_current_ceiling(c):
                superseded.append(
                    (c, f"produced under output ceiling "
                        f"{recorded_ceiling(c)}, now {config.MAX_OUTPUT_TOKENS}"))
                continue        # re-ask this draw under the ceiling now in force
            if c.get("ok"):
                schema.model_validate(c["parsed"])       # re-validate, never trust "ok"
                keep[idx] = c
            elif _settled_failure(c):
                keep[idx] = c
        _archive_superseded(out, superseded)
        return keep
    except Exception as e:  # noqa: BLE001 - any invalid cache is quarantined
        bad = out.with_suffix(out.suffix + ".invalid")
        try:
            out.replace(bad)
            moved = f" Moved to {bad.name};"
        except OSError:
            moved = ""
        print(f"    CACHE QUARANTINED: {out.name} - no longer validates "
              f"({type(e).__name__}).{moved} re-scoring this document fresh.")
        return {}


def _flush(out, by_index: dict[int, dict]) -> None:
    """Write the draws collected so far, atomically, ordered by call index."""
    with _CACHE_LOCK:
        config.atomic_write_text(
            out, json.dumps([by_index[i] for i in sorted(by_index)], indent=1))


def score_document(date: str, text: str, offline: bool = False) -> dict:
    """
    N calls for one document, cached under the configuration hash.

    A cached call is only trusted after its parsed payload RE-validates against the current
    response schema: a cache written under an older schema, truncated on disk, or edited by
    hand is quarantined and the document re-scored, rather than flowing into the combined
    table because it once said "ok".

    PARTIAL WORK IS KEPT AND REUSED. Each draw is written to the document's envelope file
    the moment it returns, so an exhausted daily budget on the fifth call no longer throws
    away the four that succeeded. On the next run only the MISSING draws are requested: a
    document that already holds enough valid draws costs nothing, and one that holds three
    good draws and two content-filtered ones is complete rather than permanently re-tried.

    `offline=True` never contacts the service: a document is accepted if the committed
    envelopes already carry config.MIN_VALID_CALLS valid draws, and the run stops with a
    clear message if they do not.
    """
    schema = _schema_model()
    out = run_dir() / f"{date}.json"
    by_index = _load_cached_calls(out, schema)
    expected = [config.CALL_INDEX_BASE + i for i in range(config.N_PARALLEL_CALLS)]
    n_valid = sum(1 for c in by_index.values() if c.get("ok"))
    missing = [i for i in expected if i not in by_index]

    if not missing:
        config.ledger_add("words", out)
        return {"meeting_date": date, "calls": [by_index[i] for i in sorted(by_index)],
                "cached": True, "fresh": []}

    if offline:
        if n_valid >= config.MIN_VALID_CALLS:
            config.ledger_add("words", out)
            return {"meeting_date": date,
                    "calls": [by_index[i] for i in sorted(by_index)],
                    "cached": True, "fresh": []}
        raise RuntimeError(
            f"--offline: {date} has only {n_valid} valid committed draws, and the minimum "
            f"is {config.MIN_VALID_CALLS}. Offline mode replays committed evidence and "
            f"never calls the service; re-run without --offline to fill the gap.")

    if _RUN_ABORTED.is_set():
        raise RuntimeError(f"{date}: run already stopped by a run-wide failure")

    if n_valid or by_index:
        print(f"    {date}: reusing {n_valid} committed draw(s), requesting "
              f"{len(missing)}")
    with ThreadPoolExecutor(max_workers=config.N_PARALLEL_CALLS) as ex:
        futs = {ex.submit(score_once, text, i): i for i in missing}
        try:
            for f in as_completed(futs):
                call = f.result()
                by_index[futs[f]] = call
                _flush(out, by_index)          # persisted the moment it lands
        except BaseException:
            # A run-wide failure (quota, credentials) makes every QUEUED call pointless,
            # but the calls already in flight will land anyway and were already paid for.
            # Cancel what has not started, collect what has, and keep the lot: losing four
            # good draws because the fifth hit the daily cap is how a stopped run used to
            # cost a document its entire evidence.
            _RUN_ABORTED.set()
            for pending in futs:
                pending.cancel()
            for done, idx in futs.items():
                if done.cancelled() or idx in by_index:
                    continue
                try:
                    by_index[idx] = done.result()
                except Exception:  # noqa: BLE001 - this draw is the one that failed
                    continue
            _flush(out, by_index)
            raise
    config.ledger_add("words", out)
    return {"meeting_date": date, "calls": [by_index[i] for i in sorted(by_index)],
            "cached": False,
            # THE DRAWS THIS RUN ACTUALLY PAID FOR. A resumed document is not "fresh":
            # reporting all five of its draws as fresh calls inflated a one-call resume
            # into five, and its 11 tokens into 55 - which then fed the budget planning
            # the brief asks teams to do.
            "fresh": [by_index[i] for i in missing if i in by_index]}


#: Quotation marks a model may wrap around an otherwise verbatim quote - straight and
#: curly, single and double. gpt-5.4-mini returns `"..."` where gpt-4o-mini returned
#: `...`, which is a presentation habit and not a paraphrase.
_ENCLOSING_QUOTES = "\"'“”‘’«»"


def _validate_evidence(quote: str, source: str) -> bool:
    """
    A quote must appear verbatim in the document. Whitespace-normalised comparison.

    ENCLOSING QUOTATION MARKS ARE STRIPPED FIRST, and that is a normalisation rather than
    a loosening: a model that hands back `"members noted..."` has quoted the document
    exactly and merely punctuated the fact that it is quoting. Requiring the bare form
    would mark a correct quote wrong on a typographic habit - which is precisely what
    happened when this stage moved to gpt-5.4-mini, taking the verbatim rate from ~84% to
    0% without a single word of any quote changing.

    Everything that WOULD be a loosening is still refused: the quote must still occur, in
    order, inside the document. A paraphrase fails, a stitched-together quote fails, and a
    quote shorter than 12 characters fails as too short to evidence anything.
    """
    if not quote:
        return False
    q = quote.strip()
    # only a MATCHED enclosing pair comes off, so an internal quotation is untouched
    while (len(q) >= 2 and q[0] in _ENCLOSING_QUOTES and q[-1] in _ENCLOSING_QUOTES):
        q = q[1:-1].strip()
    if len(q) < 12:
        return False
    norm = lambda s: " ".join(s.split()).lower()
    return norm(q) in norm(source)


def aggregate(records: list[dict], docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mean and spread per construct, plus per-call evidence and its validation status."""
    span = float(SCORE_MAX - SCORE_MIN)
    source = dict(zip(docs["meeting_date"], docs["text_scored"]))
    rows, ev_stats = [], {"checked": 0, "verbatim": 0}
    for rec in records:
        ok = [c["parsed"] for c in rec["calls"] if c.get("ok")]
        if len(ok) < config.MIN_VALID_CALLS:
            # WHY it is short decides what to do about it, and "re-run" is the wrong
            # advice for most of the reasons. A filtered document is settled: an
            # unchanged re-run makes no calls at all by design, so telling a team to
            # re-run sends them round a loop that cannot terminate.
            reasons = {}
            for call in rec["calls"]:
                if call.get("ok"):
                    continue
                cat = (failure_category(call)
                       or _error_name(call) or "unrecorded")
                reasons[cat] = reasons.get(cat, 0) + 1
            # CAN THIS DOCUMENT STILL REACH THE MINIMUM? That is the only question that
            # decides between "here is how to recover it" and "this one is genuinely
            # blocked". The previous condition compared the filtered count against
            # `N_PARALLEL_CALLS - len(ok)`, which double-counts the valid draws, so two
            # valid draws plus one filtered plus two TRUNCATED ones - recoverable by
            # raising the ceiling - were declared blocked and sent to the waiver.
            by_ceiling = reasons.get("truncated", 0)
            by_retry = sum(reasons.get(c, 0) for c in
                           ("transport", "rate_limit", "nonterminal", "unrecorded"))
            missing = config.N_PARALLEL_CALLS - len(rec["calls"])
            unrecoverable = sum(reasons.get(c, 0) for c in
                                ("content_filter", "refusal", "request_rejected",
                                 "schema"))
            reachable = len(ok) + by_ceiling + by_retry + missing

            if reachable >= config.MIN_VALID_CALLS:
                steps = []
                if by_ceiling:
                    steps.append(
                        f"{by_ceiling} draw(s) hit the output ceiling - raise "
                        f"config.MAX_OUTPUT_TOKENS and re-run; only those draws are "
                        f"re-asked")
                if by_retry:
                    steps.append(
                        f"{by_retry} draw(s) failed in transit - re-run to retry them")
                if missing:
                    steps.append(f"{missing} draw(s) were never made - re-run")
                if unrecoverable:
                    steps.append(
                        f"the {unrecoverable} settled draw(s) stay settled and are NOT "
                        f"re-asked, which is correct - report them")
                advice = ("This document can still reach the minimum: "
                          + "; ".join(steps) + ".")
            else:
                advice = (
                    f"Only {reachable} of the {config.MIN_VALID_CALLS} required draws "
                    f"are reachable: {unrecoverable} are settled and deterministic, so "
                    f"re-running the same document at the same settings makes no new "
                    f"calls and will not clear them. DO NOT edit the corpus - it is "
                    f"hash-checked and editing it invalidates the run. This is an "
                    f"approved-service failure: report the blocked document and the "
                    f"service's own reason in your Words write-up, email the lecturer "
                    f"before the deadline (brief S6 step 3), and see the rubric's "
                    f"approved-service table for what is waived.")
            raise RuntimeError(
                f"{rec['meeting_date']}: only {len(ok)} valid calls, minimum is "
                f"{config.MIN_VALID_CALLS}. Failed draws: "
                f"{ {k: v for k, v in sorted(reasons.items())} }.\n{advice}")
        row = {"meeting_date": rec["meeting_date"], "n_calls_valid": len(ok)}
        for f in FIELDS:
            v = np.array([c[f] for c in ok if f in c], dtype=float) / span
            row[f] = v.mean()
            row[f"{f}_sd"] = v.std(ddof=1) if len(v) > 1 else 0.0
            if REQUIRE_EVIDENCE:
                quotes = [c.get(f"{f}_evidence", "") for c in ok]
                good = [q for q in quotes
                        if _validate_evidence(q, source.get(rec["meeting_date"], ""))]
                ev_stats["checked"] += len(quotes)
                ev_stats["verbatim"] += len(good)
                # The quote from the call whose score is CLOSEST TO THE AGGREGATE, not the
                # longest one. The longest quote can come from the outlier call, so the
                # evidence shown could argue for a score the row does not report.
                order = np.argsort(np.abs(v - row[f]))
                row[f"{f}_evidence"] = next(
                    (quotes[i] for i in order
                     if i < len(quotes) and quotes[i] in good), 
                    (good or quotes or [""])[0])
                row[f"{f}_evidence_verbatim"] = len(good) / max(1, len(quotes))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("meeting_date").reset_index(drop=True), ev_stats


# -------------------------------------------------------------------------------------------
# The audit
# -------------------------------------------------------------------------------------------

def check_construct_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three properties, each with an explicit criterion.

    DISCRIMINATION - binned concentration. Scores are binned at config.BIN_WIDTH and the
        share falling in the busiest bin is reported. This is a CONCENTRATION statistic, not
        the share at an exact value, and the bin width is part of its definition. Standard
        deviation alone passes a construct that is constant for most of the corpus, because
        a minority of dispersed documents drags the sd up.

    SEPARATION - the between/within variance ratio. Between-document variance divided by
        mean within-document (call-to-call) variance. Below 1 the construct varies more
        between repeat calls on one document than between different documents, which means
        it is measuring noise.

        This is NOT reliability in the psychometric sense, and it was called that until a
        reviewer pointed out the overstatement. It is an ad hoc signal-to-noise diagnostic
        computed on five calls per document: no variance-component model, no uncertainty,
        no intraclass correlation. Read it as "does this dimension distinguish documents
        by more than it wobbles", which is a useful question, and not as a reliability
        coefficient you could quote in a methods section.

    COVERAGE - effective number of bins, from the entropy of the binned distribution. A
        construct using three effective bins out of twenty is coarse even if no single bin
        dominates.
    """
    rows = []
    for f in FIELDS:
        v = df[f].dropna()
        binned = (v / config.BIN_WIDTH).round().astype(int)
        share = binned.value_counts(normalize=True)
        conc = float(share.iloc[0])
        entropy = float(-(share * np.log(share)).sum())
        within = float((df[f"{f}_sd"] ** 2).mean())
        snr = float(v.var() / within) if within > 0 else np.inf
        rows.append({
            "construct": f,
            "sd": float(v.std()),
            "binned_concentration": conc,
            "modal_bin_centre": float(share.index[0] * config.BIN_WIDTH),
            "effective_bins": float(np.exp(entropy)),
            "mean_call_sd": float(df[f"{f}_sd"].mean()),
            "between_within_ratio": snr,
            "evidence_verbatim_rate": (float(df[f"{f}_evidence_verbatim"].mean())
                                       if f"{f}_evidence_verbatim" in df.columns else np.nan),
            "passes_spread": float(v.std()) >= config.MIN_CONSTRUCT_SPREAD,
            "passes_concentration": conc <= config.MAX_BINNED_CONCENTRATION,
            "passes_coverage": float(np.exp(entropy)) >= config.MIN_EFFECTIVE_BINS,
            "passes_separation": snr >= config.MIN_SIGNAL_TO_NOISE,
        })
    out = pd.DataFrame(rows)
    print(f"  gates: sd >= {config.MIN_CONSTRUCT_SPREAD}, concentration <= "
          f"{config.MAX_BINNED_CONCENTRATION:.0%} at bin width {config.BIN_WIDTH}, "
          f"effective bins >= {config.MIN_EFFECTIVE_BINS}, "
          f"between/within >= {config.MIN_SIGNAL_TO_NOISE}")
    for r in out.itertuples():
        fails = [n for n, ok in (("spread", r.passes_spread),
                                 ("concentration", r.passes_concentration),
                                 ("coverage", r.passes_coverage),
                                 ("separation", r.passes_separation)) if not ok]
        flag = f"   <-- FAILS {', '.join(fails)}" if fails else ""
        print(f"  {r.construct:30s} sd={r.sd:.3f} conc={r.binned_concentration:5.1%} "
              f"eff_bins={r.effective_bins:4.1f} B/W={r.between_within_ratio:5.1f} "
              f"quotes_ok={r.evidence_verbatim_rate:5.1%}{flag}")
    return out


# What each construct SHOULD do as the decision moves cut -> hold -> hike.
#   "up"    higher on hikes  (hawkishness, inflation concern)
#   "down"  higher on cuts   (downside-risk emphasis - it runs the other way, and an earlier
#           version demanded cut < hold < hike for every construct, so the one dimension
#           that is correctly inverted was reported as failing)
#   None    no directional expectation; reported but not judged
EXPECTED_ORIENTATION = {
    "policy_stance": "up",
    "inflation_concern": "up",
    "downside_risk_emphasis": "down",
    "financial_conditions_concern": None,
    "uncertainty_language": None,
    "vigilance": None,
    "global_risk_salience": None,
}


def orientation_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does each SIGNED dimension point the right way?

    Mean score grouped by the decision actually taken. `policy_stance` must be monotonic
    with a visible gap between cuts and holds. This is the check that catches a dimension
    reading the Board's forcefulness as hawkishness, and no spread or stability statistic
    can substitute for it.
    """
    panel_path = config.DATA_PROCESSED / "panel.parquet"
    if not panel_path.exists():
        print("  (no panel yet - run `python src/data_panel.py` to enable this check)")
        return pd.DataFrame()
    p = pd.read_parquet(panel_path)
    p["meeting_date"] = pd.to_datetime(p["meeting_date"])
    s = df.copy()
    s["meeting_date"] = pd.to_datetime(s["meeting_date"])
    j = s.merge(p[["meeting_date", "decision"]], on="meeting_date")

    rows = []
    for f in FIELDS:
        g = j.groupby("decision")[f].mean()
        cut, hold, hike = g.get(-1, np.nan), g.get(0, np.nan), g.get(1, np.nan)
        want = EXPECTED_ORIENTATION.get(f)
        if want == "up":
            ok = bool(cut < hold < hike)
            gap = float(hold - cut)
        elif want == "down":
            ok = bool(cut > hold > hike)
            gap = float(cut - hold)
        else:
            ok, gap = None, float(abs(hike - cut))
        rows.append({"construct": f, "mean_on_cut": cut, "mean_on_hold": hold,
                     "mean_on_hike": hike, "expected": want,
                     "monotonic_as_expected": ok, "cut_hold_gap": gap})
    out = pd.DataFrame(rows)
    print("\n  ORIENTATION (mean score by the decision actually taken)")
    print("  NOTE: for policy_stance this is an IMPLEMENTATION CHECK, not external "
          "validation -")
    print("  the supplied rubric names the decision explicitly, so monotonicity confirms "
          "the rubric")
    print("  was applied, not that the construct measures something independent of it.")
    for r in out.itertuples():
        if r.expected is None:
            note = "   (no directional expectation)"
        elif r.monotonic_as_expected and r.cut_hold_gap > 0.1:
            note = "   OK"
        else:
            note = f"   <-- expected to run {r.expected} across cut/hold/hike"
        print(f"  {r.construct:30s} cut={r.mean_on_cut:.3f} hold={r.mean_on_hold:.3f} "
              f"hike={r.mean_on_hike:.3f}{note}")
    return out


# -------------------------------------------------------------------------------------------
# Episode separation
# -------------------------------------------------------------------------------------------
# Prompt iteration and prompt validation must not use the same episodes. If you tune a rubric
# until it produces the right answer on the GFC, then report that it produces the right answer
# on the GFC, you have reported your own tuning back to yourself.
#
# DEVELOPMENT episodes are what you may read, argue about and tune against.
# VALIDATION episodes are scored ONCE, after the rubrics are final, and reported as they fall.
# The supplied exemplars name no episode at all, for the same reason.

DEV_EPISODES = {
    "pre-GFC tightening":   ("2007-08-07", "2008-03-04"),
    "mining boom plateau":  ("2011-02-01", "2011-11-01"),
    "post-taper easing":    ("2013-05-07", "2013-08-06"),
}

VALIDATION_EPISODES = {
    "GFC":                  ("2008-09-02", "2009-04-07"),
    "COVID onset":          ("2020-03-03", "2020-11-03"),
    "2022 tightening":      ("2022-05-03", "2022-12-06"),
    "calm (2015-16)":       ("2015-01-01", "2016-12-31"),
}


def episode_check(df: pd.DataFrame, which: str = "validation") -> pd.DataFrame:
    """
    Mean construct scores over named episodes. `which` is 'dev' or 'validation'.

    Run 'dev' as often as you like while writing rubrics. Run 'validation' ONCE, when the
    rubrics are frozen, and report what it gives you - including the dimensions that come
    out wrong. A construct that fails here and is then re-tuned must be revalidated, and the
    report must say that it was.
    """
    eps = DEV_EPISODES if which == "dev" else VALIDATION_EPISODES
    d = df.copy()
    d["meeting_date"] = pd.to_datetime(d["meeting_date"])
    rows = []
    for name, (a, b) in eps.items():
        w = d[(d.meeting_date >= a) & (d.meeting_date <= b)]
        if not len(w):
            continue
        r = {"episode": name, "n": len(w)}
        r.update({f: float(w[f].mean()) for f in FIELDS})
        rows.append(r)
    out = pd.DataFrame(rows)
    print(f"\n  {which.upper()} EPISODES (mean score)")
    if len(out):
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.round(3).to_string(index=False))
    return out


def load_documents() -> pd.DataFrame:
    docs = pd.read_parquet(config.DOCUMENTS)
    docs["meeting_date"] = pd.to_datetime(docs["meeting_date"]).dt.strftime("%Y-%m-%d")
    docs["text_scored"] = docs["text_full"]
    return docs


# -------------------------------------------------------------------------------------------
# Development and validation are SEPARATE RUNS over DISJOINT documents
# -------------------------------------------------------------------------------------------
#
#     python src/text_features.py                 development: everything EXCEPT the
#                                                 validation meetings. Run as often as you like.
#     python src/text_features.py --validate      one shot, validation meetings only, stamped
#     python src/text_features.py --validate --revalidate
#                                                 do it again after a prompt change, recorded
#                                                 as a second exposure. Declare it in the report.
#     python src/text_features.py --dry-run       no API calls
#
# WHY THIS IS TWO COMMANDS AND NOT ONE. An earlier version scored ALL 211 documents on every
# run and stamped the validation table on the first one. The validation episodes were
# therefore scored under every prompt a team ever tried; freezing the first table afterwards
# only hid that, it did not prevent it. Held out means NOT SCORED, not "scored and then not
# shown".

VALIDATION_STAMP = config.OUTPUTS / "words_validation.json"


def validation_meetings(docs: pd.DataFrame) -> set[str]:
    """The meeting dates reserved for validation. Never scored in a development run."""
    d = pd.to_datetime(docs["meeting_date"])
    keep: set[str] = set()
    for a, b in VALIDATION_EPISODES.values():
        keep |= set(docs.loc[(d >= a) & (d <= b), "meeting_date"])
    return keep


#: Documents in the iteration pilot. Large enough for the audit gates to mean something,
#: small enough that a rubric rewrite costs about a seventh of a full development pass.
PILOT_N = 25


def pilot_meetings(docs: pd.DataFrame) -> list[str]:
    """
    The FIXED pilot sample: where rubric iteration happens.

    WHY A PILOT EXISTS. The cache is keyed on your prompts, so editing a rubric re-scores
    every document it touches - a full development pass, every time. That made the honest
    workflow (revise, re-score, re-audit) the expensive one, and the rubric awards marks
    for exactly that iteration. Iterating here costs about 0.9M tokens instead of 5.7M.

    DEVELOPMENT MEETINGS ONLY, evenly spaced across the corpus so the sample spans easing,
    tightening and quiet periods rather than clustering in one regime. It is deterministic:
    every team gets the same documents from the same corpus, so before-and-after audit
    tables are comparable, and yours are comparable with your own from last week.

    The pilot NEVER writes the construct-score tables. It prints the audit and stops.
    """
    held = validation_meetings(docs)
    dev = (docs.loc[~docs["meeting_date"].isin(held), "meeting_date"]
           .sort_values().tolist())
    if len(dev) <= PILOT_N:
        return dev
    step = len(dev) / PILOT_N
    return [dev[int(i * step)] for i in range(PILOT_N)]


def _score_documents(docs: pd.DataFrame, label: str,
                     offline: bool = False) -> tuple[list[dict], dict]:
    t0, records = time.time(), []
    _RUN_ABORTED.clear()
    with ThreadPoolExecutor(max_workers=config.N_DOC_WORKERS) as ex:
        futs = [ex.submit(score_document, r.meeting_date, r.text_scored, offline)
                for r in docs.itertuples()]
        try:
            for i, f in enumerate(as_completed(futs), 1):
                records.append(f.result())
                if i % 50 == 0:
                    print(f"    {i}/{len(docs)} ({time.time()-t0:.0f}s)")
        except BaseException:
            # Stop scheduling on a run-wide failure. Every completed draw is already on
            # disk (score_document flushes as each one lands), so the next run resumes
            # instead of starting over.
            _RUN_ABORTED.set()
            for pending in futs:
                pending.cancel()
            raise
    # THE DRAWS THIS RUN PAID FOR, per draw rather than per document. Treating every
    # call in a partly-resumed document as fresh reported a one-call resume as five.
    fresh = [c for r in records for c in r.get("fresh", [])]
    usage = {"scope": label,
             "api_calls": len(fresh),
             "failed_calls": sum(1 for c in fresh if not c.get("ok")),
             "retried_calls": sum(1 for c in fresh if c.get("attempt", 1) > 1),
             "prompt_tokens": sum((c.get("usage") or {}).get("prompt_tokens", 0)
                                  for c in fresh),
             "completion_tokens": sum((c.get("usage") or {}).get("completion_tokens", 0)
                                      for c in fresh),
             # Reasoning tokens are billed as output but never appear in the answer.
             # gpt-5.4-mini reports 0 while no reasoning effort is requested; if this
             # ever goes non-zero, the model configuration changed underneath us.
             "reasoning_tokens": sum((c.get("usage") or {}).get("reasoning_tokens", 0)
                                     for c in fresh),
             "wall_seconds": round(time.time() - t0, 1),
             "documents_cached": sum(1 for r in records if r["cached"])}
    print(f"\n  {usage['api_calls']} fresh calls "
          f"({usage['failed_calls']} failed, {usage['retried_calls']} retried), "
          f"{usage['prompt_tokens']:,} in / {usage['completion_tokens']:,} out tokens, "
          f"{usage['wall_seconds']:.0f}s, {usage['documents_cached']} from cache")
    return records, usage


def _guard(dry_run: bool, docs: pd.DataFrame) -> bool:
    _require_exemplars()
    print(f"  {len(docs)} documents, median {docs.text_scored.str.len().median():,.0f} chars")
    print(f"  config hash {config_hash()}  ->  {run_dir()}")
    if dry_run:
        print("  --dry-run: stopping before any API call.")
        return False
    if not _prompts_written():
        missing = [c for c in STUDENT_CONSTRUCTS if "TODO" in CONSTRUCTS[c]]
        raise NotImplementedError(
            f"Write SYSTEM_PROMPT and your four rubrics first. Still TODO: "
            f"{missing or ['SYSTEM_PROMPT']}.")
    return True


def _write_scores(df: pd.DataFrame, which: str) -> int:
    """
    Write this run's partial and rebuild the combined table from whatever exists.

    Development and validation score DISJOINT meetings, so neither alone covers the corpus.
    The panel needs both: until validation has been run, its meetings simply have no
    construct values and `data_panel` says so rather than silently carrying NaNs into
    Replay and Shock.
    """
    target = (config.CONSTRUCT_SCORES_DEV if which == "development"
              else config.CONSTRUCT_SCORES_VAL)
    config.atomic_to_parquet(df, target)
    target.with_suffix("").with_suffix(".provenance.json").write_text(json.dumps({
        # the CALL hash: the identity of the scores themselves. The stage hash also
        # covers gates and validation settings, which do not change what was scored -
        # stamping it here would force a needless re-score on every gate adjustment.
        "config_hash": call_config_hash(), "scope": which, "n_rows": len(df),
        # the TABLE's identity, not just its configuration's: a sidecar that only named
        # the configuration once let an edited score file keep certifying itself
        "scores_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "written_at": datetime.now(timezone.utc).isoformat()}, indent=1),
        encoding="utf-8")
    # ONE CONFIGURATION PER COMBINED TABLE. Development and validation partials produced
    # under different prompts must not be concatenated: a team could otherwise keep old
    # development scores, revalidate under revised rubrics, and quote a combined table no
    # single configuration ever produced. A partial with no provenance sidecar predates
    # this check and is tolerated; two sidecars that disagree are not.
    hashes = {}
    for p in (config.CONSTRUCT_SCORES_DEV, config.CONSTRUCT_SCORES_VAL):
        prov = p.with_suffix("").with_suffix(".provenance.json")
        if p.exists() and prov.exists():
            hashes[p.name] = json.loads(prov.read_text(encoding="utf-8"))["config_hash"]
    if len(set(hashes.values())) > 1:
        raise RuntimeError(
            f"the development and validation score partials were produced under DIFFERENT "
            f"configurations ({hashes}); refusing to combine them. Re-run the stale side "
            f"under the current prompts (a repeat validation exposure needs --revalidate "
            f"and must be declared).")
    parts = [pd.read_parquet(p) for p in
             (config.CONSTRUCT_SCORES_DEV, config.CONSTRUCT_SCORES_VAL) if p.exists()]
    combined = (pd.concat(parts, ignore_index=True)
                  .drop_duplicates(subset="meeting_date", keep="last")
                  .sort_values("meeting_date").reset_index(drop=True))
    config.atomic_to_parquet(combined, config.CONSTRUCT_SCORES)
    # The COMBINED table's identity, tied to the partials it was merged from. The
    # submission suite requires every hash here to match the file on disk, so a score
    # edited consistently across all three tables still fails: the sidecars no longer
    # certify it, and the downstream stage hashes (panel provenance, Shock inputs) no
    # longer agree either.
    config.atomic_write_text(
        config.CONSTRUCT_SCORES.with_suffix("").with_suffix(".provenance.json"),
        json.dumps({
            "config_hash": call_config_hash(),
            "combined_sha256": hashlib.sha256(
                config.CONSTRUCT_SCORES.read_bytes()).hexdigest(),
            "partials": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (config.CONSTRUCT_SCORES_DEV,
                                   config.CONSTRUCT_SCORES_VAL) if p.exists()},
            "written_at": datetime.now(timezone.utc).isoformat()}, indent=1))
    return len(combined)


#: Where pilot audits are kept. One file per AUDIT, not per scoring configuration: a
#: rubric edit, a gate change, a ceiling change or a different sample each produce a new
#: file beside the old one, so the before-and-after pair survives the edit that made it
#: worth keeping.
PILOT_DIR = config.OUTPUTS / "words_pilot"


def audit_id(documents) -> str:
    """Identity of one pilot AUDIT: the stage configuration plus the sample audited.

    `call_config_hash()` is deliberately blind to gates and to the output ceiling, because
    it keys the response cache and neither of those changes a stored answer. That makes it
    the wrong name for an audit FILE: two audits that differ only in the gate they were
    judged against are two different pieces of evidence, and the first version wrote both
    to the same path.
    """
    return _sha(json.dumps({"stage": config_hash(),
                            "documents": list(documents)}, sort_keys=True), 8)


def run_pilot(dry_run: bool = False, offline: bool = False) -> pd.DataFrame | None:
    """
    ITERATION PASS on the fixed pilot sample, SAVED as evidence.

    Run this while you are still changing rubrics. It reports the same construct audit
    the full pass reports - spread, concentration, effective bins, separation and
    orientation - for about a seventh of the tokens.

    IT WRITES AN AUDIT FILE, one per AUDIT, under
    `outputs/words_pilot/<call-hash>-<audit-id>.json`. The call hash identifies the
    scoring configuration; the audit id additionally covers the gates, the output ceiling
    and the sample, so two audits of the same calls under different gates are two files. That is deliberate: the rubric awards the
    iteration marks for a before-and-after audit pair, and an earlier version printed
    those tables to the terminal and threw them away - so following the supplied command
    produced none of the evidence the marks are for. Each file records the documents
    scored, the call hash, every gate, the orientation table, the evidence rate and the
    usage, so two of them side by side ARE the before-and-after.

    It does NOT write the construct-score tables or `words_audit.json`: a 25-document
    sample is not a corpus measurement, and only the frozen development pass may produce
    the numbers the report quotes.
    """
    all_docs = load_documents()
    pilot = pilot_meetings(all_docs)
    docs = all_docs[all_docs["meeting_date"].isin(pilot)].reset_index(drop=True)
    print(f"  PILOT run - {len(docs)} fixed development documents "
          f"({docs.meeting_date.min()} to {docs.meeting_date.max()})")
    print("  The score tables and words_audit.json are NOT written - but the pilot audit")
    print("  IS saved, and API responses are cached like any other call.")
    if not _guard(dry_run, docs):
        return None
    records, usage = _score_documents(docs, "pilot", offline=offline)
    df, ev = aggregate(records, docs)
    print(f"{chr(10)}  evidence quotes verbatim: {ev['verbatim']}/{ev['checked']} "
          f"({ev['verbatim']/max(1,ev['checked']):.0%})")
    print(f"{chr(10)}  CONSTRUCT QUALITY AUDIT (pilot sample - iteration evidence, "
          f"not a corpus measurement)")
    quality = check_construct_quality(df)
    orient = orientation_check(df)

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    audit = audit_id(sorted(docs["meeting_date"]))
    out = PILOT_DIR / f"{call_config_hash()}-{audit}.json"
    # The ceilings the draws behind this audit were ACTUALLY produced under - which is
    # not necessarily the one now configured, since a document may be reusing draws made
    # earlier under a lower ceiling that still satisfy the compatibility rule.
    ceilings = sorted({c for r in records for c in
                       [recorded_ceiling(call) for call in r.get("calls", [])]
                       if c is not None})
    config.atomic_write_text(out, json.dumps({
        "scope": "pilot",
        # THE AUDIT'S OWN IDENTITY, distinct from the call configuration's. Naming the
        # file after `call_config_hash()` alone meant a run after a GATE change wrote to
        # the same path and overwrote its predecessor - so the before-and-after pair the
        # rubric's iteration marks are for could be destroyed by the very edit it was
        # supposed to document. The gates and the ceiling are in the stage hash; the
        # sample is in here too, because the same gates on a different sample is a
        # different audit.
        "audit_id": audit,
        "stage_config_hash": config_hash(),
        # kept as its own field: this is what decides CACHE REUSE, and it is deliberately
        # blind to gates and to the ceiling.
        "call_config_hash": call_config_hash(),
        "prompt_hash": _prompt_hash(),
        "n_documents": len(docs),
        "documents": sorted(docs["meeting_date"]),
        "model": config.MODEL, "temperature": config.SAMPLING_TEMPERATURE,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "observed_call_ceilings": ceilings,
        "n_calls_per_document": config.N_PARALLEL_CALLS,
        # THE THRESHOLDS THIS AUDIT WAS JUDGED AGAINST. A pass/fail column means nothing
        # a month later without the numbers behind it, and a saved audit that omits them
        # cannot be compared with one taken under different gates.
        "gates": {"min_spread": config.MIN_CONSTRUCT_SPREAD,
                  "max_binned_concentration": config.MAX_BINNED_CONCENTRATION,
                  "bin_width": config.BIN_WIDTH,
                  "min_effective_bins": getattr(config, "MIN_EFFECTIVE_BINS", None),
                  "min_signal_to_noise": config.MIN_SIGNAL_TO_NOISE,
                  "min_valid_calls": config.MIN_VALID_CALLS},
        "expected_orientation": EXPECTED_ORIENTATION,
        "constructs": sorted(CONSTRUCTS),
        "usage": usage,
        "evidence": ev,
        "quality": quality.round(4).to_dict("records"),
        "orientation": orient.round(4).to_dict("records") if len(orient) else [],
        "written_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=float))
    kept = sorted(p.name for p in PILOT_DIR.glob("*.json"))
    print(f"{chr(10)}  pilot audit saved: outputs/words_pilot/{out.name}")
    if len(kept) > 1:
        print(f"  {len(kept)} pilot audits on file. A BEFORE-AND-AFTER PAIR IS NOT ANY "
              f"TWO OF THEM: it is two audits of comparable pilot samples whose prompt "
              f"or gate change you can state. Each file records its documents, its "
              f"prompt hash and its gates - cite those when you name the pair.")
    print(f"  {usage['api_calls']} fresh calls, "
          f"{usage['prompt_tokens'] + usage['completion_tokens']:,} tokens "
          f"(cached to {run_dir().name}).")
    print("  When the audit stops changing your mind, freeze the rubrics and run the full")
    print("  development pass:  python src/text_features.py")
    return df


def run(dry_run: bool = False, offline: bool = False) -> pd.DataFrame | None:
    """
    DEVELOPMENT run. Scores every document EXCEPT the validation meetings.

    Iterate here as much as you like: the episodes you will be judged on are not in this
    sample and cannot be inspected from it.
    """
    all_docs = load_documents()
    held = validation_meetings(all_docs)
    docs = all_docs[~all_docs["meeting_date"].isin(held)].reset_index(drop=True)
    print(f"  DEVELOPMENT run - {len(held)} validation meetings withheld "
          f"({len(docs)} of {len(all_docs)} scored)")
    if not _guard(dry_run, docs):
        return None
    config.stage_begin("words", config_hash())
    config.ledger_reset("words")
    records, usage = _score_documents(docs, "development", offline=offline)
    df, ev = aggregate(records, docs)
    print(f"\n  evidence quotes verbatim: {ev['verbatim']}/{ev['checked']} "
          f"({ev['verbatim']/max(1,ev['checked']):.0%})")
    print("\n  CONSTRUCT QUALITY AUDIT (development documents only)")
    quality = check_construct_quality(df)
    orient = orientation_check(df)
    episode_check(df, "dev")

    n_total = _write_scores(df, "development")
    print(f"  construct_scores.parquet now covers {n_total} of 211 meetings")
    (config.OUTPUTS / "words_audit.json").write_text(json.dumps({
        "scope": "development",
        "config_hash": config_hash(),
        # Repository-relative: an absolute path here leaked the author's machine layout
        # into a committed artefact and is meaningless on anyone else's.
        "run_dir": str(run_dir().relative_to(config.ROOT)).replace("\\", "/"),
        "n_documents_scored": len(docs),
        "n_validation_withheld": len(held),
        # the development table this audit describes - the audit must not keep
        # certifying a score file that was edited after it was written
        "scores_sha256": hashlib.sha256(
            config.CONSTRUCT_SCORES_DEV.read_bytes()).hexdigest(),
        "usage": usage,
        "evidence": ev,
        "quality": quality.round(4).to_dict("records"),
        "orientation": orient.round(4).to_dict("records") if len(orient) else [],
    }, indent=2, default=float), encoding="utf-8")
    config.ledger_commit("words", name="words_development")
    config.stage_complete("words", config_hash())
    print(f"\n  when your rubrics are final: python src/text_features.py --validate")
    return df


def validate(dry_run: bool = False, revalidate: bool = False,
             offline: bool = False) -> pd.DataFrame | None:
    """
    VALIDATION run. Scores ONLY the held-out meetings, once, and records what produced it.

    The stamp stores the prompt text, the config hash and the exact meeting ids, so a marker
    can confirm that the rubrics which produced the validation numbers are the rubrics you
    submitted. Running it a second time requires `--revalidate` and is recorded as a second
    exposure - which is a defensible choice you must declare, not a silent one.
    """
    if VALIDATION_STAMP.exists() and not revalidate:
        rec = json.loads(VALIDATION_STAMP.read_text())
        same = rec["config_hash"] == config_hash()
        print(f"\n  VALIDATION ALREADY RUN on {rec['run_at'][:10]} under config "
              f"{rec['config_hash']}")
        print(f"  your prompts are {'UNCHANGED' if same else 'DIFFERENT'} since then")
        if not same:
            print("  re-running would be a SECOND exposure of the held-out episodes. If you "
                  "need it,")
            print("  pass --revalidate and say so in your report.")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(pd.DataFrame(rec["episodes"]).round(3).to_string(index=False))
        return pd.DataFrame(rec["episodes"])

    all_docs = load_documents()
    held = validation_meetings(all_docs)
    docs = all_docs[all_docs["meeting_date"].isin(held)].reset_index(drop=True)
    print(f"  VALIDATION run - {len(docs)} held-out meetings only")
    if revalidate and VALIDATION_STAMP.exists():
        print("  --revalidate: this is a REPEAT exposure and is recorded as one")
    if not _guard(dry_run, docs):
        return None
    config.stage_begin("words", config_hash())
    config.ledger_reset("words")
    records, _ = _score_documents(docs, "validation", offline=offline)
    df, _ = aggregate(records, docs)
    n_total = _write_scores(df, "validation")
    print(f"  construct_scores.parquet now covers {n_total} of 211 meetings")
    out = episode_check(df, "validation")

    prior = []
    if VALIDATION_STAMP.exists():
        old = json.loads(VALIDATION_STAMP.read_text())
        prior = old.get("previous_exposures", []) + [{
            "run_at": old["run_at"], "config_hash": old["config_hash"]}]
    VALIDATION_STAMP.write_text(json.dumps({
        "config_hash": config_hash(),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "exposure_number": len(prior) + 1,
        "previous_exposures": prior,
        "meetings": sorted(held),
        # the validation table this stamp describes, by content
        "scores_sha256": hashlib.sha256(
            config.CONSTRUCT_SCORES_VAL.read_bytes()).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
        "constructs": CONSTRUCTS,
        "episodes": out.round(4).to_dict("records"),
    }, indent=2, default=float), encoding="utf-8")
    config.ledger_commit("words", name="words_validation")
    config.stage_complete("words", config_hash())
    if prior:
        print(f"\n  RECORDED AS EXPOSURE {len(prior) + 1}. Your report must say why you "
              f"revalidated.")
    else:
        print(f"\n  frozen to {VALIDATION_STAMP.name} with the prompts that produced it")
    return out


if __name__ == "__main__":
    # --offline replays the committed envelopes and never contacts the service. Use it to
    # reproduce a run from someone else's evidence, or to carry on after a quota stop
    # without spending more budget.
    _offline = "--offline" in sys.argv
    if "--pilot" in sys.argv:
        run_pilot(dry_run="--dry-run" in sys.argv, offline=_offline)
    elif "--validate" in sys.argv:
        validate(dry_run="--dry-run" in sys.argv, revalidate="--revalidate" in sys.argv,
                 offline=_offline)
    else:
        run(dry_run="--dry-run" in sys.argv, offline=_offline)
