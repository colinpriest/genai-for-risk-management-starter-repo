"""courseapi - the OpenAI client shape this assignment was written against,
served by the UNSW course proxy.

The proxy exposes the Responses API only: no Chat Completions, no `messages`
field, and the GPT-5 family no longer accepts `seed`. Rather than rewrite every
call site, this adapter presents the small slice of the OpenAI client the
assignment actually uses - `client_().beta.chat.completions.parse(...)` and
`client_().chat.completions.create(...)` - and translates underneath.

What changes behind the adapter:

  messages=[...]        -> input=[...]              (same role/content dicts)
  response_format=Model -> instructor response_model, over the Responses API
  max_tokens            -> max_output_tokens        (config.MAX_OUTPUT_TOKENS)
  seed=N                -> REJECTED. Not sent, and not silently accepted.

WHY THERE IS NO `seed` PARAMETER HERE. An adapter that quietly accepted `seed`
and dropped it would let every call site - and every reproducibility envelope -
record a request parameter that was never transmitted. In an assignment whose
whole claim is that a marker can establish what was asked and what was answered,
a false request record is worse than a missing feature. So the parameter is
`call_index`, it is named for what it actually does, it never goes to the API,
and an unsupported parameter now raises instead of vanishing into `**_ignored`.

BOTH CALL PATHS ARE GUARDED. Free text used to go straight to the SDK's
`responses.create`, which skipped the wrapper's error translation, its 429
backoff and every completion-status check; a response that came back
`status="incomplete"` with `reason="content_filter"` and partial text was
recorded as an ordinary success. Structured and free-text calls now share one
guarded path: the same error translation, the same backoff, and the same
response-status validation before anything is returned.

Each response carries `.provenance`: what was transmitted (model, temperature,
output ceiling, a hash of the input and of the response schema) AND what came
back (the model the service actually reports, the response status, any
incomplete reason, the wrapper and SDK versions). It is deliberately NOT called
`.request`: it is a record of the call, not a replayable request body. The older
`.request` attribute is kept as the transmitted-parameters subset so existing
envelope writers keep working.

Token usage is normalised. The Responses API reports `input_tokens` /
`output_tokens`; Chat Completions reported `prompt_tokens` /
`completion_tokens`. `.usage` carries BOTH spellings plus `reasoning_tokens`, so
existing totals keep working and new envelopes are self-describing.

Credentials come from .env: STUDENT_AI_PROXY_URL, STUDENT_AI_ACCESS_CODE and
STUDENT_ID. There is no OpenAI key; the university proxy holds the credential.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from typing import Any

import config
import unsw_ai

__all__ = ["client_", "CourseClient", "ShimResponse", "normalise_usage",
           "NON_TRANSIENT_ERRORS", "IncompleteResponseError",
           "validate_call_kwargs", "response_status",
           "FAILURE_CATEGORIES", "SETTLED_CATEGORIES", "RUN_LEVEL_CATEGORIES",
           "failure_category", "describe_failure", "transport_budget_spent",
           "transport_attempts", "failed_attempt_usage", "ENVELOPE_VERSION",
           "effective_request", "RunConfigurationError", "call_usage",
           "ENVELOPE_CONTRACT", "envelope_contradictions", "envelope_generation"]


class RunConfigurationError(unsw_ai.UNSWAIError):
    """How this RUN is set up is wrong - not what this request asked for.

    The distinction decides whether the failure is cached. `ParameterNotSupportedError`
    was doing both jobs: it is the right answer for "this request carried a parameter the
    API will not take", which is that request's problem and settles, and it was also the
    answer for "this client has model fallback enabled", which is the run's problem and
    must not settle. Raised as the second, it wrote five settled failures per document,
    and fixing the configuration then produced zero fresh calls and zero valid draws.
    """


class IncompleteResponseError(unsw_ai.UNSWAIError):
    """The service returned 200 but did not finish the answer.

    Truncation at the output ceiling, a failed or cancelled response, or an
    explicit refusal. None of these are transient: the same request at the same
    ceiling truncates again, so retrying just spends the shared token budget.
    """

    def __init__(self, message: str, *, status: str | None = None,
                 reason: str | None = None):
        super().__init__(message)
        self.status = status
        self.reason = reason


#: Wrapper failures that retrying cannot fix, by exception NAME so the stage
#: modules can extend their own `_NON_TRANSIENT` tuples without importing
#: unsw_ai. A content filter, an oversized prompt, a rejected parameter, an
#: undeployed model and a truncated answer are all deterministic: backing off
#: and asking again just spends the shared token budget to get the same refusal.
NON_TRANSIENT_ERRORS = (
    "ContentFilteredError",
    "RequestTooLargeError",
    "ParameterNotSupportedError",
    "ModelNotAvailableError",
    "MissingCredentialsError",
    "InvalidAccessCodeError",
    "DailyQuotaExceededError",
    "UnsupportedModeError",
    "StructuredOutputError",
    "IncompleteResponseError",
)

# --------------------------------------------------------------------------- #
# ONE FAILURE VOCABULARY, SHARED BY EVERY STAGE
#
# Words, Replay and Shock each used to decide what a failure meant by testing
# whether an exception NAME appeared as a substring of a recorded error string.
# That is fragile in both directions: "IncompleteResponseError" covers a
# truncation, a refusal and a still-running response, which need three different
# answers; and a message that happens to quote an exception name classifies
# itself. The adapter is the only layer that sees the response object, so it is
# the layer that classifies, once, into a CATEGORY the stages can act on.
# --------------------------------------------------------------------------- #

#: What went wrong, as a fixed vocabulary. Recorded on every failed envelope.
FAILURE_CATEGORIES = (
    "truncated",         # hit the output ceiling; more room may finish it
    "content_filter",    # Foundry stopped the answer; ceiling-independent
    "refusal",           # the model declined; a completed, final answer
    "nonterminal",       # 200 but status not completed (in_progress, failed)
    "schema",            # completed, well-formed, wrong shape
    "request_rejected",  # the request itself is invalid (400/422/too large)
    "rate_limit",        # 429: rate, request-count or token quota
    "daily_quota",       # the daily allowance is gone; waiting is the only fix
    "credentials",       # missing/wrong access code or header
    "deployment",        # the model is not deployed on this proxy
    "run_configuration", # how THIS RUN is set up is wrong; fix it and resume
    "unreachable",       # the proxy itself did not answer
    "transport",         # timeout, dropped connection, 5xx
    "programming",       # a bug in this repository, not a service failure
    "unknown",
)

#: Exception NAME -> category, for failures that do not carry their own reason.
_CATEGORY_BY_NAME = {
    "ContentFilteredError": "content_filter",
    "StructuredOutputError": "schema",
    "ValidationError": "schema",
    "RequestTooLargeError": "request_rejected",
    "ParameterNotSupportedError": "request_rejected",
    "BadRequestError": "request_rejected",
    "UnprocessableEntityError": "request_rejected",
    "QuotaExceededError": "rate_limit",
    "RateLimitError": "rate_limit",
    "DailyQuotaExceededError": "daily_quota",
    "MissingCredentialsError": "credentials",
    "InvalidAccessCodeError": "credentials",
    "MissingHeaderError": "credentials",
    "AuthenticationError": "credentials",
    "PermissionDeniedError": "credentials",
    "ModelNotAvailableError": "deployment",
    "NotFoundError": "deployment",
    # HOW THE RUN IS CONFIGURED, not what this request asked for. Both are raised before
    # any HTTP request leaves, both are fixed by editing settings, and both must therefore
    # abort rather than settle - see RUN_LEVEL_CATEGORIES.
    "RunConfigurationError": "run_configuration",
    "UnsupportedModeError": "run_configuration",
    "UnsupportedEndpointError": "run_configuration",
    "ProxyUnreachableError": "unreachable",
    # TRANSIENT TRANSPORT, under BOTH spellings. The wrapper translates the SDK's
    # exceptions into its own before this adapter ever sees them, so a map that listed
    # only the SDK names classified every timeout and 5xx as `unknown`, marked its budget
    # spent, and stopped the stage retry that those failures exist to get. The
    # completeness test below keeps this table in step with the wrapper.
    "ProxyTimeoutError": "transport",
    "UpstreamServiceError": "transport",
    "APIConnectionError": "transport",
    "APITimeoutError": "transport",
    "InternalServerError": "transport",
    "ConnectionError": "transport",
    "TimeoutError": "transport",
    # A TypeError from the adapter is not a service failure and must never be
    # cached as this document's outcome: see RUN_LEVEL_CATEGORIES.
    "TypeError": "programming",
    "KeyError": "programming",
    "AttributeError": "programming",
    "IndexError": "programming",
    "NameError": "programming",
    "ImportError": "programming",
    "ModuleNotFoundError": "programming",
    "ZeroDivisionError": "programming",
}

#: THIS DOCUMENT's outcome, and deterministic at these settings. Re-asking spends
#: the shared budget to re-learn what the envelope already records. `truncated` is
#: settled only while the ceiling is unchanged - see the compatibility rule in
#: text_features.reusable_under_current_ceiling.
SETTLED_CATEGORIES = (
    "truncated", "content_filter", "refusal", "schema", "request_rejected",
)

#: Properties of the RUN, not of any one call. Every remaining call fails the
#: same way and the fix is outside the repository - edit .env, deploy the model,
#: wait for the allowance, or repair the code. These must never be written as a
#: settled document failure: doing so poisons the cache, so that correcting the
#: fault produces zero fresh calls and zero valid draws.
RUN_LEVEL_CATEGORIES = (
    "credentials", "deployment", "daily_quota", "unreachable", "programming",
    # A forbidden model fallback, an unserviceable instructor mode and an endpoint the
    # proxy does not expose are all properties of the RUN. Classifying them as
    # `request_rejected` made them settled document outcomes: five draws were written as
    # settled failures, and correcting the client configuration then produced zero fresh
    # calls and zero valid draws, exactly as a mistyped access code once did.
    "run_configuration",
)

#: The only category a stage is right to re-ask itself: nothing below the stage
#: retried it, and a second attempt is one more HTTP request rather than five.
STAGE_RETRYABLE_CATEGORIES = ("transport",)

_TRANSPORT_STAMP = "_courseapi_transport"


def failure_category(exc: BaseException) -> str:
    """Classify a failure into FAILURE_CATEGORIES.

    `IncompleteResponseError` carries its own `reason`, because one exception
    class covers three genuinely different outcomes: an answer cut off at the
    ceiling (raise the ceiling), a refusal (a final answer, and re-asking is
    pointless), and a response that had not finished when the body arrived.
    """
    if isinstance(exc, IncompleteResponseError):
        reason = str(getattr(exc, "reason", "") or "")
        if "max_output_tokens" in reason:
            return "truncated"
        if "content_filter" in reason:
            return "content_filter"
        if reason == "refusal":
            return "refusal"
        return "nonterminal"
    for klass in type(exc).__mro__:
        if klass.__name__ in _CATEGORY_BY_NAME:
            return _CATEGORY_BY_NAME[klass.__name__]
    return "unknown"


def stamp_transport(exc: BaseException, *, attempts: int, budget_spent: bool,
                    usage: dict | None = None) -> BaseException:
    """Record on the exception how much transport this failure already cost.

    THE POINT OF THIS STAMP is that a stage must not open a second retry budget
    around one the adapter has already spent. One persistent 429 used to cost
    5 (wrapper) x 4 (stage) = 20 HTTP requests against a 60/minute class budget
    for a single logical draw.
    """
    setattr(exc, _TRANSPORT_STAMP, {
        "attempts": int(attempts), "budget_spent": bool(budget_spent),
        "usage": usage})
    return exc


def transport_attempts(exc: BaseException) -> int | None:
    """HTTP requests this failure actually cost, when the adapter counted them."""
    stamp = getattr(exc, _TRANSPORT_STAMP, None)
    return None if stamp is None else stamp.get("attempts")


def stamp_unsent(exc: BaseException) -> BaseException:
    """Stamp a failure raised BEFORE any request left: zero requests, no usage.

    An unstamped exception used to reach the stages, which counted a failure with no
    stamp as ONE request - so a missing access code or a forbidden model fallback was
    recorded as `transport_requests: 1` for a draw that never contacted the service.
    Zero is a real answer here, not an unknown one: nothing was sent. An existing stamp
    is left alone, because a stamp is only ever written by code that counted.
    """
    if getattr(exc, _TRANSPORT_STAMP, None) is None:
        stamp_transport(
            exc, attempts=0,
            budget_spent=failure_category(exc) not in STAGE_RETRYABLE_CATEGORIES,
            usage=None)
    return exc


def failed_attempt_usage(exc: BaseException) -> dict | None:
    """Tokens a FAILED attempt still spent, when the service reported them.

    A truncated answer is billed. Dropping its usage understates the run against
    a shared allowance, which is the opposite of the error to make.
    """
    stamp = getattr(exc, _TRANSPORT_STAMP, None)
    return None if stamp is None else stamp.get("usage")


def transport_budget_spent(exc: BaseException) -> bool:
    """Has the transport budget for this logical call already been spent?

    True means: do not loop again in stage code. Either a lower layer already
    retried this failure as many times as it is worth retrying, or the failure is
    deterministic and a second identical request buys nothing.
    """
    stamp = getattr(exc, _TRANSPORT_STAMP, None)
    if stamp is not None:
        return bool(stamp.get("budget_spent"))
    return failure_category(exc) not in STAGE_RETRYABLE_CATEGORIES


def describe_failure(exc: BaseException, *, request: dict | None = None) -> dict:
    """The classified failure record written onto a FAILED envelope.

    A failed draw used to record only `ok`, `error`, `call_index`, `config_hash`
    and `timestamp`. That is not enough to decide anything later: it cannot say
    what ceiling the attempt ran under (so a raised ceiling could not retry the
    truncations it was raised for), and it forced every consumer back to
    substring-matching the exception name.
    """
    category = failure_category(exc)
    return {
        "category": category,
        "error_type": type(exc).__name__,
        "message": str(exc).splitlines()[0][:200] if str(exc) else "",
        "status": getattr(exc, "status", None),
        "reason": getattr(exc, "reason", None),
        "run_level": category in RUN_LEVEL_CATEGORIES,
        "settled": category in SETTLED_CATEGORIES,
        "transport_attempts": transport_attempts(exc),
        "request": dict(request) if request else None,
        "envelope_version": ENVELOPE_VERSION,
    }


#: Version of the envelope CONTRACT - the set of fields a record written by this
#: code is guaranteed to carry. Evidence committed before the move to the course
#: proxy has no version and genuinely lacks fields that did not exist then; the
#: loaders must be able to tell "legacy, fields unavailable" from "current, field
#: missing", and inventing values for the first case is how false provenance gets
#: into a report. See text_features._envelope_generation.
ENVELOPE_VERSION = 2

#: Versions this code knows how to read. An envelope declaring anything else is a record
#: from a future contract, and guessing at it is worse than refusing it.
SUPPORTED_ENVELOPE_VERSIONS = (2,)

# --------------------------------------------------------------------------- #
# THE ENVELOPE CONTRACT - ONE DEFINITION, ALL THREE STAGES
#
# Words, Replay and Shock each grew their own idea of what an envelope contains
# and what a loader checks, and the differences were not design: Replay and Shock
# wrote no version at all, so evidence written five minutes ago was classified as
# "legacy", and their loaders never compared the served model or the response
# status, so a cache edited to name a different deployment and an `incomplete`
# status was reused by both. The contract below is what every writer promises and
# every loader checks.
#
# THE THREE CASES ARE DIFFERENT, and pretending otherwise is how the previous
# version made promises it could not keep:
#
#   success           a completed, parsed answer: full response-side provenance
#   response_failure  a response arrived and was rejected (truncated, filtered,
#                     refused, nonterminal, schema): status is known, no payload
#   pretransport      NO USABLE RESPONSE. The name is historical, and it covers two
#                     different situations that the request count tells apart:
#                       - NOTHING WAS SENT (bad credentials, a forbidden fallback):
#                         `failure.transport_attempts` is 0, `transport_requests`
#                         is null and `usage` is null;
#                       - REQUESTS WERE SENT and none was answered usably (a
#                         timeout, a 429, a 5xx): `transport_requests` is the real
#                         count, and `usage` says `tokens_known: false` with every
#                         attempt in `attempts_unknown`.
#                     Either way there is no response id and no served model, and
#                     inventing one would be a fabricated audit trail
#
# Free-text calls cut across all three: they have no response schema, so
# `response_schema_sha256_16` is absent from their provenance and is not required
# of any case here.
# --------------------------------------------------------------------------- #

ENVELOPE_CONTRACT = {
    "success": {
        "required": ("envelope_version", "ok", "call_index", "config_hash", "request",
                     "model", "temperature", "provenance", "model_served", "usage",
                     "transport_requests", "timestamp"),
        # WHAT THOSE BLOCKS MUST ACTUALLY CONTAIN. A `provenance: {}` satisfies a
        # presence check and records nothing; a `request` holding only the output ceiling
        # cannot say which model or temperature produced the answer.
        "nested": {
            "provenance": ("response_status", "model_requested", "model_served",
                           "input_sha256_16", "transport_requests"),
            "request": ("model", "max_output_tokens"),
        },
        "forbidden": ("failure",),
    },
    "response_failure": {
        "required": ("envelope_version", "ok", "call_index", "config_hash", "request",
                     "model", "temperature", "failure", "transport_requests",
                     "timestamp"),
        "forbidden": (),
    },
    "pretransport": {
        "required": ("envelope_version", "ok", "call_index", "config_hash", "request",
                     "model", "temperature", "failure", "timestamp"),
        "forbidden": (),
    },
}

#: Failure categories that mean NO RESPONSE EVER ARRIVED, so response-side fields
#: (served model, response status, response id, usage) are genuinely unavailable.
PRETRANSPORT_CATEGORIES = (
    "credentials", "deployment", "run_configuration", "unreachable", "transport",
    "rate_limit", "daily_quota", "request_rejected", "programming", "unknown",
)


def failure_envelope(exc: BaseException, *, request: dict, config_hash: str,
                     call_index: int | None = None, transport: int = 0,
                     usage: dict | None = None, **extra) -> dict:
    """THE failure envelope. One builder, so no stage can emit an invalid one.

    Replay and Shock each hand-assembled their own and each omitted different required
    fields - Replay had no call index, config hash, model, temperature or transport
    count; Shock had no config hash, temperature or transport count - while both declared
    `envelope_version: 2`. A version marker on a record that does not keep the contract is
    worse than no marker, because a loader believes it.
    """
    #: `transport` and `usage` are the COMPLETE totals for this logical call, current
    #: attempt included. The builder adds nothing: it used to add the exception's own
    #: attempts on top of a caller that had already accumulated them, which double-counted
    #: every stage retry - a persistent 429 read as ten requests where five were made.
    failure = describe_failure(exc, request=request)
    envelope = {
        "ok": False,
        "error": f"{failure['error_type']}: {failure['message']}",
        "failure": failure,
        "config_hash": config_hash,
        "request": dict(request),
        "model": request.get("model", config.MODEL),
        "temperature": request.get("temperature", config.SAMPLING_TEMPERATURE),
        "usage": usage,
        "transport_requests": transport or None,
        "envelope_version": ENVELOPE_VERSION,
        "timestamp": _now(),
    }
    if call_index is not None:
        envelope["call_index"] = call_index
    envelope.update(extra)
    return envelope


def success_envelope(response: Any, *, request: dict, config_hash: str,
                     call_index: int | None = None, transport: int = 0,
                     usage: dict | None = None, **extra) -> dict:
    """THE success envelope, for the fields every stage promises.

    `transport` and `usage` are what the STAGE has already spent on earlier attempts of
    this same logical call; the response carries what the adapter spent on the last one.
    A stage retry loop that reported only the final attempt undercounted both.
    """
    provenance = getattr(response, "provenance", None) or {}
    envelope = {
        "ok": True,
        "config_hash": config_hash,
        "request": getattr(response, "request", None) or dict(request),
        "model": request.get("model", config.MODEL),
        "temperature": request.get("temperature", config.SAMPLING_TEMPERATURE),
        "model_served": getattr(response, "model", None),
        "provenance": provenance,
        "request_id": getattr(response, "id", None),
        "usage": merge_usage(usage, getattr(response, "call_usage", None)
                             or normalise_usage(getattr(response, "usage", None))),
        "usage_final_response": normalise_usage(getattr(response, "usage", None)),
        "transport_requests": transport + (provenance.get("transport_requests") or 0),
        "envelope_version": ENVELOPE_VERSION,
        "timestamp": _now(),
    }
    if call_index is not None:
        envelope["call_index"] = call_index
    envelope.update(extra)
    return envelope


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def envelope_case(call: dict) -> str:
    """Which contract case this record is: success, response_failure or pretransport."""
    if call.get("ok"):
        return "success"
    category = (call.get("failure") or {}).get("category")
    return "pretransport" if category in PRETRANSPORT_CATEGORIES else "response_failure"


def envelope_generation(call: dict) -> str:
    """"current" if written under the versioned contract, else "legacy".

    "legacy" means ONLY "written before this contract existed" - which includes records
    from the first proxy implementation, not just pre-proxy ones. It is not a statement
    about which service produced them, and nothing back-fills their missing fields.
    """
    v = call.get("envelope_version")
    return "current" if isinstance(v, int) and not isinstance(v, bool) else "legacy"


def envelope_contradictions(call: dict, *, config_hash: str | None = None,
                            prompt_hash: str | None = None,
                            prompt_field: str = "prompt_hash") -> list[str]:
    """Fields inside one envelope that disagree with each other or with this run.

    A MATCHING CONFIG HASH IS NOT PROVENANCE. It says the run was configured the same
    way; it says nothing about what the envelope itself claims. ONLY FIELDS THAT ARE
    PRESENT are checked - absence in a legacy record means the field did not exist when
    it was written, and inventing a value now is exactly the false provenance this
    validation exists to prevent. Absence in a CURRENT record is a defect, and the
    required-field check below catches it.
    """
    bad: list[str] = []

    def check(field, actual, expected):
        if actual is not None and expected is not None and actual != expected:
            bad.append(f"{field}={actual!r}, but this run is {expected!r}")

    version = call.get("envelope_version")
    if version is not None:
        if isinstance(version, bool) or not isinstance(version, int):
            bad.append(f"envelope_version={version!r} is not an integer")
        elif version not in SUPPORTED_ENVELOPE_VERSIONS:
            bad.append(f"envelope_version={version} is not one of "
                       f"{list(SUPPORTED_ENVELOPE_VERSIONS)}; this evidence was written "
                       f"by a newer contract than this code can read")

    check("model", call.get("model"), config.MODEL)
    check("temperature", call.get("temperature"), config.SAMPLING_TEMPERATURE)
    check(prompt_field, call.get(prompt_field), prompt_hash)
    req = call.get("request") or {}
    check("request.model", req.get("model"), config.MODEL)
    check("request.temperature", req.get("temperature"), config.SAMPLING_TEMPERATURE)
    if config_hash is not None:
        check("config_hash", call.get("config_hash"), config_hash)

    prov = call.get("provenance") or {}
    # NO SILENT MODEL SUBSTITUTION, on the way back in as well as on the way out. An
    # envelope whose served model differs from the one it requested records an answer
    # some other deployment produced.
    served = call.get("model_served") or prov.get("model_served")
    requested = prov.get("model_requested") or call.get("model") or config.MODEL
    if served is not None and served != requested:
        bad.append(f"model_served={served!r} but model_requested={requested!r}: "
                   f"this answer came from a different deployment")
    status = prov.get("response_status")
    if call.get("ok") and status is not None and status != "completed":
        bad.append(f"marked ok with response_status={status!r}: an unfinished response "
                   f"is a fragment, not a result")

    if envelope_generation(call) == "current" and not bad:
        case = envelope_case(call)
        spec = ENVELOPE_CONTRACT[case]
        # PRESENT-BUT-EMPTY IS NOT PRESENT. `provenance: {}` and `usage: {}` passed a
        # `is None` test and so satisfied the contract while carrying nothing, which is
        # how a version-2 success with no response-side provenance at all was accepted by
        # all three loaders.
        # TEMPERATURE IS REQUIRED ONLY WHILE THE RUN SAMPLES. A run that sets no sampling
        # temperature sends none and records `temperature: null`, and every such record was
        # refused as "missing" on its first reload - so that configuration could never
        # replay its own cache. Under a sampling run it stays required, and a record that
        # carries a temperature while the run sets none is refused by the checks below.
        required = [f for f in spec["required"]
                    if not (f == "temperature" and config.SAMPLING_TEMPERATURE is None)]
        missing = [f for f in required
                   if call.get(f) is None
                   or (isinstance(call.get(f), (dict, list, str))
                       and len(call.get(f)) == 0)]
        if missing:
            bad.append(f"declares envelope_version {version} as a {case} record but is "
                       f"missing or empty: {missing}")
        present = [f for f in spec["forbidden"] if call.get(f) is not None]
        if present:
            bad.append(f"a {case} record must not carry {present}")
        # NESTED REQUIREMENTS TOO. The top-level field can be a dict that omits exactly
        # the thing it exists to record: a `provenance` without a response status, or a
        # `request` carrying only the output ceiling, told a reader nothing while passing
        # a presence check.
        for parent, children in spec.get("nested", {}).items():
            block = call.get(parent)
            if not isinstance(block, dict):
                continue
            gaps = [c for c in children
                    if block.get(c) is None
                    or (isinstance(block.get(c), (dict, list, str))
                        and len(block.get(c)) == 0)]
            if gaps:
                bad.append(f"{parent} is missing or empty: {gaps}")

        # EVERY REPRESENTATION OF THE MODEL MUST AGREE WITH THE CONFIGURED ONE, not just
        # with whichever other copy `or` happened to pick first. Changing the top-level
        # served model AND the nested requested model to the same wrong value used to
        # pass, because the two were compared to each other and one of them supplied the
        # expectation.
        prov_block = call.get("provenance") or {}
        for field, value in (("model", call.get("model")),
                             ("model_served", call.get("model_served")),
                             ("provenance.model_served", prov_block.get("model_served")),
                             ("provenance.model_requested",
                              prov_block.get("model_requested")),
                             ("request.model", (call.get("request") or {}).get("model"))):
            if value is not None and value != config.MODEL:
                bad.append(f"{field}={value!r}, but this run is configured for "
                           f"{config.MODEL!r}")
        top_temperature = call.get("temperature")
        if (top_temperature is not None
                and top_temperature != config.SAMPLING_TEMPERATURE):
            bad.append(f"temperature={top_temperature!r}, but this run is configured for "
                       f"{config.SAMPLING_TEMPERATURE!r}")

        # EVERY RECORDED COPY OF THE REQUEST, not only the top-level one. The complete
        # request is also written to `provenance.transmitted` on a success and to
        # `failure.request` on a failure. A temperature of 99 in `transmitted` sat beside
        # two correct copies and was never read, so a cache whose own record of what was
        # sent contradicted this run was reused as though it agreed.
        failure_block = call.get("failure") or {}
        request_copies = [("request", call.get("request"))]
        if "transmitted" in prov_block:
            request_copies.append(("provenance.transmitted", prov_block.get("transmitted")))
        if failure_block.get("request") is not None:
            request_copies.append(("failure.request", failure_block.get("request")))
        ceilings = {}
        for label, copy in request_copies:
            if copy is None:
                continue          # a missing top-level request is the contract check's job
            if not isinstance(copy, dict):
                bad.append(f"{label} is a {type(copy).__name__}, not a request record")
                continue
            if label != "request":            # request.model is checked above
                sent_model = copy.get("model")
                if sent_model is not None and sent_model != config.MODEL:
                    bad.append(f"{label}.model={sent_model!r}, but this run is "
                               f"configured for {config.MODEL!r}")
            # TEMPERATURE IS REQUIRED WHEN THE RUN SAMPLES. Every stage passes
            # `config.SAMPLING_TEMPERATURE` explicitly, so a current request that omits it
            # cannot say what produced the answer - and absence used to satisfy a check
            # that only compared values that were present.
            sent_temperature = copy.get("temperature")
            if config.SAMPLING_TEMPERATURE is not None:
                if sent_temperature is None:
                    bad.append(f"{label} records no temperature, but this run samples at "
                               f"{config.SAMPLING_TEMPERATURE!r}: a request without its "
                               f"temperature cannot say what produced the answer")
                elif sent_temperature != config.SAMPLING_TEMPERATURE:
                    bad.append(f"{label}.temperature={sent_temperature!r}, but this run "
                               f"is configured for {config.SAMPLING_TEMPERATURE!r}")
            elif sent_temperature is not None:
                bad.append(f"{label}.temperature={sent_temperature!r}, but this run sets "
                           f"no sampling temperature, so none was sent")
            # The CEILING is not compared with the one now in force - a different ceiling
            # is a re-ask decided by `reusable_under_current_ceiling`, not a contradiction.
            # It must be a real ceiling, and every copy must agree on which one it was.
            ceiling = copy.get("max_output_tokens")
            if ceiling is not None:
                if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= 0:
                    bad.append(f"{label}.max_output_tokens={ceiling!r} is not a positive "
                               f"integer")
                else:
                    ceilings[label] = ceiling
        if len(set(ceilings.values())) > 1:
            bad.append(f"the recorded output ceilings disagree with each other: {ceilings}")

        # COUNTS ARE COUNTS. A negative number of requests, attempts or tokens is not a
        # record of anything that happened, and a bool is not a count. Consistency is
        # checked only WITHIN one kind of figure: whole-call totals and the final adapter
        # attempt's own metadata legitimately differ after a stage retry.
        def _count(label, value, *, minimum=0):
            if value is None:
                return
            if isinstance(value, bool) or not isinstance(value, int):
                bad.append(f"{label}={value!r} is not an integer count")
            elif value < minimum:
                bad.append(f"{label}={value} is below {minimum}: "
                           + ("a completed answer took at least one request"
                              if minimum else "a count cannot be negative"))

        succeeded = bool(call.get("ok"))
        _count("transport_requests", call.get("transport_requests"),
               minimum=1 if succeeded else 0)
        _count("provenance.transport_requests", prov_block.get("transport_requests"),
               minimum=1 if succeeded else 0)
        _count("provenance.adapter_attempts", prov_block.get("adapter_attempts"),
               minimum=1 if succeeded else 0)
        _count("failure.transport_attempts", failure_block.get("transport_attempts"))
        for usage_label in ("usage", "usage_final_response"):
            usage = call.get(usage_label)
            if usage is None:
                continue
            if not isinstance(usage, dict):
                bad.append(f"{usage_label} is a {type(usage).__name__}, not a usage record")
                continue
            for key in _ADDITIVE_USAGE:
                _count(f"{usage_label}.{key}", usage.get(key))
            _count(f"{usage_label}.attempts_counted", usage.get("attempts_counted"))
            _count(f"{usage_label}.attempts_unknown", usage.get("attempts_unknown"))
            for flag in ("tokens_known", "is_complete"):
                if flag in usage and not isinstance(usage[flag], bool):
                    bad.append(f"{usage_label}.{flag}={usage[flag]!r} is not true or false")
            unknown = usage.get("attempts_unknown")
            if (usage.get("is_complete") is True and isinstance(unknown, int)
                    and not isinstance(unknown, bool) and unknown > 0):
                bad.append(f"{usage_label} is marked complete while {unknown} attempt(s) "
                           f"have an unknown cost")
            # The writers leave token keys OUT when nothing was reported. A record that
            # says the cost is unknown and then states one is a zero bill in disguise.
            if usage.get("tokens_known") is False:
                stated = [k for k in _ADDITIVE_USAGE if k in usage]
                if stated:
                    bad.append(f"{usage_label} says no token usage was reported but "
                               f"records {stated}: an unreported cost is absent, not a "
                               f"number")

    # AN EXPLICIT `null` IS NOT AN ABSENT MARKER. A record written before the contract
    # existed has no `envelope_version` key at all; one that carries the key with no value
    # is not legacy evidence, it is an invalid marker - and reading it as legacy silently
    # switched off every current-contract check above, which is how a record with its
    # version nulled and its provenance, served model, transport count and usage removed
    # was accepted as a clean cache hit.
    if "envelope_version" in call and call.get("envelope_version") is None:
        bad.append("envelope_version is present but null: a record from before the "
                   "contract has no marker at all, so this is an invalid marker rather "
                   "than a legacy one, and the weaker legacy checks do not apply")
    return bad


#: Everything this adapter is willing to be handed. Anything else raises rather
#: than being dropped: `seed` and `reasoning` were both silently ignored, and a
#: silently ignored parameter is a false entry in the audit trail.
_ACCEPTED_KWARGS = frozenset({
    "model", "messages", "temperature", "max_tokens", "response_format",
    "call_index",
})

#: Parameters the Responses API (or the GPT-5 family on it) does not honour.
#: Named individually so the error can say why rather than "unexpected keyword".
_REJECTED_KWARGS = {
    "seed": "the Responses API does not accept `seed`; use `call_index`, which "
            "names the draw without pretending to control sampling",
    "reasoning": "reasoning effort is not part of the assessed configuration - "
                 "setting it would change results without appearing in the "
                 "stage hash",
    "reasoning_effort": "see `reasoning`",
    "n": "not accepted by the Responses API; make N separate calls so each one "
         "has its own envelope",
    "stop": "not accepted by the Responses API",
    "logit_bias": "not accepted by the Responses API",
    "logprobs": "not available through the proxy",
    "stream": "the envelope writers need a complete response",
}


def validate_call_kwargs(kwargs: dict) -> None:
    """Reject anything that would not actually be transmitted."""
    for name in kwargs:
        if name in _ACCEPTED_KWARGS:
            continue
        why = _REJECTED_KWARGS.get(name)
        # Refused before anything is sent, so stamped as zero requests: a rejected
        # parameter must not be recorded as a call that reached the service.
        raise stamp_unsent(unsw_ai.ParameterNotSupportedError(
            f"courseapi does not accept {name!r}: "
            + (why or "it is not transmitted by this adapter, and an adapter "
                      "that accepted it would put a parameter in the "
                      "reproducibility record that never reached the service")
            + ". Remove it from the call."))


def normalise_usage(raw: Any) -> dict | None:
    """Token usage under BOTH the Responses and Chat Completions spellings.

    Returns None when the response carried no usage at all, so a caller can tell
    "nothing reported" apart from "reported zero"."""
    if raw is None:
        return None
    u = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)
    inp = u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
    out = u.get("output_tokens", u.get("completion_tokens", 0)) or 0
    details = u.get("output_tokens_details") or {}
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": u.get("total_tokens", inp + out) or (inp + out),
        # the Chat Completions spelling, kept so older readers still total up
        "prompt_tokens": inp,
        "completion_tokens": out,
        "reasoning_tokens": details.get("reasoning_tokens", 0) or 0,
        "cached_input_tokens": cached,
    }


def response_status(raw: Any) -> tuple[str | None, str | None]:
    """(status, incomplete reason) from a Responses object, both optional."""
    if raw is None:
        return None, None
    status = getattr(raw, "status", None)
    details = getattr(raw, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details is not None else None
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return (str(status) if status is not None else None,
            str(reason) if reason is not None else None)


def _refusal_text(raw: Any) -> str:
    out = ""
    for item in getattr(raw, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    out += getattr(part, "refusal", "") or ""
    return out.strip()


def validate_response(raw: Any, settings=None) -> None:
    """Refuse a response that did not finish, BEFORE it becomes an envelope.

    A Responses result carries its own verdict in `status`. A 200 with
    `status="incomplete"` is not a success: the text present is a fragment, and
    the two reasons that produce one here are opposite problems - the content
    filter stopped the model, or the answer hit the output ceiling. Both used to
    be recorded as ordinary completed calls with `finish_reason="stop"`.

    ONLY `completed` IS A RESULT. `in_progress` used to be accepted here, which
    was wrong twice over: this adapter makes synchronous requests and never polls
    for a background response, so an `in_progress` body carries whatever text had
    been generated when the server replied - a fragment, handed on with
    `finish_reason="stop"`. `None` is still accepted because a stub or an older
    response object may carry no status field at all.
    """
    status, reason = response_status(raw)
    refusal = _refusal_text(raw)
    if refusal:
        raise IncompleteResponseError(
            f"the model refused this request: {refusal[:200]}",
            status=status, reason="refusal")
    if status in (None, "completed"):
        return
    if status == "in_progress":
        raise IncompleteResponseError(
            "the response is still in progress: this adapter is synchronous and "
            "does not poll for background responses, so the text in this body is "
            "a fragment rather than an answer",
            status=status, reason=reason)
    if status == "incomplete" and reason and "content_filter" in reason:
        raise unsw_ai.ContentFilteredError(
            "Foundry's content filter stopped the model mid-answer: the "
            "response came back marked incomplete with reason 'content_filter', "
            "so what text exists is a fragment and must not be scored. This is "
            "a deployment policy, not a model limitation.",
            categories=("completion_filter",), source="completion")
    if status == "incomplete" and reason and "max_output_tokens" in reason:
        raise IncompleteResponseError(
            f"the answer hit the output ceiling of "
            f"{config.MAX_OUTPUT_TOKENS} tokens and stopped mid-sentence. The "
            f"partial text is not a result. Raise config.MAX_OUTPUT_TOKENS or "
            f"ask for a shorter answer - retrying unchanged truncates again.",
            status=status, reason=reason)
    raise IncompleteResponseError(
        f"the response did not complete (status={status!r}, reason={reason!r}), "
        f"so anything it contains is a fragment",
        status=status, reason=reason)


def effective_request(*, model=None, temperature=None, max_tokens=None) -> dict:
    """EXACTLY the parameters that go to the API - no implicit defaults left
    unstated. The output ceiling is in here because it CHANGES THE ANSWER: the
    same prompt under a lower ceiling truncates.

    Module-level and callable WITHOUT a response, because a failed draw needs it
    too. A failure envelope that does not record the ceiling the attempt ran
    under cannot afterwards be compared against the ceiling now in force, which
    is precisely the comparison that decides whether raising it should re-ask.
    """
    # Temperature is included ONLY when the caller gave one. Defaulting it here would put
    # a parameter into the transmitted record that the caller chose not to send, which is
    # the same false-audit-trail problem `seed` was removed for. Every stage passes
    # config.SAMPLING_TEMPERATURE explicitly.
    req: dict[str, Any] = {"model": model or config.MODEL}
    if temperature is not None:
        req["temperature"] = temperature
    req["max_output_tokens"] = max(
        int(max_tokens or config.MAX_OUTPUT_TOKENS), 16)
    return req


#: How many times instructor may re-ask when the model returns a well-formed but
#: INVALID object. Two means one repair attempt.
REPAIR_ATTEMPTS = 2

#: Exception names that mean "the model answered, and the answer does not fit the
#: schema". These - and only these - are worth re-asking: the model can produce a
#: different object next time. Every other failure either cannot change (a
#: refusal, a rejected request) or is somebody else's budget to spend (a 429 is
#: the wrapper's, a dropped connection is the stage's).
_REPAIRABLE_ERRORS = (
    "ValidationError", "InstructorRetryException", "IncompleteOutputException",
    "JSONDecodeError", "StructuredOutputError",
)

#: THE PER-CALL ACCOUNT for the logical call in flight on this thread: how many real HTTP
#: requests it has cost, and every usage figure the service actually returned along the
#: way. Reset by `_guarded`, so it spans the wrapper's 429 retries AND instructor's
#: repairs - everything one call to `.parse()` costs.
#:
#: USAGE IS ACCUMULATED, NOT OVERWRITTEN. Keeping only the last response's usage reported
#: 20 tokens for a call that made two billed attempts of 20 - understating the run against
#: a shared allowance, which is the wrong direction to be wrong in. Attempts whose usage
#: the service never reported (a timeout, a dropped connection) are counted separately and
#: labelled, rather than being silently treated as zero.
_http = threading.local()


def _account_reset() -> None:
    _http.count = 0
    _http.usage = []
    _http.unknown = 0
    _http.seen = []
    # NO ATTEMPT IS OPEN until a request is dispatched. Opening one here booked a call the
    # client refused itself - over the per-request cap, past the daily budget - as an
    # attempt of unknown cost, although nothing was ever sent.
    _http.accounted = True


def _account_attempt_start() -> None:
    """Begin an attempt whose cost is not yet known."""
    _http.accounted = False


def _account_attempt_unknown() -> None:
    """Close an attempt that produced no usable response, ONCE.

    Both the repair policy and the guard's error handler look for a response, and both
    used to record 'cost unknown' when they did not find one - so a single free-text
    truncation, already accounted from its own response, was also counted as an attempt
    of unknown cost and reported `attempts_unknown: 1` beside the figure it knew.
    """
    if not getattr(_http, "accounted", False):
        _account_usage(None)
        _http.accounted = True


def _http_count() -> int:
    return int(getattr(_http, "count", 0) or 0)


def _http_tick(_retry_state=None) -> None:
    """One HTTP request has been DISPATCHED: count it, and open its attempt.

    Called by the wrapper's transport, through `unsw_ai.dispatch_sink`, after its local
    token and rate guards have admitted the request and immediately before the network.
    It used to be called at instructor's attempt hook and just before the raw SDK call -
    both BEFORE those guards - so a prompt the client refused itself was recorded as one
    real request of unknown cost, in envelopes that passed the contract.

    THE ATTEMPT BOUNDARY IS THE REQUEST, not the adapter call. Opening it once per
    adapter call meant the wrapper's five internal 429 retries shared a single attempt
    slot, so four of the five billed-but-unreported requests vanished from the account.
    """
    _http.count = _http_count() + 1
    _http.accounted = False


def _account_usage(usage: dict | None) -> None:
    """Record what one attempt cost, or that its cost is unknown."""
    if usage:
        getattr(_http, "usage", []).append(dict(usage))
    else:
        _http.unknown = int(getattr(_http, "unknown", 0) or 0) + 1


#: Token fields that are sums over a call, and may therefore be added across attempts.
_ADDITIVE_USAGE = ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens",
                   "completion_tokens", "reasoning_tokens", "cached_input_tokens")


def call_usage() -> dict | None:
    """Everything this logical call is known to have cost, across every attempt.

    `attempts_counted` and `attempts_unknown` are carried with the totals so a reader can
    tell "this is the whole bill" from "this is the part of the bill we were told about".
    """
    records = list(getattr(_http, "usage", []) or [])
    unknown = int(getattr(_http, "unknown", 0) or 0)
    if not records and not unknown:
        return None                      # nothing was attempted; there is no bill
    if not records:
        # ATTEMPTS WHOSE COST WAS NEVER REPORTED. The token keys are ABSENT rather than
        # zero: a 429 that cost five requests did not cost zero tokens, it cost an amount
        # nobody told us, and a record of zeros would be read as the former.
        return {"attempts_counted": 0, "attempts_unknown": unknown,
                "tokens_known": False, "is_complete": False}
    total = {k: 0 for k in _ADDITIVE_USAGE}
    for rec in records:
        for k in _ADDITIVE_USAGE:
            v = rec.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] += int(v)
    total["attempts_counted"] = len(records)
    total["attempts_unknown"] = unknown
    total["tokens_known"] = True
    total["is_complete"] = unknown == 0
    return total


def merge_usage(*records: dict | None) -> dict | None:
    """Add up usage the service actually reported, keeping the unknown count honest.

    Used where a stage retries around the adapter: each adapter call reports its own
    total, and the stage's envelope has to carry the sum rather than the last one.
    """
    known = [r for r in records if r]
    if not known:
        return None
    total = {k: 0 for k in _ADDITIVE_USAGE}
    counted = unknown = 0
    for rec in known:
        for k in _ADDITIVE_USAGE:
            v = rec.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] += int(v)
        counted += int(rec.get("attempts_counted", 1) or 0)
        unknown += int(rec.get("attempts_unknown", 0) or 0)
    total["attempts_counted"] = counted
    total["attempts_unknown"] = unknown
    total["tokens_known"] = any(r.get("tokens_known", True) for r in known)
    total["is_complete"] = unknown == 0
    if not total["tokens_known"]:
        for k in _ADDITIVE_USAGE:
            total.pop(k, None)
    return total


def _account_response(raw: Any, *, attempted: bool = True) -> None:
    """Record one attempt's usage exactly once.

    Keyed on the response OBJECT, because the same attempt is seen twice - once by the
    repair policy deciding whether to re-ask, and again by `_guarded` translating the
    final failure - and counting it twice would overstate the bill as surely as keeping
    only the last understates it.
    """
    seen = getattr(_http, "seen", None)
    if seen is None:
        seen = _http.seen = []
    if raw is None:
        if attempted:
            _account_attempt_unknown()
        return
    if id(raw) in seen:
        _http.accounted = True            # this attempt is already on the account
        return
    seen.append(id(raw))
    _http.accounted = True
    _account_usage(normalise_usage(getattr(raw, "usage", None)))


def _repair_policy():
    """Retry a schema failure ONLY when the response actually completed.

    Instructor's repair loop is built for a model that returned a complete object
    of the wrong shape: asking again often fixes it. It is useless for a response
    the service cut short - a truncated body fails to parse, gets re-asked, is
    truncated again at the same ceiling, and costs three requests to reach the
    refusal it reached on the first. This policy reads the raw response the
    wrapper stashed and repairs only a `completed` one.

    Falls back to "no repair retry" when tenacity is unavailable: one refusal is
    always better than three.
    """
    if unsw_ai.tenacity is None:                      # pragma: no cover
        return 1
    tenacity = unsw_ai.tenacity

    def worth_repairing(retry_state) -> bool:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        if exc is None or isinstance(exc, unsw_ai._DETERMINISTIC_FAILURES):
            _account_response(getattr(unsw_ai._recent, "response", None))
            return False
        # ONLY a schema failure is repairable. A dropped connection is transport
        # and belongs to the stage's backoff; a 429 belongs to the wrapper. Asking
        # instructor to re-ask them doubles the request count for no gain.
        if not any(k.__name__ in _REPAIRABLE_ERRORS for k in type(exc).__mro__):
            return False
        raw = getattr(unsw_ai._recent, "response", None)
        # THIS ATTEMPT IS BILLED WHETHER OR NOT IT IS REPAIRED, so account for it here -
        # this is the only place a repaired-away attempt is ever visible.
        _account_response(raw)
        # A REFUSAL IS A COMPLETED ANSWER. The status says `completed` and the
        # body parses as nothing, so a status-only test read it as repairable and
        # spent a second request to be refused again in the same words.
        if _refusal_text(raw):
            return False
        status, _reason = response_status(raw)
        return status in (None, "completed")
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(REPAIR_ATTEMPTS),
        retry=worth_repairing, reraise=True)


def _sha(value: Any, n: int = 16) -> str:
    if not isinstance(value, (str, bytes)):
        value = json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:n]


class _Message:
    """Stands in for `choices[0].message`."""

    def __init__(self, parsed: Any = None, content: str | None = None,
                 refusal: str | None = None):
        self.parsed = parsed
        self.content = content
        self.refusal = refusal


class _Choice:
    def __init__(self, message: _Message, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.index = 0


class ShimResponse:
    """Stands in for a ChatCompletion: `.choices`, `.id`, `.usage`, `.model`,
    plus `.provenance` - the full record of the call - and `.request`, the
    transmitted-parameters subset of it."""

    def __init__(self, *, parsed=None, content=None, raw=None, model=None,
                 request: dict | None = None, provenance: dict | None = None):
        status, reason = response_status(raw)
        finish = "stop"
        if status == "incomplete":
            finish = ("content_filter" if (reason or "").find("content_filter") >= 0
                      else "length")
        self.choices = [_Choice(_Message(parsed=parsed, content=content,
                                         refusal=_refusal_text(raw) or None),
                                finish_reason=finish)]
        self.id = getattr(raw, "id", None)
        self.model = getattr(raw, "model", model)
        self.usage = normalise_usage(getattr(raw, "usage", None))
        self.request = dict(request or {})
        self.provenance = dict(provenance or {})
        self.raw = raw
        #: What the WHOLE logical call cost, across instructor's repairs and the
        #: wrapper's retries. `.usage` is this response's own figure, as on an SDK
        #: response; `.call_usage` is the bill. A success reached after one repaired
        #: attempt costs both attempts, and recording only the second understated it.
        self.call_usage = call_usage()


class _Completions:
    """The `.chat.completions` / `.beta.chat.completions` surface."""

    def __init__(self, structured: bool):
        self._structured = structured

    @staticmethod
    def _transmitted(model, temperature, max_tokens) -> dict:
        """EXACTLY the parameters that go to the API."""
        return effective_request(model=model, temperature=temperature,
                                 max_tokens=max_tokens)

    @staticmethod
    def _provenance(req: dict, messages, raw, *, schema=None,
                    attempts: int = 1) -> dict:
        """What was sent and what came back, separately.

        The requested model and the served model are recorded as two fields on
        purpose: `config.MODEL` is a mutable alias, and an envelope that records
        only what was asked for cannot show that the service answered with
        something else.
        """
        status, reason = response_status(raw)
        return {
            "transmitted": dict(req),
            # TRUNCATED digests, and named so. These identify the caller's messages and
            # the response schema; they are NOT a record of the transformed body the SDK
            # and instructor actually transmitted (tool definitions, system framing),
            # which this adapter does not see.
            "input_sha256_16": _sha(list(messages)),
            "response_schema_sha256_16": (
                _sha(schema.model_json_schema()) if schema is not None else None),
            "digest_note": "16 hex chars of sha256; identifies the CALLER's input and "
                           "schema, not the transmitted transport body",
            "model_requested": req.get("model"),
            "model_served": getattr(raw, "model", None),
            "response_status": status,
            "incomplete_reason": reason,
            # ADAPTER attempts only: how many times this method called through.
            "adapter_attempts": attempts,
            # REAL HTTP requests for this logical call, counted at the one place
            # that sees each one: the instructor attempt hook on the structured
            # path, and the per-attempt tick on the raw path. Counted separately
            # from `adapter_attempts` because they are different numbers - the
            # wrapper's 429 backoff and instructor's repair both sit between them.
            "transport_requests": _http_count() or None,
            "envelope_version": ENVELOPE_VERSION,
            "openai_version": getattr(unsw_ai.openai, "__version__", None),
            "unsw_ai_mode": str(unsw_ai.DEFAULT_MODE),
        }

    @staticmethod
    def _no_substitution(client, requested: str) -> None:
        """The assessed adapter NEVER silently answers from a different model.

        The wrapper supports an opt-in fallback list for teaching contexts where any
        answer beats none. Here it is the opposite: a result produced by a different
        model than the envelope records is a false audit trail, and every number in the
        report is supposed to be attributable to one pinned deployment.
        """
        fallbacks = tuple(
            getattr(getattr(client, "settings", None), "fallback_models", ()) or ())
        if fallbacks:
            raise RunConfigurationError(
                f"model fallback is enabled ({list(fallbacks)}), and the assessed "
                f"pipeline forbids it: a silent substitution would put {requested!r} in "
                f"the reproducibility envelope while another model produced the answer. "
                f"Unset STUDENT_AI_FALLBACK_MODELS and re-run.")

    def _guarded(self, fn, req: dict, *, wrapper_retries: bool):
        """Error translation and response validation for BOTH call paths.

        THERE IS EXACTLY ONE RETRY BUDGET, and it does not live here. The
        structured path runs inside `unsw_ai._guard()`, which already waits out
        a 429 up to DEFAULT_RATE_LIMIT_RETRIES times. An earlier version of this
        method wrapped a second loop of the same size around it, so one
        persistent rate limit became 5 x 5 = 25 HTTP requests - and Words' own
        retry loop turned that into 100 for a single logical draw, against a
        shared class budget.

        So: when the wrapper already retries (`wrapper_retries=True`) this method
        only translates. When it does not - the raw `responses.create` path has
        no guard of its own - this method supplies the ONE retry budget.

        AND IT SAYS SO ON THE WAY OUT. Every exception leaving here is stamped
        with how many HTTP requests it cost and whether the budget for this
        failure is spent, because the previous version's "exactly one retry
        budget" was only true of the adapter: Words, Replay and Shock each still
        wrapped four attempts of their own around it, so one persistent 429 cost
        20 requests for a single logical draw. `transport_budget_spent()` is the
        signal that stops them. Only `transport` failures - a timeout, a dropped
        connection, a 5xx - leave here unspent, because nothing below the stage
        retried them and each stage attempt is one more request rather than five.
        """
        _account_reset()
        try:
            client = unsw_ai.get_client(req["model"])
            self._no_substitution(client, req["model"])
            settings = client.settings
        except Exception as exc:  # noqa: BLE001 - stamped, then re-raised unchanged
            # A forbidden fallback is refused HERE, before any request - and it used to
            # leave this method unstamped, so the stages recorded it as one request.
            stamp_unsent(exc)
            raise
        retries = 0 if wrapper_retries else unsw_ai.DEFAULT_RATE_LIMIT_RETRIES
        for attempt in range(retries + 1):
            # The wrapper stashes each raw response on a THREAD-LOCAL. Clear it
            # first: a request that fails before any response arrives (a 429, a
            # dropped connection) would otherwise be diagnosed from whatever the
            # previous call on this thread left behind, and reported as that
            # call's content filter.
            unsw_ai._recent.response = None
            try:
                # COUNTED AT DISPATCH, on both paths: the transport ticks this call's account
                # for each request its guards admit - and for nothing they refuse.
                with unsw_ai.dispatch_sink(_http_tick):
                    value = fn()      # accounts for its own response before returning
                return value, attempt + 1
            except Exception as exc:  # noqa: BLE001 - re-raised below
                # A response that came back 200-but-unfinished makes instructor
                # fail while parsing, and the useful verdict is on the response
                # object rather than the exception. Check it FIRST, so truncation
                # and completion filtering surface as themselves instead of as a
                # schema ValueError.
                recent = getattr(unsw_ai._recent, "response", None)
                _account_response(recent)
                spent_usage = call_usage()
                try:
                    if recent is not None:
                        validate_response(recent, settings)
                    translated = unsw_ai.translate_error(exc, settings)
                except Exception as verdict:  # noqa: BLE001 - stamped below
                    raise self._stamped(verdict, spent_usage) from exc
                if (isinstance(translated, unsw_ai.QuotaExceededError)
                        and not isinstance(translated,
                                           unsw_ai.DailyQuotaExceededError)
                        and attempt < retries):
                    time.sleep(unsw_ai.backoff_seconds(translated, attempt))
                    continue
                if translated is exc:
                    raise self._stamped(exc, spent_usage)
                raise self._stamped(translated, spent_usage) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _stamped(exc: BaseException, usage: dict | None) -> BaseException:
        """Attach the transport cost and the budget verdict before re-raising."""
        return stamp_transport(
            exc, attempts=_http_count(),
            budget_spent=failure_category(exc) not in STAGE_RETRYABLE_CATEGORIES,
            usage=usage)

    # -- .create: free text ------------------------------------------------ #

    def create(self, *, model=None, messages, temperature=None, call_index=None,
               max_tokens=None, response_format=None, **rejected):
        validate_call_kwargs(rejected)
        if response_format is not None:   # tolerate the structured form here too
            return self.parse(model=model, messages=messages,
                              response_format=response_format,
                              temperature=temperature, call_index=call_index,
                              max_tokens=max_tokens)
        try:
            req = self._transmitted(model, temperature, max_tokens)
            client = unsw_ai.get_client(req["model"])
        except Exception as exc:  # noqa: BLE001 - stamped, then re-raised unchanged
            # Nothing has been sent yet: a missing access code fails right here.
            stamp_unsent(exc)
            raise

        # VALIDATION HAPPENS INSIDE THE GUARD, so a truncated free-text answer is
        # accounted for like any other failure. Validating after `_guarded` returned
        # meant the resulting IncompleteResponseError carried neither the transport
        # count nor the usage the service had just reported - the one failure path that
        # knew exactly what it had cost was the one that threw the figure away.
        session = getattr(client, "usage", None)
        count_session = getattr(session, "count_attempt", None)

        def _ask():
            # THE SESSION TOTAL SEES THIS PATH TOO. The raw SDK call bypasses the wrapper's
            # own guard, so neither its requests nor its tokens reached `usage_report()`.
            with (unsw_ai.dispatch_sink(count_session) if count_session is not None
                  else contextlib.nullcontext()):
                raw = client.client.responses.create(input=list(messages), **req)
            if session is not None and hasattr(session, "record"):
                session.record(raw)
            _account_response(raw)
            validate_response(raw, client.settings)
            return raw

        # The raw SDK path has no wrapper guard of its own, so the retry budget
        # belongs to _guarded here.
        raw, attempts = self._guarded(_ask, req, wrapper_retries=False)
        return ShimResponse(
            content=raw.output_text, raw=raw, model=req["model"], request=req,
            provenance=self._provenance(req, messages, raw, attempts=attempts))

    # -- .parse: structured output ----------------------------------------- #

    def parse(self, *, model=None, messages, response_format, temperature=None,
              call_index=None, max_tokens=None, **rejected):
        validate_call_kwargs(rejected)
        try:
            req = self._transmitted(model, temperature, max_tokens)
            client = unsw_ai.get_client(req["model"])
        except Exception as exc:  # noqa: BLE001 - stamped, then re-raised unchanged
            # Nothing has been sent yet: a missing access code fails right here.
            stamp_unsent(exc)
            raise
        # `_repair_policy()` lets INSTRUCTOR re-ask once for a completed-but-
        # invalid object, and not at all for a truncated or filtered one. Its
        # repair loop used to re-ask three times for a body the service had cut
        # short, arriving at the same refusal each time. The wrapper's own 429
        # budget is untouched - that is a different failure with its own policy.
        def _ask():
            parsed, raw = client.chat.completions.create_with_completion(
                response_model=response_format, messages=list(messages),
                max_retries=_repair_policy(), **req)
            _account_response(raw)
            validate_response(raw, client.settings)
            return parsed, raw

        (parsed, raw), attempts = self._guarded(_ask, req, wrapper_retries=True)
        return ShimResponse(
            parsed=parsed, raw=raw, model=req["model"], request=req,
            provenance=self._provenance(req, messages, raw,
                                        schema=response_format,
                                        attempts=attempts))


class _Chat:
    def __init__(self, structured: bool):
        self.completions = _Completions(structured)


class _Beta:
    def __init__(self):
        self.chat = _Chat(structured=True)


class CourseClient:
    """Exposes just enough of the OpenAI client for this assignment."""

    def __init__(self):
        self.chat = _Chat(structured=False)
        self.beta = _Beta()

    @property
    def usage(self) -> dict:
        """Tokens THIS PROCESS has spent, from the shared wrapper.

        A dict, not an object: `client.usage.report()` does not exist here. Use
        `courseapi.usage_report()` for the printable summary.
        """
        return unsw_ai.get_client().usage.summary()


def usage_report() -> str:
    """The printable token summary for this process.

    THIS PROCESS ONLY. The wrapper counts what it sent; it cannot see calls made
    from another terminal, another machine or a teammate's account, so this is a
    floor on your usage and never an authority on your remaining quota.
    """
    return unsw_ai.get_client().usage.report()


_client: CourseClient | None = None


def client_() -> CourseClient:
    global _client
    if _client is None:
        _client = CourseClient()
    return _client
