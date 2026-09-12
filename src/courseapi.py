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
           "effective_request"]


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
    "UnsupportedModeError": "request_rejected",
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
    "ProxyUnreachableError": "unreachable",
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
        raise unsw_ai.ParameterNotSupportedError(
            f"courseapi does not accept {name!r}: "
            + (why or "it is not transmitted by this adapter, and an adapter "
                      "that accepted it would put a parameter in the "
                      "reproducibility record that never reached the service")
            + ". Remove it from the call.")


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

#: Counts REAL HTTP requests for the logical call in flight on this thread.
#: Incremented once per instructor attempt, which is one request each. Reset by
#: `_guarded`, so it spans the wrapper's 429 retries and instructor's repairs -
#: everything one call to `.parse()` costs.
_http = threading.local()


def _http_reset() -> None:
    _http.count = 0


def _http_count() -> int:
    return int(getattr(_http, "count", 0) or 0)


def _http_tick(_retry_state=None) -> None:
    _http.count = _http_count() + 1


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
            return False
        # ONLY a schema failure is repairable. A dropped connection is transport
        # and belongs to the stage's backoff; a 429 belongs to the wrapper. Asking
        # instructor to re-ask them doubles the request count for no gain.
        if not any(k.__name__ in _REPAIRABLE_ERRORS for k in type(exc).__mro__):
            return False
        raw = getattr(unsw_ai._recent, "response", None)
        # A REFUSAL IS A COMPLETED ANSWER. The status says `completed` and the
        # body parses as nothing, so a status-only test read it as repairable and
        # spent a second request to be refused again in the same words.
        if _refusal_text(raw):
            return False
        status, _reason = response_status(raw)
        return status in (None, "completed")
    return tenacity.Retrying(
        stop=tenacity.stop_after_attempt(REPAIR_ATTEMPTS),
        retry=worth_repairing, before=_http_tick, reraise=True)


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
            raise unsw_ai.ParameterNotSupportedError(
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
        client = unsw_ai.get_client(req["model"])
        self._no_substitution(client, req["model"])
        settings = client.settings
        retries = 0 if wrapper_retries else unsw_ai.DEFAULT_RATE_LIMIT_RETRIES
        _http_reset()
        for attempt in range(retries + 1):
            # The wrapper stashes each raw response on a THREAD-LOCAL. Clear it
            # first: a request that fails before any response arrives (a 429, a
            # dropped connection) would otherwise be diagnosed from whatever the
            # previous call on this thread left behind, and reported as that
            # call's content filter.
            unsw_ai._recent.response = None
            try:
                if not wrapper_retries:
                    _http_tick()      # the raw path is one request per attempt
                return fn(), attempt + 1
            except Exception as exc:  # noqa: BLE001 - re-raised below
                # A response that came back 200-but-unfinished makes instructor
                # fail while parsing, and the useful verdict is on the response
                # object rather than the exception. Check it FIRST, so truncation
                # and completion filtering surface as themselves instead of as a
                # schema ValueError.
                recent = getattr(unsw_ai._recent, "response", None)
                spent_usage = normalise_usage(getattr(recent, "usage", None))
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
            exc, attempts=_http_count() or 1,
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
        req = self._transmitted(model, temperature, max_tokens)
        client = unsw_ai.get_client(req["model"])
        # The raw SDK path has no wrapper guard of its own, so the retry budget
        # belongs to _guarded here.
        raw, attempts = self._guarded(
            lambda: client.client.responses.create(input=list(messages), **req),
            req, wrapper_retries=False)
        validate_response(raw, client.settings)
        return ShimResponse(
            content=raw.output_text, raw=raw, model=req["model"], request=req,
            provenance=self._provenance(req, messages, raw, attempts=attempts))

    # -- .parse: structured output ----------------------------------------- #

    def parse(self, *, model=None, messages, response_format, temperature=None,
              call_index=None, max_tokens=None, **rejected):
        validate_call_kwargs(rejected)
        req = self._transmitted(model, temperature, max_tokens)
        client = unsw_ai.get_client(req["model"])
        # `_repair_policy()` lets INSTRUCTOR re-ask once for a completed-but-
        # invalid object, and not at all for a truncated or filtered one. Its
        # repair loop used to re-ask three times for a body the service had cut
        # short, arriving at the same refusal each time. The wrapper's own 429
        # budget is untouched - that is a different failure with its own policy.
        (parsed, raw), attempts = self._guarded(
            lambda: client.chat.completions.create_with_completion(
                response_model=response_format, messages=list(messages),
                max_retries=_repair_policy(), **req),
            req, wrapper_retries=True)
        validate_response(raw, client.settings)
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
