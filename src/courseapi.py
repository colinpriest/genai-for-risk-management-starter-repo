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
import time
from typing import Any

import config
import unsw_ai

__all__ = ["client_", "CourseClient", "ShimResponse", "normalise_usage",
           "NON_TRANSIENT_ERRORS", "IncompleteResponseError",
           "validate_call_kwargs", "response_status"]


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


#: How many times instructor may re-ask when the model returns a well-formed but
#: INVALID object. Two means one repair attempt.
REPAIR_ATTEMPTS = 2


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
        status, _reason = response_status(
            getattr(unsw_ai._recent, "response", None))
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


class _Completions:
    """The `.chat.completions` / `.beta.chat.completions` surface."""

    def __init__(self, structured: bool):
        self._structured = structured

    @staticmethod
    def _transmitted(model, temperature, max_tokens) -> dict:
        """EXACTLY the parameters that go to the API - no implicit defaults left
        unstated. The output ceiling is in here because it CHANGES THE ANSWER:
        the same prompt under a lower ceiling truncates."""
        req: dict[str, Any] = {"model": model or config.MODEL}
        if temperature is not None:
            req["temperature"] = temperature
        req["max_output_tokens"] = max(
            int(max_tokens or config.MAX_OUTPUT_TOKENS), 16)
        return req

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
            # ADAPTER attempts only. Instructor may repair once inside a single adapter
            # attempt (see _repair_policy), and the SDK is configured with max_retries=0,
            # so this is a lower bound on HTTP requests rather than a count of them.
            "adapter_attempts": attempts,
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
        """
        client = unsw_ai.get_client(req["model"])
        self._no_substitution(client, req["model"])
        settings = client.settings
        retries = 0 if wrapper_retries else unsw_ai.DEFAULT_RATE_LIMIT_RETRIES
        for attempt in range(retries + 1):
            # The wrapper stashes each raw response on a THREAD-LOCAL. Clear it
            # first: a request that fails before any response arrives (a 429, a
            # dropped connection) would otherwise be diagnosed from whatever the
            # previous call on this thread left behind, and reported as that
            # call's content filter.
            unsw_ai._recent.response = None
            try:
                return fn(), attempt + 1
            except Exception as exc:  # noqa: BLE001 - re-raised below
                # A response that came back 200-but-unfinished makes instructor
                # fail while parsing, and the useful verdict is on the response
                # object rather than the exception. Check it FIRST, so truncation
                # and completion filtering surface as themselves instead of as a
                # schema ValueError.
                recent = getattr(unsw_ai._recent, "response", None)
                if recent is not None:
                    validate_response(recent, settings)
                translated = unsw_ai.translate_error(exc, settings)
                if (isinstance(translated, unsw_ai.QuotaExceededError)
                        and not isinstance(translated,
                                           unsw_ai.DailyQuotaExceededError)
                        and attempt < retries):
                    time.sleep(unsw_ai.backoff_seconds(translated, attempt))
                    continue
                if translated is exc:
                    raise
                raise translated from exc
        raise AssertionError("unreachable")  # pragma: no cover

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
