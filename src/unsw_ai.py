"""
unsw_ai.py - an `instructor`-compatible client for the UNSW AI Foundry student proxy.

Upstream: https://github.com/colinpriest/UNSW-student-ChatGPT-API-wrapper
This file is VENDORED, not installed: the copy here is the one the assignment is tested
against, and it does not update underneath you. Report problems with it to the course
staff rather than editing it - `src/courseapi.py` is the layer the assignment calls.

The UNSW proxy (see `student-guide.docx`) is a thin pass-through to the Azure
OpenAI **Responses API**.  It differs from a stock OpenAI endpoint in two ways:

1.  Authentication uses the headers ``x-student-access-code`` and
    ``x-student-id`` instead of a bearer token.  The real APIM subscription key
    lives server-side and is never seen by student code.
2.  Every request goes to one fixed URL - there is no ``/v1/responses`` path to
    append, and the Chat Completions API is *not* available through it.

This module hides both differences behind the ordinary ``instructor`` interface:

    from unsw_ai import get_client
    from pydantic import BaseModel

    class Answer(BaseModel):
        capital: str

    client = get_client()                           # reads .env, then os.environ
    answer = client.chat.completions.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "Capital of France?"}],
    )

``get_client()`` returns a process-wide shared client for a given configuration,
which is what teaching code should use - constructing a client per notebook cell
would open a new connection pool each time.  ``UNSWInstructor(...)`` remains
available for tests and for anything wanting an isolated instance.

`UNSWInstructor` subclasses ``instructor.Instructor``, so everything instructor
offers is available unchanged: ``create``, ``create_with_completion``,
``create_partial``, ``create_iterable``, the ``chat.completions`` / ``messages``
/ ``responses`` namespaces, ``max_retries``, ``validation_context``, hooks, and
per-client default settings.  ``AsyncUNSWInstructor`` is the async twin.

Credential lookup order (as required for the course): the ``.env`` file first,
then the process environment.  Missing or placeholder values raise
``MissingCredentialsError`` with instructions rather than a KeyError, and the
proxy's documented failure modes are translated into named exceptions.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import httpx
import instructor
import openai
from dotenv import dotenv_values

try:  # exact token counting when available; a char heuristic otherwise
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]

try:  # instructor's retry loop is built on tenacity
    import tenacity
except ImportError:  # pragma: no cover
    tenacity = None  # type: ignore[assignment]

try:  # instructor raises this when a response_model never validates
    from instructor.core import InstructorRetryException
except ImportError:  # pragma: no cover - older instructor releases
    try:
        from instructor.exceptions import InstructorRetryException
    except ImportError:
        InstructorRetryException = None  # type: ignore[assignment]

__all__ = [
    "UNSWInstructor",
    "AsyncUNSWInstructor",
    "ProxySettings",
    "get_client",
    "reset_clients",
    "from_env",
    "UNSWAIError",
    "MissingCredentialsError",
    "InvalidAccessCodeError",
    "MissingHeaderError",
    "QuotaExceededError",
    "RequestTooLargeError",
    "DailyQuotaExceededError",
    "ParameterNotSupportedError",
    "ModelNotAvailableError",
    "ProxyUnreachableError",
    "ProxyTimeoutError",
    "UpstreamServiceError",
    "ContentFilteredError",
    "UnsupportedEndpointError",
    "UnsupportedModeError",
    "StructuredOutputError",
    "RateLimiter",
    "TokenLimiter",
    "estimate_tokens",
    "normalise_student_id",
    "SAMPLING_PARAMETERS",
    "SUPPORTED_REASONING_EFFORTS",
    "UsageTracker",
    "shared_limiter",
    "shared_token_limiter",
    "backoff_seconds",
    "DEFAULT_MODEL",
    "DEFAULT_FALLBACK_MODELS",
    "DEFAULT_REQUESTS_PER_MINUTE",
    "ANNOUNCED_DEPLOYMENTS",
    "KNOWN_DEPLOYMENTS",
    "REQUIRED_VARIABLES",
]

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

#: The models UNSW IT says are available for UNSW Foundry (Sina Ameli, 2026-09-08).
#: Australian data-residency rules limit the catalogue to these three.
ANNOUNCED_DEPLOYMENTS = ("gpt-5.4-mini", "gpt-5.4", "gpt-5.2")

#: The only deployments on the student proxy, verified 2026-09-11. UNSW IT
#: retired gpt-4o, gpt-4o-mini and gpt-4.1-mini on 2026-09-10.
KNOWN_DEPLOYMENTS = ("gpt-5.4-mini", "gpt-5.4")

DEFAULT_MODEL = "gpt-5.4-mini"

#: Empty by default since 2026-09-09: gpt-5.4-mini and gpt-5.4 are deployed, so
#: a 404 now means a genuine mistake and should fail loudly rather than silently
#: switch model.  Set STUDENT_AI_FALLBACK_MODELS (comma separated) to re-enable
#: the safety net; each substitution warns, because a silent model swap would
#: wreck reproducibility.
DEFAULT_FALLBACK_MODELS: tuple[str, ...] = ()

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2

# THE LIMITS, as they actually stand. Verified against the live student proxy on
# 2026-09-11; supersedes every earlier note in this file.
#
#   60 requests / minute          per student
#   70,000 tokens / minute        per student - the only limit that paces a bulk run
#   10,000,000 tokens / day       per student, raised from ~100,000 on 2026-09-11
#   10,000 requests / 30 days     per student
#
# TEMPORARY. UNSW IT (Sina Ameli) raised the daily cap for Semester 3 only;
# continuing beyond that needs fresh approval. Re-verify before each teaching
# session with `test_api_connection.py` rather than trusting these numbers.
#
# These are PER STUDENT, keyed on the x-student-id header. An earlier version of
# this comment described them as class-wide, and a second one a few lines down
# gave a daily figure that contradicted the constant beneath it; both were wrong.
DEFAULT_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_PERIOD = 60.0

# Two token limits behave differently, so they stay separate:
#   - the per-REQUEST ceiling makes a call FAIL (estimated pre-flight, so
#     waiting cannot help);
#   - the per-MINUTE budget makes a call WAIT.
# The ~1,000-token per-request ceiling seen on 2026-09-08/09 is gone: a
# 10,305-token request now succeeds.
#
# WHAT THE TRACKER CAN AND CANNOT KNOW. These counters live in THIS PROCESS. They
# cannot see calls made from another terminal, another machine, or by a teammate
# on the same budget. Treat the numbers this wrapper reports as a floor on what
# you have spent, never as an authority on what you have left.
DEFAULT_MAX_REQUEST_TOKENS = 70000
DEFAULT_TOKENS_PER_MINUTE = 70000
DEFAULT_TOKENS_PER_DAY = 10_000_000

REQUEST_CEILING_TOLERANCE = 1.5

#: How many times a call may wait out a 429 before giving up.
DEFAULT_RATE_LIMIT_RETRIES = 4

#: The proxy speaks the Responses API only, so only instructor's Responses
#: modes can work through it.  Chat-Completions modes (TOOLS, JSON, MD_JSON...)
#: fail upstream with "Unsupported parameter: 'messages'".
SUPPORTED_MODES = frozenset(
    {
        instructor.Mode.RESPONSES_TOOLS,
        instructor.Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
    }
)
DEFAULT_MODE = instructor.Mode.RESPONSES_TOOLS

REQUIRED_VARIABLES = ("STUDENT_AI_PROXY_URL", "STUDENT_AI_ACCESS_CODE", "STUDENT_ID")
OPTIONAL_VARIABLES = (
    "STUDENT_AI_MODEL",
    "STUDENT_AI_TIMEOUT",
    "STUDENT_AI_REQUESTS_PER_MINUTE",
    "STUDENT_AI_TOKENS_PER_MINUTE",
    "STUDENT_AI_MAX_REQUEST_TOKENS",
    "STUDENT_AI_TOKENS_PER_DAY",
    "STUDENT_AI_FALLBACK_MODELS",
)

#: Values that look like the placeholders shipped in `.env.example`.
_PLACEHOLDER_MARKERS = ("paste_", "paste-", "your_", "<", "xxxx")

_ENV_FILE_NAME = ".env"


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class UNSWAIError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingCredentialsError(UNSWAIError):
    """A required environment variable was absent, blank, or still a placeholder."""


class InvalidAccessCodeError(UNSWAIError):
    """HTTP 401 - the course access code was rejected by the proxy."""


class MissingHeaderError(UNSWAIError):
    """HTTP 400 - a required ``x-student-*`` header was missing or misspelt."""


class QuotaExceededError(UNSWAIError):
    """HTTP 403 / 429 - the service quota or rate limit has been reached.

    ``retry_after`` holds the number of seconds the proxy asked us to wait, when
    it said; :class:`UNSWInstructor` uses it to back off automatically.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RequestTooLargeError(UNSWAIError):
    """The request cannot fit the token-per-minute budget, so it can never run.

    Distinct from :class:`QuotaExceededError` because the remedy is different and
    retrying is pointless: the proxy estimates the request size before running it
    and refuses anything larger than the budget, however long you wait.
    """

    def __init__(self, message: str, estimated_tokens: int | None = None,
                 budget: int | None = None):
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.budget = budget


class ParameterNotSupportedError(UNSWAIError):
    """A request parameter the target model will reject.

    Raised before the call goes out, so the message can explain the constraint
    and the way round it rather than leaving a bare 400 to be decoded.
    """


class DailyQuotaExceededError(UNSWAIError):
    """The 100,000 token/day budget is spent.

    Separate from :class:`QuotaExceededError` because there is no useful wait:
    the daily window refills over hours, not seconds, so a long-running job
    should stop and resume rather than block.
    """

    def __init__(self, message: str, used_today: int = 0, budget: int = 0):
        super().__init__(message)
        self.used_today = used_today
        self.budget = budget


class ModelNotAvailableError(UNSWAIError):
    """HTTP 404 DeploymentNotFound - the requested model is not deployed."""


class ProxyUnreachableError(UNSWAIError):
    """DNS failure or connection refused - usually a wrong proxy URL."""


class ProxyTimeoutError(UNSWAIError):
    """The request timed out before the proxy answered."""


class UpstreamServiceError(UNSWAIError):
    """HTTP 5xx (e.g. 502) - the proxy could not reach the AI service in time."""


class ContentFilteredError(UNSWAIError):
    """Foundry's content filters blocked the prompt or the completion.

    Attributes carry what actually fired, so calling code can react to it:

    ``categories``
        Filter categories that blocked, e.g. ``("personally_identifiable_information",)``.
    ``subcategories``
        Finer detail where the filter gives it, e.g. ``("Person",)``.
    ``source``
        ``"prompt"`` or ``"completion"`` - was it what we sent or what came back.
    ``details``
        The raw ``content_filters`` block, for anything not summarised above.
    """

    def __init__(
        self,
        message: str,
        categories: tuple[str, ...] = (),
        subcategories: tuple[str, ...] = (),
        source: str | None = None,
        details: Any = None,
    ):
        super().__init__(message)
        self.categories = categories
        self.subcategories = subcategories
        self.source = source
        self.details = details


class UnsupportedEndpointError(UNSWAIError):
    """Something tried to call an endpoint the proxy does not expose."""


class UnsupportedModeError(UNSWAIError):
    """An instructor mode that the proxy cannot serve was requested."""


class StructuredOutputError(UNSWAIError):
    """The model could not be coaxed into valid output for the response_model."""


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


#: A UNSW student ID: the letter z followed by 7 digits, e.g. z1234567.  The
#: proxy rejects anything non-numeric (measured 2026-09-09), so the leading z
#: has to be stripped before the header is sent - but students should still be
#: able to put their real ID in .env.
_ZID_PATTERN = re.compile(r"^[zZ]?(\d{7,8})$")


def normalise_student_id(value: str) -> str:
    """Return the digits of a UNSW student ID, accepting the usual zID form.

    ``z1234567`` -> ``1234567``.  The proxy requires a numeric value, so the
    canonical zID cannot be sent as-is; this keeps `.env` readable while sending
    what the API accepts.
    """
    match = _ZID_PATTERN.match(str(value).strip())
    if not match:
        raise MissingCredentialsError(
            f"STUDENT_ID must be a UNSW student ID - the letter z followed by 7 "
            f"digits, e.g. z1234567 - but got {value!r}.\n"
            "The proxy itself accepts digits only, so the leading z is stripped "
            "automatically before the request is sent; put your normal zID in .env."
        )
    return match.group(1)


def _looks_like_placeholder(value: str) -> bool:
    low = value.strip().lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


#: Overrides the ``.env`` search from the environment. Set it to a path to read that
#: file and nothing else; set it to "none" (or an empty string) to DISABLE the search
#: entirely, so the process environment is the only source of credentials.
#:
#: This exists because an offline test that carefully sets empty credentials and a
#: loopback endpoint could still be overridden by an ancestor ``.env`` it never knew
#: about - the walk up the parents found the developer's real file, and file values beat
#: the environment. A test that cannot switch the discovery off cannot promise isolation.
DOTENV_PATH_VAR = "DOTENV_PATH"

_DOTENV_DISABLED = ("none", "", "off", "0", "false")


def find_env_file(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate the nearest ``.env``.

    ``DOTENV_PATH`` in the environment overrides everything: a path is used directly,
    and "none" disables the search so nothing on disk can contribute credentials.

    Otherwise searches, in order: ``start`` (default: the current working directory) and
    each of its parents, then the directory holding this module.  Returns
    ``None`` when no ``.env`` exists anywhere on that path - the process
    environment is then the only source of credentials.
    """
    override = os.environ.get(DOTENV_PATH_VAR)
    if override is not None:
        if override.strip().lower() in _DOTENV_DISABLED:
            return None
        explicit = Path(override).expanduser()
        return explicit if explicit.is_file() else None

    candidates: list[Path] = []
    base = Path(start) if start is not None else Path.cwd()
    base = base.resolve()
    candidates.extend(parent / _ENV_FILE_NAME for parent in (base, *base.parents))
    candidates.append(Path(__file__).resolve().parent / _ENV_FILE_NAME)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class ProxySettings:
    """Connection settings for the UNSW student AI proxy.

    Instances are immutable and their ``repr`` masks the access code, so a
    settings object can safely be printed or logged in a notebook.
    """

    proxy_url: str
    access_code: str
    student_id: str
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE
    max_request_tokens: int = DEFAULT_MAX_REQUEST_TOKENS
    tokens_per_day: int = DEFAULT_TOKENS_PER_DAY
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    env_file: Path | None = None
    sources: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    # -- construction ------------------------------------------------------- #

    @classmethod
    def load(
        cls,
        *,
        env_file: str | os.PathLike[str] | None = None,
        proxy_url: str | None = None,
        access_code: str | None = None,
        student_id: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_request_tokens: int | None = None,
        tokens_per_day: int | None = None,
        fallback_models: Sequence[str] | None = None,
        search_from: str | os.PathLike[str] | None = None,
    ) -> "ProxySettings":
        """Read settings from ``.env`` first, then the process environment.

        Explicit keyword arguments beat both.  Raises
        :class:`MissingCredentialsError` listing every variable that is still
        missing, where the search looked, and how to fix it.
        """
        if env_file is not None:
            path: Path | None = Path(env_file).expanduser().resolve()
            if path is not None and not path.is_file():
                raise MissingCredentialsError(
                    f"No .env file at the path you supplied: {path}\n"
                    "Pass env_file=None to search the working directory and this "
                    "module's folder instead, or create the file."
                )
        else:
            path = find_env_file(search_from)

        file_values = dotenv_values(path) if path is not None else {}
        sources: dict[str, str] = {}
        problems: dict[str, str] = {}

        def pick(name: str, override: str | None) -> str | None:
            if override is not None:
                sources[name] = "argument"
                return str(override).strip()
            raw = file_values.get(name)
            origin = f".env ({path})"
            if raw is None or not str(raw).strip():
                raw = os.environ.get(name)
                origin = "process environment"
            if raw is None or not str(raw).strip():
                problems[name] = "not set"
                return None
            value = str(raw).strip().strip('"').strip("'")
            if _looks_like_placeholder(value):
                problems[name] = f"still the placeholder value {value!r}"
                return None
            sources[name] = origin
            return value

        url = pick("STUDENT_AI_PROXY_URL", proxy_url)
        code = pick("STUDENT_AI_ACCESS_CODE", access_code)
        sid = pick("STUDENT_ID", student_id)

        if problems:
            raise MissingCredentialsError(_missing_credentials_message(problems, path))

        assert url and code and sid  # narrowed by the check above

        normalise_student_id(sid)  # raises MissingCredentialsError if malformed

        if not url.lower().startswith("https://"):
            raise MissingCredentialsError(
                f"STUDENT_AI_PROXY_URL must be an https:// URL, got {url!r}.\n"
                "Copy the complete URL from the course administrator exactly; do "
                "not guess the azurewebsites.net hostname."
            )

        resolved_model = model or _optional("STUDENT_AI_MODEL", file_values) or DEFAULT_MODEL
        raw_timeout = timeout if timeout is not None else _optional("STUDENT_AI_TIMEOUT", file_values)
        try:
            resolved_timeout = float(raw_timeout) if raw_timeout is not None else DEFAULT_TIMEOUT
        except (TypeError, ValueError):
            raise MissingCredentialsError(
                f"STUDENT_AI_TIMEOUT must be a number of seconds, got {raw_timeout!r}."
            ) from None

        raw_rpm = (
            requests_per_minute
            if requests_per_minute is not None
            else _optional("STUDENT_AI_REQUESTS_PER_MINUTE", file_values)
        )
        try:
            resolved_rpm = int(raw_rpm) if raw_rpm is not None else DEFAULT_REQUESTS_PER_MINUTE
        except (TypeError, ValueError):
            raise MissingCredentialsError(
                f"STUDENT_AI_REQUESTS_PER_MINUTE must be a whole number, got {raw_rpm!r}."
            ) from None
        if resolved_rpm < 1:
            raise MissingCredentialsError(
                f"STUDENT_AI_REQUESTS_PER_MINUTE must be at least 1, got {resolved_rpm}."
            )

        raw_tpm = (
            tokens_per_minute
            if tokens_per_minute is not None
            else _optional("STUDENT_AI_TOKENS_PER_MINUTE", file_values)
        )
        try:
            resolved_tpm = int(raw_tpm) if raw_tpm is not None else DEFAULT_TOKENS_PER_MINUTE
        except (TypeError, ValueError):
            raise MissingCredentialsError(
                f"STUDENT_AI_TOKENS_PER_MINUTE must be a whole number, got {raw_tpm!r}."
            ) from None
        if resolved_tpm < 1:
            raise MissingCredentialsError(
                f"STUDENT_AI_TOKENS_PER_MINUTE must be at least 1, got {resolved_tpm}."
            )

        raw_mrt = (
            max_request_tokens
            if max_request_tokens is not None
            else _optional("STUDENT_AI_MAX_REQUEST_TOKENS", file_values)
        )
        try:
            resolved_mrt = int(raw_mrt) if raw_mrt is not None else DEFAULT_MAX_REQUEST_TOKENS
        except (TypeError, ValueError):
            raise MissingCredentialsError(
                f"STUDENT_AI_MAX_REQUEST_TOKENS must be a whole number, got {raw_mrt!r}."
            ) from None
        if resolved_mrt < 1:
            raise MissingCredentialsError(
                f"STUDENT_AI_MAX_REQUEST_TOKENS must be at least 1, got {resolved_mrt}."
            )

        raw_tpd = (
            tokens_per_day
            if tokens_per_day is not None
            else _optional("STUDENT_AI_TOKENS_PER_DAY", file_values)
        )
        try:
            resolved_tpd = int(raw_tpd) if raw_tpd is not None else DEFAULT_TOKENS_PER_DAY
        except (TypeError, ValueError):
            raise MissingCredentialsError(
                f"STUDENT_AI_TOKENS_PER_DAY must be a whole number, got {raw_tpd!r}."
            ) from None

        if fallback_models is not None:
            resolved_fallbacks = tuple(fallback_models)
        else:
            raw_fallbacks = _optional("STUDENT_AI_FALLBACK_MODELS", file_values)
            if raw_fallbacks is None:
                resolved_fallbacks = DEFAULT_FALLBACK_MODELS
            else:
                resolved_fallbacks = tuple(
                    name.strip() for name in raw_fallbacks.split(",") if name.strip()
                )

        return cls(
            proxy_url=url.rstrip("/"),
            access_code=code,
            student_id=sid,
            model=resolved_model,
            timeout=resolved_timeout,
            requests_per_minute=resolved_rpm,
            tokens_per_minute=resolved_tpm,
            max_request_tokens=resolved_mrt,
            tokens_per_day=resolved_tpd,
            fallback_models=resolved_fallbacks,
            env_file=path,
            sources=sources,
        )

    # -- presentation ------------------------------------------------------- #

    @property
    def student_id_header(self) -> str:
        """The student ID as the proxy wants it: digits only, no leading z."""
        return normalise_student_id(self.student_id)

    @property
    def headers(self) -> dict[str, str]:
        """The proxy authentication headers, exactly as the guide spells them."""
        return {
            "Content-Type": "application/json",
            "x-student-access-code": self.access_code,
            "x-student-id": self.student_id_header,
        }

    def masked_access_code(self) -> str:
        code = self.access_code
        return f"{code[:4]}...{code[-4:]}" if len(code) > 10 else "***"

    def __repr__(self) -> str:  # never leak the access code into a notebook
        return (
            f"ProxySettings(proxy_url={self.proxy_url!r}, "
            f"access_code={self.masked_access_code()!r}, "
            f"student_id={self.student_id!r}, model={self.model!r}, "
            f"timeout={self.timeout!r}, env_file={str(self.env_file)!r})"
        )

    def describe(self) -> str:
        """A multi-line, credential-safe summary of where each value came from."""
        lines = [
            f"proxy URL   : {self.proxy_url}",
            f"access code : {self.masked_access_code()}",
            f"student ID  : {self.student_id} (sent as {self.student_id_header})",
            f"model       : {self.model}",
            f"fallbacks   : {', '.join(self.fallback_models) or '(none)'}",
            f"rate limit  : {self.requests_per_minute} requests/minute, "
            f"{self.tokens_per_minute:,} tokens/minute (client-side)",
            f"request cap : {self.max_request_tokens:,} tokens in a single request",
            f"daily budget: {self.tokens_per_day:,} tokens/day",
            f"timeout     : {self.timeout:g}s",
            f".env file   : {self.env_file or '(none found - using process environment)'}",
        ]
        if self.sources:
            lines.append("sources     : " + ", ".join(f"{k}<-{v}" for k, v in self.sources.items()))
        return "\n".join(lines)


def _optional(name: str, file_values: Mapping[str, str | None]) -> str | None:
    raw = file_values.get(name)
    if raw is None or not str(raw).strip():
        raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().strip('"').strip("'")


def _missing_credentials_message(problems: Mapping[str, str], path: Path | None) -> str:
    listed = "\n".join(f"  - {name}: {why}" for name, why in sorted(problems.items()))
    where = f"read from {path}" if path is not None else "no .env file was found"
    sample = "\n".join(
        [
            "STUDENT_AI_PROXY_URL=https://<the-complete-url-from-your-lecturer>/api/student-ai-proxy",
            "STUDENT_AI_ACCESS_CODE=<the course access code>",
            "STUDENT_ID=z1234567",
        ]
    )
    return (
        "UNSW AI proxy credentials are not usable:\n"
        f"{listed}\n\n"
        f"Lookup order is the .env file first, then the process environment ({where}).\n\n"
        "Fix it either by creating a .env file next to your script:\n"
        f"{sample}\n\n"
        "or by setting the variables for the current PowerShell window:\n"
        '  $env:STUDENT_AI_PROXY_URL = "..."\n'
        '  $env:STUDENT_AI_ACCESS_CODE = "..."\n'
        '  $env:STUDENT_ID = "z1234567"\n\n'
        "Ask the course administrator for the values; do not publish them."
    )


# --------------------------------------------------------------------------- #
# Client-side rate limiting
# --------------------------------------------------------------------------- #


class RateLimiter:
    """Sliding-window limiter: at most ``max_requests`` per ``period`` seconds.

    APIM enforces 60 requests/minute across the entire class, so staying under
    the limit locally is the difference between a lab that runs and a lab that
    spends its hour collecting 429s.  Every HTTP request the SDK makes passes
    through here, including instructor's re-asks and streamed calls.

    The limiter is thread-safe and usable from sync or async code.  After a 429
    it can also be told to hold *every* caller back for a while - see
    :meth:`penalise` - so one rejected request slows the whole process down
    instead of the next thread charging straight into the same wall.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_REQUESTS_PER_MINUTE,
        period: float = RATE_LIMIT_PERIOD,
    ):
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        self.max_requests = max_requests
        self.period = float(period)
        self._times: deque[float] = deque()
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self.total_acquired = 0
        self.total_wait_seconds = 0.0

    # -- internals ---------------------------------------------------------- #

    def _reserve(self) -> float:
        """Take a slot, or report how long to wait for one. Call under the lock."""
        now = time.monotonic()
        horizon = now - self.period
        while self._times and self._times[0] <= horizon:
            self._times.popleft()

        if now < self._cooldown_until:
            return self._cooldown_until - now
        if len(self._times) < self.max_requests:
            self._times.append(now)
            self.total_acquired += 1
            return 0.0
        # Wait until the oldest request in the window falls out of it.
        return max(self._times[0] + self.period - now, 0.01)

    # -- public API --------------------------------------------------------- #

    def acquire(self) -> float:
        """Block until a request slot is free. Returns the seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                delay = self._reserve()
                if delay <= 0:
                    self.total_wait_seconds += waited
                    return waited
            # A little jitter stops parallel workers waking in lockstep.
            delay += random.uniform(0.0, 0.05)
            time.sleep(delay)
            waited += delay

    async def aacquire(self) -> float:
        """Async twin of :meth:`acquire`."""
        waited = 0.0
        while True:
            with self._lock:
                delay = self._reserve()
                if delay <= 0:
                    self.total_wait_seconds += waited
                    return waited
            delay += random.uniform(0.0, 0.05)
            await asyncio.sleep(delay)
            waited += delay

    def penalise(self, seconds: float) -> None:
        """Hold every caller back for ``seconds`` - used after a 429."""
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)

    def set_capacity(self, max_requests: int) -> None:
        """Lower the ceiling. Never raises it - the most cautious setting wins."""
        with self._lock:
            self.max_requests = min(self.max_requests, max_requests)

    def snapshot(self) -> dict[str, Any]:
        """Current state, for troubleshooting a slow lab."""
        with self._lock:
            now = time.monotonic()
            horizon = now - self.period
            in_window = sum(1 for t in self._times if t > horizon)
            return {
                "max_requests_per_period": self.max_requests,
                "period_seconds": self.period,
                "used_in_window": in_window,
                "cooling_down_for": max(self._cooldown_until - now, 0.0),
                "total_requests": self.total_acquired,
                "total_wait_seconds": round(self.total_wait_seconds, 2),
            }

    def __repr__(self) -> str:
        return f"RateLimiter({self.max_requests} requests / {self.period:g}s)"


class UsageTracker:
    """Running token and request totals for a client.

    APIM caps us on *request count*, which is a poor proxy for what a workload
    actually costs when the calls are small and schema-constrained.  Having real
    token figures per lab turns "we need a higher limit" into a number, and lets
    a lab be designed against a token or dollar budget instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.by_model: dict[str, dict[str, int]] = {}

    def record(self, response: Any) -> None:
        """Accumulate one response.  Silently ignores anything without usage."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        model = str(getattr(response, "model", "unknown"))
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
        in_details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(in_details, "cached_tokens", 0) or 0) if in_details else 0

        with self._lock:
            self.requests += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.reasoning_tokens += reasoning
            self.cached_tokens += cached
            row = self.by_model.setdefault(
                model, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
            )
            row["requests"] += 1
            row["input_tokens"] += inp
            row["output_tokens"] += out

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cached_input_tokens": self.cached_tokens,
                "mean_tokens_per_request": (
                    round((self.input_tokens + self.output_tokens) / self.requests, 1)
                    if self.requests
                    else 0.0
                ),
                "by_model": {m: dict(r) for m, r in self.by_model.items()},
            }

    def report(self) -> str:
        """One-paragraph summary suitable for pasting into a capacity discussion."""
        s = self.summary()
        if not s["requests"]:
            return "No requests recorded."
        lines = [
            f"{s['requests']} requests, {s['total_tokens']:,} tokens "
            f"({s['input_tokens']:,} in / {s['output_tokens']:,} out), "
            f"mean {s['mean_tokens_per_request']:g} tokens per request."
        ]
        for model, row in s["by_model"].items():
            lines.append(
                f"  {model}: {row['requests']} requests, "
                f"{row['input_tokens'] + row['output_tokens']:,} tokens"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self.requests = self.input_tokens = self.output_tokens = 0
            self.reasoning_tokens = self.cached_tokens = 0
            self.by_model.clear()


def estimate_tokens(body: Mapping[str, Any]) -> int:
    """Estimate the tokens a Responses request will be charged for.

    Counts the prompt-bearing fields only.  Measured 2026-09-08: a tiny prompt
    with ``max_output_tokens=3000`` is admitted, while a ~1,260-token prompt with
    ``max_output_tokens=16`` is refused - so the proxy's pre-flight check budgets
    for the *prompt*, not the reply.  Counting the output allowance here would
    block legitimate calls that ask for a long answer.

    Uses ``tiktoken`` when installed and a 4-chars-per-token heuristic otherwise.
    Calibrated against real usage figures it runs 3-8% high, which is the safe
    direction: better to pace slightly early than to send a doomed request.
    """
    chunks: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                walk(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v)

    for field_name in ("input", "instructions", "tools", "text", "tool_choice"):
        if field_name in body:
            walk(body[field_name])
    text = "\n".join(chunks)

    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("o200k_base")
            prompt_tokens = len(enc.encode(text, disallowed_special=()))
        except Exception:  # pragma: no cover - fall back to the heuristic
            prompt_tokens = len(text) // 4 + 1
    else:  # pragma: no cover
        prompt_tokens = len(text) // 4 + 1

    overhead = 16  # role scaffolding and the response envelope
    return prompt_tokens + overhead


class TokenLimiter:
    """Enforces the proxy's two token limits, which are different animals.

    ``max_request_tokens`` is a ceiling on a *single* request: the proxy
    estimates the size before running it and refuses anything larger, so no
    amount of waiting helps and retrying is futile.  ``max_tokens_per_minute``
    is an ordinary sliding-window budget: exceed it and you wait.

    Keeping them separate matters. A request can be small enough to be admitted
    yet still have to queue for budget, and a request can be refused outright on
    a completely idle bucket.
    """

    def __init__(
        self,
        max_tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
        max_request_tokens: int = DEFAULT_MAX_REQUEST_TOKENS,
        period: float = RATE_LIMIT_PERIOD,
        max_tokens_per_day: int = DEFAULT_TOKENS_PER_DAY,
    ):
        if min(max_tokens_per_minute, max_request_tokens, max_tokens_per_day) < 1:
            raise ValueError("token limits must be at least 1")
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_request_tokens = max_request_tokens
        self.max_tokens_per_day = max_tokens_per_day
        self.period = float(period)
        self.day_period = 24 * 60 * 60.0
        self._events: deque[tuple[float, int]] = deque()
        self._day_events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self.total_tokens = 0
        self.total_wait_seconds = 0.0
        self.refused_requests = 0

    # -- internals ---------------------------------------------------------- #

    def _day_used(self, now: float) -> int:
        horizon = now - self.day_period
        while self._day_events and self._day_events[0][0] <= horizon:
            self._day_events.popleft()
        return sum(t for _, t in self._day_events)

    def _reserve(self, tokens: int) -> float:
        now = time.monotonic()
        horizon = now - self.period
        while self._events and self._events[0][0] <= horizon:
            self._events.popleft()

        # The daily budget is the tightest limit for document-sized prompts, and
        # unlike the per-minute one there is no sensible "just wait" - a wait
        # could be hours. Fail with the arithmetic instead of hanging.
        day_used = self._day_used(now)
        if day_used + tokens > self.max_tokens_per_day:
            self.refused_requests += 1
            raise DailyQuotaExceededError(
                f"This request needs about {tokens:,} tokens, but only "
                f"{max(self.max_tokens_per_day - day_used, 0):,} of the "
                f"{self.max_tokens_per_day:,} token daily budget remain "
                f"({day_used:,} already used by this process).\n"
                "The daily budget does not refill for up to 24 hours, so this is not "
                "something to wait out mid-run. Cache what you have, reduce the corpus "
                "for this pass, or continue tomorrow.",
                used_today=day_used,
                budget=self.max_tokens_per_day,
            )

        used = sum(t for _, t in self._events)
        if used + tokens <= self.max_tokens_per_minute:
            self._events.append((now, tokens))
            self._day_events.append((now, tokens))
            self.total_tokens += tokens
            return 0.0
        return max(self._events[0][0] + self.period - now, 0.01)

    def _check_request_ceiling(self, tokens: int) -> None:
        if tokens <= self.max_request_tokens * REQUEST_CEILING_TOLERANCE:
            return
        self.refused_requests += 1
        raise RequestTooLargeError(
            f"This request is about {tokens:,} tokens, but the proxy refuses a single "
            f"request above roughly {self.max_request_tokens:,} tokens - it estimates "
            "the size before running it, so waiting and retrying cannot help.\n"
            "This is separate from the per-minute budget: an oversized request is "
            "refused even when nothing else is running.\n"
            "Shorten the prompt - send retrieved passages rather than whole documents, "
            "or use fewer n-shot examples - or ask the course administrator to raise "
            "the per-request limit. If it has already been raised, set "
            "STUDENT_AI_MAX_REQUEST_TOKENS to the new figure.",
            estimated_tokens=tokens,
            budget=self.max_request_tokens,
        )

    # -- public API --------------------------------------------------------- #

    def acquire(self, tokens: int) -> float:
        """Check the per-request ceiling, then wait for per-minute budget."""
        self._check_request_ceiling(tokens)
        effective = min(tokens, self.max_tokens_per_minute)
        waited = 0.0
        while True:
            with self._lock:
                delay = self._reserve(effective)
                if delay <= 0:
                    self.total_wait_seconds += waited
                    return waited
            delay += random.uniform(0.0, 0.05)
            time.sleep(delay)
            waited += delay

    async def aacquire(self, tokens: int) -> float:
        """Async twin of :meth:`acquire`."""
        self._check_request_ceiling(tokens)
        effective = min(tokens, self.max_tokens_per_minute)
        waited = 0.0
        while True:
            with self._lock:
                delay = self._reserve(effective)
                if delay <= 0:
                    self.total_wait_seconds += waited
                    return waited
            delay += random.uniform(0.0, 0.05)
            await asyncio.sleep(delay)
            waited += delay

    def set_capacity(
        self,
        max_tokens_per_minute: int | None = None,
        max_request_tokens: int | None = None,
        max_tokens_per_day: int | None = None,
    ) -> None:
        """Lower any ceiling. Never raises one - the most cautious wins."""
        with self._lock:
            if max_tokens_per_minute is not None:
                self.max_tokens_per_minute = min(
                    self.max_tokens_per_minute, max_tokens_per_minute
                )
            if max_request_tokens is not None:
                self.max_request_tokens = min(self.max_request_tokens, max_request_tokens)
            if max_tokens_per_day is not None:
                self.max_tokens_per_day = min(self.max_tokens_per_day, max_tokens_per_day)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            horizon = now - self.period
            used = sum(t for ts, t in self._events if ts > horizon)
            day_used = self._day_used(now)
            return {
                "max_request_tokens": self.max_request_tokens,
                "max_tokens_per_period": self.max_tokens_per_minute,
                "max_tokens_per_day": self.max_tokens_per_day,
                "tokens_used_today": day_used,
                "tokens_left_today": max(self.max_tokens_per_day - day_used, 0),
                "period_seconds": self.period,
                "tokens_used_in_window": used,
                "tokens_available": max(self.max_tokens_per_minute - used, 0),
                "total_tokens": self.total_tokens,
                "total_wait_seconds": round(self.total_wait_seconds, 2),
                "requests_refused_as_too_large": self.refused_requests,
            }

    def __repr__(self) -> str:
        return (
            f"TokenLimiter(max {self.max_request_tokens:,} tokens/request, "
            f"{self.max_tokens_per_minute:,} tokens/{self.period:g}s)"
        )


_shared_limiter = RateLimiter()
_shared_token_limiter = TokenLimiter()


def shared_limiter(max_requests: int = DEFAULT_REQUESTS_PER_MINUTE) -> RateLimiter:
    """The process-wide limiter.

    Every client shares one budget by default, because APIM counts requests per
    subscription rather than per Python object - two clients in one script must
    not each think they own 60 requests a minute.  A lower ``max_requests``
    tightens the shared ceiling; a higher one is ignored.
    """
    _shared_limiter.set_capacity(max_requests)
    return _shared_limiter


def shared_token_limiter(
    max_tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
    max_request_tokens: int = DEFAULT_MAX_REQUEST_TOKENS,
    max_tokens_per_day: int = DEFAULT_TOKENS_PER_DAY,
) -> TokenLimiter:
    """The process-wide token limiter; the most cautious setting wins."""
    _shared_token_limiter.set_capacity(
        max_tokens_per_minute, max_request_tokens, max_tokens_per_day
    )
    return _shared_token_limiter


# --------------------------------------------------------------------------- #
# HTTP transports - rewrite every SDK request onto the single proxy URL
# --------------------------------------------------------------------------- #

_DROP_HEADERS = ("authorization", "host", "content-length", "api-key", "openai-organization")


def _rebuild_request(request: httpx.Request, settings: ProxySettings) -> httpx.Request:
    path = request.url.path.rstrip("/")
    if not path.endswith("/responses"):
        raise UnsupportedEndpointError(
            f"The UNSW proxy only exposes the Responses API, but the SDK tried to call "
            f"{request.method} {request.url.path}.\n"
            "Chat Completions, embeddings, files and model listings are not available "
            "through the proxy. Use instructor's Responses modes "
            "(instructor.Mode.RESPONSES_TOOLS, the default here)."
        )

    headers = httpx.Headers(request.headers)
    for name in _DROP_HEADERS:
        if name in headers:
            del headers[name]
    for name, value in settings.headers.items():
        headers[name] = value

    return httpx.Request(
        method="POST",
        url=settings.proxy_url,
        headers=headers,
        content=request.read(),
        extensions=request.extensions,
    )


def _estimate_request_tokens(request: httpx.Request) -> int:
    """Token estimate for an outgoing request, from its JSON body."""
    try:
        body = json.loads(request.content.decode("utf-8"))
    except Exception:  # pragma: no cover - non-JSON body should not happen here
        return 0
    if not isinstance(body, dict):
        return 0
    return estimate_tokens(body)


class ProxyTransport(httpx.BaseTransport):
    """Sync transport that redirects SDK traffic to the UNSW proxy URL.

    Every outgoing request waits for a slot from ``limiter`` first, so the
    60/minute ceiling holds no matter which instructor entry point was used.
    """

    def __init__(
        self,
        settings: ProxySettings,
        inner: httpx.BaseTransport | None = None,
        limiter: RateLimiter | None = None,
        token_limiter: "TokenLimiter | None" = None,
    ):
        self._settings = settings
        self._inner = inner or httpx.HTTPTransport()
        self._limiter = limiter
        self._token_limiter = token_limiter

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        proxied = _rebuild_request(request, self._settings)
        if self._token_limiter is not None:
            self._token_limiter.acquire(_estimate_request_tokens(proxied))
        if self._limiter is not None:
            self._limiter.acquire()
        return self._inner.handle_request(proxied)


class AsyncProxyTransport(httpx.AsyncBaseTransport):
    """Async twin of :class:`ProxyTransport`."""

    def __init__(
        self,
        settings: ProxySettings,
        inner: httpx.AsyncBaseTransport | None = None,
        limiter: RateLimiter | None = None,
        token_limiter: "TokenLimiter | None" = None,
    ):
        self._settings = settings
        self._inner = inner or httpx.AsyncHTTPTransport()
        self._limiter = limiter
        self._token_limiter = token_limiter

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        proxied = _rebuild_request(request, self._settings)
        await proxied.aread()
        if self._token_limiter is not None:
            await self._token_limiter.aacquire(_estimate_request_tokens(proxied))
        if self._limiter is not None:
            await self._limiter.aacquire()
        return await self._inner.handle_async_request(proxied)


def build_openai_client(
    settings: ProxySettings,
    *,
    is_async: bool = False,
    limiter: RateLimiter | None = None,
    token_limiter: "TokenLimiter | None" = None,
):
    """Return an ``openai`` client whose every call lands on the UNSW proxy."""
    common = dict(
        api_key="unsw-proxy-supplies-the-key-server-side",  # never sent; required by the SDK
        base_url=settings.proxy_url,
        max_retries=0,  # retries are handled by instructor, not the SDK
    )
    if is_async:
        return openai.AsyncOpenAI(
            http_client=httpx.AsyncClient(
                transport=AsyncProxyTransport(settings, limiter=limiter, token_limiter=token_limiter),
                timeout=settings.timeout,
            ),
            **common,
        )
    return openai.OpenAI(
        http_client=httpx.Client(
            transport=ProxyTransport(settings, limiter=limiter, token_limiter=token_limiter),
            timeout=settings.timeout,
        ),
        **common,
    )


# --------------------------------------------------------------------------- #
# Error translation
# --------------------------------------------------------------------------- #


def _body_text(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.text or ""
        except Exception:  # pragma: no cover - defensive
            return ""
    return str(getattr(exc, "body", "") or "")


#: Per-category advice.  The PII entry matters most here: the deployment blocks
#: person names outright, which stops ordinary source documents - board minutes,
#: tribunal decisions, annual reports - from being analysed at all.
_FILTER_GUIDANCE = {
    "personally_identifiable_information": (
        "The deployment is configured to BLOCK personal information rather than "
        "annotate it, and ordinary source documents routinely contain names - "
        "board minutes, tribunal decisions, company reports.\n"
        "Options: redact or pseudonymise names before sending (spaCy or a regex "
        "over a known name list is usually enough), work from an extract that "
        "omits them, or ask the course administrator to set the PII filter to "
        "annotate instead of block."
    ),
    "jailbreak": (
        "The prompt looked like an attempt to override the model's instructions. "
        "This fires on quoted instructions inside a document as easily as on a "
        "real attack - try quoting less of the source, or summarising it."
    ),
    "indirect_attack": (
        "The prompt looked like an indirect prompt-injection attempt, usually "
        "because the document itself contains instruction-like text. Consider "
        "clearly delimiting the document and telling the model to treat it as "
        "data."
    ),
    "hate": "Rephrase, or work from an extract that omits the offending passage.",
    "sexual": "Rephrase, or work from an extract that omits the offending passage.",
    "violence": "Rephrase, or work from an extract that omits the offending passage.",
    "self_harm": "Rephrase, or work from an extract that omits the offending passage.",
    "protected_material_text": (
        "The text matched known copyrighted material. Quote less of it, or "
        "paraphrase."
    ),
}


def _parse_content_filters(payload: Any) -> tuple[tuple[str, ...], tuple[str, ...], str | None, Any]:
    """Pull the blocking categories out of an Azure content-filter response."""
    blocks = None
    if isinstance(payload, Mapping):
        blocks = payload.get("content_filters")
        if blocks is None and isinstance(payload.get("error"), Mapping):
            blocks = payload["error"].get("content_filters")
    if not isinstance(blocks, (list, tuple)):
        return (), (), None, None

    categories: list[str] = []
    subcategories: list[str] = []
    source: str | None = None
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        if block.get("blocked"):
            source = block.get("source_type") or source
        results = block.get("content_filter_results")
        if not isinstance(results, Mapping):
            continue
        for name, info in results.items():
            if not isinstance(info, Mapping) or not info.get("filtered"):
                continue
            categories.append(str(name))
            source = source or block.get("source_type")
            for sub in info.get("sub_categories") or []:
                if isinstance(sub, Mapping) and sub.get("filtered"):
                    subcategories.append(str(sub.get("sub_category")))
    return tuple(dict.fromkeys(categories)), tuple(dict.fromkeys(subcategories)), source, blocks


def _content_filter_error(exc: Exception, body: str, tail: str) -> ContentFilteredError:
    """Build an informative error from whatever detail the filter returned."""
    payload: Any = None
    try:
        payload = json.loads(body) if body else None
    except Exception:  # pragma: no cover - non-JSON body
        payload = None
    if payload is None:
        payload = getattr(exc, "body", None)

    categories, subcategories, source, details = _parse_content_filters(payload)

    where = {"prompt": "what we sent", "completion": "what the model replied"}.get(
        str(source), "the request"
    )
    if categories:
        named = ", ".join(categories)
        if subcategories:
            named += f" ({', '.join(subcategories)})"
        headline = f"Foundry's content filter blocked {where}: {named}."
    else:
        headline = f"Foundry's content filter blocked {where}."

    advice = [_FILTER_GUIDANCE[c] for c in categories if c in _FILTER_GUIDANCE]
    if not advice:
        advice = [
            "Rephrase the prompt, or work from an extract without the offending "
            "passage. The full filter detail is on the exception's .details."
        ]

    return ContentFilteredError(
        headline + "\n" + "\n".join(advice) +
        "\nThis is a deployment policy, not a model limitation, and retrying the "
        "same text will fail the same way." + tail,
        categories=categories,
        subcategories=subcategories,
        source=str(source) if source else None,
        details=details,
    )


#: The most recent raw response on this thread.  Completion-side filtering does
#: not raise an API error - the call returns 200 with status "incomplete" and a
#: refusal message where the tool call should be - so instructor fails while
#: parsing and the useful detail is only on the response object.  Hooks swallow
#: exceptions, so the response is stashed here and consulted when a call fails.
_recent = threading.local()


def _remember_response(response: Any) -> None:
    _recent.response = response


def completion_filter_error(settings: "ProxySettings") -> "ContentFilteredError | None":
    """A ContentFilteredError if the last response was cut short by the filter.

    Completion-side filtering does not raise an API error: the call returns 200
    with ``status="incomplete"``, ``incomplete_details.reason="content_filter"``
    and a refusal message where the tool call should be.  Instructor then fails
    while parsing, and the useful detail is only on the response object.
    """
    response = getattr(_recent, "response", None)
    if response is None:
        return None
    if str(getattr(response, "status", "")) != "incomplete":
        return None
    details = getattr(response, "incomplete_details", None)
    if "content_filter" not in str(getattr(details, "reason", "") or ""):
        return None

    refusal = ""
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for part in getattr(item, "content", None) or []:
                refusal += getattr(part, "text", "") or ""

    said = "The model said: " + refusal.strip()[:200] + "\n" if refusal.strip() else ""
    tail = "\n(Proxy: {url}, student ID: {sid}, model default: {model})".format(
        url=settings.proxy_url, sid=settings.student_id, model=settings.model
    )
    return ContentFilteredError(
        "Foundry's content filter blocked what the model replied: the response came "
        "back marked incomplete with reason 'content_filter', so the structured "
        "output was never finished.\n"
        + said
        + "Unlike a blocked prompt this fires on the generated text. The usual fix is "
        "to ask for less sensitive output - do not ask the model to reproduce names or "
        "personal details from the source, and prefer placeholders. This is a "
        "deployment policy, not a model limitation." + tail,
        categories=("completion_filter",),
        source="completion",
        details=details,
    )


def _retry_after_seconds(exc: Exception, body: str) -> float | None:
    """How long the proxy wants us to wait, from the header or the message text."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in (\d+(?:\.\d+)?)\s*second", body, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _last_failed_attempt(exc: Exception) -> Exception | None:
    """The real exception behind an InstructorRetryException, if there is one."""
    attempts = getattr(exc, "failed_attempts", None) or []
    for attempt in reversed(attempts):
        inner = getattr(attempt, "exception", None)
        if isinstance(inner, Exception):
            return inner
    cause = exc.__cause__ or exc.__context__
    return cause if isinstance(cause, Exception) and cause is not exc else None


def _unwrap(exc: BaseException) -> BaseException:
    """Recover a UNSWAIError that the OpenAI SDK wrapped in APIConnectionError."""
    seen: set[int] = set()
    cursor: BaseException | None = exc
    while cursor is not None and id(cursor) not in seen:
        if isinstance(cursor, UNSWAIError):
            return cursor
        seen.add(id(cursor))
        cursor = cursor.__cause__ or cursor.__context__
    return exc


def translate_error(exc: Exception, settings: ProxySettings) -> Exception:
    """Map SDK / instructor failures onto the documented proxy failure modes."""
    unwrapped = _unwrap(exc)
    if isinstance(unwrapped, UNSWAIError):
        return unwrapped

    tail = "\n(Proxy: {url}, student ID: {sid}, model default: {model})".format(
        url=settings.proxy_url, sid=settings.student_id, model=settings.model
    )
    body = _body_text(exc)
    snippet = body.strip()[:400]

    if isinstance(exc, openai.AuthenticationError):
        return InvalidAccessCodeError(
            "401 from the proxy: the student access code was rejected.\n"
            "Check STUDENT_AI_ACCESS_CODE for typos, stray spaces or quotes, and "
            "confirm it is the current code for this course." + tail
        )

    if isinstance(exc, openai.NotFoundError):
        if "DeploymentNotFound" in body or "deployment" in body.lower():
            return ModelNotAvailableError(
                "404 DeploymentNotFound: that model is not deployed on this proxy.\n"
                f"Available here: {', '.join(KNOWN_DEPLOYMENTS)}.\n"
                "UNSW IT retired everything else on 2026-09-10, including gpt-4o, "
                "gpt-4o-mini and gpt-4.1-mini.\n"
                "Set STUDENT_AI_MODEL in .env to one of the models above." + tail
            )
        return ProxyUnreachableError(
            "404 from the proxy URL itself - the path is probably wrong.\n"
            "Copy the complete proxy URL from the course administrator exactly." + tail
        )

    if isinstance(exc, openai.BadRequestError):
        low = body.lower()
        if "missing required header" in low or "x-student" in low:
            return MissingHeaderError(
                "400 Missing required header. The proxy needs exactly these header "
                "names: x-student-access-code and x-student-id." + tail
            )
        if "content_filter" in low or "responsibleai" in low or "content management policy" in low:
            return _content_filter_error(exc, body, tail)
        if "unsupported parameter" in low and "messages" in low:
            return UnsupportedModeError(
                "The proxy exposes the Responses API only, and this request was sent in "
                "Chat Completions shape.\nUse instructor.Mode.RESPONSES_TOOLS (the "
                "default for UNSWInstructor)." + tail
            )
        return UNSWAIError(f"400 Bad Request from the proxy: {snippet}{tail}")

    if isinstance(exc, (openai.PermissionDeniedError, openai.RateLimitError)):
        # The DAILY budget speaks differently from the per-minute one: a 403
        # saying "Token quota will exceed ... Try again in 23 hours". Backing off
        # is useless at that timescale, so this must not be retried.
        low_all = body.lower()
        if "token quota" in low_all or "quota will exceed" in low_all:
            wait_note = ""
            match = re.search(r"try again in ([^.\"]+)", body, re.IGNORECASE)
            if match:
                wait_note = f"\nThe proxy says to try again in {match.group(1).strip()}."
            return DailyQuotaExceededError(
                "The daily token budget is exhausted." + wait_note + "\n"
                "Smaller requests may still get through on what is left, but this one "
                "will not - the check is made against the estimated request size.\n"
                "Retrying will not help on any useful timescale: cache what you have, "
                "reduce this pass, or continue tomorrow." + tail
            )

        # APIM refuses on estimated size BEFORE running the request, and the same
        # message covers two different situations:
        #
        #   {"statusCode":429,"message":"Token limit is exceeded. Try again in 25
        #    seconds."}                                  <- the per-minute bucket
        #   ...the same wording with no retry time       <- it will never fit
        #
        # This used to be classified as permanent in both cases, which was right
        # when the per-request ceiling was about 1,000 tokens and a whole document
        # could not fit however long you waited. Since the per-minute budget was
        # raised to 70,000 the usual meaning is the first one. Measured 2026-09-11:
        # in one Words run identical documents were refused twice and served three
        # times, which a fixed size ceiling cannot do - and 20 calls were abandoned
        # as permanently too large when waiting would have served them.
        #
        # So the RETRY TIME decides. A refusal offering one is the bucket, and the
        # caller should back off and try again; a refusal offering none is a size
        # problem no amount of waiting fixes. A request genuinely over the ceiling
        # is caught locally before it is sent (TokenLimiter._check_request_ceiling),
        # so this branch is the proxy disagreeing with our local estimate.
        wait = _retry_after_seconds(exc, body)
        if "estimated request tokens" in low_all or "token limit" in low_all:
            if wait is not None:
                return QuotaExceededError(
                    f"The proxy refused this request on its estimated size: the "
                    f"token-per-minute budget is momentarily full.\nIt asked us to "
                    f"wait {wait:g}s. This is a rate condition, not an oversized "
                    f"request - the same request will be served once the window "
                    f"moves." + tail,
                    retry_after=wait,
                )
            return RequestTooLargeError(
                "The proxy refused this request before running it, on its estimated "
                "size, and offered no retry time - so this is a size limit rather "
                "than a momentarily full budget.\n"
                "Retrying will not help - the request has to get smaller, or the limit "
                "has to get larger. Send retrieved passages rather than whole documents, "
                "use fewer n-shot examples, or lower max_output_tokens; and ask the "
                "course administrator to raise the per-request limit." + tail
            )
        code = getattr(exc, "status_code", "403/429")
        hint = f"\nThe proxy asked us to wait {wait:g}s." if wait else ""
        return QuotaExceededError(
            f"{code} from the proxy: the request-rate, request-count or token quota has "
            f"been reached.{hint}\nWait and try again, reduce max_output_tokens, or "
            "contact the course administrator if it persists." + tail,
            retry_after=wait,
        )

    if isinstance(exc, openai.APITimeoutError):
        return ProxyTimeoutError(
            f"The proxy did not respond within {settings.timeout:g}s.\n"
            "Retry once; if it keeps happening raise the timeout "
            "(STUDENT_AI_TIMEOUT) or report it to the course administrator." + tail
        )

    if isinstance(exc, openai.InternalServerError):
        return UpstreamServiceError(
            f"{getattr(exc, 'status_code', '5xx')} from the proxy: it could not get a "
            "timely response from the AI service.\nWait briefly and retry once; if the "
            "problem continues, report it to the course administrator." + tail
        )

    if isinstance(exc, openai.APIConnectionError):
        cause = exc.__cause__
        detail = f" ({type(cause).__name__}: {cause})" if cause else ""
        return ProxyUnreachableError(
            "Could not reach the proxy host - DNS failure, no internet, or a blocked "
            f"network{detail}.\nCheck your connection and confirm STUDENT_AI_PROXY_URL "
            "is the exact URL supplied by the course administrator (do not guess the "
            "azurewebsites.net hostname)." + tail
        )

    if InstructorRetryException is not None and isinstance(exc, InstructorRetryException):
        # Instructor wraps whatever finally went wrong; a 429 or a 404 hiding in
        # here should be reported as such, not as a schema problem.
        inner = _last_failed_attempt(exc)
        if inner is not None:
            translated = translate_error(inner, settings)
            if isinstance(translated, UNSWAIError):
                return translated
        text = f"{body}\n{exc}".lower()
        if "token limit is exceeded" in text or "error code: 429" in text:
            match = re.search(r"try again in (\d+(?:\.\d+)?)\s*second", str(exc), re.IGNORECASE)
            wait = float(match.group(1)) if match else None
            return QuotaExceededError(
                "429 from the proxy: the token-per-minute quota was exhausted while "
                "instructor was retrying."
                + (f"\nThe proxy asked us to wait {wait:g}s." if wait else "")
                + "\nWait and try again, or reduce max_output_tokens." + tail,
                retry_after=wait,
            )
        return StructuredOutputError(
            "The model's output never satisfied the response_model after "
            f"{getattr(exc, 'n_attempts', 'several')} attempts.\n"
            "Simplify the Pydantic model, add field descriptions, raise "
            "max_output_tokens, or increase max_retries.\n"
            f"Last error: {str(exc)[:400]}" + tail
        )

    return exc


#: Cap on how long a single automatic rate-limit wait may be.
MAX_RATE_LIMIT_WAIT = 65.0


def backoff_seconds(exc: QuotaExceededError, attempt: int) -> float:
    """How long to wait before retrying a 429.

    The proxy usually names an interval ("try again in 24 seconds") and that
    figure beats any guess we could make, so it wins when present.  Otherwise
    back off exponentially - 1, 2, 4, 8, 16, 32 seconds.  Either way a jitter
    term is added: without it, every student whose script tripped the same
    shared quota would retry at the same instant and trip it again.
    """
    if exc.retry_after is not None:
        base = float(exc.retry_after) + 1.0
    else:
        base = min(2.0**attempt, 32.0)
    jitter = random.uniform(0.0, max(0.25 * base, 0.5))
    return min(base + jitter, MAX_RATE_LIMIT_WAIT)


#: Models this process has already seen 404 on.  Without this every call would
#: re-pay one request from the shared 60/minute budget to rediscover the same
#: missing deployment.  Cleared by :func:`forget_unavailable_models`.
_unavailable_models: set[str] = set()
_unavailable_lock = threading.Lock()


def forget_unavailable_models() -> None:
    """Forget which models 404'd - call after IT deploys something new."""
    with _unavailable_lock:
        _unavailable_models.clear()


def _candidate_models(kwargs: dict[str, Any], fallbacks: Sequence[str]) -> list[str | None]:
    """The model to try, then any fallbacks, without repeats.

    Models already known to be missing sink to the back rather than being
    dropped: if every candidate has failed before, we still want a real request
    and a real error rather than a silent no-op.
    """
    first = kwargs.get("model")
    ordered: list[str | None] = [first]
    for name in fallbacks:
        if name != first and name not in ordered:
            ordered.append(name)
    with _unavailable_lock:
        missing = set(_unavailable_models)
    live = [m for m in ordered if m not in missing]
    return live + [m for m in ordered if m in missing] if live else ordered


def _warn_fallback(failed: str | None, nxt: str | None) -> None:
    warnings.warn(
        f"Model {failed!r} is not deployed on the UNSW proxy; falling back to {nxt!r}. "
        "Results are NOT comparable across models - pin STUDENT_AI_MODEL to a deployed "
        "model before generating anything you intend to keep.",
        RuntimeWarning,
        stacklevel=4,
    )


def _guard(
    create_fn: Callable[..., Any],
    settings: ProxySettings,
    rate_limit_retries: int,
    limiter: RateLimiter | None = None,
    fallback_models: Sequence[str] = (),
) -> Callable[..., Any]:
    """Wrap instructor's patched ``create`` with kwarg fixes, backoff and error translation."""

    def create(*args: Any, **kwargs: Any) -> Any:
        kwargs = _normalise_kwargs(kwargs)
        candidates = _candidate_models(kwargs, fallback_models)
        for index, model in enumerate(candidates):
            # Rebuilt and re-validated per candidate: a GPT-5 fallback target
            # has different parameter rules from gpt-4o.
            call_kwargs = dict(kwargs)
            if model is not None:
                call_kwargs["model"] = model
            validate_parameters(call_kwargs)
            for attempt in range(rate_limit_retries + 1):
                try:
                    return create_fn(*args, **call_kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    translated = translate_error(exc, settings)
                    if translated is exc:
                        filtered = completion_filter_error(settings)
                        if filtered is not None:
                            raise filtered from exc
                    # The shared quota is small; waiting out a 429 is almost
                    # always right, and the limiter holds other callers too.
                    if isinstance(translated, QuotaExceededError) and attempt < rate_limit_retries:
                        wait = backoff_seconds(translated, attempt)
                        if limiter is not None:
                            limiter.penalise(wait)
                        time.sleep(wait)
                        continue
                    if isinstance(translated, ModelNotAvailableError):
                        if model:
                            with _unavailable_lock:
                                _unavailable_models.add(model)
                        if index + 1 < len(candidates):
                            _warn_fallback(model, candidates[index + 1])
                            break  # try the next model
                    if translated is exc:
                        raise
                    raise translated from exc
        raise AssertionError("unreachable")  # pragma: no cover

    create.__name__ = getattr(create_fn, "__name__", "create")
    create.__doc__ = getattr(create_fn, "__doc__", None)
    return create


def _aguard(
    create_fn: Callable[..., Any],
    settings: ProxySettings,
    rate_limit_retries: int,
    limiter: RateLimiter | None = None,
    fallback_models: Sequence[str] = (),
) -> Callable[..., Any]:
    async def acreate(*args: Any, **kwargs: Any) -> Any:
        kwargs = _normalise_kwargs(kwargs, is_async=True)
        candidates = _candidate_models(kwargs, fallback_models)
        for index, model in enumerate(candidates):
            call_kwargs = dict(kwargs)
            if model is not None:
                call_kwargs["model"] = model
            validate_parameters(call_kwargs)
            for attempt in range(rate_limit_retries + 1):
                try:
                    return await create_fn(*args, **call_kwargs)
                except Exception as exc:  # noqa: BLE001 - re-raised below
                    translated = translate_error(exc, settings)
                    if translated is exc:
                        filtered = completion_filter_error(settings)
                        if filtered is not None:
                            raise filtered from exc
                    if isinstance(translated, QuotaExceededError) and attempt < rate_limit_retries:
                        wait = backoff_seconds(translated, attempt)
                        if limiter is not None:
                            limiter.penalise(wait)
                        await asyncio.sleep(wait)
                        continue
                    if isinstance(translated, ModelNotAvailableError):
                        if model:
                            with _unavailable_lock:
                                _unavailable_models.add(model)
                        if index + 1 < len(candidates):
                            _warn_fallback(model, candidates[index + 1])
                            break
                    if translated is exc:
                        raise
                    raise translated from exc
        raise AssertionError("unreachable")  # pragma: no cover

    acreate.__name__ = getattr(create_fn, "__name__", "acreate")
    acreate.__doc__ = getattr(create_fn, "__doc__", None)
    return acreate


#: Failures that will never fix themselves on a second identical attempt.  Left
#: to itself instructor retries every exception, so a 404 for an undeployed
#: model costs three requests out of a 60/minute class budget instead of one.
#: 429s are excluded here too - :func:`backoff_seconds` handles those better.
_DETERMINISTIC_FAILURES = (
    openai.NotFoundError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.BadRequestError,
    openai.RateLimitError,
)


def _retrying_for(max_retries: Any, is_async: bool) -> Any:
    """Turn ``max_retries=N`` into a Retrying that skips deterministic failures.

    Transient problems (timeouts, dropped connections) and validation failures
    still get their retries; a wrong model name or a bad access code does not.
    A caller who passes their own tenacity object keeps full control.
    """
    if tenacity is None or not isinstance(max_retries, int):
        return max_retries
    retry_policy = tenacity.retry_if_not_exception_type(_DETERMINISTIC_FAILURES)
    stop = tenacity.stop_after_attempt(max_retries)
    if is_async:
        return tenacity.AsyncRetrying(stop=stop, retry=retry_policy, reraise=True)
    return tenacity.Retrying(stop=stop, retry=retry_policy, reraise=True)


#: Sampling parameters the GPT-5 family accepts ONLY while reasoning is off.
#: Measured on the UNSW proxy 2026-09-09 for gpt-5.4-mini and gpt-5.4: with
#: reasoning.effort unset or "none" these all work; with any other effort the
#: API returns 400.
SAMPLING_PARAMETERS = ("temperature", "top_p", "logprobs", "top_logprobs")

#: reasoning.effort values these deployments accept.  The API advertises
#: 'minimal' and 'max' too, but both are rejected by gpt-5.4-mini / gpt-5.4.
SUPPORTED_REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh")

#: Efforts that switch reasoning on, and therefore forbid the sampling knobs.
_REASONING_ON = ("low", "medium", "high", "xhigh")

#: Never accepted on the GPT-5 family, at any effort: "Unknown parameter".
_GPT5_REMOVED = ("seed",)


def _is_gpt5_family(model: str | None) -> bool:
    """True for gpt-5, gpt-5.4, gpt-5.4-mini and later reasoning models."""
    if not model:
        return False
    name = str(model).lower().lstrip("gpt").lstrip("-")
    return bool(re.match(r"^[5-9](\.|$|-)", name)) or bool(
        re.match(r"^o[3-9](-|$)", str(model).lower())
    )


def _reasoning_effort(kwargs: Mapping[str, Any]) -> str | None:
    """The requested reasoning effort, if any."""
    reasoning = kwargs.get("reasoning")
    if isinstance(reasoning, Mapping):
        effort = reasoning.get("effort")
        return str(effort) if effort is not None else None
    return None


def validate_parameters(kwargs: Mapping[str, Any]) -> None:
    """Reject parameter combinations the target model cannot serve.

    Verified against the UNSW proxy on 2026-09-09 for gpt-5.4-mini and gpt-5.4.
    Doing this client-side turns a bare 400 into an explanation, and costs no
    request from the class budget.
    """
    model = kwargs.get("model")

    temp = kwargs.get("temperature")
    if temp is not None and not (0 <= float(temp) <= 2):
        raise ParameterNotSupportedError(
            f"temperature must be between 0 and 2, got {temp}. "
            "0 gives greedy decoding (the most consistent, literal answers); the "
            "default is 1."
        )

    if not _is_gpt5_family(model):
        return  # gpt-4o and friends accept the whole classic parameter set

    removed = [name for name in _GPT5_REMOVED if name in kwargs]
    if removed:
        raise ParameterNotSupportedError(
            f"{model} does not support {', '.join(removed)} - the API rejects it as "
            "an unknown parameter, and there is no replacement.\n"
            "For reproducibility, cache responses and key the cache on the prompt "
            "plus a call index. To force N *different* answers to one prompt, rely "
            "on the model's own sampling variation (repeat the call), or append an "
            "opaque correlation id to the END of the prompt so the cacheable prefix "
            "is preserved."
        )

    effort = _reasoning_effort(kwargs)
    if effort is not None and effort not in SUPPORTED_REASONING_EFFORTS:
        raise ParameterNotSupportedError(
            f"reasoning effort {effort!r} is not accepted by {model}.\n"
            f"Supported here: {', '.join(SUPPORTED_REASONING_EFFORTS)}. "
            "(The API advertises 'minimal' and 'max' as well, but this deployment "
            "rejects both.)"
        )

    used = [name for name in SAMPLING_PARAMETERS if name in kwargs]
    if used and effort in _REASONING_ON:
        raise ParameterNotSupportedError(
            f"{model} cannot combine {', '.join(used)} with "
            f"reasoning effort {effort!r} - the sampling controls are only "
            "available while reasoning is off.\n"
            "Choose one:\n"
            f"  - drop {', '.join(used)} and let the reasoning effort do the work; or\n"
            "  - set reasoning={'effort': 'none'} (or omit reasoning entirely) and "
            "keep the sampling controls.\n"
            "For extraction and scoring, temperature=0 with reasoning off is usually "
            "the more predictable choice; for multi-step problems, reasoning effort "
            "usually wins."
        )


def _normalise_kwargs(kwargs: dict[str, Any], is_async: bool = False) -> dict[str, Any]:
    """Accept Chat-Completions habits and translate them to Responses arguments."""
    if "max_retries" in kwargs:
        kwargs["max_retries"] = _retrying_for(kwargs["max_retries"], is_async)
    for chat_name in ("max_tokens", "max_completion_tokens"):
        if chat_name in kwargs:
            value = kwargs.pop(chat_name)
            kwargs.setdefault("max_output_tokens", value)

    if "response_format" in kwargs:
        raise UnsupportedModeError(
            "response_format= is a Chat Completions argument. With instructor, pass a "
            "Pydantic class as response_model= instead (or text={'format': ...} if you "
            "are calling the raw Responses API)."
        )
    return kwargs


def _check_mode(mode: instructor.Mode) -> instructor.Mode:
    if mode not in SUPPORTED_MODES:
        raise UnsupportedModeError(
            f"instructor.Mode.{mode.name} sends a Chat Completions request, which the "
            "UNSW proxy rejects with \"Unsupported parameter: 'messages'\".\n"
            "Supported here: "
            + ", ".join(f"instructor.Mode.{m.name}" for m in sorted(SUPPORTED_MODES, key=lambda m: m.name))
        )
    return mode


# --------------------------------------------------------------------------- #
# The clients
# --------------------------------------------------------------------------- #


class UNSWInstructor(instructor.Instructor):
    """``instructor.Instructor`` wired to the UNSW student AI proxy.

    Parameters
    ----------
    model:
        Default deployment name for every call (per-call ``model=`` wins).
        Defaults to ``STUDENT_AI_MODEL`` from the environment, else
        :data:`DEFAULT_MODEL` (``gpt-5.4-mini``).
    env_file:
        Explicit path to a ``.env``.  By default the working directory and its
        parents are searched, then this module's folder.
    mode:
        An instructor Responses mode; see :data:`SUPPORTED_MODES`.
    timeout, proxy_url, access_code, student_id:
        Overrides for the corresponding environment variables.
    settings:
        A ready-made :class:`ProxySettings`, bypassing environment lookup.
    rate_limit_retries:
        How many times a call may wait out a 429 and try again (default 4).  The
        quota is small and shared across the class, so transient rate limits are
        normal; set 0 to fail immediately instead.
    requests_per_minute:
        Client-side ceiling, default 60 to match the APIM limit.  Enforced
        before every HTTP request, so instructor's re-asks and streamed calls
        count too.
    tokens_per_minute, max_request_tokens:
        The proxy's two token limits, which behave differently: the per-minute
        budget makes a call *wait*, while the per-request ceiling makes it
        *fail* - an oversized request is refused before it runs, so retrying
        cannot help.  Defaults are set a little under the measured figures.
    rate_limiter:
        A specific :class:`RateLimiter`.  By default every client in the process
        shares one budget, because APIM counts requests per subscription rather
        than per Python object.
    fallback_models:
        Models to try if the chosen one is not deployed, each substitution
        warning loudly.  Defaults to :data:`DEFAULT_FALLBACK_MODELS`; pass ``()``
        to fail fast instead.
    **defaults:
        Any other keyword becomes a per-client default merged into every call
        (e.g. ``temperature=0``, ``max_output_tokens=800``) - the same mechanism
        instructor itself uses.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        env_file: str | os.PathLike[str] | None = None,
        settings: ProxySettings | None = None,
        mode: instructor.Mode = DEFAULT_MODE,
        timeout: float | None = None,
        proxy_url: str | None = None,
        access_code: str | None = None,
        student_id: str | None = None,
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_request_tokens: int | None = None,
        tokens_per_day: int | None = None,
        rate_limiter: RateLimiter | None = None,
        token_limiter: TokenLimiter | None = None,
        fallback_models: Sequence[str] | None = None,
        **defaults: Any,
    ):
        mode = _check_mode(mode)
        resolved = settings or ProxySettings.load(
            env_file=env_file,
            proxy_url=proxy_url,
            access_code=access_code,
            student_id=student_id,
            model=model,
            timeout=timeout,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            max_request_tokens=max_request_tokens,
            tokens_per_day=tokens_per_day,
            fallback_models=fallback_models,
        )
        limiter = rate_limiter or shared_limiter(resolved.requests_per_minute)
        tokens = token_limiter or shared_token_limiter(
            resolved.tokens_per_minute, resolved.max_request_tokens, resolved.tokens_per_day
        )
        openai_client = build_openai_client(resolved, is_async=False, limiter=limiter, token_limiter=tokens)
        patched = instructor.from_openai(openai_client, mode=mode)
        defaults.setdefault("model", resolved.model)

        super().__init__(
            client=openai_client,
            create=_guard(patched.create_fn, resolved, rate_limit_retries, limiter,
                          resolved.fallback_models),
            mode=mode,
            provider=patched.provider,
            **defaults,
        )
        self._settings = resolved
        self.default_model = resolved.model
        self.rate_limit_retries = rate_limit_retries
        self.rate_limiter = limiter
        self.token_limiter = tokens
        # Token accounting: APIM caps request count, but tokens are what the
        # workload actually costs, so make the real figures available.
        self.usage = UsageTracker()
        self.on("completion:response", self.usage.record)
        self.on("completion:response", _remember_response)

    # -- extras on top of the instructor interface -------------------------- #

    @property
    def settings(self) -> ProxySettings:
        """The resolved connection settings (access code masked in ``repr``)."""
        return self._settings

    def describe(self) -> str:
        """Credential-safe summary of the connection, for troubleshooting."""
        return f"{self._settings.describe()}\nmode        : instructor.Mode.{self.mode.name}"

    def check_connection(self, *, max_output_tokens: int = 16, model: str | None = None) -> dict[str, Any]:
        """Send a minimal round-trip and return id/model/usage. Raises on failure.

        Uses the same 429 backoff as a normal call, but never falls back to a
        different model - a connection check should tell you the truth about the
        model you asked for.
        """
        for attempt in range(self.rate_limit_retries + 1):
            try:
                response = self.client.responses.create(
                    model=model or self.default_model,
                    input="Reply with exactly: OK",
                    max_output_tokens=max_output_tokens,
                )
                break
            except Exception as exc:  # noqa: BLE001
                translated = translate_error(exc, self._settings)
                if isinstance(translated, QuotaExceededError) and attempt < self.rate_limit_retries:
                    wait = backoff_seconds(translated, attempt)
                    self.rate_limiter.penalise(wait)
                    time.sleep(wait)
                    continue
                if translated is exc:
                    raise
                raise translated from exc
        return {
            "id": response.id,
            "model": response.model,
            "status": response.status,
            "text": response.output_text,
            "usage": response.usage.model_dump() if response.usage else None,
        }

    def __repr__(self) -> str:
        return (
            f"UNSWInstructor(model={self.default_model!r}, "
            f"mode=instructor.Mode.{self.mode.name}, "
            f"student_id={self._settings.student_id!r})"
        )


class AsyncUNSWInstructor(instructor.AsyncInstructor):
    """Async twin of :class:`UNSWInstructor` (``await client.chat.completions.create``)."""

    def __init__(
        self,
        model: str | None = None,
        *,
        env_file: str | os.PathLike[str] | None = None,
        settings: ProxySettings | None = None,
        mode: instructor.Mode = DEFAULT_MODE,
        timeout: float | None = None,
        proxy_url: str | None = None,
        access_code: str | None = None,
        student_id: str | None = None,
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        max_request_tokens: int | None = None,
        tokens_per_day: int | None = None,
        rate_limiter: RateLimiter | None = None,
        token_limiter: TokenLimiter | None = None,
        fallback_models: Sequence[str] | None = None,
        **defaults: Any,
    ):
        mode = _check_mode(mode)
        resolved = settings or ProxySettings.load(
            env_file=env_file,
            proxy_url=proxy_url,
            access_code=access_code,
            student_id=student_id,
            model=model,
            timeout=timeout,
            requests_per_minute=requests_per_minute,
            tokens_per_minute=tokens_per_minute,
            max_request_tokens=max_request_tokens,
            tokens_per_day=tokens_per_day,
            fallback_models=fallback_models,
        )
        limiter = rate_limiter or shared_limiter(resolved.requests_per_minute)
        tokens = token_limiter or shared_token_limiter(
            resolved.tokens_per_minute, resolved.max_request_tokens, resolved.tokens_per_day
        )
        openai_client = build_openai_client(resolved, is_async=True, limiter=limiter, token_limiter=tokens)
        patched = instructor.from_openai(openai_client, mode=mode)
        defaults.setdefault("model", resolved.model)

        super().__init__(
            client=openai_client,
            create=_aguard(patched.create_fn, resolved, rate_limit_retries, limiter,
                           resolved.fallback_models),
            mode=mode,
            provider=patched.provider,
            **defaults,
        )
        self._settings = resolved
        self.default_model = resolved.model
        self.rate_limit_retries = rate_limit_retries
        self.rate_limiter = limiter
        self.token_limiter = tokens
        # Token accounting: APIM caps request count, but tokens are what the
        # workload actually costs, so make the real figures available.
        self.usage = UsageTracker()
        self.on("completion:response", self.usage.record)
        self.on("completion:response", _remember_response)

    @property
    def settings(self) -> ProxySettings:
        return self._settings

    def describe(self) -> str:
        return f"{self._settings.describe()}\nmode        : instructor.Mode.{self.mode.name}"

    def __repr__(self) -> str:
        return (
            f"AsyncUNSWInstructor(model={self.default_model!r}, "
            f"mode=instructor.Mode.{self.mode.name}, "
            f"student_id={self._settings.student_id!r})"
        )


# --------------------------------------------------------------------------- #
# Shared client accessor
# --------------------------------------------------------------------------- #

_client_cache: dict[Any, Any] = {}
_client_cache_lock = threading.Lock()


def _freeze(value: Any) -> Any:
    """A hashable stand-in for a config value, so kwargs can key the cache."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_freeze(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def get_client(
    model: str | None = None,
    *,
    is_async: bool = False,
    **kwargs: Any,
) -> "UNSWInstructor | AsyncUNSWInstructor":
    """Return a shared client, building it on first use.

    Prefer this to constructing :class:`UNSWInstructor` directly in teaching
    code.  A notebook whose every cell opens with ``UNSWInstructor()`` builds a
    fresh HTTP connection pool each time and re-reads ``.env`` each time; this
    hands back the same object for the same configuration.

    Note what this is *not* fixing: the rate and token limiters are already
    process-wide, so quota accounting was never at risk from multiple clients.
    This is about resource reuse and about giving students one obvious way in.

    Distinct configurations get distinct clients, so asking for a different
    model or different defaults still works:

        fast = get_client()                                  # gpt-5.4-mini
        deep = get_client("gpt-5.4", reasoning={"effort": "high"})

    Call :func:`reset_clients` after changing ``.env`` or credentials.
    """
    key = (is_async, model, tuple(sorted((k, _freeze(v)) for k, v in kwargs.items())))
    with _client_cache_lock:
        client = _client_cache.get(key)
        if client is None:
            cls = AsyncUNSWInstructor if is_async else UNSWInstructor
            client = cls(model, **kwargs)
            _client_cache[key] = client
        return client


def reset_clients() -> None:
    """Discard cached clients, so the next :func:`get_client` rebuilds.

    Use after editing ``.env``, rotating the access code, or in tests.  Does not
    reset the shared limiters - those track a real server-side budget that does
    not care how many client objects we have made.
    """
    with _client_cache_lock:
        for client in _client_cache.values():
            inner = getattr(client, "client", None)
            http_client = getattr(inner, "_client", None)
            close = getattr(http_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
        _client_cache.clear()


def from_env(
    model: str | None = None,
    *,
    is_async: bool = False,
    **kwargs: Any,
) -> UNSWInstructor | AsyncUNSWInstructor:
    """Convenience factory mirroring ``instructor.from_provider``."""
    cls = AsyncUNSWInstructor if is_async else UNSWInstructor
    return cls(model, **kwargs)


if __name__ == "__main__":  # a 10-second smoke test: python unsw_ai.py
    from pydantic import BaseModel

    class _Ping(BaseModel):
        answer: str

    client = UNSWInstructor()
    print(client.describe())
    print()

    # A real call, so the model fallback and the rate limiter both take part.
    result = client.chat.completions.create(
        response_model=_Ping,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_output_tokens=64,
    )
    print("structured reply :", result.model_dump())
    print("rate limiter     :", client.rate_limiter.snapshot())
