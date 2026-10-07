"""ALGOFORGE Gemini LLM Integration & Generation Resilience Layer.

Uses google-genai SDK background interactions with bounded polling,
exponential retry backoff, jitter, and sustained-503 capacity fast-fail.
"""

import os
import re
import sys
import time
import random

from google import genai
from google.genai import types

import obs
from .prompts import SYSTEM_PROMPT

_PERF_ENABLED = True


def _perf(stage, started_at, extra=None):
    if not _PERF_ENABLED:
        return
    try:
        duration = time.monotonic() - started_at
        obs.record_stage(stage, duration, extra)
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-3.7-flash"
GEMINI_HTTP_TIMEOUT_MS = 30000
GEMINI_POLL_INTERVAL_SECONDS = 5
GEMINI_MAX_WAIT_SECONDS = 90
GEMINI_MAX_TOTAL_SECONDS = 200
TERMINAL_TEXT_STATUSES = ("completed", "incomplete", "budget_exceeded")
GEMINI_MAX_OUTPUT_TOKENS = 1800
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_SECONDS = 2.0
GEMINI_RETRY_MAX_SECONDS = 20.0
GEMINI_503_MAX_CONSECUTIVE = 3
GEMINI_503_DEGRADED_MAX_SECONDS = 20.0


# ============================================================
# INITIALIZE CLIENT
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or not api_key.strip():
    print(
        "\nERROR: GEMINI_API_KEY environment variable is not set or empty.\n"
        "Set it with:\n"
        'setx GEMINI_API_KEY "your-key"\n'
        "Then restart the terminal."
    )
    sys.exit(1)

try:
    genai_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_HTTP_TIMEOUT_MS
        )
    )
except Exception as e:
    print(f"\nERROR: Failed to initialize Gemini client: {e}")
    sys.exit(1)


# ============================================================
# ERROR CLASSIFICATION & CIRCUIT BREAKER
# ============================================================

def _extract_api_error_code(error):
    for attr in ("code", "status_code", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    text = str(error)
    match = re.search(r"\b([45]\d{2})\b", text)
    return int(match.group(1)) if match else None


def _is_daily_quota_error(error):
    text = str(error).lower()
    return (
        "requests per day" in text
        or ("per day" in text and "limit" in text)
    )


def _is_transient_gemini_error(error):
    if _is_daily_quota_error(error):
        return False
    code = _extract_api_error_code(error)
    if code in {408, 429, 500, 502, 503, 504}:
        return True

    text = str(error).lower()
    transient_phrases = (
        "high demand",
        "temporarily unavailable",
        "temporary",
        "overloaded",
        "rate limit",
        "rate-limit",
        "too many requests",
        "service unavailable",
        "internal server error",
        "deadline exceeded",
        "timeout",
        "timed out",
    )
    return any(phrase in text for phrase in transient_phrases)


def _retry_delay(attempt):
    base = min(
        GEMINI_RETRY_MAX_SECONDS,
        GEMINI_RETRY_BASE_SECONDS * (2 ** attempt),
    )
    return min(
        GEMINI_RETRY_MAX_SECONDS,
        base + random.uniform(0.0, 0.75),
    )


class _GeminiPermanentError(Exception):
    """A Gemini failure that no amount of retrying can fix."""


class _Gemini503UnavailableError(Exception):
    """Sustained Gemini 503 capacity failure -- fail this request fast."""


def _is_503_service_unavailable(error):
    code = _extract_api_error_code(error)
    if code == 503:
        return True
    text = str(error).lower()
    return "service_unavailable" in text


def _register_503_error(error, streak):
    if not _is_503_service_unavailable(error):
        return
    streak["count"] += 1
    if streak["count"] >= GEMINI_503_MAX_CONSECUTIVE:
        raise _Gemini503UnavailableError(
            f"Gemini capacity unavailable: {streak['count']} consecutive "
            "503 service_unavailable responses. Failing fast instead of "
            "waiting out the outage."
        ) from error


# ============================================================
# BACKGROUND INTERACTION LIFECYCLE
# ============================================================

def _create_gemini_interaction(user_message, deadline=None, error_503_streak=None):
    if error_503_streak is None:
        error_503_streak = {"count": 0}
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        if (
            deadline is not None
            and attempt > 0
            and time.monotonic() >= deadline
        ):
            print("  Gemini create retry budget exhausted.")
            return None

        _t_create = time.monotonic()
        try:
            _created = genai_client.interactions.create(
                model=GEMINI_MODEL,
                input=user_message,
                system_instruction=SYSTEM_PROMPT,
                background=True,
                generation_config={
                    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                },
            )
            _perf("gemini_create_call", _t_create)
            return _created
        except Exception as error:
            if not _is_transient_gemini_error(error):
                raise _GeminiPermanentError(
                    f"Gemini create failed permanently: {error}"
                ) from error

            _register_503_error(error, error_503_streak)

            if attempt >= GEMINI_MAX_RETRIES:
                print(f"  Failed to start Gemini task: {error}")
                return None

            delay = _retry_delay(attempt)
            print(
                f"  Gemini start failed temporarily. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    return None


def _poll_gemini_interaction(interaction_id):
    return genai_client.interactions.get(id=interaction_id)


def _generate_with_single_interaction(
    user_message, deadline, error_503_streak=None, degraded_deadline=None
):
    if error_503_streak is None:
        error_503_streak = {"count": 0}

    interaction = _create_gemini_interaction(
        user_message,
        deadline=deadline,
        error_503_streak=error_503_streak,
    )
    if interaction is None:
        return None

    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        print("  Gemini returned no interaction ID.")
        return None

    print(f"  Gemini task started: {interaction_id}")
    print(
        "  Initial status: "
        f"{getattr(interaction, 'status', 'unknown')}"
    )

    start_time = time.monotonic()
    consecutive_poll_failures = 0
    poll_count = 0

    while True:
        if (
            degraded_deadline is not None
            and error_503_streak["count"] > 0
            and time.monotonic() >= degraded_deadline
        ):
            print(
                "  Gemini degraded budget (post-503) exhausted; "
                "stopping this interaction."
            )
            return None

        elapsed = time.monotonic() - start_time
        if (
            elapsed >= GEMINI_MAX_WAIT_SECONDS
            or time.monotonic() >= deadline
        ):
            print(
                f"  Gemini task exceeded the local "
                f"{GEMINI_MAX_WAIT_SECONDS}s wait limit."
            )
            return None

        _t_poll = time.monotonic()
        try:
            result = _poll_gemini_interaction(interaction_id)

            poll_count += 1
            _perf(
                "gemini_poll_get",
                _t_poll,
                extra=(
                    f"poll={poll_count} "
                    f"status={getattr(result, 'status', 'unknown')}"
                ),
            )
            consecutive_poll_failures = 0

        except Exception as error:
            consecutive_poll_failures += 1
            code = _extract_api_error_code(error)

            print(
                f"  Gemini polling error "
                f"(attempt {consecutive_poll_failures}/"
                f"{GEMINI_MAX_RETRIES + 1}, code={code}): {error}"
            )

            if _is_daily_quota_error(error):
                raise _GeminiPermanentError(
                    f"Gemini daily quota exhausted: {error}"
                ) from error

            _register_503_error(error, error_503_streak)

            if (
                not _is_transient_gemini_error(error)
                or consecutive_poll_failures > GEMINI_MAX_RETRIES
            ):
                print("  Non-retryable Gemini polling failure.")
                return None

            delay = _retry_delay(consecutive_poll_failures - 1)
            print(
                f"  Gemini appears temporarily unavailable. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            continue

        status = getattr(result, "status", "unknown")
        print(f"  Gemini status: {status}")

        if status in TERMINAL_TEXT_STATUSES:
            response_text = getattr(result, "output_text", None)

            if response_text and response_text.strip():
                if status != "completed":
                    print(
                        f"  Gemini stopped early with status '{status}'; "
                        "using the partial output produced."
                    )
                print("  Gemini task completed.")
                return response_text.strip()

            print(
                f"  Gemini finished with status '{status}' "
                "but returned no text."
            )
            return None

        if status == "failed":
            print(
                f"  Gemini task failed: "
                f"{getattr(result, 'error', None)}"
            )
            return None

        if status in ("cancelled", "canceled"):
            print("  Gemini task was cancelled.")
            return None

        if status == "requires_action":
            print("  Gemini interaction requires action; not supported.")
            return None

        time.sleep(GEMINI_POLL_INTERVAL_SECONDS)


def _run_gemini_generation(user_message):
    """Execute the generation retry loop across fresh interactions."""
    deadline = time.monotonic() + GEMINI_MAX_TOTAL_SECONDS
    attempt = 0
    error_503_streak = {"count": 0}
    degraded_deadline = time.monotonic() + GEMINI_503_DEGRADED_MAX_SECONDS

    while True:
        if (
            error_503_streak["count"] > 0
            and time.monotonic() >= degraded_deadline
        ):
            print(
                "  Gemini degraded budget exhausted; not retrying "
                "after 503."
            )
            return None

        attempt += 1
        print("  Starting Gemini background task...")

        _t_gen = time.monotonic()
        try:
            answer = _generate_with_single_interaction(
                user_message,
                deadline,
                error_503_streak=error_503_streak,
                degraded_deadline=degraded_deadline,
            )
            _perf("gemini_interaction_success", _t_gen, extra=f"attempt={attempt}")
        except _Gemini503UnavailableError as error:
            print(f"  {error}")
            print("  Failing fast (Gemini capacity incident).")
            raise
        except _GeminiPermanentError as error:
            print(f"  {error}")
            raise

        if answer is not None:
            return answer

        if attempt > GEMINI_MAX_RETRIES or time.monotonic() >= deadline:
            print(f"  Gemini generation failed after {attempt} attempt(s).")
            return None

        delay = _retry_delay(attempt - 1)
        print(
            f"  Retrying generation with a fresh interaction "
            f"in {delay:.1f}s..."
        )
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))


def ask_gemini(user_message):
    """Generate a grounded answer via LLM Gateway (with backwards-compatible contract)."""
    try:
        from .gateway import get_default_gateway
        response = get_default_gateway().generate(user_message)
        return response.text
    except Exception as error:
        print(f"  Gemini/Gateway generation failed: {error}")
        return None
