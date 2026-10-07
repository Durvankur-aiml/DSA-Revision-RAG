"""ALGOFORGE Gemini Provider Adapter.

Encapsulates Google GenAI SDK background interactions with bounded polling,
exponential retry backoff with jitter, and 503 circuit-breaker fast-fail.
"""

import os
import time
from typing import Optional

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTransientError,
    LLMPermanentError,
    redact_secrets,
)
from . import gemini as gemini_mod


class GeminiAdapter(LLMProvider):
    """Adapter wrapping the battle-tested Gemini resilience layer."""

    def __init__(self, model_name: Optional[str] = None):
        self._custom_model = model_name

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return (
            self._custom_model
            or os.environ.get("GEMINI_MODEL")
            or gemini_mod.GEMINI_MODEL
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute generation using the resilient background interaction lifecycle."""
        start_time = time.monotonic()
        deadline = start_time + gemini_mod.GEMINI_MAX_TOTAL_SECONDS
        attempt = 0
        error_503_streak = {"count": 0}
        degraded_deadline = (
            start_time + gemini_mod.GEMINI_503_DEGRADED_MAX_SECONDS
        )

        last_transient_error = None

        while True:
            if (
                error_503_streak["count"] > 0
                and time.monotonic() >= degraded_deadline
            ):
                msg = (
                    "Gemini degraded budget exhausted; stopping retries "
                    f"after {error_503_streak['count']} 503 error(s)."
                )
                print(f"  {msg}")
                raise LLMTransientError(
                    msg,
                    provider=self.provider_name,
                    model=self.model_name,
                    error_code=503,
                )

            attempt += 1
            print("  Starting Gemini background task...")

            _t_gen = time.monotonic()
            try:
                answer = gemini_mod._generate_with_single_interaction(
                    request.prompt,
                    deadline,
                    error_503_streak=error_503_streak,
                    degraded_deadline=degraded_deadline,
                )
                gemini_mod._perf(
                    "gemini_interaction_success",
                    _t_gen,
                    extra=f"attempt={attempt}",
                )
            except gemini_mod._Gemini503UnavailableError as error:
                print(f"  {error}")
                print("  Failing fast (Gemini capacity incident).")
                raise LLMTransientError(
                    redact_secrets(str(error)),
                    provider=self.provider_name,
                    model=self.model_name,
                    error_code=503,
                    original_exception=error,
                ) from error
            except gemini_mod._GeminiPermanentError as error:
                print(f"  {error}")
                raise LLMPermanentError(
                    redact_secrets(str(error)),
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=error,
                ) from error
            except Exception as error:
                clean_msg = redact_secrets(str(error))
                if gemini_mod._is_transient_gemini_error(error):
                    last_transient_error = error
                    print(f"  Gemini transient error on attempt {attempt}: {clean_msg}")
                else:
                    raise LLMPermanentError(
                        clean_msg,
                        provider=self.provider_name,
                        model=self.model_name,
                        original_exception=error,
                    ) from error

            if answer is not None:
                latency = time.monotonic() - start_time
                return LLMResponse(
                    text=answer,
                    provider=self.provider_name,
                    model=self.model_name,
                    latency_seconds=round(latency, 4),
                    finish_reason="completed",
                    request_id=request.request_id,
                )

            if attempt > gemini_mod.GEMINI_MAX_RETRIES or time.monotonic() >= deadline:
                err_msg = (
                    f"Gemini generation failed after {attempt} attempt(s) "
                    f"or deadline exceeded."
                )
                print(f"  {err_msg}")
                raise LLMTransientError(
                    err_msg,
                    provider=self.provider_name,
                    model=self.model_name,
                    original_exception=last_transient_error,
                )

            delay = gemini_mod._retry_delay(attempt - 1)
            print(
                f"  Retrying generation with a fresh interaction "
                f"in {delay:.1f}s..."
            )
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
