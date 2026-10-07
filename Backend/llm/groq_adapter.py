"""ALGOFORGE Groq Provider Adapter.

Uses official Groq SDK with bounded retries, transient error mapping,
usage extraction, and secret redaction.
"""

import os
import random
import re
import time
from typing import Any, Optional

from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMTransientError,
    LLMPermanentError,
    LLMConfigError,
    redact_secrets,
)
from .prompts import SYSTEM_PROMPT

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_HTTP_TIMEOUT_SECONDS = 30.0
GROQ_MAX_RETRIES = 2
GROQ_RETRY_BASE_SECONDS = 1.0
GROQ_RETRY_MAX_SECONDS = 8.0


def _extract_status_code(error: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "http_status", "code"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
    match = re.search(r"\b([45]\d{2})\b", str(error))
    return int(match.group(1)) if match else None


def _is_transient_groq_error(error: Exception) -> bool:
    code = _extract_status_code(error)
    if code in {408, 429, 500, 502, 503, 504}:
        return True
    text = str(error).lower()
    transient_phrases = (
        "rate limit",
        "rate_limit",
        "too many requests",
        "service unavailable",
        "service_unavailable",
        "temporarily unavailable",
        "overloaded",
        "connection",
        "connecterror",
        "timeout",
        "timed out",
        "deadline exceeded",
    )
    return any(p in text for p in transient_phrases)


def _groq_retry_delay(attempt: int) -> float:
    base = min(
        GROQ_RETRY_MAX_SECONDS,
        GROQ_RETRY_BASE_SECONDS * (2 ** attempt),
    )
    return min(GROQ_RETRY_MAX_SECONDS, base + random.uniform(0.0, 0.5))


class GroqAdapter(LLMProvider):
    """Adapter wrapping Groq API for secondary generation and fallback."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self._client = client
        self._custom_model = model_name
        self._custom_api_key = api_key

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return (
            self._custom_model
            or os.environ.get("GROQ_MODEL")
            or GROQ_DEFAULT_MODEL
        )

    def _get_api_key(self) -> str:
        key = self._custom_api_key or os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise LLMConfigError(
                "GROQ_API_KEY environment variable is not set or empty.",
                provider=self.provider_name,
                model=self.model_name,
            )
        return key

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from groq import Groq
            api_key = self._get_api_key()
            self._client = Groq(
                api_key=api_key,
                timeout=GROQ_HTTP_TIMEOUT_SECONDS,
            )
            return self._client
        except LLMConfigError:
            raise
        except Exception as e:
            raise LLMPermanentError(
                f"Failed to initialize Groq client: {redact_secrets(str(e))}",
                provider=self.provider_name,
                model=self.model_name,
                original_exception=e,
            ) from e

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute chat completion with Groq with retry on transient failures."""
        client = self._get_client()
        start_time = time.monotonic()
        system_content = request.system_instruction or SYSTEM_PROMPT
        max_tokens = request.max_output_tokens or 1800

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.prompt},
        ]

        last_error = None

        for attempt in range(GROQ_MAX_RETRIES + 1):
            _t_call = time.monotonic()
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                latency = time.monotonic() - start_time

                # Extract content
                choice = completion.choices[0]
                text = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", "completed")

                # Extract token usage if present
                usage = None
                raw_usage = getattr(completion, "usage", None)
                if raw_usage is not None:
                    usage = LLMUsage(
                        prompt_tokens=getattr(raw_usage, "prompt_tokens", None),
                        completion_tokens=getattr(raw_usage, "completion_tokens", None),
                        total_tokens=getattr(raw_usage, "total_tokens", None),
                    )

                return LLMResponse(
                    text=text.strip(),
                    provider=self.provider_name,
                    model=self.model_name,
                    latency_seconds=round(latency, 4),
                    usage=usage,
                    finish_reason=finish_reason,
                    request_id=request.request_id,
                    raw_response=completion,
                )

            except Exception as error:
                last_error = error
                code = _extract_status_code(error)
                clean_msg = redact_secrets(str(error))

                if not _is_transient_groq_error(error):
                    print(f"  Groq non-retryable error (code={code}): {clean_msg}")
                    raise LLMPermanentError(
                        clean_msg,
                        provider=self.provider_name,
                        model=self.model_name,
                        error_code=code,
                        original_exception=error,
                    ) from error

                if attempt >= GROQ_MAX_RETRIES:
                    print(f"  Groq retries exhausted (attempt {attempt + 1}): {clean_msg}")
                    raise LLMTransientError(
                        f"Groq generation failed after {attempt + 1} attempt(s): {clean_msg}",
                        provider=self.provider_name,
                        model=self.model_name,
                        error_code=code,
                        original_exception=error,
                    ) from error

                delay = _groq_retry_delay(attempt)
                print(
                    f"  Groq transient error (code={code}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)

        # Fallback if loop terminates unexpectedly
        raise LLMTransientError(
            f"Groq request failed: {redact_secrets(str(last_error))}",
            provider=self.provider_name,
            model=self.model_name,
            original_exception=last_error,
        )
