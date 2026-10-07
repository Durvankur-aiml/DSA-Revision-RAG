"""ALGOFORGE LLM Base Contracts & Normalized Provider Interfaces.

Defines the normalized request, response, error hierarchy, and abstract
provider interface for provider-agnostic LLM execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import re
from typing import Any, Dict, Optional


def redact_secrets(text: str) -> str:
    """Scrub known API keys and sensitive tokens from messages and logs."""
    if not text:
        return ""
    # Groq API keys: gsk_...
    text = re.sub(r"gsk_[a-zA-Z0-9]{20,}", "[REDACTED_GROQ_KEY]", text)
    # Gemini / Google API keys: AIza...
    text = re.sub(r"AIza[a-zA-Z0-9_-]{30,}", "[REDACTED_GEMINI_KEY]", text)
    # Generic Bearer tokens
    text = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}", "Bearer [REDACTED_TOKEN]", text)
    # Dynamic redaction of active environment variables
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val and len(val) >= 8:
            text = text.replace(val, f"[REDACTED_{var}]")
    return text


@dataclass
class LLMRequest:
    """Normalized request passed into any LLMProvider."""

    prompt: str
    system_instruction: Optional[str] = None
    max_output_tokens: Optional[int] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsage:
    """Token consumption metrics for a generation."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMResponse:
    """Normalized, provider-neutral response."""

    text: str
    provider: str
    model: str
    latency_seconds: float
    usage: Optional[LLMUsage] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    raw_response: Optional[Any] = None
    fallback_used: bool = False
    error_info: Optional[str] = None


# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM gateway and provider errors."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        error_code: Optional[int] = None,
        is_transient: bool = False,
        original_exception: Optional[Exception] = None,
    ):
        clean_msg = redact_secrets(message)
        super().__init__(clean_msg)
        self.message = clean_msg
        self.provider = provider
        self.model = model
        self.error_code = error_code
        self.is_transient = is_transient
        self.original_exception = original_exception

    def __str__(self) -> str:
        prov = f" [{self.provider}]" if self.provider else ""
        code = f" (code={self.error_code})" if self.error_code else ""
        return f"{self.__class__.__name__}{prov}{code}: {self.message}"


class LLMTransientError(LLMError):
    """Transient errors eligible for retry and bounded fallback (429, 503, timeouts)."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        error_code: Optional[int] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            error_code=error_code,
            is_transient=True,
            original_exception=original_exception,
        )


class LLMPermanentError(LLMError):
    """Permanent errors that must fail fast without fallback (400, 401/403 auth, schema)."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        error_code: Optional[int] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            error_code=error_code,
            is_transient=False,
            original_exception=original_exception,
        )


class LLMConfigError(LLMPermanentError):
    """Configuration error such as missing credentials or unresolvable provider."""


# ---------------------------------------------------------------------------
# Provider Interface
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract interface that each provider adapter must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Normalized lowercase provider name (e.g. 'gemini', 'groq')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response synchronously or wait for background completion.

        Raises LLMTransientError on transient issues or LLMPermanentError on hard failures.
        """
        pass
