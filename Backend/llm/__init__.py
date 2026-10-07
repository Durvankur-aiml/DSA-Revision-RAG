"""ALGOFORGE LLM Gateway, Multi-Provider Architecture, and Prompt Construction."""

from .prompts import (
    SYSTEM_PROMPT,
    MAX_QUESTION_LENGTH,
    validate_question,
    classify_sources,
    build_prompt,
)
from .gemini import (
    ask_gemini,
    GEMINI_MODEL,
    GEMINI_POLL_INTERVAL_SECONDS,
    GEMINI_503_MAX_CONSECUTIVE,
    GEMINI_503_DEGRADED_MAX_SECONDS,
)
from .base import (
    LLMRequest,
    LLMUsage,
    LLMResponse,
    LLMError,
    LLMTransientError,
    LLMPermanentError,
    LLMConfigError,
    LLMProvider,
    redact_secrets,
)
from .gemini_adapter import GeminiAdapter
from .groq_adapter import GroqAdapter
from .gateway import (
    LLMGateway,
    get_default_gateway,
    reset_default_gateway,
)

__all__ = [
    # Prompts
    "SYSTEM_PROMPT",
    "MAX_QUESTION_LENGTH",
    "validate_question",
    "classify_sources",
    "build_prompt",
    # Legacy Gemini Public API
    "ask_gemini",
    "GEMINI_MODEL",
    "GEMINI_POLL_INTERVAL_SECONDS",
    "GEMINI_503_MAX_CONSECUTIVE",
    "GEMINI_503_DEGRADED_MAX_SECONDS",
    # Base Contracts
    "LLMRequest",
    "LLMUsage",
    "LLMResponse",
    "LLMError",
    "LLMTransientError",
    "LLMPermanentError",
    "LLMConfigError",
    "LLMProvider",
    "redact_secrets",
    # Adapters & Gateway
    "GeminiAdapter",
    "GroqAdapter",
    "LLMGateway",
    "get_default_gateway",
    "reset_default_gateway",
]
