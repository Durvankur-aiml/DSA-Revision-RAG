"""ALGOFORGE LLM Gateway & Provider Router.

Provides deterministic provider selection, bounded transient fallback,
normalized response delivery, secret scrubbing, and observability integration.
"""

import os
import time
from typing import Dict, Optional, Union

import obs
from .base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMError,
    LLMTransientError,
    LLMPermanentError,
    LLMConfigError,
    redact_secrets,
)
from .gemini_adapter import GeminiAdapter
from .groq_adapter import GroqAdapter
from .prompts import SYSTEM_PROMPT


class LLMGateway:
    """Central gateway routing LLM requests to configured providers with bounded fallback."""

    def __init__(self, providers: Optional[Dict[str, LLMProvider]] = None):
        self._providers: Dict[str, LLMProvider] = {}
        if providers:
            self._providers.update(providers)
        else:
            # Register defaults lazily
            self._providers["gemini"] = GeminiAdapter()
            self._providers["groq"] = GroqAdapter()

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """Register or override a provider adapter."""
        self._providers[name.strip().lower()] = provider

    def get_provider(self, name: str) -> LLMProvider:
        """Retrieve a registered provider by name."""
        key = name.strip().lower()
        if key not in self._providers:
            raise LLMConfigError(
                f"Unknown LLM provider '{key}'. Registered: {list(self._providers.keys())}",
                provider=key,
            )
        return self._providers[key]

    @property
    def primary_provider_name(self) -> str:
        return os.environ.get("LLM_PROVIDER", "gemini").strip().lower()

    @property
    def fallback_enabled(self) -> bool:
        return os.environ.get("LLM_FALLBACK_ENABLED", "false").strip().lower() in (
            "true",
            "1",
            "yes",
        )

    @property
    def fallback_provider_name(self) -> str:
        return os.environ.get("LLM_FALLBACK_PROVIDER", "groq").strip().lower()

    def generate(self, request: Union[str, LLMRequest]) -> LLMResponse:
        """Execute LLM generation with deterministic primary routing and bounded fallback.

        Parameters
        ----------
        request : str or LLMRequest
            The question prompt string or fully configured LLMRequest.

        Returns
        -------
        LLMResponse
            Normalized provider-neutral response.

        Raises
        ------
        LLMError
            Normalized error if generation fails.
        """
        if isinstance(request, str):
            request = LLMRequest(
                prompt=request,
                system_instruction=SYSTEM_PROMPT,
            )

        if not request.request_id:
            ctx = obs.current_request()
            if ctx and getattr(ctx, "request_id", None):
                request.request_id = ctx.request_id
            else:
                request.request_id = obs.adopt_or_create_request_id(None)

        primary_name = self.primary_provider_name
        fallback_name = self.fallback_provider_name
        allow_fallback = (
            self.fallback_enabled
            and fallback_name != primary_name
            and fallback_name in self._providers
        )

        primary_provider = self.get_provider(primary_name)
        t_start = time.monotonic()

        try:
            response = primary_provider.generate(request)
            gen_duration = time.monotonic() - t_start

            # Record primary metrics
            obs.record_stage(
                "llm_generation",
                gen_duration,
                extra=f"provider={primary_name} model={response.model}",
            )
            obs.set_phase_latency("generation", gen_duration)
            if hasattr(obs, "set_llm_meta"):
                tokens = response.usage.total_tokens if response.usage else None
                obs.set_llm_meta(
                    provider=primary_name,
                    model=response.model,
                    fallback_used=False,
                    tokens=tokens,
                )

            return response

        except LLMTransientError as primary_err:
            primary_duration = time.monotonic() - t_start
            obs.record_stage(
                "llm_transient_error",
                primary_duration,
                extra=f"provider={primary_name} err={primary_err.error_code or 'transient'}",
            )

            if not allow_fallback:
                print(
                    f"  [LLM_GATEWAY] Primary provider '{primary_name}' failed "
                    f"transiently and fallback is disabled. Re-raising."
                )
                raise primary_err

            # Bounded fallback (exactly 1 fallback attempt)
            print(
                f"\n  [LLM_GATEWAY] Primary provider '{primary_name}' transient failure: "
                f"{primary_err.message}. Initiating bounded fallback to '{fallback_name}'."
            )
            obs.record_stage(
                "llm_fallback_attempt",
                primary_duration,
                extra=f"from={primary_name} to={fallback_name}",
            )

            fallback_provider = self.get_provider(fallback_name)
            fb_start = time.monotonic()
            try:
                fb_response = fallback_provider.generate(request)
                fb_duration = time.monotonic() - fb_start
                fb_response.fallback_used = True

                print(
                    f"  [LLM_GATEWAY] Fallback to '{fallback_name}' succeeded "
                    f"in {fb_duration:.2f}s."
                )
                obs.record_stage(
                    "llm_fallback_success",
                    fb_duration,
                    extra=f"provider={fallback_name} model={fb_response.model}",
                )
                obs.set_phase_latency("generation", primary_duration + fb_duration)
                if hasattr(obs, "set_llm_meta"):
                    tokens = fb_response.usage.total_tokens if fb_response.usage else None
                    obs.set_llm_meta(
                        provider=fallback_name,
                        model=fb_response.model,
                        fallback_used=True,
                        fallback_provider=fallback_name,
                        tokens=tokens,
                    )

                return fb_response

            except Exception as fb_err:
                fb_duration = time.monotonic() - fb_start
                clean_fb_err = redact_secrets(str(fb_err))
                print(
                    f"  [LLM_GATEWAY] Fallback provider '{fallback_name}' also failed: "
                    f"{clean_fb_err}"
                )
                obs.record_stage(
                    "llm_fallback_failure",
                    fb_duration,
                    extra=f"provider={fallback_name}",
                )
                raise LLMTransientError(
                    f"Both primary provider '{primary_name}' and fallback '{fallback_name}' failed. "
                    f"Primary: {primary_err.message}; Fallback: {clean_fb_err}",
                    provider=primary_name,
                    original_exception=fb_err,
                ) from fb_err

        except LLMPermanentError as perm_err:
            # Permanent errors (auth, bad request) fail fast without fallback
            duration = time.monotonic() - t_start
            obs.record_stage(
                "llm_permanent_error",
                duration,
                extra=f"provider={primary_name}",
            )
            print(
                f"  [LLM_GATEWAY] Permanent failure with '{primary_name}': "
                f"{perm_err.message}. Fast-failing."
            )
            raise perm_err


# ---------------------------------------------------------------------------
# Singleton Gateway Access
# ---------------------------------------------------------------------------

_DEFAULT_GATEWAY: Optional[LLMGateway] = None


def get_default_gateway() -> LLMGateway:
    """Retrieve or initialize the process-wide default LLM gateway."""
    global _DEFAULT_GATEWAY
    if _DEFAULT_GATEWAY is None:
        _DEFAULT_GATEWAY = LLMGateway()
    return _DEFAULT_GATEWAY


def reset_default_gateway() -> None:
    """Reset the default gateway instance (for test isolation and config changes)."""
    global _DEFAULT_GATEWAY
    _DEFAULT_GATEWAY = None
