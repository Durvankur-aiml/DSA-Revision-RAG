"""ALGOFORGE LLM Gateway & Multi-Provider Architecture Test Suite.

Deterministic, 100% offline unit tests verifying:
1. Gemini provider selection (default / explicit)
2. Groq provider selection
3. Normalized LLMResponse schema and fields
4. Gemini provider success (offline mock)
5. Groq provider success (offline mock)
6. Gemini transient failure -> Groq fallback
7. Gemini permanent failure -> NO fallback
8. Groq provider failure handling
9. Both providers unavailable -> unified failure report
10. Retry limits and backoff bounds
11. Bounded fallback (max 1 attempt, no loops)
12. Provider and model environment configuration
13. Secret redaction in exceptions and log messages
14. Observability integration (stages, request context, metadata)
15. Request correlation ID propagation
"""

import os
import types
from unittest.mock import MagicMock, patch
import pytest

from llm.base import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LLMError,
    LLMTransientError,
    LLMPermanentError,
    LLMConfigError,
    LLMProvider,
    redact_secrets,
)
from llm.gateway import (
    LLMGateway,
    get_default_gateway,
    reset_default_gateway,
)
from llm.gemini_adapter import GeminiAdapter
from llm.groq_adapter import GroqAdapter
import obs


# ---------------------------------------------------------------------------
# Test Helpers & Stubs
# ---------------------------------------------------------------------------


class MockGroqResponse:
    def __init__(
        self,
        content: str = "Groq generated answer",
        finish_reason: str = "stop",
        prompt_tokens: int = 15,
        completion_tokens: int = 45,
    ):
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
        self.usage = types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class MockGroqClient:
    def __init__(self, script=None):
        self.script = list(script or [])
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return MockGroqResponse()


class StubProvider(LLMProvider):
    def __init__(self, name: str, model: str = "stub-model", generate_fn=None):
        self._name = name
        self._model = model
        self.generate_fn = generate_fn or (
            lambda req: LLMResponse(
                text="stub answer",
                provider=self._name,
                model=self._model,
                latency_seconds=0.05,
                request_id=req.request_id,
            )
        )
        self.calls = []

    @property
    def provider_name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return self.generate_fn(request)


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch):
    """Ensure clean gateway environment defaults for every test."""
    reset_default_gateway()
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    yield
    reset_default_gateway()


# ---------------------------------------------------------------------------
# 1. Provider Selection (Gemini Default & Explicit)
# ---------------------------------------------------------------------------


def test_1_gemini_provider_selected_by_default():
    gw = LLMGateway()
    assert gw.primary_provider_name == "gemini"
    assert gw.fallback_enabled is False
    assert gw.fallback_provider_name == "groq"


def test_1b_gemini_provider_selected_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    gw = LLMGateway()
    assert gw.primary_provider_name == "gemini"


# ---------------------------------------------------------------------------
# 2. Groq Provider Selection
# ---------------------------------------------------------------------------


def test_2_groq_provider_selected_via_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    gw = LLMGateway()
    assert gw.primary_provider_name == "groq"


# ---------------------------------------------------------------------------
# 3. Normalized Response Structure Validation
# ---------------------------------------------------------------------------


def test_3_normalized_response_structure():
    usage = LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    resp = LLMResponse(
        text="Normalized test response",
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_seconds=0.42,
        usage=usage,
        finish_reason="stop",
        request_id="req-12345",
        fallback_used=False,
    )
    assert resp.text == "Normalized test response"
    assert resp.provider == "groq"
    assert resp.model == "llama-3.3-70b-versatile"
    assert resp.latency_seconds == 0.42
    assert resp.usage.total_tokens == 150
    assert resp.finish_reason == "stop"
    assert resp.request_id == "req-12345"
    assert resp.fallback_used is False


# ---------------------------------------------------------------------------
# 4. Gemini Provider Success (Offline Mock)
# ---------------------------------------------------------------------------


def test_4_gemini_success(monkeypatch):
    adapter = GeminiAdapter(model_name="gemini-test-model")
    fake_answer = "Binary search runs in O(log N) time."

    with patch(
        "llm.gemini._generate_with_single_interaction",
        return_value=fake_answer,
    ):
        req = LLMRequest(prompt="Explain binary search complexity.", request_id="gemini-req-1")
        resp = adapter.generate(req)

        assert resp.text == fake_answer
        assert resp.provider == "gemini"
        assert resp.model == "gemini-test-model"
        assert resp.request_id == "gemini-req-1"
        assert resp.latency_seconds >= 0.0


# ---------------------------------------------------------------------------
# 5. Groq Provider Success (Offline Mock)
# ---------------------------------------------------------------------------


def test_5_groq_success():
    mock_client = MockGroqClient()
    adapter = GroqAdapter(
        client=mock_client,
        model_name="llama-3.3-70b-versatile",
        api_key="gsk_testdummykey1234567890abcdef",
    )

    req = LLMRequest(prompt="What is dynamic programming?", request_id="groq-req-1")
    resp = adapter.generate(req)

    assert resp.text == "Groq generated answer"
    assert resp.provider == "groq"
    assert resp.model == "llama-3.3-70b-versatile"
    assert resp.request_id == "groq-req-1"
    assert resp.usage.total_tokens == 60
    assert len(mock_client.calls) == 1
    assert mock_client.calls[0]["model"] == "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# 6. Gemini Transient Failure -> Groq Fallback
# ---------------------------------------------------------------------------


def test_6_gemini_transient_failure_triggers_groq_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")

    gemini_stub = StubProvider(
        name="gemini",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMTransientError("Gemini 503 high demand", provider="gemini", error_code=503)
        ),
    )
    groq_stub = StubProvider(
        name="groq",
        model="llama-3.3-70b-versatile",
        generate_fn=lambda req: LLMResponse(
            text="Groq fallback answer",
            provider="groq",
            model="llama-3.3-70b-versatile",
            latency_seconds=0.3,
            request_id=req.request_id,
        ),
    )

    gw = LLMGateway({"gemini": gemini_stub, "groq": groq_stub})
    resp = gw.generate(LLMRequest(prompt="Find cycle in graph", request_id="req-fb-1"))

    assert resp.text == "Groq fallback answer"
    assert resp.provider == "groq"
    assert resp.fallback_used is True
    assert len(gemini_stub.calls) == 1
    assert len(groq_stub.calls) == 1


# ---------------------------------------------------------------------------
# 7. Gemini Permanent Failure -> NO Fallback
# ---------------------------------------------------------------------------


def test_7_gemini_permanent_failure_does_not_trigger_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")

    gemini_stub = StubProvider(
        name="gemini",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMPermanentError("Invalid Argument (400)", provider="gemini", error_code=400)
        ),
    )
    groq_stub = StubProvider(name="groq")

    gw = LLMGateway({"gemini": gemini_stub, "groq": groq_stub})

    with pytest.raises(LLMPermanentError) as exc_info:
        gw.generate(LLMRequest(prompt="Invalid prompt", request_id="req-perm-1"))

    assert "Invalid Argument (400)" in str(exc_info.value)
    assert len(gemini_stub.calls) == 1
    # Fallback must NEVER be called on permanent failure
    assert len(groq_stub.calls) == 0


# ---------------------------------------------------------------------------
# 8. Groq Provider Failure Handling
# ---------------------------------------------------------------------------


def test_8_groq_permanent_error_fails_immediately():
    err_401 = types.SimpleNamespace(status_code=401, message="Invalid API Key")
    # Simulate SDK raising APIStatusError
    class MockAPIStatusError(Exception):
        status_code = 401

    mock_client = MockGroqClient(script=[MockAPIStatusError("Unauthorized access")])
    adapter = GroqAdapter(client=mock_client, api_key="gsk_invalid")

    with pytest.raises(LLMPermanentError) as exc_info:
        adapter.generate(LLMRequest(prompt="Hello"))

    assert exc_info.value.error_code == 401
    assert len(mock_client.calls) == 1


# ---------------------------------------------------------------------------
# 9. Both Providers Unavailable -> Unified Error
# ---------------------------------------------------------------------------


def test_9_both_providers_unavailable_raises_unified_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")

    gemini_stub = StubProvider(
        name="gemini",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMTransientError("Gemini unavailable", provider="gemini", error_code=503)
        ),
    )
    groq_stub = StubProvider(
        name="groq",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMTransientError("Groq overloaded", provider="groq", error_code=503)
        ),
    )

    gw = LLMGateway({"gemini": gemini_stub, "groq": groq_stub})

    with pytest.raises(LLMTransientError) as exc_info:
        gw.generate(LLMRequest(prompt="Both down query"))

    msg = str(exc_info.value)
    assert "Both primary provider 'gemini' and fallback 'groq' failed" in msg
    assert "Gemini unavailable" in msg
    assert "Groq overloaded" in msg


# ---------------------------------------------------------------------------
# 10. Retry Limits and Backoff Bounds
# ---------------------------------------------------------------------------


def test_10_groq_retry_limits(monkeypatch):
    class Transient500(Exception):
        status_code = 500

    # 3 consecutive 500 errors (initial + 2 retries)
    mock_client = MockGroqClient(
        script=[
            Transient500("Internal 500 error"),
            Transient500("Internal 500 error"),
            Transient500("Internal 500 error"),
        ]
    )
    adapter = GroqAdapter(client=mock_client, api_key="gsk_test")

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    with pytest.raises(LLMTransientError) as exc_info:
        adapter.generate(LLMRequest(prompt="Test retries"))

    assert "failed after 3 attempt(s)" in str(exc_info.value)
    assert len(mock_client.calls) == 3
    assert len(sleeps) == 2  # slept between attempt 1->2 and 2->3
    assert all(s <= 8.5 for s in sleeps)


# ---------------------------------------------------------------------------
# 11. Bounded Fallback (Max 1 Attempt, No Ping-Pong)
# ---------------------------------------------------------------------------


def test_11_bounded_fallback_no_cascading(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")

    gemini_stub = StubProvider(
        name="gemini",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMTransientError("Gemini down", provider="gemini")
        ),
    )
    groq_stub = StubProvider(
        name="groq",
        generate_fn=lambda req: (_ for _ in ()).throw(
            LLMTransientError("Groq down", provider="groq")
        ),
    )

    gw = LLMGateway({"gemini": gemini_stub, "groq": groq_stub})

    with pytest.raises(LLMTransientError):
        gw.generate(LLMRequest(prompt="Test bounded"))

    # Exactly 1 attempt on primary, exactly 1 attempt on fallback
    assert len(gemini_stub.calls) == 1
    assert len(groq_stub.calls) == 1


# ---------------------------------------------------------------------------
# 12. Provider and Model Configuration
# ---------------------------------------------------------------------------


def test_12_custom_model_configuration(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")

    groq_ad = GroqAdapter(api_key="gsk_dummy")
    gemini_ad = GeminiAdapter()

    assert groq_ad.model_name == "mixtral-8x7b-32768"
    assert gemini_ad.model_name == "gemini-2.0-flash"


def test_12b_unknown_provider_raises_config_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic_claude")
    gw = LLMGateway()

    with pytest.raises(LLMConfigError) as exc_info:
        gw.generate(LLMRequest(prompt="Hello"))

    assert "Unknown LLM provider 'anthropic_claude'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 13. Secret Redaction
# ---------------------------------------------------------------------------


def test_13_secret_redaction(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realSuperSecretKey1234567890abcdef")
    raw_error = "Connection failed to https://api.groq.com with key gsk_realSuperSecretKey1234567890abcdef and Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

    scrubbed = redact_secrets(raw_error)
    assert "gsk_realSuperSecretKey1234567890abcdef" not in scrubbed
    assert "[REDACTED_GROQ_KEY]" in scrubbed or "[REDACTED_GROQ_API_KEY]" in scrubbed
    assert "Bearer [REDACTED_TOKEN]" in scrubbed

    # Verify exception message redaction
    err = LLMTransientError(raw_error, provider="groq")
    assert "gsk_realSuperSecretKey1234567890abcdef" not in str(err)
    assert "gsk_realSuperSecretKey1234567890abcdef" not in err.message


# ---------------------------------------------------------------------------
# 14. Observability Fields Integration
# ---------------------------------------------------------------------------


def test_14_observability_fields_recorded(monkeypatch):
    recorded_stages = []
    recorded_meta = []

    monkeypatch.setattr(
        obs,
        "record_stage",
        lambda stage, duration, extra=None: recorded_stages.append((stage, duration, extra)),
    )
    monkeypatch.setattr(
        obs,
        "set_llm_meta",
        lambda **kwargs: recorded_meta.append(kwargs),
    )

    provider = StubProvider(name="groq", model="llama-3.3-70b-versatile")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    gw = LLMGateway({"groq": provider})

    gw.generate(LLMRequest(prompt="Explain Trie", request_id="obs-req-99"))

    assert any(s[0] == "llm_generation" for s in recorded_stages)
    assert any("provider=groq" in (s[2] or "") for s in recorded_stages)
    assert len(recorded_meta) >= 1
    assert recorded_meta[0]["provider"] == "groq"
    assert recorded_meta[0]["model"] == "llama-3.3-70b-versatile"
    assert recorded_meta[0]["fallback_used"] is False


# ---------------------------------------------------------------------------
# 15. Request Correlation ID Propagation
# ---------------------------------------------------------------------------


def test_15_request_correlation_id_propagation():
    req_id = "trace-corr-xyz-987"
    provider = StubProvider(name="gemini")
    gw = LLMGateway({"gemini": provider})

    resp = gw.generate(LLMRequest(prompt="Find LCA", request_id=req_id))

    assert resp.request_id == req_id
    assert provider.calls[0].request_id == req_id


def test_15b_gateway_creates_request_id_when_omitted():
    provider = StubProvider(name="gemini")
    gw = LLMGateway({"gemini": provider})

    resp = gw.generate("Prompt as simple string")

    assert resp.request_id is not None
    assert len(resp.request_id) >= 8
