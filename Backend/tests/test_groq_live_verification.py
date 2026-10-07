"""ALGOFORGE Groq Live Gateway Integration Verification.

Executes a live call through LLMGateway with provider='groq' and model='openai/gpt-oss-120b'.
Prints only safe diagnostics (provider, model, latency, response text).
NEVER logs or exposes any API keys or secrets.
"""

import os
import sys
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Ensure Backend directory is on sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load project-root .env safely
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from llm.base import LLMRequest, LLMResponse
from llm.gateway import LLMGateway, reset_default_gateway


def test_groq_gateway_live_call(monkeypatch):
    """Verify end-to-end generation through LLMGateway -> GroqAdapter -> openai/gpt-oss-120b."""
    # Verify presence of key safely without printing
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        pytest.skip("GROQ_API_KEY is not configured in .env; skipping live integration test.")

    # Configure gateway routing to Groq with openai/gpt-oss-120b
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    reset_default_gateway()

    gateway = LLMGateway()

    # Ensure provider and model match target
    assert gateway.primary_provider_name == "groq"
    groq_provider = gateway.get_provider("groq")
    assert groq_provider.model_name == "openai/gpt-oss-120b"

    # Construct request expected by the project's LLM abstraction
    prompt = "Reply with exactly: ALGOFORGE_GROQ_OK"
    request = LLMRequest(
        prompt=prompt,
        max_output_tokens=60,
        request_id="groq-live-verify-1",
    )

    # Execute generation through the gateway
    response: LLMResponse = gateway.generate(request)

    # Print safe diagnostics
    print("\n" + "=" * 60)
    print("GROQ GATEWAY INTEGRATION VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Provider:        {response.provider}")
    print(f"Model:           {response.model}")
    print(f"Latency:         {response.latency_seconds:.3f}s")
    print(f"Finish Reason:   {response.finish_reason}")
    print(f"Tokens:          {response.usage.total_tokens if response.usage else 'N/A'}")
    print(f"Response Text:   {response.text}")
    print("=" * 60 + "\n")

    # Assertions
    assert isinstance(response, LLMResponse), "Expected valid LLMResponse instance"
    assert response.provider == "groq", f"Expected provider 'groq', got '{response.provider}'"
    assert response.model == "openai/gpt-oss-120b", f"Expected model 'openai/gpt-oss-120b', got '{response.model}'"
    assert response.text and len(response.text.strip()) > 0, "Response text must not be empty"
    assert "ALGOFORGE_GROQ_OK" in response.text, f"Expected 'ALGOFORGE_GROQ_OK' in response, got: {response.text}"


def test_groq_gateway_string_prompt_call(monkeypatch):
    """Verify that LLMGateway.generate(str) also auto-wraps into LLMRequest cleanly."""
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not groq_key:
        pytest.skip("GROQ_API_KEY is not configured in .env; skipping live integration test.")

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    reset_default_gateway()

    gateway = LLMGateway()
    response = gateway.generate("Reply with exactly: ALGOFORGE_GROQ_STRING_OK")

    assert isinstance(response, LLMResponse)
    assert response.provider == "groq"
    assert response.model == "openai/gpt-oss-120b"
    assert "ALGOFORGE_GROQ_STRING_OK" in response.text


if __name__ == "__main__":
    test_groq_gateway_live_call()
    test_groq_gateway_string_prompt_call()
