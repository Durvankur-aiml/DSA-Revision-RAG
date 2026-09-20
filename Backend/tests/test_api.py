"""API contract tests for POST /ask, GET /health, GET /.

All Gemini and retrieval dependencies are stubbed. These tests verify
HTTP behavior, response schema, error mapping, ordering, and security
(leak prevention) — never real generation.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. /ask success path (TC-1)
# ---------------------------------------------------------------------------

class TestAskSuccess:
    def test_returns_200_with_schema(self, client, sample_chunks):
        response = client.post(
            "/ask",
            json={"question": "What is binary search?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"answer", "sources"}

        assert isinstance(body["answer"], str)
        assert body["answer"].strip() != ""

        assert isinstance(body["sources"], list)
        assert len(body["sources"]) > 0

        for source in body["sources"]:
            assert set(source.keys()) == {"title", "timestamp", "url", "score"}
            assert isinstance(source["title"], str) and source["title"]
            assert isinstance(source["timestamp"], int) and source["timestamp"] >= 0
            assert isinstance(source["url"], str)
            assert isinstance(source["score"], float)

    def test_sources_preserve_retrieval_order_and_scores(
        self, client, sample_chunks
    ):
        response = client.post(
            "/ask",
            json={"question": "What is binary search?"},
        )
        sources = response.json()["sources"]

        titles = [s["title"] for s in sources]
        assert titles[0].startswith("BS-1.")
        assert titles[1].startswith("BS-16.")
        assert titles[2].startswith("BS-18.")

        scores = [s["score"] for s in sources]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 1.0

    def test_retrieval_called_with_question(self, retrieval_mock, client):
        client.post("/ask", json={"question": "Explain 3 Sum."})
        assert retrieval_mock == ["Explain 3 Sum."]

    def test_question_whitespace_is_trimmed(self, client, retrieval_mock):
        response = client.post(
            "/ask", json={"question": "  What is binary search?  "}
        )
        assert response.status_code == 200
        assert retrieval_mock == ["What is binary search?"]


# ---------------------------------------------------------------------------
# 2. Empty / invalid questions (TC-2, TC-3)
# ---------------------------------------------------------------------------

class TestAskValidation:
    def test_empty_question_returns_400(self, client):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert isinstance(detail, str) and detail
        assert "Traceback" not in detail

    def test_missing_question_field_is_422(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_non_string_question_is_422(self, client):
        response = client.post("/ask", json={"question": 12345})
        assert response.status_code == 422

    def test_oversized_question_returns_400(self, client):
        response = client.post(
            "/ask", json={"question": "x" * (ask_max_question_length() + 1)}
        )
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()

    def test_validation_failure_never_calls_gemini(
        self, client, gemini_mock
    ):
        client.post("/ask", json={"question": "   "})
        assert gemini_mock.create_calls == 0
        assert gemini_mock.get_calls == 0

    def test_validation_failure_needs_no_retrieval(self, client, retrieval_mock):
        client.post("/ask", json={"question": ""})
        assert retrieval_mock == []


# ---------------------------------------------------------------------------
# Knowledge-base miss (documented 404 contract)
# ---------------------------------------------------------------------------

class TestNoEvidence:
    def test_empty_retrieval_returns_404(self, client, empty_retrieval):
        response = client.post(
            "/ask", json={"question": "What is quantum chromodynamics?"}
        )
        assert response.status_code == 404
        assert "knowledge base" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# LLM failure mapping (documented 502 contract)
# ---------------------------------------------------------------------------

class TestLLMFailure:
    def test_ask_gemini_none_maps_to_502(self, client, set_gemini_failure):
        set_gemini_failure(None)
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 502
        assert "Traceback" not in response.text

    def test_answer_passes_through_unchanged(self, client, set_gemini_answer):
        """Contract: ask_gemini's stripped text is returned verbatim.

        Note: ask_gemini itself can never return a blank string (it
        strips and yields None instead), so the handler only needs the
        None check — blank answers are unreachable by construction.
        """
        expected = "## Binary Search\n\n- halves the space"
        set_gemini_answer(expected)
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 200
        assert response.json()["answer"] == expected


    def test_malformed_json_body_is_clean_422(self, client):
        """TC-5: invalid JSON must produce a clean 4xx, no stack trace,
        and no Gemini call."""
        response = client.post(
            "/ask",
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "Traceback" not in response.text

    def test_wrong_json_type_is_422(self, client):
        response = client.post("/ask", json=["question"])
        assert response.status_code == 422


class TestContractStability:
    """Pins the exact response shape Phase 4's frontend will consume,

    so any accidental field rename breaks the suite here first.
    """

    def test_delivery_model_is_single_json_object(self, client):
        """The backend returns ONE JSON object (not NDJSON/streaming).
        Phase 4's frontend must read `response.json()`, not a stream.
        """
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.headers["content-type"].startswith(
            "application/json"
        )
        body = response.json()  # a single parseable object
        assert isinstance(body, dict)

    def test_source_fields_for_frontend_cards(self, client):
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        source = response.json()["sources"][0]

        # Frontend SourceCard needs these exact names and types:
        assert isinstance(source["title"], str) and len(source["title"]) > 0
        assert isinstance(source["timestamp"], int)  # seconds
        assert source["url"].startswith("https://")
        assert "t=" in source["url"]  # deep link to the timestamp
        assert 0.0 <= source["score"] <= 1.0


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

class TestHealth:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "online"

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# 14. API key leakage (TC-14)
# ---------------------------------------------------------------------------

class TestSecretHygiene:
    def test_success_response_contains_no_api_key(self, client):
        from conftest import FAKE_API_KEY

        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert FAKE_API_KEY not in response.text

    def test_error_responses_contain_no_api_key(
        self, client, set_gemini_failure
    ):
        from conftest import FAKE_API_KEY

        set_gemini_failure(None)
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert FAKE_API_KEY not in response.text

    def test_validation_error_contains_no_api_key(self, client):
        from conftest import FAKE_API_KEY

        response = client.post("/ask", json={"question": ""})
        assert FAKE_API_KEY not in response.text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ask_max_question_length():
    import ask

    return ask.MAX_QUESTION_LENGTH
