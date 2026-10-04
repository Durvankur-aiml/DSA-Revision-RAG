"""Shared pytest fixtures for Mission Anthropic backend tests.

Strategy
--------
``Backend/ask.py`` performs module-level side effects on import:
connecting to Qdrant, loading the SentenceTransformer, and failing hard
(via ``sys.exit``) if GEMINI_API_KEY or the Qdrant store are missing.
To keep this suite deterministic and fully offline, ``conftest``:

1. Sets a FAKE ``GEMINI_API_KEY`` before importing ``ask`` (the leak
   tests then assert this fake secret never reaches API responses).
2. Stubs ``qdrant_client`` and ``sentence_transformers`` in
   ``sys.modules`` so the import-time validation passes without any
   real database, model weights, or network access.

Production Gemini calls go through the module attribute
``ask.genai_client``; tests replace it with a scriptable fake
(``gemini_mock``). NOTE: ``main.py`` imports handlers *by value*
(``from ask import retrieve_chunks``), so fixtures that fake retrieval
or generation patch BOTH ``ask`` and ``main`` namespaces.

No test in this suite performs a real network call.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Route the observability layer's rotating perf log into a throwaway
# session directory BEFORE any application module is imported (obs.py
# configures itself at import time). Keeps the test run fully offline
# and prevents tests from writing into the repo's logs/ directory.
os.environ["ALGOFORGE_LOG_DIR"] = tempfile.mkdtemp(
    prefix="algoforge-test-logs-"
)

# Fake credential used as the production module's api_key for the whole
# test session. Leak tests assert this never appears in responses.
FAKE_API_KEY = "test-fake-api-key-ABCDEF123456"

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _install_offline_stubs() -> None:
    """Stub heavy external libraries BEFORE ask.py is imported."""
    os.environ["GEMINI_API_KEY"] = FAKE_API_KEY

    # ---- qdrant_client ----------------------------------------------------
    if "qdrant_client" not in sys.modules:
        qdrant_client = types.ModuleType("qdrant_client")

        class _StubQdrantClient:
            """Satisfies ask.py's import-time validation only."""

            def __init__(self, *args, **kwargs):
                pass

            def get_collections(self):
                name = types.SimpleNamespace(name="striver_a2z")
                return types.SimpleNamespace(collections=[name])

            def get_collection(self, collection_name):
                return types.SimpleNamespace(points_count=4850)

            def close(self):
                pass

        qdrant_client.QdrantClient = _StubQdrantClient

        qdrant_models = types.ModuleType("qdrant_client.models")
        qdrant_models.Distance = types.SimpleNamespace(COSINE="cosine")
        qdrant_models.VectorParams = lambda **kw: kw
        qdrant_models.PointStruct = lambda **kw: kw
        qdrant_client.models = qdrant_models
        sys.modules["qdrant_client"] = qdrant_client
        sys.modules["qdrant_client.models"] = qdrant_models

    # ---- sentence_transformers --------------------------------------------
    if "sentence_transformers" not in sys.modules:
        st_module = types.ModuleType("sentence_transformers")

        class _FakeVector(list):
            """Mimics the numpy array returned by SentenceTransformer.encode.

            ask.py's import-time validation calls .tolist() on the result.
            """

            def tolist(self):
                return list(self)

        class _StubSentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, text):
                return _FakeVector([0.0] * 384)

        st_module.SentenceTransformer = _StubSentenceTransformer
        sys.modules["sentence_transformers"] = st_module


_install_offline_stubs()

# Import the application modules once, with the stubs in place.
sys.path.insert(0, str(BACKEND_DIR))
import ask  # noqa: E402
import obs  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Gemini SDK surface
# ---------------------------------------------------------------------------

class FakeInteraction:
    """Mimics google.genai.interactions.Interaction for tests.

    ask.py only uses: .id, .status, .output_text, .error
    """

    def __init__(self, id="fake-interaction", status="in_progress",
                 output_text=None, error=None):
        self.id = id
        self.status = status
        self.output_text = output_text
        self.error = error


class TransientError(Exception):
    """Exception classified as transient by ask._is_transient_gemini_error."""

    def __init__(self, message):
        super().__init__(message)


def transient_503():
    return TransientError(
        "Error code: 503 - {'error': {'message': 'The model is currently "
        "experiencing high demand.', 'code': 'service_unavailable'}}"
    )


def invalid_request_400():
    return TransientError(
        "Error code: 400 - {'error': {'message': 'Request contains an "
        "invalid argument.', 'code': 'invalid_request'}}"
    )


def daily_quota_429():
    return TransientError(
        "Error code: 429 - {'error': {'message': 'Rate limit exceeded for "
        "model gemini-3.7-flash (limit: 20 requests per day on Free Tier). "
        "Please retry in 21s.', 'code': 'too_many_requests'}}"
    )


class FakeInteractionsAPI:
    """Scriptable stand-in for client.interactions.

    ``get_script`` items are either Exception instances (raised once by
    get()) or FakeInteraction objects (returned once). When the script
    is exhausted, get() keeps returning a stuck in_progress interaction,
    mirroring a hung generation.
    """

    def __init__(self, get_script=None, create_error=None):
        self.create_calls = 0
        self.get_calls = 0
        self.create_error = create_error
        self.get_script = list(get_script or [])
        self.created_ids = []
        self.create_kwargs_seen = []

    def create(self, **kwargs):
        self.create_calls += 1
        self.create_kwargs_seen.append(kwargs)
        if self.create_error is not None:
            error, self.create_error = self.create_error, None
            raise error
        interaction_id = f"fake-interaction-{self.create_calls}"
        self.created_ids.append(interaction_id)
        return FakeInteraction(id=interaction_id, status="in_progress")

    def get(self, id=None, **kwargs):  # noqa: A002 - matches SDK signature
        self.get_calls += 1
        if self.get_script:
            item = self.get_script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        # Script exhausted: interaction appears permanently stuck.
        return FakeInteraction(id="stuck-interaction", status="in_progress")

    def cancel(self, **kwargs):
        pass


class FakeGenaiClient:
    def __init__(self, interactions_api):
        self.interactions = interactions_api


class FakeClock:
    """Deterministic clock + sleep for deadline/retry logic."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds  # sleeping advances the fake clock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ask.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ask.time, "sleep", clock.sleep)
    return clock


@pytest.fixture()
def gemini_mock(monkeypatch):
    """Install a fresh FakeInteractionsAPI as ask.genai_client."""
    api = FakeInteractionsAPI()
    monkeypatch.setattr(ask, "genai_client", FakeGenaiClient(api))
    return api


def _sample_chunks():
    """Deterministic retrieval results shaped like Qdrant ScoredPoints.

    Ordered best-first, mirroring the live 'What is binary search?'
    retrieval observed during Phase 1 verification.
    """

    def _chunk(point_id, video_title, start, text, score):
        return types.SimpleNamespace(
            id=point_id,
            score=score,
            payload={
                "video_id": point_id,
                "video_title": video_title,
                "youtube_url": f"https://youtu.be/{point_id}",
                "start": start,
                "end": start + 90,
                "text": text,
                "chunk_index": 0,
            },
        )

    return [
        _chunk(
            "BS1VIDEO",
            "BS-1. Binary Search Introduction | Real Life Example",
            86.0,
            "Binary search works on sorted arrays by repeatedly halving "
            "the search space.",
            1.0,
        ),
        _chunk(
            "BS16VIDEO",
            "BS-16. Kth Missing Positive Number | Maths + Binary Search",
            261.0,
            "We can apply binary search on the answer to find the kth "
            "missing positive number.",
            0.7285,
        ),
        _chunk(
            "BS18VIDEO",
            "BS-18. Allocate Books or Book Allocation | Hard Binary Search",
            1060.0,
            "Allocate books to students using binary search on the "
            "answer space.",
            0.7112,
        ),
    ]


@pytest.fixture()
def sample_chunks():
    return _sample_chunks()


@pytest.fixture()
def main_module():
    """Import (once) and return the FastAPI app module."""
    import main

    return main


@pytest.fixture()
def retrieval_mock(monkeypatch, main_module):
    """Replace retrieve_chunks with a deterministic, recording fake.

    Patches both ask.py and main.py namespaces (main imports by value).
    Mirrors production score bookkeeping: the real retrieve_chunks
    streams each selected chunk's hybrid score into the per-request
    ContextVar, which the /ask handler reads when building sources.
    """
    calls = []
    chunks = _sample_chunks()

    def _fake_retrieve(question, top_k=ask.TOP_K):
        calls.append(question)
        selected = list(chunks[:top_k])
        # Mirror production: retrieve_chunks streams each selected chunk's
        # hybrid score through the same per-request ContextVar API, so the
        # test merges into the existing var instead of overwriting it.
        for hit in selected:
            ask._set_retrieval_scores(
                {**ask._retrieval_scores_var.get(), str(hit.id): float(hit.score)}
            )
        # Mirror production observability bookkeeping (obs.set_retrieval_meta
        # calls inside retrieve_chunks) using the real pure functions.
        topic = ask.extract_topic(question)
        exact = any(
            ask.title_exact_match(question, hit.payload.get("video_title", ""))
            >= 0.95
            for hit in selected
        )
        obs.set_retrieval_meta(
            strategy="hybrid",
            topic=topic,
            exact_topic=exact,
            candidate_count=len(selected),
        )
        return selected

    monkeypatch.setattr(ask, "retrieve_chunks", _fake_retrieve)
    monkeypatch.setattr(main_module, "retrieve_chunks", _fake_retrieve)
    return calls


@pytest.fixture()
def empty_retrieval(monkeypatch, main_module):
    """Force retrieval to return no results (knowledge-base miss)."""

    def _empty(question, top_k=ask.TOP_K):
        return []

    monkeypatch.setattr(ask, "retrieve_chunks", _empty)
    monkeypatch.setattr(main_module, "retrieve_chunks", _empty)
    return _empty


@pytest.fixture()
def set_gemini_answer(monkeypatch, main_module):
    """Force ask_gemini to return a canned answer (API-level stub)."""

    def _set(answer_text):
        stub = lambda message: answer_text  # noqa: E731
        monkeypatch.setattr(ask, "ask_gemini", stub)
        monkeypatch.setattr(main_module, "ask_gemini", stub)

    return _set


@pytest.fixture()
def set_gemini_failure(monkeypatch, main_module):
    """Force ask_gemini to fail (LLM outage at API level)."""

    def _set(return_value=None):
        stub = lambda message: return_value  # noqa: E731
        monkeypatch.setattr(ask, "ask_gemini", stub)
        monkeypatch.setattr(main_module, "ask_gemini", stub)

    return _set


@pytest.fixture()
def client(retrieval_mock, set_gemini_answer):
    """FastAPI TestClient with retrieval and Gemini stubbed offline."""
    set_gemini_answer("Grounded answer about binary search.")
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)
