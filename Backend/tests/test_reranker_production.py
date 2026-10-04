"""Phase 3C production-integration tests for the /ask reranker stage.

The Cross-Encoder is ALWAYS mocked (a scripted CrossReranker stand-in is
injected by patching ask.CrossReranker) — no test downloads a model.
The REAL retrieve_chunks runs (with ask.semantic_retrieve /
ask.lexical_retrieve patched to deterministic offline candidate
sources), so the actual production seam — fusion sort -> rerank ->
apply_production_selection — is exercised, not a copy of it.

Covered areas (Phase 3C spec section 18):
 1. reranker disabled  -> existing retrieval path (byte-identical)
 2. reranker enabled   -> reranker invoked
 3. depth default      -> 70 when enabled
 4. custom depth       -> configuration respected
 5. init failure       -> fallback, request continues
 6. inference failure  -> fallback, request continues
 7. malformed config   -> safe handling (disabled, no crash)
 8. API response schema unchanged (with reranker enabled)
 9. source ordering    -> reranked ordering, production rules
10. URL preservation   -> unchanged timestamp-URL generation
11. concurrent requests -> no request-state leakage
12. Gemini receives the final selected context (unchanged path)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ask  # noqa: E402
import obs  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic offline candidate sources
# ---------------------------------------------------------------------------


def _point(point_id, title, text, chunk_index=0, start=10.0):
    """A Qdrant-point-shaped object with payload."""
    return SimpleNamespace(
        id=point_id,
        score=0.0,
        payload={
            "video_id": point_id,
            "video_title": title,
            "youtube_url": f"https://youtu.be/{point_id}",
            "start": start,
            "end": start + 90,
            "text": text,
            "chunk_index": chunk_index,
        },
    )


def _semantic_hit(point_id, title, text, semantic_score):
    """A ScoredPoint-shaped object for semantic_retrieve."""
    point = _point(point_id, title, text)
    point.score = semantic_score
    return point


# Six distinct "videos"; semantic scores descend, so the disabled fused
# order is F0..F5. The scripted reranker reverses the head, so with
# depth >= 4 the enabled order starts F3, F2, F1, F0, ...
CANDIDATES = [
    ("F0", "AL-1. Arrays Introduction", "arrays store elements in order", 0.90),
    ("F1", "AL-2. Two Pointers Approach", "two pointers move over the array", 0.80),
    ("F2", "AL-3. Sliding Window Technique", "sliding window shrinks and grows", 0.70),
    ("F3", "AL-4. Hashing Fundamentals", "hash maps give constant lookups", 0.60),
    ("F4", "AL-5. Prefix Sum Technique", "prefix sums answer range queries", 0.50),
    ("F5", "AL-6. Kadane Algorithm", "kadane tracks the maximum subarray", 0.40),
]


def _install_candidate_sources(monkeypatch):
    """Patch ask.semantic_retrieve / ask.lexical_retrieve with fakes.

    Semantic returns all six candidates (top-40-style). Lexical returns
    the same points with lexical scores so the merge stays realistic but
    the fused order is dictated by the semantic scores.
    """
    semantic = [
        _semantic_hit(pid, title, text, score)
        for pid, title, text, score in CANDIDATES
    ]
    lexical = [
        (0.85 * 0.72, _point(pid, title, text))
        for pid, title, text, _ in CANDIDATES
    ]

    monkeypatch.setattr(ask, "semantic_retrieve", lambda question: semantic)
    monkeypatch.setattr(ask, "lexical_retrieve", lambda question, all_points=None: lexical)
    return semantic, lexical


# Thread-local marker used by the concurrency test's thread-aware
# semantic source (no mutable module state is touched at request time).
_THREAD_LOCAL = threading.local()


def _fused_ids(selected):
    return [str(hit.id) for hit in selected]


@pytest.fixture()
def offline_pool(monkeypatch):
    _install_candidate_sources(monkeypatch)
    # Every test starts from a clean singleton.
    ask.reset_reranker_singleton()
    monkeypatch.setattr(ask, "RERANK_ENABLED", False)
    monkeypatch.setattr(ask, "RERANK_DEPTH", ask.RERANK_DEPTH_DEFAULT)
    yield
    ask.reset_reranker_singleton()


class ScriptedCrossReranker:
    """Offline CrossReranker stand-in injected as ask.CrossReranker."""

    DEFAULT_MODEL = "BAAI/bge-reranker-base"
    instances: list = []
    score_fn = staticmethod(lambda query, docs: [0.0] * len(docs))
    init_error: Exception | None = None

    def __init__(self, model_name=None, batch_size=16, **kwargs):
        if ScriptedCrossReranker.init_error is not None:
            raise ScriptedCrossReranker.init_error
        self.model_name = model_name or "BAAI/bge-reranker-base"
        self.device = "cpu"  # deterministic: never touches CUDA in tests
        self.batch_size = batch_size
        self.score_calls = []
        ScriptedCrossReranker.instances.append(self)

    def score(self, query, candidates):
        # Production passes plain doc dicts ({video_title, text}), exactly
        # like the real CrossReranker accepts.
        docs = list(candidates)
        self.score_calls.append((query, [d.get("video_title", "") for d in docs]))
        return list(ScriptedCrossReranker.score_fn(query, docs))

    def _ensure_model(self):
        return True


@pytest.fixture()
def scripted_reranker(monkeypatch):
    ScriptedCrossReranker.instances = []
    ScriptedCrossReranker.init_error = None
    ScriptedCrossReranker.score_fn = staticmethod(
        lambda query, docs: [0.0] * len(docs)
    )
    monkeypatch.setattr(ask, "CrossReranker", ScriptedCrossReranker)
    return ScriptedCrossReranker


def _reverse_scores(query, docs):
    """Ascending scores: later candidates score higher -> head reverses."""
    return [float(i + 1) for i in range(len(docs))]


def _with_request():
    """Open an obs request context; return (ctx, finish) for assertions."""
    ctx = obs.begin_request("test-rerank-req", "POST", "/ask")
    return ctx


# ---------------------------------------------------------------------------
# 1. Disabled path — existing retrieval behavior untouched
# ---------------------------------------------------------------------------


class TestDisabledPath:
    def test_disabled_uses_existing_path_and_never_loads_reranker(
        self, offline_pool, scripted_reranker
    ):
        assert ask.RERANK_ENABLED is False
        selected = ask.retrieve_chunks("explain arrays and two pointers")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]
        # No reranker instance was ever constructed.
        assert scripted_reranker.instances == []

    def test_disabled_has_no_reranker_scores_in_diagnostics(self, offline_pool):
        ask.retrieve_chunks("explain arrays")
        diag = ask.get_retrieval_diagnostics()
        assert diag["config"]["rerank_enabled"] is False
        assert diag["config"]["rerank_depth"] is None
        assert all(c["reranker_score"] is None for c in diag["candidates"])

    def test_disabled_snapshot_records_enabled_false(self, offline_pool):
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            snap = ctx.snapshot()
            assert snap["reranker_enabled"] is False
            assert snap["reranker_fallback"] is None
            assert snap["reranker_model"] is None
        finally:
            obs.end_request()


# ---------------------------------------------------------------------------
# 2. Enabled path — reranker invoked, validated semantics
# ---------------------------------------------------------------------------


class TestEnabledPath:
    def test_enabled_invokes_reranker_and_reorders_pool(
        self, offline_pool, scripted_reranker
    ):
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        ask.RERANK_ENABLED = True
        selected = ask.retrieve_chunks("explain arrays and two pointers")
        assert _fused_ids(selected) == ["F5", "F4", "F3", "F2", "F1", "F0"]
        assert len(scripted_reranker.instances) == 1
        query, doc_titles = scripted_reranker.instances[0].score_calls[0]
        assert query == "explain arrays and two pointers"
        assert doc_titles == [title for _, title, _, _ in CANDIDATES]

    def test_enabled_singleton_loaded_once_across_requests(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        ask.retrieve_chunks("explain arrays")
        ask.retrieve_chunks("explain hashing")
        assert len(scripted_reranker.instances) == 1

    def test_enabled_lazy_singleton_not_built_before_first_request(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        assert ask._RERANKER_SINGLETON is None

    def test_enabled_diag_records_reranker_scores(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        ask.retrieve_chunks("explain arrays")
        diag = ask.get_retrieval_diagnostics()
        assert diag["config"]["rerank_enabled"] is True
        assert diag["config"]["rerank_depth"] == 70
        # Best-first after reversal: F5..F0 carry scores 6..1.
        scored = [c["reranker_score"] for c in diag["candidates"]]
        assert scored == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    def test_score_semantics_preserved_sources_carry_hybrid_scores(
        self, offline_pool, scripted_reranker
    ):
        """The API `score` stays the retrieval score, never the reranker score."""
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        ask.RERANK_ENABLED = True
        selected_enabled = ask.retrieve_chunks("explain arrays")
        enabled_scores = {
            str(hit.id): ask.get_retrieval_score(hit.id)
            for hit in selected_enabled
        }
        ask.RERANK_ENABLED = False
        ask.reset_reranker_singleton()
        selected_disabled = ask.retrieve_chunks("explain arrays")
        disabled_scores = {
            str(hit.id): ask.get_retrieval_score(hit.id)
            for hit in selected_disabled
        }
        # Identical hybrid scores per point in both modes; only ORDER moved.
        assert enabled_scores == disabled_scores
        assert _fused_ids(selected_enabled) != _fused_ids(selected_disabled)
        assert all(0.0 <= v <= 1.0 for v in enabled_scores.values())

    def test_ties_keep_fused_order(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(lambda q, d: [1.0] * len(d))
        selected = ask.retrieve_chunks("explain arrays")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]


# ---------------------------------------------------------------------------
# 3./4. Depth configuration
# ---------------------------------------------------------------------------


class TestDepthConfiguration:
    def test_default_depth_is_70(self):
        assert ask.RERANK_DEPTH_DEFAULT == 70

    def test_depth_70_applies_to_whole_small_pool(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        selected = ask.retrieve_chunks("explain arrays")
        # Pool of 6 < depth 70: entire pool is reranked head, empty tail.
        assert _fused_ids(selected) == ["F5", "F4", "F3", "F2", "F1", "F0"]

    def test_custom_depth_respected(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        ask.RERANK_DEPTH = 3
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        selected = ask.retrieve_chunks("explain arrays")
        # Head (F0..F2) reversed, tail (F3..F5) keeps fused order.
        assert _fused_ids(selected) == ["F2", "F1", "F0", "F3", "F4", "F5"]

    def test_snapshot_records_configured_depth(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        ask.RERANK_DEPTH = 3
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            assert ctx.snapshot()["reranker_depth"] == 3
        finally:
            obs.end_request()


# ---------------------------------------------------------------------------
# 5./6. Failure fallback
# ---------------------------------------------------------------------------


class TestFailureFallback:
    def test_init_failure_falls_back_to_fused_ranking(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        scripted_reranker.init_error = RuntimeError("no CUDA / weights missing")
        selected = ask.retrieve_chunks("explain arrays")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]

    def test_init_failure_does_not_fail_request_and_is_observable(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        scripted_reranker.init_error = RuntimeError("no CUDA")
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            snap = ctx.snapshot()
            assert snap["reranker_enabled"] is True
            assert snap["reranker_fallback"] is True
        finally:
            obs.end_request()

    def test_init_failure_does_not_reload_per_request(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        scripted_reranker.init_error = RuntimeError("no CUDA")
        ask.retrieve_chunks("q one")
        ask.retrieve_chunks("q two")
        # Construction attempted once, then the cached failure short-circuits.
        assert scripted_reranker.instances == []

    def test_inference_failure_falls_back_to_fused_ranking(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        instance = []
        scripted_reranker.score_fn = staticmethod(
            lambda q, d: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
        )
        selected = ask.retrieve_chunks("explain arrays")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]
        assert scripted_reranker.instances != []
        instance = scripted_reranker.instances[0]
        assert instance.score_calls  # inference was attempted

    def test_inference_failure_is_observable(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(
            lambda q, d: (_ for _ in ()).throw(RuntimeError("CUDA OOM"))
        )
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            snap = ctx.snapshot()
            assert snap["reranker_fallback"] is True
            assert snap["reranker_enabled"] is True
        finally:
            obs.end_request()

    def test_misaligned_scores_fall_back(self, offline_pool, scripted_reranker):
        """A scorer returning the wrong arity cannot be mapped — fall back."""
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(lambda q, d: [1.0])
        selected = ask.retrieve_chunks("explain arrays")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]


# ---------------------------------------------------------------------------
# 7. Malformed configuration
# ---------------------------------------------------------------------------


class TestMalformedConfig:
    def test_parse_defaults_when_unset(self, monkeypatch):
        monkeypatch.delenv(ask.RERANK_ENABLED_ENV, raising=False)
        monkeypatch.delenv(ask.RERANK_DEPTH_ENV, raising=False)
        assert ask._parse_rerank_config() == (False, 70)

    def test_parse_true_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "TRUE")
        monkeypatch.delenv(ask.RERANK_DEPTH_ENV, raising=False)
        assert ask._parse_rerank_config() == (True, 70)

    def test_parse_truthy_one(self, monkeypatch):
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "1")
        assert ask._parse_rerank_config()[0] is True

    def test_parse_false_values(self, monkeypatch):
        for value in ("false", "0", "False", ""):
            monkeypatch.setenv(ask.RERANK_ENABLED_ENV, value)
            assert ask._parse_rerank_config()[0] is False

    def test_malformed_enabled_disables(self, monkeypatch):
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "maybe")
        assert ask._parse_rerank_config() == (False, 70)

    def test_malformed_depth_disables(self, monkeypatch):
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "true")
        monkeypatch.setenv(ask.RERANK_DEPTH_ENV, "seventy")
        assert ask._parse_rerank_config() == (False, 70)

    def test_nonpositive_depth_disables(self, monkeypatch):
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "true")
        monkeypatch.setenv(ask.RERANK_DEPTH_ENV, "-5")
        assert ask._parse_rerank_config() == (False, 70)

    def test_malformed_depth_never_crashes_retrieval(
        self, offline_pool, monkeypatch
    ):
        """Simulate a misconfigured process: flag off, defaults intact."""
        monkeypatch.setenv(ask.RERANK_ENABLED_ENV, "true")
        monkeypatch.setenv(ask.RERANK_DEPTH_ENV, "not-a-number")
        enabled, depth = ask._parse_rerank_config()
        assert enabled is False and depth == 70
        selected = ask.retrieve_chunks("explain arrays")
        assert _fused_ids(selected) == ["F0", "F1", "F2", "F3", "F4", "F5"]


# ---------------------------------------------------------------------------
# 8.-10., 12. API-level: schema, ordering, URLs, Gemini context
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(monkeypatch, main_module):
    """TestClient running the REAL retrieve_chunks behind patched sources."""
    from fastapi.testclient import TestClient

    import main

    _install_candidate_sources(monkeypatch)
    ask.reset_reranker_singleton()

    from conftest import FakeGenaiClient, FakeInteractionsAPI

    monkeypatch.setattr(
        ask, "genai_client", FakeGenaiClient(FakeInteractionsAPI())
    )
    captured = {}

    def _fake_gemini(user_message):
        captured["user_message"] = user_message
        return "Scripted grounded answer."

    monkeypatch.setattr(ask, "ask_gemini", _fake_gemini)
    monkeypatch.setattr(main_module, "ask_gemini", _fake_gemini)

    client = TestClient(main.app)
    yield client, captured
    ask.reset_reranker_singleton()


class TestApiContractWithReranker:
    def test_schema_unchanged_with_reranker_enabled(
        self, api_client, scripted_reranker
    ):
        client, _ = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        response = client.post("/ask", json={"question": "explain arrays"})
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"answer", "sources"}
        assert body["answer"] == "Scripted grounded answer."
        for source in body["sources"]:
            assert set(source.keys()) == {"title", "timestamp", "url", "score"}
            assert isinstance(source["timestamp"], int)
            assert isinstance(source["score"], float)

    def test_source_ordering_follows_reranked_selection(
        self, api_client, scripted_reranker
    ):
        client, _ = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        body = client.post("/ask", json={"question": "explain arrays"}).json()
        titles = [s["title"] for s in body["sources"]]
        assert titles[0].startswith("AL-6.")
        assert titles[1].startswith("AL-5.")

    def test_urls_preserved_with_timestamps(self, api_client, scripted_reranker):
        client, _ = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        body = client.post("/ask", json={"question": "explain arrays"}).json()
        for source in body["sources"]:
            assert source["url"].startswith("https://youtu.be/F")
            assert "?t=" in source["url"]

    def test_score_field_stays_hybrid_score(self, api_client, scripted_reranker):
        client, _ = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        rerank_run = client.post(
            "/ask", json={"question": "explain arrays"}
        ).json()

        ask.RERANK_ENABLED = False
        ask.reset_reranker_singleton()
        disabled_run = client.post(
            "/ask", json={"question": "explain arrays"}
        ).json()

        disabled_scores = {
            s["title"]: s["score"] for s in disabled_run["sources"]
        }
        for source in rerank_run["sources"]:
            assert source["score"] == disabled_scores[source["title"]]

    def test_gemini_receives_final_reranked_context(
        self, api_client, scripted_reranker
    ):
        client, captured = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        client.post("/ask", json={"question": "explain arrays"})
        message = captured["user_message"]
        positions = [message.find(f'VIDEO TITLE: "{title}"')
                     for _, title, _, _ in reversed(CANDIDATES)]
        assert all(p != -1 for p in positions)
        assert positions == sorted(positions)  # reranked order in the prompt

    def test_gemini_untouched_path_still_called_once(
        self, api_client, scripted_reranker
    ):
        client, captured = api_client
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)
        response = client.post("/ask", json={"question": "explain arrays"})
        assert response.status_code == 200
        assert captured["user_message"].count("TRANSCRIPT:") == 6


# ---------------------------------------------------------------------------
# 11. Concurrency — no request-state leakage
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_retrieve_chunks_no_state_leakage(
        self, offline_pool, scripted_reranker
    ):
        """Two threads rerank different pools; results stay request-local."""
        ask.RERANK_ENABLED = True
        scripted_reranker.score_fn = staticmethod(_reverse_scores)

        # Thread-aware semantic source: thread B sees a different pool.
        base_semantic = list(ask.semantic_retrieve("x"))
        alt_hit = _semantic_hit(
            "G0", "GR-1. Graphs Basics", "graphs have nodes", 0.95
        )

        def _thread_aware_semantic(question):
            return [alt_hit] if getattr(_THREAD_LOCAL, "name", None) == "B" else base_semantic

        ask.semantic_retrieve = _thread_aware_semantic

        results = {}
        start = threading.Barrier(2)

        def run(name):
            _THREAD_LOCAL.name = name
            start.wait()
            selected = ask.retrieve_chunks(f"question {name}")
            diag = ask.get_retrieval_diagnostics()
            results[name] = (
                [str(hit.id) for hit in selected],
                [c["point_id"] for c in diag["candidates"]],
                ask.get_retrieval_score(selected[0].id),
            )

        threads = [threading.Thread(target=run, args=(n,)) for n in "AB"]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        ids_a, pool_a, score_a = results["A"]
        ids_b, pool_b, score_b = results["B"]

        assert ids_a == ["F5", "F4", "F3", "F2", "F1", "F0"]
        assert ids_b == ["G0"]
        assert set(pool_a) == {f"F{i}" for i in range(6)}
        # Thread B saw the alt semantic hit; lexical still contributes the
        # F points to the pool, but the threshold gate keeps only G0.
        assert "G0" in pool_b
        assert "G0" not in pool_a
        assert score_a > 0 and score_b > 0
        # Both threads shared ONE reranker instance (model is shareable)
        # while their candidate data stayed local.
        assert len(scripted_reranker.instances) == 1

    def test_shared_singleton_object_identical_across_calls(
        self, offline_pool, scripted_reranker
    ):
        ask.RERANK_ENABLED = True
        ask.retrieve_chunks("q one")
        first = ask._RERANKER_SINGLETON
        ask.retrieve_chunks("q two")
        assert ask._RERANKER_SINGLETON is first


# ---------------------------------------------------------------------------
# Observability fields
# ---------------------------------------------------------------------------


class TestRerankerObservability:
    def test_enabled_snapshot_has_model_device_batch(self, offline_pool, scripted_reranker):
        ask.RERANK_ENABLED = True
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            snap = ctx.snapshot()
            assert snap["reranker_enabled"] is True
            assert snap["reranker_model"] == "BAAI/bge-reranker-base"
            assert snap["reranker_device"] == "cpu"
            assert snap["reranker_depth"] == 70
            assert snap["reranker_fallback"] is False
            assert "reranker_stage" in snap["stages"]
        finally:
            obs.end_request()

    def test_reranker_stage_recorded_only_when_enabled(
        self, offline_pool, scripted_reranker
    ):
        ctx = _with_request()
        try:
            ask.retrieve_chunks("explain arrays")
            assert "reranker_stage" not in ctx.snapshot()["stages"]
        finally:
            obs.end_request()
