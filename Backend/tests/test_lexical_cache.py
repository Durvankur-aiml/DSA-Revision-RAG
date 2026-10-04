"""Static lexical cache: retrieval-equivalence and safety tests.

The optimization moved query-independent lexical work (title
normalization, title tokenization, transcript tokenization) from
request time to a one-time synchronized cache. These tests prove:

1. the cached path produces EXACTLY the same (score, point_id)
   candidates as the original formulas (TC-EQ);
2. the wrapper path (explicit points) matches the reference body;
3. the cache is built once and shared, never rebuilt per request
   (TC-CACHE);
4. the cache initialization lock actually serializes (TC-THREAD);
5. cache failure behavior mirrors the old failure behavior (TC-FAIL).

Fully offline: no Qdrant, no network. Points are lightweight fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ask


def _point(point_id, title, text):
    return SimpleNamespace(
        id=point_id,
        payload={
            "video_id": point_id,
            "video_title": title,
            "youtube_url": f"https://youtu.be/{point_id}",
            "start": 12.0,
            "text": text,
        },
    )


def _sample_points():
    """Small representative set: intro, problem, BST, unrelated, empty text."""
    return [
        _point(
            "BS1VIDEO",
            "BS-1. Binary Search Introduction | Real Life Example",
            "Binary search works on sorted arrays by repeatedly halving "
            "the search space.",
        ),
        _point(
            "BS16VIDEO",
            "BS-16. Kth Missing Positive Number | Maths + Binary Search",
            "We can apply binary search on the answer to find the kth "
            "missing positive number.",
        ),
        _point(
            "L39VIDEO",
            "L39. Introduction to Binary Search Tree | BST",
            "A binary search tree orders nodes so left is smaller and "
            "right is larger.",
        ),
        _point(
            "G32VIDEO",
            "G-32. Dijkstra's Algorithm - Using Priority Queue",
            "Dijkstra computes shortest paths using a priority queue.",
        ),
        _point(
            "NOTEXT",
            "BS-18. Allocate Books or Book Allocation",
            "",
        ),
    ]


QUERIES = [
    "What is binary search?",                 # broad topic, exact intro
    "What is a binary search tree?",          # longer distinct topic
    "Explain 3 Sum.",                         # numeric sum handling
    "Explain Two Sum problem.",               # numeric sum, partial
    "How does Kadane's algorithm work?",      # known topic list
    "Explain Kth Missing Positive Number.",   # fallback topic (known limitation)
    "What is the time complexity of merge sort?",  # multiple words
    "Dijkstra",                               # single token, unrelated set
    "Explain recursion.",                     # no exact title match present
    "xyzzy irrelevant nonsense tokens",       # weak/zero lexical signal
]


# ---------------------------------------------------------------------------
# TC-EQ: cached path == original formulas
# ---------------------------------------------------------------------------


class TestRetrievalEquivalence:
    def test_cached_path_matches_reference_for_all_queries(self):
        points = _sample_points()

        results = ask._verify_lexical_equivalence(QUERIES, points=points)

        failures = [r for r in results if not r[1]]
        assert not failures, (
            "lexical cache changed retrieval results: "
            + "; ".join(f"{q} -> {d}" for q, ok, d in failures)
        )

    @pytest.mark.parametrize("query", QUERIES)
    def test_each_query_equivalent_individually(self, query):
        points = _sample_points()
        (q, ok, detail) = ask._verify_lexical_equivalence([query], points=points)[0]

        assert ok, f"{q}: {detail}"

    def test_wrapper_title_exact_match_matches_reference(self):
        """title_exact_match (wrapper) == _title_exact_match_original_body."""
        questions = [
            "What is binary search?",
            "What is a binary search tree?",
            "Explain 3 Sum.",
            "Explain Two Sum.",
            "Explain Kth Missing Positive Number.",
            "xyzzy nonsense",
        ]
        titles = [
            "BS-1. Binary Search Introduction | Real Life Example",
            "L39. Introduction to Binary Search Tree | BST",
            "3 Sum | Brute-Better-Optimal",
            "Two Sum | Check Pair Exists",
            "G-32. Dijkstra's Algorithm - Using Priority Queue",
            "",
            None,
        ]

        for question in questions:
            for title in titles:
                if title is None:
                    continue
                assert ask.title_exact_match(question, title) == (
                    ask._title_exact_match_original_body(question, title)
                ), (question, title)

    def test_title_exact_match_none_title_scores_zero(self):
        # Original body returned 0.0 via `if not video_title`.
        assert ask.title_exact_match("What is binary search?", "") == 0.0

    def test_hybrid_candidates_object_identity_preserved(self):
        """lexical_retrieve must return the SAME point objects it scored."""
        points = _sample_points()
        results = ask.lexical_retrieve("What is binary search?", points)

        assert results
        for score, point in results:
            assert point in points
            assert score > 0


# ---------------------------------------------------------------------------
# TC-CACHE: built once, shared, not rebuilt per request
# ---------------------------------------------------------------------------


class TestCacheBuildOnce:
    def test_cache_initialized_once_across_calls(self, monkeypatch):
        calls = {"scroll": 0}

        def fake_load():
            calls["scroll"] += 1
            return _sample_points()

        monkeypatch.setattr(ask, "load_all_points", fake_load)
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)

        c1 = ask._get_lexical_cache()
        c2 = ask._get_lexical_cache()
        c3 = ask.lexical_retrieve("What is binary search?")

        assert calls["scroll"] == 1
        assert c1 is c2
        # lexical_retrieve returned the cached point objects themselves.
        assert c3
        assert all(
            any(returned_point is p for p in c1["points"])
            for _, returned_point in c3
        )

    def test_cached_points_are_the_scored_points(self, monkeypatch):
        cached_points = _sample_points()
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)

        monkeypatch.setattr(
            ask, "load_all_points", lambda: list(cached_points)
        )

        results = ask.lexical_retrieve("What is binary search?")

        assert results
        for score, point in results:
            assert any(point is p for p in cached_points)

    def test_cache_failure_yields_empty_results_not_crash(self, monkeypatch):
        def failing_load():
            raise RuntimeError("qdrant unavailable")

        monkeypatch.setattr(ask, "load_all_points", failing_load)
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)

        cache = ask._get_lexical_cache()

        assert cache["points"] == []
        assert cache["features"] == []
        assert ask.lexical_retrieve("What is binary search?") == []

    def test_failed_init_retries_next_request_until_success(
        self, monkeypatch
    ):
        state = {"fail": True}

        def flaky_load():
            if state["fail"]:
                raise RuntimeError("temporary")
            return _sample_points()

        monkeypatch.setattr(ask, "load_all_points", flaky_load)
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)

        first = ask._get_lexical_cache()
        assert first["points"] == []

        state["fail"] = False
        second = ask._get_lexical_cache()
        assert len(second["points"]) == len(_sample_points())

    def test_feature_values_match_fresh_computation(self, monkeypatch):
        """Cached features must equal freshly computed ones, bit for bit."""
        points = _sample_points()
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)
        monkeypatch.setattr(ask, "load_all_points", lambda: list(points))

        cache = ask._get_lexical_cache()

        for point, features in zip(cache["points"], cache["features"]):
            payload = point.payload or {}
            title = payload.get("video_title", "")
            text = payload.get("text", "")

            assert features.title_normalized == ask.normalize_text(title)
            assert set(features.title_token_set) == set(ask.tokenize(title))
            assert set(features.text_token_set) == set(ask.tokenize(text))


# ---------------------------------------------------------------------------
# TC-THREAD: initialization synchronization
# ---------------------------------------------------------------------------


class TestCacheThreadSafety:
    def test_concurrent_first_access_initializes_once(self, monkeypatch):
        import threading
        import time as _time

        calls = {"scroll": 0}

        def slow_load():
            calls["scroll"] += 1
            _time.sleep(0.15)  # widen the race window
            return _sample_points()

        monkeypatch.setattr(ask, "load_all_points", slow_load)
        monkeypatch.setattr(ask, "_LEXICAL_CACHE", None)

        results = []
        errors = []

        def worker():
            try:
                results.append(ask._get_lexical_cache())
            except Exception as e:  # surfaced below
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert calls["scroll"] == 1
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# TC-IMMUT: features are immutable read-only structures
# ---------------------------------------------------------------------------


class TestCacheImmutability:
    def test_feature_dataclass_is_frozen(self):
        features = ask.__dict__["__PointLexicalFeatures"](
            title_normalized="t",
            title_token_set=frozenset(),
            text_token_set=frozenset(),
        )

        with pytest.raises(Exception):
            features.title_normalized = "mutated"

    def test_cached_title_token_set_is_frozenset(self):
        points = _sample_points()
        features = ask.__dict__["__build_point_lexical_features"](
            points[0].payload
        )

        assert isinstance(features.title_token_set, frozenset)
        assert isinstance(features.text_token_set, frozenset)
