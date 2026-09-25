"""Concurrency isolation tests for request-local retrieval scores.

The retrieval-score snapshot MUST be per-request (ContextVar), not a
module-level global: two concurrent /ask requests with different
questions must never see each other's scores. These tests fail loudly
if the score storage ever regresses to shared mutable global state.

Fully offline: retrieval and Gemini are stubbed; no Qdrant, no network.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import ask


@pytest.fixture(autouse=True)
def _clean_score_var():
    """Keep this module's tests from inheriting or leaking score state."""
    ask._set_retrieval_scores({})
    yield
    ask._set_retrieval_scores({})


# ---------------------------------------------------------------------------
# Unit level: raw ContextVar semantics
# ---------------------------------------------------------------------------


class TestContextVarUnit:
    def test_helper_replaces_snapshot(self):
        ask._set_retrieval_scores({"A": 1.0})
        ask._set_retrieval_scores({"B": 0.5})

        assert ask.get_retrieval_score("B") == 0.5
        # The second call replaces the snapshot; it does not merge.
        assert ask.get_retrieval_score("A") == 0.0

    def test_missing_point_id_scores_zero(self):
        ask._set_retrieval_scores({})
        assert ask.get_retrieval_score("NOPE") == 0.0

    def test_scores_do_not_leak_across_threads(self):
        """Two threads setting scores simultaneously must stay isolated.

        With a module-level dict this test is flaky-by-design: whichever
        thread sets last wins and the other reads 0.0. With a ContextVar
        it passes deterministically.
        """
        results = {}
        errors = []
        barrier = threading.Barrier(2, timeout=10)

        def worker(name, point_id, score):
            try:
                ask._set_retrieval_scores({point_id: score})
                barrier.wait()  # maximize overlap of both critical sections
                time.sleep(0.05)  # widen the race window
                results[name] = {
                    "own": ask.get_retrieval_score(point_id),
                    "foreign": ask.get_retrieval_score("FOREIGN_ID"),
                }
            except Exception as error:  # surfaced below, never hidden
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=("first", "FIRST_ID", 0.25)),
            threading.Thread(target=worker, args=("second", "SECOND_ID", 0.75)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not any(thread.is_alive() for thread in threads), "worker hung"
        assert errors == []

        assert results["first"] == {"own": 0.25, "foreign": 0.0}
        assert results["second"] == {"own": 0.75, "foreign": 0.0}


# ---------------------------------------------------------------------------
# API level: concurrent /ask requests with disjoint scores
# ---------------------------------------------------------------------------


def _chunk(point_id, title, start):
    return SimpleNamespace(
        id=point_id,
        payload={
            "video_title": title,
            "youtube_url": f"https://youtu.be/{point_id}",
            "start": start,
            "text": f"transcript text for {point_id}",
        },
    )


@pytest.fixture()
def dual_question_app(monkeypatch, main_module, set_gemini_answer):
    """App where two questions retrieve disjoint chunks with own scores."""
    set_gemini_answer("Grounded answer.")

    registry = {
        "What is binary search?": (
            [_chunk("BS1VIDEO", "BS-1. Binary Search Introduction", 86.0)],
            {"BS1VIDEO": 0.9001},
        ),
        "Explain 3 Sum.": (
            [_chunk("THREESUM", "3 Sum | Brute-Better-Optimal", 12.0)],
            {"THREESUM": 0.3003},
        ),
    }

    def _fake_retrieve(question, top_k=ask.TOP_K):
        chunks, scores = registry[question]
        # Production contract: the request's retrieve_chunks call stores
        # its own scores through the request-local ContextVar.
        ask._set_retrieval_scores(scores)
        return list(chunks)

    monkeypatch.setattr(ask, "retrieve_chunks", _fake_retrieve)
    monkeypatch.setattr(main_module, "retrieve_chunks", _fake_retrieve)

    return main_module


class TestConcurrentAskNoLeak:
    def test_concurrent_ask_requests_keep_scores_isolated(
        self, dual_question_app
    ):
        """10 rounds of two simultaneous /ask requests.

        Each response must carry exactly its own question's score. If
        score storage regresses to a shared global, the second request's
        snapshot clobbers the first and its sources read 0.0.
        """
        from fastapi.testclient import TestClient

        rounds = 10
        failures = []
        errors = []
        barrier = threading.Barrier(2, timeout=10)

        def ask_worker(question, expected_point_id, expected_score):
            try:
                client = TestClient(dual_question_app.app)
                barrier.wait()
                response = client.post("/ask", json={"question": question})
                body = response.json()
                sources = body["sources"]
                if len(sources) != 1:
                    failures.append(
                        f"{question}: expected 1 source, got {len(sources)}"
                    )
                    return
                source = sources[0]
                if source["score"] != pytest.approx(expected_score):
                    failures.append(
                        f"{question}: score leaked - expected "
                        f"{expected_score}, got {source['score']}"
                    )
                if expected_point_id not in source["url"]:
                    failures.append(
                        f"{question}: wrong source url {source['url']}"
                    )
            except Exception as error:  # surfaced below, never hidden
                errors.append(error)

        for _ in range(rounds):
            threads = [
                threading.Thread(
                    target=ask_worker,
                    args=("What is binary search?", "BS1VIDEO", 0.9001),
                ),
                threading.Thread(
                    target=ask_worker,
                    args=("Explain 3 Sum.", "THREESUM", 0.3003),
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=15)
                assert not thread.is_alive(), "/ask worker hung"

        assert errors == []
        assert failures == [], "score leakage detected:\n" + "\n".join(
            failures
        )
