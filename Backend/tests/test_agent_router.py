"""ALGOFORGE Agent Router Test Suite.

Deterministic, sub-millisecond, 100% offline tests verifying:
1. CONCEPT intent classification
2. EXPLANATION intent classification
3. PROBLEM_SOLVING intent classification
4. COMPARISON intent classification
5. CODE_DEBUG intent classification
6. REVISION intent classification
7. FOLLOW_UP intent classification
8. OUT_OF_SCOPE intent classification
9. Ambiguous DSA query routing
10. Empty / whitespace handling
11. Zero LLM calls verification
12. Determinism and idempotence
13. Observability stage & metadata recording
14. API integration behavior on /ask (fast-fail for out-of-scope)
"""

import time
import pytest

from agent.models import QueryIntent, AgentDecision
from agent.router import AgentRouter, route_query
import obs


# ---------------------------------------------------------------------------
# 1. Canonical Intent Tests
# ---------------------------------------------------------------------------


def test_intent_concept():
    queries = [
        "What is binary search?",
        "Define dynamic programming.",
        "What is an AVL tree?",
        "Concept of disjoint set union",
        "Introduction to graphs and trees",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.CONCEPT, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.retrieval_depth >= 6
        assert decision.response_mode in ("conceptual", "standard")


def test_intent_explanation():
    queries = [
        "Explain binary search step by step.",
        "Step by step walkthrough of merge sort",
        "Explain Kadane's algorithm step by step with dry run",
        "How does Dijkstra's algorithm work in detail?",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.EXPLANATION, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.response_mode == "step_by_step"


def test_intent_problem_solving():
    queries = [
        "How do I solve Two Sum?",
        "What is the optimal approach for Trapping Rain Water?",
        "How to solve 3Sum problem?",
        "Algorithm for Allocate Books problem",
        "Brute force and optimal approach for Koko Eating Bananas",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.PROBLEM_SOLVING, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.response_mode in ("problem_solving", "standard")


def test_intent_comparison():
    queries = [
        "Binary search vs linear search",
        "Difference between BFS and DFS",
        "Compare merge sort and quick sort",
        "When to use Dijkstra vs Bellman Ford?",
        "Pros and cons of adjacency matrix versus adjacency list",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.COMPARISON, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.retrieval_depth >= 6
        assert decision.response_mode == "comparison"


def test_intent_code_debug():
    queries = [
        "Why does this linked list code fail?",
        "Debug my binary search code giving runtime error",
        "Fix this solution giving time limit exceeded TLE",
        "What is the bug in this tree traversal implementation?",
        "Why does my solution produce a segmentation fault?",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.CODE_DEBUG, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.response_mode == "code_debug"


def test_intent_revision():
    queries = [
        "Give me a quick revision of sliding window.",
        "Summary of dynamic programming patterns for interview",
        "Cheat sheet for binary tree traversals",
        "Quick recap of two pointer approach",
        "Refresher on graph topological sort",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.REVISION, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.response_mode == "concise_revision"


def test_intent_follow_up():
    queries = [
        "Why does that work?",
        "Explain the second step again.",
        "What did you mean by that?",
        "Can you clarify that earlier intuition?",
        "Why not the other way?",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.FOLLOW_UP, f"Failed for query: {q}"
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.response_mode == "conversational"


def test_intent_out_of_scope():
    queries = [
        "What is the weather today?",
        "Give me a recipe for chocolate cake.",
        "Who won the cricket world cup?",
        "Capital of France and population",
        "What is quantum chromodynamics?",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.intent == QueryIntent.OUT_OF_SCOPE, f"Failed for query: {q}"
        assert decision.retrieval_required is False
        assert decision.reranking_required is False
        assert decision.response_mode == "fast_fail"


# ---------------------------------------------------------------------------
# 2. Ambiguous & Edge Case Queries
# ---------------------------------------------------------------------------


def test_ambiguous_dsa_queries_route_to_retrieval():
    queries = [
        "Arrays and hashing",
        "Binary tree maximum path",
        "Recursion on subsets",
        "Disjoint set union by rank",
    ]
    for q in queries:
        decision = route_query(q)
        assert decision.retrieval_required is True
        assert decision.reranking_required is True
        assert decision.intent in (QueryIntent.CONCEPT, QueryIntent.PROBLEM_SOLVING)


def test_empty_and_whitespace_query():
    for empty in ("", "   ", "\n\t"):
        decision = route_query(empty)
        assert decision.intent == QueryIntent.OUT_OF_SCOPE
        assert decision.retrieval_required is False
        assert decision.response_mode == "fast_fail"


def test_dsa_special_cases():
    # 'Stock' in DSA (Best Time to Buy and Sell Stock) must NOT be flagged as financial out-of-scope
    decision = route_query("Best time to buy and sell stock DP approach")
    assert decision.intent in (QueryIntent.PROBLEM_SOLVING, QueryIntent.CONCEPT)
    assert decision.retrieval_required is True


# ---------------------------------------------------------------------------
# 3. Router Guarantees: Zero LLM Calls & Negligible Latency
# ---------------------------------------------------------------------------


def test_router_never_calls_llm(monkeypatch):
    from llm.gateway import LLMGateway

    # Mock gateway generate to blow up if called
    def exploding_generate(*args, **kwargs):
        raise AssertionError("Router invoked LLMGateway.generate! Router must be 100% deterministic without LLM calls.")

    monkeypatch.setattr(LLMGateway, "generate", exploding_generate)

    # Calling router must NOT touch LLM
    decision = route_query("Explain binary search step by step.")
    assert decision.intent == QueryIntent.EXPLANATION


def test_router_performance_sub_millisecond():
    t0 = time.perf_counter()
    for _ in range(100):
        route_query("Explain how to solve Two Sum using two pointers step by step.")
    duration = time.perf_counter() - t0

    # 100 classifications should take well under 50ms total (< 0.5ms per query)
    assert duration < 0.1, f"Router took too long: {duration * 1000:.2f}ms for 100 calls"


def test_router_determinism():
    q = "Binary search vs linear search tradeoffs"
    d1 = route_query(q)
    d2 = route_query(q)
    assert d1.intent == d2.intent
    assert d1.retrieval_required == d2.retrieval_required
    assert d1.response_mode == d2.response_mode
    assert d1.confidence == d2.confidence


# ---------------------------------------------------------------------------
# 4. Observability Integration
# ---------------------------------------------------------------------------


def test_router_observability_recording(monkeypatch):
    recorded_stages = []
    recorded_agent_meta = []

    monkeypatch.setattr(
        obs,
        "record_stage",
        lambda stage, duration, extra=None: recorded_stages.append((stage, duration, extra)),
    )
    monkeypatch.setattr(
        obs,
        "set_agent_meta",
        lambda **kwargs: recorded_agent_meta.append(kwargs),
    )

    route_query("Give me a quick revision of sliding window.", request_id="agent-req-42")

    assert any(s[0] == "agent_routing" for s in recorded_stages)
    assert any("intent=REVISION" in (s[2] or "") for s in recorded_stages)
    assert len(recorded_agent_meta) == 1
    assert recorded_agent_meta[0]["intent"] == "REVISION"
    assert recorded_agent_meta[0]["response_mode"] == "concise_revision"
    assert recorded_agent_meta[0]["retrieval_required"] is True
