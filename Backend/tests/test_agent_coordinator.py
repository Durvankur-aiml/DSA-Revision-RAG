"""Deterministic unit and integration tests for the ALGOFORGE Agent Coordinator.

Tests all 31 scenarios required by the specification:
  Plan Tests: 1-8
  Execution Tests: 9-15
  Recovery Tests: 16-19
  Safety Tests: 20-25
  Determinism Tests: 26-27
  Out of Scope Tests: 28-30
  API Compatibility: 31
"""

from __future__ import annotations

import socket
import types
from typing import Any, List
import pytest

from agent.coordinator import (
    AgentCoordinator,
    coordinate_query,
    get_default_coordinator,
    reset_default_coordinator,
)
from agent.coordinator_models import (
    AgentExecutionResult,
    AgentPlan,
    AgentStep,
    StepOperation,
    StepStatus,
)
from agent.models import AgentDecision, QueryIntent
from citations.models import CitationVerificationResult, VerificationStatus
from llm.base import LLMResponse, LLMTransientError
from llm.gateway import LLMGateway


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_coordinator():
    reset_default_coordinator()
    yield
    reset_default_coordinator()


def make_sample_chunks():
    def _chunk(point_id, title, text, start=0.0):
        return types.SimpleNamespace(
            id=point_id,
            score=0.95,
            payload={
                "video_id": point_id,
                "video_title": title,
                "youtube_url": f"https://youtu.be/{point_id}",
                "start": start,
                "end": start + 60.0,
                "text": text,
            },
        )

    return [
        _chunk("v1", "BS Intro", "Binary search halves search space on sorted array [1]."),
        _chunk("v2", "Two Pointers", "Two pointers move from ends toward center [2]."),
    ]


class FakeGateway(LLMGateway):
    def __init__(self, script: List[Any]):
        super().__init__(providers={})
        self.script = list(script)
        self.calls: List[str] = []

    def generate(self, request):
        prompt_text = request.prompt if hasattr(request, "prompt") else str(request)
        self.calls.append(prompt_text)
        if not self.script:
            raise RuntimeError("FakeGateway script exhausted!")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            provider="fake_provider",
            model="fake_model",
            latency_seconds=0.01,
        )


# ---------------------------------------------------------------------------
# 1-8. PLAN TESTS
# ---------------------------------------------------------------------------

class TestPlanGeneration:
    def test_concept_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        plan = coordinator.create_plan(decision)

        assert plan.summary() == "retrieve→rerank→generate→verify"
        assert [s.operation for s in plan.steps] == [
            StepOperation.RETRIEVE,
            StepOperation.RERANK,
            StepOperation.GENERATE,
            StepOperation.VERIFY,
        ]

    def test_explanation_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.EXPLANATION, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_problem_solving_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.PROBLEM_SOLVING, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_comparison_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.COMPARISON, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_code_debug_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.CODE_DEBUG, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_revision_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.REVISION, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_follow_up_plan(self):
        coordinator = AgentCoordinator()
        # Case A: context available
        decision = AgentDecision(intent=QueryIntent.FOLLOW_UP, retrieval_required=True)
        plan_with_context = coordinator.create_plan(
            decision, context={"chunks": make_sample_chunks(), "retrieval_required": False}
        )
        assert plan_with_context.summary() == "generate→verify"

        # Case B: contextual evidence unavailable
        plan_no_context = coordinator.create_plan(decision, context=None)
        assert plan_no_context.summary() == "retrieve→rerank→generate→verify"

    def test_out_of_scope_plan(self):
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.OUT_OF_SCOPE, retrieval_required=False)
        plan = coordinator.create_plan(decision)

        assert plan.is_out_of_scope is True
        assert plan.summary() == "stop_out_of_scope"
        assert len(plan.steps) == 0


# ---------------------------------------------------------------------------
# 9-15. EXECUTION TESTS
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def test_successful_execution_pipeline(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway(["Binary search divides search space [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            reranker_fn=lambda q, c, depth: c,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.PASS, valid=True, citation_count=1
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.final_status == "SUCCESS"
        assert result.final_answer == "Binary search divides search space [1]."
        assert len(result.completed_steps) == 4
        assert [s.status for s in result.completed_steps] == [
            StepStatus.SUCCESS,
            StepStatus.SUCCESS,
            StepStatus.SUCCESS,
            StepStatus.SUCCESS,
        ]
        assert result.recovery_attempted is False
        assert result.generation_attempts == 1

    def test_retrieval_failure(self):
        def failing_retriever(q, top_k):
            raise RuntimeError("Database unavailable")

        coordinator = AgentCoordinator(retriever_fn=failing_retriever)
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.final_status == "FAILED"
        assert "Retrieval failed" in result.error
        assert len(result.completed_steps) == 1
        assert result.completed_steps[0].operation == StepOperation.RETRIEVE
        assert result.completed_steps[0].status == StepStatus.FAILED

    def test_reranker_failure(self):
        chunks = make_sample_chunks()

        def failing_reranker(q, c, depth):
            raise RuntimeError("Cross-Encoder CUDA out of memory")

        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            reranker_fn=failing_reranker,
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.final_status == "FAILED"
        assert "Reranking failed" in result.error
        assert len(result.completed_steps) == 2
        assert result.completed_steps[1].operation == StepOperation.RERANK
        assert result.completed_steps[1].status == StepStatus.FAILED

    def test_llm_failure(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway([LLMTransientError("Provider rate limit reached", provider="gemini")])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.final_status == "FAILED"
        assert "Generation failed" in result.error
        assert len(result.completed_steps) == 3
        assert result.completed_steps[2].operation == StepOperation.GENERATE
        assert result.completed_steps[2].status == StepStatus.FAILED

    def test_verification_pass(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway(["Binary search divides search space [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.PASS, valid=True, citation_count=1
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.verification_status == "PASS"
        assert result.final_status == "SUCCESS"
        assert result.recovery_attempted is False

    def test_verification_warning(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway(["Binary search operates on sorted array."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.WARNING,
                valid=True,
                citation_count=0,
                warnings=["DSA claims without citations"],
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.verification_status == "WARNING"
        assert result.final_status == "SUCCESS"
        assert result.final_answer == "Binary search operates on sorted array."
        assert result.recovery_attempted is False

    def test_verification_fail_triggers_safe_handling(self):
        chunks = make_sample_chunks()
        # Fail with no recovery possible (e.g. max attempts reached or regeneration disabled)
        gateway = FakeGateway(["Hallucinated claims [99].", "Still hallucinated [99]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.FAIL,
                valid=False,
                citation_count=1,
                cleaned_answer="Hallucinated claims.",
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        # Recovers sanitized answer from verifier
        assert result.recovery_attempted is True
        assert "[99]" not in result.final_answer
        assert result.final_answer == "Hallucinated claims."


# ---------------------------------------------------------------------------
# 16-19. RECOVERY TESTS
# ---------------------------------------------------------------------------

class TestBoundedRecovery:
    def test_fail_then_regeneration_pass(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway([
            "Initial answer with bad citation [99].",
            "Regenerated answer with valid citation [1].",
        ])

        verif_script = [
            CitationVerificationResult(
                status=VerificationStatus.FAIL,
                valid=False,
                citation_count=1,
                cleaned_answer="Initial answer with bad citation.",
            ),
            CitationVerificationResult(
                status=VerificationStatus.PASS,
                valid=True,
                citation_count=1,
            ),
        ]

        def scriptable_verifier(ans, c):
            return verif_script.pop(0)

        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=scriptable_verifier,
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.recovery_attempted is True
        assert result.generation_attempts == 2
        assert result.verification_status == "PASS"
        assert result.final_answer == "Regenerated answer with valid citation [1]."

    def test_fail_then_regeneration_warning(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway([
            "Bad citation [99].",
            "Acceptable revision answer.",
        ])

        verif_script = [
            CitationVerificationResult(
                status=VerificationStatus.FAIL,
                valid=False,
                citation_count=1,
                cleaned_answer="Bad citation.",
            ),
            CitationVerificationResult(
                status=VerificationStatus.WARNING,
                valid=True,
                citation_count=0,
                warnings=["Low citation density"],
            ),
        ]

        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: verif_script.pop(0),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.recovery_attempted is True
        assert result.generation_attempts == 2
        assert result.verification_status == "WARNING"
        assert result.final_answer == "Acceptable revision answer."

    def test_fail_then_regeneration_fail_stops_and_cleans(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway([
            "Bad citation 1 [99].",
            "Bad citation 2 [100].",
        ])

        verif_script = [
            CitationVerificationResult(
                status=VerificationStatus.FAIL,
                valid=False,
                citation_count=1,
                cleaned_answer="Bad citation 1.",
            ),
            CitationVerificationResult(
                status=VerificationStatus.FAIL,
                valid=False,
                citation_count=1,
                cleaned_answer="Bad citation 2.",
            ),
        ]

        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: verif_script.pop(0),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.recovery_attempted is True
        assert result.generation_attempts == 2
        assert result.verification_status == "FAIL"
        # Bounded recovery stops; returns safest cleaned answer without third regeneration
        assert result.final_answer == "Bad citation 2."
        assert len(gateway.calls) == 2

    def test_third_generation_is_strictly_impossible(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway([
            "Bad citation 1 [99].",
            "Bad citation 2 [100].",
            "Should never be reached [101].",
        ])

        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.FAIL, valid=False, cleaned_answer="Safe text."
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("What is binary search?", decision)

        assert result.generation_attempts == 2
        assert len(gateway.calls) == 2  # Exactly 2 calls made, never 3


# ---------------------------------------------------------------------------
# 20-25. SAFETY & BOUNDEDNESS TESTS
# ---------------------------------------------------------------------------

class TestSafetyAndIsolation:
    def test_no_direct_gemini_sdk_invocation(self, monkeypatch):
        import ask

        def forbid_sdk(*args, **kwargs):
            raise AssertionError("Coordinator called Gemini SDK directly!")

        monkeypatch.setattr(ask, "genai_client", forbid_sdk)

        chunks = make_sample_chunks()
        gateway = FakeGateway(["Answer from gateway [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.PASS, valid=True
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        res = coordinator.coordinate("Query", decision)
        assert res.final_status == "SUCCESS"

    def test_no_direct_groq_sdk_invocation(self, monkeypatch):
        class ForbidGroq:
            def __init__(self, *args, **kwargs):
                raise AssertionError("Coordinator instantiated Groq SDK directly!")

        monkeypatch.setattr("llm.groq_adapter.GroqAdapter", ForbidGroq)
        chunks = make_sample_chunks()
        gateway = FakeGateway(["Answer [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.PASS, valid=True
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        res = coordinator.coordinate("Query", decision)
        assert res.final_status == "SUCCESS"

    def test_llm_gateway_is_used_for_generation(self):
        chunks = make_sample_chunks()
        gateway = FakeGateway(["Grounded explanation [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.PASS, valid=True
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        coordinator.coordinate("Question", decision)

        assert len(gateway.calls) == 1
        assert "Question" in gateway.calls[0]

    def test_existing_citation_verifier_is_used(self):
        verifier_called = []

        def tracked_verifier(ans, c):
            verifier_called.append(ans)
            return CitationVerificationResult(status=VerificationStatus.PASS, valid=True)

        chunks = make_sample_chunks()
        gateway = FakeGateway(["Answer text [1]."])
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=gateway,
            verifier_fn=tracked_verifier,
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        coordinator.coordinate("Question", decision)

        assert len(verifier_called) == 1
        assert verifier_called[0] == "Answer text [1]."

    def test_no_network_call_during_planning(self, monkeypatch):
        def no_network(*args, **kwargs):
            raise AssertionError("Network called during plan creation!")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        coordinator = AgentCoordinator()
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        plan = coordinator.create_plan(decision)
        assert plan.summary() == "retrieve→rerank→generate→verify"

    def test_no_infinite_loops(self):
        chunks = make_sample_chunks()
        # Even with infinite FAIL results, execution terminates after attempt 2
        coordinator = AgentCoordinator(
            retriever_fn=lambda q, top_k: chunks,
            gateway=FakeGateway(["Bad 1", "Bad 2", "Bad 3", "Bad 4"]),
            verifier_fn=lambda ans, c: CitationVerificationResult(
                status=VerificationStatus.FAIL, valid=False, cleaned_answer="Cleaned."
            ),
        )
        decision = AgentDecision(intent=QueryIntent.CONCEPT, retrieval_required=True)
        result = coordinator.coordinate("Question", decision)

        assert result.generation_attempts == 2
        assert result.final_answer == "Cleaned."


# ---------------------------------------------------------------------------
# 26-27. DETERMINISM TESTS
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_produces_same_plan(self):
        coordinator = AgentCoordinator()
        decision1 = AgentDecision(intent=QueryIntent.PROBLEM_SOLVING, retrieval_required=True)
        decision2 = AgentDecision(intent=QueryIntent.PROBLEM_SOLVING, retrieval_required=True)

        plan1 = coordinator.create_plan(decision1)
        plan2 = coordinator.create_plan(decision2)

        assert plan1.summary() == plan2.summary()
        assert [s.operation for s in plan1.steps] == [s.operation for s in plan2.steps]

    def test_same_decision_produces_identical_plan_structure(self):
        coordinator = AgentCoordinator()
        for intent in QueryIntent:
            req = intent != QueryIntent.OUT_OF_SCOPE
            d = AgentDecision(intent=intent, retrieval_required=req)
            p1 = coordinator.create_plan(d)
            p2 = coordinator.create_plan(d)
            assert p1.summary() == p2.summary()
            assert p1.is_out_of_scope == p2.is_out_of_scope


# ---------------------------------------------------------------------------
# 28-30. OUT OF SCOPE TESTS
# ---------------------------------------------------------------------------

class TestOutOfScopeBehavior:
    def test_out_of_scope_performs_zero_retrieval(self):
        retrieval_called = []
        coordinator = AgentCoordinator(retriever_fn=lambda q, k: retrieval_called.append(q))
        decision = AgentDecision(intent=QueryIntent.OUT_OF_SCOPE, retrieval_required=False)
        result = coordinator.coordinate("What is the weather?", decision)

        assert len(retrieval_called) == 0
        assert result.final_status == "OUT_OF_SCOPE"

    def test_out_of_scope_performs_zero_reranking(self):
        rerank_called = []
        coordinator = AgentCoordinator(reranker_fn=lambda q, c, d: rerank_called.append(q))
        decision = AgentDecision(intent=QueryIntent.OUT_OF_SCOPE, retrieval_required=False)
        result = coordinator.coordinate("What is the weather?", decision)

        assert len(rerank_called) == 0
        assert result.final_status == "OUT_OF_SCOPE"

    def test_out_of_scope_performs_zero_llm_generation(self):
        gateway = FakeGateway(["Should never generate"])
        coordinator = AgentCoordinator(gateway=gateway)
        decision = AgentDecision(intent=QueryIntent.OUT_OF_SCOPE, retrieval_required=False)
        result = coordinator.coordinate("What is the weather?", decision)

        assert len(gateway.calls) == 0
        assert result.final_status == "OUT_OF_SCOPE"
        assert result.generation_attempts == 0


# ---------------------------------------------------------------------------
# 31. API COMPATIBILITY TEST
# ---------------------------------------------------------------------------

class TestApiCompatibility:
    def test_existing_ask_behavior_remains_compatible(self, client, sample_chunks, gemini_mock):
        gemini_mock.get_script = [
            types.SimpleNamespace(
                status="completed",
                output_text="Binary search halves the search space [1].",
            )
        ]

        response = client.post(
            "/ask",
            json={"question": "What is binary search?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0
        assert isinstance(data["answer"], str)
        assert data["answer"] != ""
