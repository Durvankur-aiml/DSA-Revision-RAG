"""ALGOFORGE Agent Coordinator Data Models.

Defines strongly typed representations for execution steps, plans, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .models import QueryIntent


class StepOperation(str, Enum):
    """Supported operations in an agent execution plan."""

    RETRIEVE = "RETRIEVE"
    RERANK = "RERANK"
    GENERATE = "GENERATE"
    VERIFY = "VERIFY"


class StepStatus(str, Enum):
    """Lifecycle status of an individual execution step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class AgentStep:
    """A discrete execution step in an agent plan."""

    operation: StepOperation
    status: StepStatus = StepStatus.PENDING
    latency_seconds: float = 0.0
    error: Optional[str] = None
    result_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "status": self.status.value,
            "latency_seconds": round(self.latency_seconds, 6),
            "error": self.error,
            "result_metadata": dict(self.result_metadata),
        }


@dataclass
class AgentPlan:
    """Deterministic sequence of operations planned for a user query."""

    intent: QueryIntent
    steps: List[AgentStep] = field(default_factory=list)
    is_out_of_scope: bool = False
    description: str = ""

    def summary(self) -> str:
        """Compact string representation, e.g. 'retrieve→rerank→generate→verify'."""
        if self.is_out_of_scope:
            return "stop_out_of_scope"
        if not self.steps:
            return "empty"
        return "→".join(s.operation.value.lower() for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "summary": self.summary(),
            "is_out_of_scope": self.is_out_of_scope,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class AgentExecutionResult:
    """Final result produced by the Agent Coordinator execution engine."""

    final_answer: str
    plan: AgentPlan
    completed_steps: List[AgentStep] = field(default_factory=list)
    verification_status: Optional[str] = None
    recovery_attempted: bool = False
    generation_attempts: int = 0
    final_status: str = "SUCCESS"
    total_latency_seconds: float = 0.0
    sources: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "plan": self.plan.to_dict(),
            "completed_steps": [s.to_dict() for s in self.completed_steps],
            "verification_status": self.verification_status,
            "recovery_attempted": self.recovery_attempted,
            "generation_attempts": self.generation_attempts,
            "final_status": self.final_status,
            "total_latency_seconds": round(self.total_latency_seconds, 6),
            "sources_count": len(self.sources),
            "error": self.error,
            "metadata": dict(self.metadata),
        }
