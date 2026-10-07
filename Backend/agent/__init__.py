"""ALGOFORGE Agent Layer.

Contains the Agent Router, Agent Coordinator, and execution models.
"""

from .models import QueryIntent, AgentDecision
from .router import AgentRouter, route_query
from .coordinator_models import (
    StepOperation,
    StepStatus,
    AgentStep,
    AgentPlan,
    AgentExecutionResult,
)
from .coordinator import (
    AgentCoordinator,
    get_default_coordinator,
    reset_default_coordinator,
    coordinate_query,
)

__all__ = [
    "QueryIntent",
    "AgentDecision",
    "AgentRouter",
    "route_query",
    "StepOperation",
    "StepStatus",
    "AgentStep",
    "AgentPlan",
    "AgentExecutionResult",
    "AgentCoordinator",
    "get_default_coordinator",
    "reset_default_coordinator",
    "coordinate_query",
]
