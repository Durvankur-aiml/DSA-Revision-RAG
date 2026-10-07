"""ALGOFORGE Agent Router Data Models.

Defines the query intent classification and structured agent routing decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class QueryIntent(str, Enum):
    """Canonical query intents supported by ALGOFORGE Agent Router."""

    CONCEPT = "CONCEPT"
    EXPLANATION = "EXPLANATION"
    PROBLEM_SOLVING = "PROBLEM_SOLVING"
    COMPARISON = "COMPARISON"
    CODE_DEBUG = "CODE_DEBUG"
    REVISION = "REVISION"
    FOLLOW_UP = "FOLLOW_UP"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


@dataclass
class AgentDecision:
    """Structured decision produced by the Agent Router before retrieval/generation.

    Attributes
    ----------
    intent : QueryIntent
        The classified user intent.
    retrieval_required : bool
        Whether Qdrant hybrid retrieval must be executed.
    retrieval_depth : int
        Number of final chunks to pass to generation (default TOP_K = 6).
    reranking_required : bool
        Whether BGE cross-encoder reranking should be applied.
    response_mode : str
        Target response formatting ('standard', 'concise_revision', 'step_by_step',
        'comparison', 'code_debug', 'fast_fail', 'conversational').
    confidence : float
        Classification confidence score (0.0 to 1.0).
    reason : str
        Human-readable explanation of why this routing strategy was selected.
    suggested_prompt_prefix : Optional[str]
        Optional system guidance injected into prompt builder for this intent.
    metadata : Dict[str, Any]
        Additional intent-specific metadata.
    """

    intent: QueryIntent
    retrieval_required: bool
    retrieval_depth: int = 6
    reranking_required: bool = True
    response_mode: str = "standard"
    confidence: float = 1.0
    reason: str = ""
    suggested_prompt_prefix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
