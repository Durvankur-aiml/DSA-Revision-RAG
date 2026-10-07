"""ALGOFORGE Citation and Grounding Verification Models.

Defines tri-state verification status, citation items, and structured verification results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    """Tri-state verification outcome."""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CitationItem:
    """Individual citation extracted from generated answer."""
    raw_text: str
    source_index: Optional[int] = None
    is_valid: bool = True
    error_reason: Optional[str] = None
    warning_reason: Optional[str] = None
    claim_text: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    overlap_score: float = 0.0
    youtube_url: Optional[str] = None
    source_payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "source_index": self.source_index,
            "is_valid": self.is_valid,
            "error_reason": self.error_reason,
            "warning_reason": self.warning_reason,
            "claim_text": self.claim_text,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "overlap_score": round(self.overlap_score, 4),
            "youtube_url": self.youtube_url,
        }


@dataclass
class CitationVerificationResult:
    """Structured result returned by the CitationVerifier."""
    status: VerificationStatus
    valid: bool
    citation_count: int = 0
    valid_citation_count: int = 0
    invalid_citation_count: int = 0
    unsupported_claim_count: int = 0
    malformed_citation_count: int = 0
    citations: List[CitationItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cleaned_answer: Optional[str] = None
    latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "valid": self.valid,
            "citation_count": self.citation_count,
            "valid_citation_count": self.valid_citation_count,
            "invalid_citation_count": self.invalid_citation_count,
            "unsupported_claim_count": self.unsupported_claim_count,
            "malformed_citation_count": self.malformed_citation_count,
            "citations": [c.to_dict() for c in self.citations],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "cleaned_answer": self.cleaned_answer,
            "latency_seconds": round(self.latency_seconds, 6),
        }
