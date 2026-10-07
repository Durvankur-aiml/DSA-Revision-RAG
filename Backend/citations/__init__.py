"""ALGOFORGE Citations & Source Formatting Package."""

from .formatter import format_sources, register_score_getter
from .models import (
    CitationItem,
    CitationVerificationResult,
    VerificationStatus,
)
from .verifier import (
    CitationVerifier,
    verify_citations,
)

__all__ = [
    "format_sources",
    "register_score_getter",
    "CitationItem",
    "CitationVerificationResult",
    "VerificationStatus",
    "CitationVerifier",
    "verify_citations",
]
