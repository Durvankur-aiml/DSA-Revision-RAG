"""ALGOFORGE Retrieval Package.

Exposes candidate search (vector + lexical), static lexical caching,
and hybrid fusion scoring.
"""

from .lexical import (
    normalize_text,
    tokenize,
    extract_topic,
    title_exact_match,
    text_match_score,
    lexical_retrieve,
    _get_lexical_cache,
)
from .hybrid import (
    retrieve_chunks,
    semantic_retrieve,
    apply_production_selection,
    get_retrieval_score,
    get_retrieval_diagnostics,
    load_all_points,
)

__all__ = [
    "normalize_text",
    "tokenize",
    "extract_topic",
    "title_exact_match",
    "text_match_score",
    "lexical_retrieve",
    "_get_lexical_cache",
    "retrieve_chunks",
    "semantic_retrieve",
    "apply_production_selection",
    "get_retrieval_score",
    "get_retrieval_diagnostics",
    "load_all_points",
]
