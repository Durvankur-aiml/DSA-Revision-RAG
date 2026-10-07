"""ALGOFORGE Backward-Compatible Facade.

Re-exports symbols from the responsibility-driven packages:
  - retrieval/ (vector search, lexical search, hybrid fusion)
  - llm/ (prompt construction, Gemini background generation)
  - citations/ (timestamp URL formatting)

Also provides the interactive CLI entry point (main).
"""

import os
import sys
import time
import random
import types

# ------------------------------------------------------------
# 1. Retrieval Layer Exports
# ------------------------------------------------------------
from retrieval.lexical import (
    normalize_text,
    tokenize,
    extract_topic,
    _is_broad_topic_question,
    title_exact_match,
    _title_exact_match_original_body,
    text_match_score,
    _get_lexical_cache,
    _verify_lexical_equivalence,
    _TITLE_INTRO_MARKERS,
    __TitleScoringContext,
    __PointLexicalFeatures,
    __build_point_lexical_features,
    _LEXICAL_CACHE,
    _LEXICAL_CACHE_LOCK,
)
import retrieval.lexical as _lexical_mod

from retrieval.hybrid import (
    BASE_DIR,
    QDRANT_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    VECTOR_SIZE,
    TOP_K,
    SEMANTIC_CANDIDATES,
    LEXICAL_CANDIDATES,
    MIN_SCORE_THRESHOLD,
    TITLE_WEIGHT,
    SEMANTIC_WEIGHT,
    TEXT_WEIGHT,
    RERANK_ENABLED_ENV,
    RERANK_DEPTH_ENV,
    RERANK_DEPTH_DEFAULT,
    RERANK_ENABLED,
    RERANK_DEPTH,
    _parse_rerank_config,
    CrossReranker,
    reset_reranker_singleton,
    get_reranker_meta,
    _get_reranker,
    _rerank_candidates,
    _RERANKER_LOCK,
    _RERANKER_SINGLETON,
    _RERANKER_INIT_FAILED,
    _retrieval_scores_var,
    _retrieval_diagnostics_var,
    _set_retrieval_scores,
    _set_retrieval_diagnostics,
    get_retrieval_score,
    get_retrieval_diagnostics,
    fail,
    qdrant_client,
    embed_model,
    collection_info,
    load_all_points,
    semantic_retrieve,
    lexical_retrieve,
    apply_production_selection,
    retrieve_chunks,
    _perf,
    _PERF_ENABLED,
)
import retrieval.hybrid as _hybrid_mod

# ------------------------------------------------------------
# 2. LLM Layer Exports
# ------------------------------------------------------------
from llm.prompts import (
    MAX_QUESTION_LENGTH,
    SYSTEM_PROMPT,
    validate_question,
    classify_sources,
    build_prompt,
    register_prompt_helpers,
)
import llm.prompts as _prompts_mod

from llm.gemini import (
    GEMINI_MODEL,
    GEMINI_HTTP_TIMEOUT_MS,
    GEMINI_POLL_INTERVAL_SECONDS,
    GEMINI_MAX_WAIT_SECONDS,
    GEMINI_MAX_TOTAL_SECONDS,
    TERMINAL_TEXT_STATUSES,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BASE_SECONDS,
    GEMINI_RETRY_MAX_SECONDS,
    GEMINI_503_MAX_CONSECUTIVE,
    GEMINI_503_DEGRADED_MAX_SECONDS,
    genai_client,
    _extract_api_error_code,
    _is_daily_quota_error,
    _is_transient_gemini_error,
    _retry_delay,
    _GeminiPermanentError,
    _Gemini503UnavailableError,
    _is_503_service_unavailable,
    _register_503_error,
    _create_gemini_interaction,
    _poll_gemini_interaction,
    _generate_with_single_interaction,
    ask_gemini,
)
import llm.gemini as _gemini_mod

from llm.base import (
    LLMRequest,
    LLMUsage,
    LLMResponse,
    LLMError,
    LLMTransientError,
    LLMPermanentError,
    LLMConfigError,
    LLMProvider,
    redact_secrets,
)
from llm.gemini_adapter import GeminiAdapter
from llm.groq_adapter import GroqAdapter
from llm.gateway import (
    LLMGateway,
    get_default_gateway,
    reset_default_gateway,
)
import llm.gateway as _gateway_mod

# ------------------------------------------------------------
# 3. Citation Layer Exports
# ------------------------------------------------------------
from citations.formatter import (
    format_sources,
    register_score_getter,
)
from citations.models import (
    VerificationStatus,
    CitationItem,
    CitationVerificationResult,
)
from citations.verifier import (
    CitationVerifier,
    verify_citations,
)
import citations.formatter as _formatter_mod
import citations.verifier as _verifier_mod

# ------------------------------------------------------------
# 4. Agent Router & Coordinator Exports
# ------------------------------------------------------------
from agent import (
    QueryIntent,
    AgentDecision,
    AgentRouter,
    route_query,
    StepOperation,
    StepStatus,
    AgentStep,
    AgentPlan,
    AgentExecutionResult,
    AgentCoordinator,
    get_default_coordinator,
    reset_default_coordinator,
    coordinate_query,
)
import agent.router as _router_mod
import agent.coordinator as _coordinator_mod

# Wire cross-subsystem helpers
register_prompt_helpers(
    extract_topic_fn=extract_topic,
    title_match_fn=title_exact_match,
    get_score_fn=get_retrieval_score,
)
register_score_getter(get_retrieval_score)


# ------------------------------------------------------------
# 4. Dynamic Attribute Forwarding for Test Monkeypatching
# ------------------------------------------------------------

class _AskModuleFacade(types.ModuleType):
    """Facade module ensuring test monkeypatching propagates to underlying subsystems."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if hasattr(_hybrid_mod, name) or name in (
            "RERANK_ENABLED", "RERANK_DEPTH", "CrossReranker",
            "semantic_retrieve", "lexical_retrieve", "retrieve_chunks",
            "_RERANKER_SINGLETON", "_retrieval_scores_var", "_retrieval_diagnostics_var",
            "TOP_K", "SEMANTIC_CANDIDATES", "LEXICAL_CANDIDATES", "MIN_SCORE_THRESHOLD",
            "TITLE_WEIGHT", "SEMANTIC_WEIGHT", "TEXT_WEIGHT", "load_all_points",
        ):
            setattr(_hybrid_mod, name, value)

        if hasattr(_gemini_mod, name) or name in (
            "genai_client", "ask_gemini", "GEMINI_MODEL", "GEMINI_POLL_INTERVAL_SECONDS",
            "_create_gemini_interaction", "_poll_gemini_interaction",
        ):
            setattr(_gemini_mod, name, value)

        if hasattr(_lexical_mod, name) or name in (
            "_LEXICAL_CACHE", "load_all_points", "normalize_text", "tokenize",
            "extract_topic", "title_exact_match",
        ):
            setattr(_lexical_mod, name, value)

        if hasattr(_prompts_mod, name) or name in (
            "build_prompt", "validate_question", "classify_sources", "SYSTEM_PROMPT",
        ):
            setattr(_prompts_mod, name, value)

        if hasattr(_formatter_mod, name) or name in ("format_sources",):
            setattr(_formatter_mod, name, value)

        if hasattr(_gateway_mod, name) or name in (
            "LLMGateway", "get_default_gateway", "reset_default_gateway",
        ):
            setattr(_gateway_mod, name, value)

        if hasattr(_router_mod, name) or name in (
            "QueryIntent", "AgentDecision", "AgentRouter", "route_query",
        ):
            setattr(_router_mod, name, value)

    def __getattr__(self, name):
        if hasattr(_lexical_mod, name):
            return getattr(_lexical_mod, name)
        if hasattr(_hybrid_mod, name):
            return getattr(_hybrid_mod, name)
        if hasattr(_gemini_mod, name):
            return getattr(_gemini_mod, name)
        if hasattr(_prompts_mod, name):
            return getattr(_prompts_mod, name)
        if hasattr(_formatter_mod, name):
            return getattr(_formatter_mod, name)
        if hasattr(_gateway_mod, name):
            return getattr(_gateway_mod, name)
        if hasattr(_router_mod, name):
            return getattr(_router_mod, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _AskModuleFacade


# ============================================================
# CLI INTERACTIVE MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("MISSION ANTHROPIC — STRIVER A2Z RAG")
    print("=" * 60)
    print(
        f"Collection: {COLLECTION_NAME} "
        f"({collection_info.points_count} chunks indexed)"
    )
    print(
        "Ask a question about the DSA course. "
        "Type 'exit' to quit.\n"
    )

    while True:
        try:
            raw_question = input("Your question: ")
        except EOFError:
            print("\nInput closed. Goodbye, bro!")
            break

        question, error = validate_question(raw_question)
        if error:
            print(f"{error}\n")
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye, bro!")
            break

        # ----------------------------------------------------
        # Agent Router — Intent Classification
        # ----------------------------------------------------
        decision = route_query(question)
        if not decision.retrieval_required:
            print(
                "\nThis query appears outside of the DSA course scope. "
                "Please ask a Data Structures & Algorithms question.\n"
            )
            continue

        print("\nSearching knowledge base...")
        chunks = retrieve_chunks(question, top_k=decision.retrieval_depth)
        if not chunks:
            print("\nNo relevant content found.\n")
            continue

        user_message = build_prompt(question, chunks)
        if user_message is None:
            print("Retrieved chunks contained no usable text.\n")
            continue

        print("Generating answer...\n")
        answer = ask_gemini(user_message)
        if answer is None:
            print(
                "\nCould not generate an answer "
                "because the Gemini request failed."
            )
            print("Check the Gemini error above.\n")
            continue

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)

        print("\n" + "-" * 60)
        print("SOURCES")
        print("-" * 60)
        print(format_sources(chunks))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye, bro!")
        sys.exit(0)
    except Exception as e:
        fail(f"Unexpected error: {e}")