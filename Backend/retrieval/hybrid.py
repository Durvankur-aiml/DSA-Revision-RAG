"""ALGOFORGE Hybrid Retrieval and Fusion Scoring Engine.

Combines semantic vector search (Qdrant) and lexical/title search, applies
linear weighting, exact title boost, optional cross-encoder reranking,
candidate deduplication, and production selection.
"""

import os
import sys
import time
import random
import contextvars
import threading

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import obs
from reranker import CrossReranker

from .lexical import (
    extract_topic,
    lexical_retrieve,
    title_exact_match,
    text_match_score,
    normalize_text,
    _perf,
)
_PERF_ENABLED = True
# ============================================================
# PROJECT PATHS & CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

QDRANT_DIR = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "striver_a2z"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

TOP_K = 6
SEMANTIC_CANDIDATES = 40
LEXICAL_CANDIDATES = 40
MIN_SCORE_THRESHOLD = 0.25

TITLE_WEIGHT = 0.60
SEMANTIC_WEIGHT = 0.30
TEXT_WEIGHT = 0.10

RERANK_ENABLED_ENV = "ALGOFORGE_RERANK_ENABLED"
RERANK_DEPTH_ENV = "ALGOFORGE_RERANK_DEPTH"
RERANK_DEPTH_DEFAULT = 70


def _parse_rerank_config():
    raw_enabled = os.environ.get(RERANK_ENABLED_ENV, "").strip().lower()
    raw_depth = os.environ.get(RERANK_DEPTH_ENV, "").strip()

    if raw_enabled not in ("true", "1", ""):
        if raw_enabled not in ("false", "0", ""):
            print(
                f"WARNING: {RERANK_ENABLED_ENV}={raw_enabled!r} is not a "
                "boolean; reranker DISABLED."
            )
        return False, RERANK_DEPTH_DEFAULT

    enabled = raw_enabled in ("true", "1")
    depth = RERANK_DEPTH_DEFAULT

    if raw_depth:
        try:
            depth = int(raw_depth)
        except ValueError:
            print(
                f"WARNING: {RERANK_DEPTH_ENV}={raw_depth!r} is not an "
                "integer; reranker DISABLED."
            )
            return False, RERANK_DEPTH_DEFAULT
        if depth < 1:
            print(
                f"WARNING: {RERANK_DEPTH_ENV}={raw_depth!r} must be >= 1; "
                "reranker DISABLED."
            )
            return False, RERANK_DEPTH_DEFAULT

    return enabled, depth


RERANK_ENABLED, RERANK_DEPTH = _parse_rerank_config()


# ============================================================
# REQUEST-LOCAL RETRIEVAL SCORE STORAGE
# ============================================================

_retrieval_scores_var = contextvars.ContextVar(
    "retrieval_scores", default={}
)

_retrieval_diagnostics_var = contextvars.ContextVar(
    "retrieval_diagnostics", default=None
)


def _set_retrieval_diagnostics(diagnostics):
    _retrieval_diagnostics_var.set(diagnostics)


def get_retrieval_diagnostics():
    return _retrieval_diagnostics_var.get()


def _set_retrieval_scores(scores):
    _retrieval_scores_var.set(dict(scores))


def get_retrieval_score(point_id):
    scores = _retrieval_scores_var.get()
    return scores.get(str(point_id), 0.0)


# ============================================================
# ERROR HANDLER
# ============================================================

def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# ============================================================
# CONNECT TO QDRANT & LOAD EMBEDDING MODEL
# ============================================================

if not os.path.isdir(QDRANT_DIR):
    fail(
        f"Qdrant storage not found at:\n{QDRANT_DIR}\n\n"
        "Run load_to_qdrant.py first."
    )

try:
    qdrant_client = QdrantClient(path=QDRANT_DIR)
except Exception as e:
    fail(f"Failed to connect to Qdrant: {e}")

try:
    collections = [c.name for c in qdrant_client.get_collections().collections]
except Exception as e:
    fail(f"Failed to list Qdrant collections: {e}")

if COLLECTION_NAME not in collections:
    fail(
        f"Collection '{COLLECTION_NAME}' does not exist.\n"
        "Run load_to_qdrant.py first."
    )

try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        fail(
            f"Collection '{COLLECTION_NAME}' exists but contains zero points.\n"
            "Run load_to_qdrant.py."
        )
except Exception as e:
    fail(f"Failed to inspect Qdrant collection: {e}")

try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    fail(f"Failed to load embedding model: {e}")

try:
    test_vector = embed_model.encode("test").tolist()
    if len(test_vector) != VECTOR_SIZE:
        fail(
            "Embedding dimension mismatch.\n"
            f"Expected: {VECTOR_SIZE}\n"
            f"Got: {len(test_vector)}"
        )
except Exception as e:
    fail(f"Failed to validate embedding model: {e}")


# ============================================================
# CROSS-ENCODER RERANKER — LAZY PRODUCTION SINGLETON (Phase 3C)
# ============================================================

_RERANKER_LOCK = threading.Lock()
_RERANKER_SINGLETON = None
_RERANKER_INIT_FAILED = False


def reset_reranker_singleton():
    global _RERANKER_SINGLETON, _RERANKER_INIT_FAILED
    with _RERANKER_LOCK:
        _RERANKER_SINGLETON = None
        _RERANKER_INIT_FAILED = False


def get_reranker_meta():
    singleton = _RERANKER_SINGLETON
    if singleton is not None:
        return {
            "model": singleton.model_name,
            "device": singleton.device,
            "depth": RERANK_DEPTH,
            "batch_size": singleton.batch_size,
        }
    return {
        "model": CrossReranker.DEFAULT_MODEL,
        "device": None,
        "depth": RERANK_DEPTH,
        "batch_size": 16,
    }


def _get_reranker():
    global _RERANKER_SINGLETON, _RERANKER_INIT_FAILED

    if _RERANKER_SINGLETON is not None:
        return _RERANKER_SINGLETON

    with _RERANKER_LOCK:
        if _RERANKER_SINGLETON is not None:
            return _RERANKER_SINGLETON

        if _RERANKER_INIT_FAILED:
            return None

        try:
            singleton = CrossReranker(
                model_name=CrossReranker.DEFAULT_MODEL,
                batch_size=16,
            )
        except Exception as error:
            print(f"ERROR: Reranker initialization failed: {error}")
            _RERANKER_INIT_FAILED = True
            obs.set_reranker_meta(enabled=True, fallback=True)
            return None

        _RERANKER_SINGLETON = singleton
        obs.set_reranker_meta(
            enabled=True,
            fallback=False,
            model=singleton.model_name,
            device=singleton.device,
            depth=RERANK_DEPTH,
        )
        return singleton


def _rerank_candidates(question, ranked, depth, reranker_instance):
    head = ranked[:depth]
    tail = ranked[depth:]

    docs = [
        {
            "video_title": (item["hit"].payload or {}).get("video_title", ""),
            "text": (item["hit"].payload or {}).get("text", ""),
        }
        for item in head
    ]

    try:
        scores = reranker_instance.score(question, docs)
    except Exception as error:
        print(f"ERROR: Reranker inference failed; using fused ranking: {error}")
        obs.set_reranker_meta(fallback=True)
        return ranked

    if len(scores) != len(docs):
        print(
            "ERROR: Reranker returned misaligned scores; "
            "using fused ranking."
        )
        obs.set_reranker_meta(fallback=True)
        return ranked

    order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
    reordered_head = [
        {**head[i], "reranker_score": scores[i]} for i in order
    ]
    return reordered_head + tail


# ============================================================
# LOAD ALL PAYLOADS
# ============================================================

def load_all_points():
    all_points = []
    _t = time.monotonic()
    offset = None

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    _perf("qdrant_scroll", _t, extra=f"points_loaded={len(all_points)}")
    return all_points


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(question):
    _t = time.monotonic()
    try:
        query_vector = embed_model.encode(question).tolist()
    except Exception as e:
        print(f"ERROR: Failed to embed question: {e}")
        return []

    _perf("embedding", _t)
    _t = time.monotonic()

    if len(query_vector) != VECTOR_SIZE:
        print("ERROR: Query vector dimension mismatch:")
        print(f"  Expected: {VECTOR_SIZE}")
        print(f"  Got: {len(query_vector)}")
        return []

    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=SEMANTIC_CANDIDATES,
        ).points
    except Exception as e:
        print(f"ERROR: Qdrant semantic search failed: {e}")
        return []

    _perf("qdrant_query", _t)
    return results


# ============================================================
# SELECTION & RETRIEVE_CHUNKS
# ============================================================

def apply_production_selection(ranked, top_k=TOP_K):
    selected = []
    seen_videos = set()

    for item in ranked:
        hit = item["hit"]
        payload = hit.payload or {}
        raw_title = payload.get("video_title", "Unknown")
        video_title = normalize_text(raw_title)

        if video_title in seen_videos:
            continue

        semantic_score = item["semantic_score"]
        title_score = item["title_score"]

        if (
            semantic_score < MIN_SCORE_THRESHOLD
            and title_score < 0.70
        ):
            continue

        selected.append(hit)
        seen_videos.add(video_title)

        if len(selected) >= top_k:
            break

    return selected


def retrieve_chunks(
    question,
    top_k=TOP_K
):
    _t = time.monotonic()
    _set_retrieval_scores({})
    obs.set_retrieval_meta(strategy="hybrid")
    obs.set_reranker_meta(enabled=RERANK_ENABLED, depth=RERANK_DEPTH)

    # 1. Topic
    topic = extract_topic(question)
    if topic:
        print(f"  Detected topic: {topic}")
    else:
        print("  Detected topic: general DSA query")
    obs.set_retrieval_meta(topic=topic)

    # 2. Semantic
    _perf("topic_detection", _t)
    _t = time.monotonic()
    semantic_results = semantic_retrieve(question)
    _perf("semantic_retrieve_total", _t)
    _t = time.monotonic()

    candidates = {}
    for rank, hit in enumerate(semantic_results):
        point_id = str(hit.id)
        semantic_score = float(hit.score)
        candidates[point_id] = {
            "hit": hit,
            "semantic_score": max(0.0, min(semantic_score, 1.0)),
            "title_score": 0.0,
            "text_score": 0.0,
            "source": "semantic",
            "semantic_rank": rank,
        }

    # 3. Load local records (cached)
    all_points = None
    _perf("load_all_points_total", _t, extra="cached")
    _t = time.monotonic()

    # 4. Lexical
    lexical_results = lexical_retrieve(question, all_points)
    _perf("lexical_retrieve", _t)
    _t = time.monotonic()

    # 5. Merge lexical
    for lexical_score, point in lexical_results:
        point_id = str(point.id)
        payload = point.payload or {}
        title = payload.get("video_title", "")
        text = payload.get("text", "")

        title_score = title_exact_match(question, title)
        text_score = text_match_score(question, text)

        if point_id not in candidates:
            candidates[point_id] = {
                "hit": point,
                "semantic_score": 0.0,
                "title_score": title_score,
                "text_score": text_score,
                "source": "lexical",
                "semantic_rank": 999999,
            }
        else:
            candidates[point_id]["title_score"] = max(
                candidates[point_id]["title_score"], title_score
            )
            candidates[point_id]["text_score"] = max(
                candidates[point_id]["text_score"], text_score
            )

    # 6. Calculate hybrid score
    _perf("merge_lexical", _t)
    _t = time.monotonic()
    ranked = []

    for data in candidates.values():
        hit = data["hit"]
        payload = hit.payload or {}
        title = payload.get("video_title", "")
        text = payload.get("text", "")

        title_score = max(data["title_score"], title_exact_match(question, title))
        text_score = max(data["text_score"], text_match_score(question, text))
        semantic_score = data["semantic_score"]

        hybrid_score = (
            TITLE_WEIGHT * title_score
            + SEMANTIC_WEIGHT * semantic_score
            + TEXT_WEIGHT * text_score
        )

        if title_score >= 0.95:
            hybrid_score += 0.20

        hybrid_score = min(hybrid_score, 1.0)

        ranked.append(
            {
                "hit": hit,
                "hybrid_score": hybrid_score,
                "title_score": title_score,
                "semantic_score": semantic_score,
                "text_score": text_score,
            }
        )

    # 7. Sort
    _perf("hybrid_scoring", _t)
    _t = time.monotonic()
    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    obs.set_retrieval_meta(candidate_count=len(ranked))

    # 7B. Optional cross-encoder rerank
    _perf("ranking_sort", _t)
    _t = time.monotonic()
    if RERANK_ENABLED:
        reranker = _get_reranker()
        if reranker is not None:
            ranked = _rerank_candidates(
                question, ranked, RERANK_DEPTH, reranker_instance=reranker
            )
        _perf("reranker_stage", _t)

    # 8. Exact topic detection
    _t = time.monotonic()
    exact_topic_candidates = [
        item for item in ranked if item["title_score"] >= 0.95
    ]
    exact_topic_found = len(exact_topic_candidates) > 0

    if topic:
        if exact_topic_found:
            print(f"  Exact topic match found: {topic}")
        else:
            print(f"  WARNING: No exact title match found for: {topic}")
            print("  Semantic results will be treated cautiously.")

    obs.set_retrieval_meta(exact_topic=exact_topic_found)

    # 9. Selection
    selected = apply_production_selection(ranked, top_k=top_k)
    seen_videos = {
        normalize_text((hit.payload or {}).get("video_title", "Unknown"))
        for hit in selected
    }

    # 10. Store final retrieval scores
    _perf("source_selection", _t)
    _t = time.monotonic()
    for item in ranked:
        point_id = str(item["hit"].id)
        _set_retrieval_scores(
            {**_retrieval_scores_var.get(), point_id: item["hybrid_score"]}
        )

    # 11. Debug output
    if selected:
        print("\n  Hybrid retrieval:")
        for index, hit in enumerate(selected, start=1):
            payload = hit.payload or {}
            title = payload.get("video_title", "Unknown")
            score = get_retrieval_score(str(hit.id))
            print(f"    {index}. {title}")
            print(f"       hybrid score: {score:.4f}")

    _perf("score_store_and_debug", _t)

    _selected_ids = {str(hit.id) for hit in selected}
    _diag = {
        "candidates": [
            {
                "point_id": str(item["hit"].id),
                "video_id": (item["hit"].payload or {}).get("video_id"),
                "chunk_index": (item["hit"].payload or {}).get("chunk_index"),
                "video_title": (item["hit"].payload or {}).get("video_title"),
                "text": (item["hit"].payload or {}).get("text"),
                "hybrid_score": item["hybrid_score"],
                "title_score": item["title_score"],
                "semantic_score": item["semantic_score"],
                "text_score": item["text_score"],
                "reranker_score": item.get("reranker_score"),
                "rank": _rank,
                "selected": str(item["hit"].id) in _selected_ids,
                "cull_reason": (
                    "selected"
                    if str(item["hit"].id) in _selected_ids
                    else (
                        "duplicate_title"
                        if normalize_text(
                            (item["hit"].payload or {}).get("video_title", "")
                        ) in seen_videos
                        else (
                            "below_threshold"
                            if item["semantic_score"] < MIN_SCORE_THRESHOLD
                            and item["title_score"] < 0.70
                            else "beyond_top_k"
                        )
                    )
                ),
            }
            for _rank, item in enumerate(ranked, start=1)
        ],
        "config": {
            "top_k": top_k,
            "min_score_threshold": MIN_SCORE_THRESHOLD,
            "title_weight": TITLE_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "text_weight": TEXT_WEIGHT,
            "exact_title_boost": 0.20,
            "rerank_enabled": RERANK_ENABLED,
            "rerank_depth": RERANK_DEPTH if RERANK_ENABLED else None,
        },
    }
    _set_retrieval_diagnostics(_diag)

    return selected
