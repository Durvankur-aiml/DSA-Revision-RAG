"""ALGOFORGE Lexical Retrieval and Precomputed Static Feature Cache.

Contains text normalization, tokenization, topic extraction, exact title matching,
text match scoring, and static point feature precomputation.
"""

import os
import re
import sys
import time
import threading
from dataclasses import dataclass

import obs

_PERF_ENABLED = True


def _perf(stage, started_at, extra=None):
    """Emit a [PERF] timing line and record the stage structurally."""
    if not _PERF_ENABLED:
        return
    try:
        duration = time.monotonic() - started_at
        obs.record_stage(stage, duration, extra)
    except Exception:
        pass


LEXICAL_CANDIDATES = 40


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Convert text into normalized representation for matching.

    Examples:
        "Three Sum" -> "3 sum"
        "3-Sum"     -> "3 sum"
        "3_sum"     -> "3 sum"
    """
    if not value:
        return ""

    value = str(value).lower()

    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    for word, number in number_words.items():
        value = re.sub(
            rf"\b{word}\b",
            number,
            value
        )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value):
    stopwords = {
        "explain",
        "problem",
        "question",
        "tell",
        "me",
        "about",
        "the",
        "a",
        "an",
        "please",
        "how",
        "what",
        "is",
        "are",
        "can",
        "you",
        "solve",
        "solution",
        "approach",
        "algorithm",
        "concept",
        "understand",
        "understanding",
        "using",
        "with",
        "for",
        "of",
        "in",
        "on",
        "to",
        "and",
        "give",
        "show",
        "teach",
    }

    normalized = normalize_text(value)

    return {
        token
        for token in normalized.split()
        if token not in stopwords
        and len(token) > 1
    }


# ============================================================
# QUERY TOPIC EXTRACTION
# ============================================================

def extract_topic(question):
    normalized = normalize_text(question)

    if not normalized:
        return None

    # --------------------------------------------------------
    # Number Sum problems
    # --------------------------------------------------------

    match = re.search(
        r"\b([0-9]+)\s+sum\b",
        normalized
    )

    if match:
        return f"{match.group(1)} sum"

    # --------------------------------------------------------
    # Common DSA topics
    # --------------------------------------------------------

    known_topics = [
        "binary search",
        "linear search",
        "merge sort",
        "quick sort",
        "selection sort",
        "bubble sort",
        "insertion sort",
        "heap sort",

        "linked list",
        "doubly linked list",
        "circular linked list",

        "binary tree",
        "binary search tree",
        "bst",

        "sliding window",
        "two pointers",

        "longest subarray",
        "maximum subarray",
        "maximum subarray sum",

        "kadane algorithm",
        "kadane",

        "stock buy and sell",
        "buy and sell stock",

        "valid parentheses",
        "balanced parentheses",

        "next greater element",
        "previous smaller element",

        "majority element",

        "set matrix zeroes",
        "rotate matrix",

        "spiral matrix",

        "subarray sum",
        "subset sum",
        "partition equal subset sum",

        "climbing stairs",
        "house robber",

        "fibonacci",

        "dfs",
        "bfs",
        "depth first search",
        "breadth first search",

        "dijkstra",
        "bellman ford",
        "floyd warshall",

        "kruskal",
        "prim",

        "topological sort",

        "knapsack",
        "0 1 knapsack",

        "lcs",
        "longest common subsequence",

        "lis",
        "longest increasing subsequence",

        "recursion",
        "backtracking",

        "n queens",
        "sudoku",

        "trie",

        "hashing",
        "hash map",
        "hash table",

        "heap",
        "priority queue",

        "stack",
        "queue",
    ]

    known_topics.sort(
        key=len,
        reverse=True
    )

    for topic in known_topics:
        if topic in normalized:
            return topic

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    tokens = tokenize(question)

    if not tokens:
        return None

    return " ".join(
        sorted(tokens)
    )


# ============================================================
# TITLE MATCHING
# ============================================================

def _is_broad_topic_question(question, topic):
    """Return True when the question is primarily asking about a topic itself."""
    question_tokens = tokenize(question)
    topic_tokens = set(topic.split())

    if not question_tokens:
        return False

    # If the query contains only the topic plus generic teaching words,
    # prefer an introductory/general video over a problem-specific video.
    return question_tokens.issubset(topic_tokens | {
        "basics", "basic", "introduction", "intro", "meaning",
        "definition", "fundamentals", "overview", "working",
        "works", "intuition", "simple", "simply"
    })


class __TitleScoringContext:
    """Query-dependent inputs of title_exact_match, computed once."""

    __slots__ = ("normalized_question", "question_tokens", "topic", "topic_phrase", "broad_topic")

    def __init__(self, question):
        self.normalized_question = normalize_text(question)
        self.question_tokens = tokenize(question)
        self.topic = extract_topic(question)
        self.topic_phrase = (
            normalize_text(self.topic) if self.topic else ""
        )
        self.broad_topic = bool(
            self.topic
            and self.question_tokens
            and _is_broad_topic_question(question, self.topic)
        )


_TITLE_INTRO_MARKERS = frozenset(
    {
        "introduction", "intro", "basics", "basic", "fundamentals",
        "overview", "concept", "concepts", "real", "life"
    }
)


def _title_score_cached(question, context, features):
    """title_exact_match with per-point static work already cached."""
    if not context.topic:
        return 0.0

    title = features.title_normalized
    topic_phrase = context.topic_phrase

    if not topic_phrase:
        return 0.0

    # Never treat a longer, distinct topic as an exact match.
    # Example: "binary search tree" != "binary search".
    if topic_phrase == "binary search" and "binary search tree" in title:
        return 0.0

    # Exact phrase must be present as complete tokens, not as a substring.
    if not re.search(rf"(?<!\w){re.escape(topic_phrase)}(?!\w)", title):
        # Special numeric Sum handling.
        sum_match = re.fullmatch(r"([0-9]+)\s+sum", topic_phrase)
        if sum_match:
            number = sum_match.group(1)
            if re.search(rf"(?<!\w){number}\s*sum(?!\w)", title):
                return 0.80
        return 0.0

    if context.broad_topic:
        # Introductory/general titles are the strongest match.
        title_tokens = set(title.split())

        if title_tokens & _TITLE_INTRO_MARKERS:
            return 1.0

        # A title consisting mostly of the topic is also a strong general match.
        topic_tokens = set(topic_phrase.split())
        non_topic_tokens = title_tokens - topic_tokens
        if len(non_topic_tokens) <= 2:
            return 0.95

        # Problem-specific videos remain useful, but are not "exact".
        return 0.72

    # For specific questions, a title containing the topic is strong,
    # but reserve 1.0 for a genuinely exact/general match.
    topic_tokens = set(topic_phrase.split())
    title_tokens = features.title_token_set

    if topic_tokens and topic_tokens.issubset(title_tokens):
        return 0.80

    return 0.0


def title_exact_match(question, video_title, context=None):
    """Score how strongly a video title matches the detected topic."""
    if context is None:
        context = __TitleScoringContext(question)

    return _title_score_cached(
        question,
        context,
        __build_point_lexical_features(
            {"video_title": video_title, "text": ""}
        ),
    )


def _title_exact_match_original_body(question, video_title):
    """Reference implementation retained for equivalence testing."""
    if not video_title:
        return 0.0

    topic = extract_topic(question)
    if not topic:
        return 0.0

    title = normalize_text(video_title)
    topic_phrase = normalize_text(topic)

    if not topic_phrase:
        return 0.0

    if topic_phrase == "binary search" and "binary search tree" in title:
        return 0.0

    if not re.search(rf"(?<!\w){re.escape(topic_phrase)}(?!\w)", title):
        sum_match = re.fullmatch(r"([0-9]+)\s+sum", topic_phrase)
        if sum_match:
            number = sum_match.group(1)
            if re.search(rf"(?<!\w){number}\s*sum(?!\w)", title):
                return 0.80
        return 0.0

    if _is_broad_topic_question(question, topic):
        title_tokens = set(title.split())

        if title_tokens & _TITLE_INTRO_MARKERS:
            return 1.0

        topic_tokens = set(topic_phrase.split())
        non_topic_tokens = title_tokens - topic_tokens
        if len(non_topic_tokens) <= 2:
            return 0.95

        return 0.72

    topic_tokens = set(topic_phrase.split())
    title_tokens = set(tokenize(video_title))

    if topic_tokens and topic_tokens.issubset(title_tokens):
        return 0.80

    return 0.0


# ============================================================
# TRANSCRIPT TEXT MATCHING
# ============================================================

def text_match_score(question, text):
    if not text:
        return 0.0

    query_tokens = tokenize(question)
    text_tokens = tokenize(text)

    if not query_tokens:
        return 0.0

    overlap = (
        len(query_tokens & text_tokens)
        / len(query_tokens)
    )

    return min(overlap, 1.0)


# ============================================================
# STATIC LEXICAL FEATURE CACHE
# ============================================================

@dataclass(frozen=True)
class __PointLexicalFeatures:
    """Query-independent lexical features of one static point."""
    title_normalized: str
    title_token_set: frozenset
    text_token_set: frozenset


def __build_point_lexical_features(payload):
    title = payload.get("video_title", "")
    text = payload.get("text", "")

    return __PointLexicalFeatures(
        title_normalized=normalize_text(title),
        title_token_set=frozenset(tokenize(title)),
        text_token_set=frozenset(tokenize(text)),
    )


_LEXICAL_CACHE_LOCK = threading.Lock()
_LEXICAL_CACHE = None
_point_loader_fn = None


def set_point_loader(fn):
    """Register custom point loader function (e.g. for offline testing)."""
    global _point_loader_fn
    _point_loader_fn = fn


def load_all_points():
    """Default point loader delegates to registered loader or hybrid module."""
    global _point_loader_fn
    if _point_loader_fn is not None:
        return _point_loader_fn()
    from .hybrid import load_all_points as _hybrid_loader
    return _hybrid_loader()


def _get_lexical_cache():
    """Return {"points", "features"} for the static collection."""
    global _LEXICAL_CACHE

    if _LEXICAL_CACHE is not None:
        return _LEXICAL_CACHE

    with _LEXICAL_CACHE_LOCK:
        if _LEXICAL_CACHE is not None:
            return _LEXICAL_CACHE

        try:
            points = load_all_points()
            features = [
                __build_point_lexical_features(point.payload or {})
                for point in points
            ]
        except Exception as e:
            print(
                f"ERROR: Failed to initialize lexical cache: {e}"
            )
            return {"points": [], "features": []}

        _LEXICAL_CACHE = {
            "points": points,
            "features": features,
        }
        _perf(
            "lexical_cache_init",
            time.monotonic(),
            extra=f"points={len(points)}",
        )
        return _LEXICAL_CACHE


# ============================================================
# LEXICAL RETRIEVAL
# ============================================================

def lexical_retrieve(question, all_points=None):
    """Lexical candidates for a question, best-first."""
    context = __TitleScoringContext(question)
    query_tokens = tokenize(question)

    if all_points is None:
        cache = _get_lexical_cache()
        points = cache["points"]
        point_features = cache["features"]
    else:
        points = all_points
        point_features = [
            __build_point_lexical_features(point.payload or {})
            for point in points
        ]

    candidates = []

    for point, features in zip(points, point_features):
        title_score = _title_score_cached(
            question,
            context,
            features
        )

        if not features.text_token_set or not query_tokens:
            text_score = 0.0
        else:
            overlap = (
                len(query_tokens & features.text_token_set)
                / len(query_tokens)
            )
            text_score = min(
                overlap,
                1.0
            )

        lexical_score = (
            0.85 * title_score
            +
            0.15 * text_score
        )

        if lexical_score > 0:
            candidates.append(
                (
                    lexical_score,
                    point
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[
        :LEXICAL_CANDIDATES
    ]


def _verify_lexical_equivalence(queries, points=None):
    """Offline equivalence harness: cached path vs original formulas."""
    if points is None:
        cache = _get_lexical_cache()
        points = cache["points"]
    else:
        points = points

    results = []

    for question in queries:
        optimized = lexical_retrieve(question, points)

        reference_candidates = []
        for point in points:
            payload = point.payload or {}
            title = payload.get("video_title", "")
            text = payload.get("text", "")

            title_score = _title_exact_match_original_body(question, title)

            if not text:
                text_score = 0.0
            else:
                query_tokens = tokenize(question)
                text_tokens = tokenize(text)
                if not query_tokens:
                    text_score = 0.0
                else:
                    overlap = (
                        len(query_tokens & text_tokens)
                        / len(query_tokens)
                    )
                    text_score = min(overlap, 1.0)

            lexical_score = (
                0.85 * title_score
                +
                0.15 * text_score
            )

            if lexical_score > 0:
                reference_candidates.append((lexical_score, point))

        reference_candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )
        reference = reference_candidates[:LEXICAL_CANDIDATES]

        optimized_pairs = [
            (score, str(point.id)) for score, point in optimized
        ]
        reference_pairs = [
            (score, str(point.id)) for score, point in reference
        ]

        ok = optimized_pairs == reference_pairs
        detail = ""
        if not ok:
            detail = (
                f"optimized={optimized_pairs[:5]} "
                f"reference={reference_pairs[:5]}"
            )
        results.append((question, ok, detail))

    return results
