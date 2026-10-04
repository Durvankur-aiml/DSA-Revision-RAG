import os
import re
import sys
import time
import random
import contextvars
import threading
from dataclasses import dataclass

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Structured observability (stdlib-only, never raises, no import cycle).
# Stage timings recorded through _perf below land in the per-request
# context; retrieval metadata is attached inside retrieve_chunks.
import obs

# Cross-encoder reranker (validated in Phase 3/3B, integrated Phase 3C).
# Used ONLY by the optional rerank stage in retrieve_chunks, which runs
# exclusively behind ALGOFORGE_RERANK_ENABLED=true (default: disabled).
from reranker import CrossReranker


_PERF_ENABLED = True


def _perf(stage, started_at, extra=None):
    """Emit a [PERF] timing line and record the stage structurally.

    Instrumentation only: never raises and never changes control flow.
    The stdout line keeps its historical format, so existing diagnostics
    and any tooling that greps for [PERF] keep working; the same duration
    is additionally recorded into the per-request observability context
    (Backend/obs.py) when a request is active (CLI runs simply skip it).
    """
    if not _PERF_ENABLED:
        return
    try:
        duration = time.monotonic() - started_at
        obs.record_stage(stage, duration, extra)
    except Exception:
        pass


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

QDRANT_DIR = os.path.join(
    BASE_DIR,
    "qdrant_db"
)


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "striver_a2z"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Final number of chunks sent to Gemini.
TOP_K = 6

# Broad semantic candidate pool.
SEMANTIC_CANDIDATES = 40

# Number of lexical/title candidates kept.
LEXICAL_CANDIDATES = 40

# Minimum semantic similarity before a purely semantic
# result can enter the final context.
MIN_SCORE_THRESHOLD = 0.25


# ------------------------------------------------------------
# Hybrid retrieval weights
# ------------------------------------------------------------

# Exact title/topic matching is strongest.
TITLE_WEIGHT = 0.60

# Semantic similarity handles natural language.
SEMANTIC_WEIGHT = 0.30

# Transcript text matching is supporting signal.
TEXT_WEIGHT = 0.10


# ------------------------------------------------------------
# Gemini
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.7-flash"

# Client-wide HTTP timeout (ms) for SDK calls (create/poll). Observed
# normal create/poll latencies are well under 5s; a 30s stall is already
# anomalous and should fail into the retry path instead of blocking an
# interactive API request for minutes.
GEMINI_HTTP_TIMEOUT_MS = 30000

GEMINI_POLL_INTERVAL_SECONDS = 5

# Bounded wait for a single background interaction to finish. Observed
# full generation latency is ~10-25s; this leaves ~4x headroom.
GEMINI_MAX_WAIT_SECONDS = 90

# Total wall-clock budget for ask_gemini across all attempts. A mid-flight
# capacity error (503) can leave an interaction permanently un-pollable
# (every GET then returns 400 invalid_request); the only recovery is a
# fresh interaction, so the budget must cover more than one attempt.
GEMINI_MAX_TOTAL_SECONDS = 200

# Terminal interaction statuses that can still carry generated text.
# "incomplete" / "budget_exceeded" mean generation stopped early
# (e.g. output budget reached) but the partial answer is usable.
TERMINAL_TEXT_STATUSES = ("completed", "incomplete", "budget_exceeded")

# Increased because we now want proper teaching answers.
GEMINI_MAX_OUTPUT_TOKENS = 1800

# Maximum retries for transient Gemini API failures.
GEMINI_MAX_RETRIES = 3

# Backoff starts here and grows exponentially, with small jitter.
GEMINI_RETRY_BASE_SECONDS = 2.0
GEMINI_RETRY_MAX_SECONDS = 20.0

# Consecutive Gemini 503 'service_unavailable' (capacity) responses --
# counted across creates AND polls within ONE /ask request -- tolerated
# before failing fast. Brief blips (1-2 consecutive 503s) keep the
# existing retry/recovery behavior; a sustained capacity incident fails
# the request in seconds instead of burning two 90-second interaction
# windows (~190 s) and quota. A successful poll GET does NOT reset
# the count and does NOT prove recovery: live evidence (2026-09-25)
# shows polls of an in_progress interaction keep succeeding during an
# incident while the actual generation 503s.
GEMINI_503_MAX_CONSECUTIVE = 3

# Once ANY capacity 503 has been seen in a request, the entire request
# must finish within this many seconds (measured from ask_gemini
# entry): either the current interaction recovers or the request
# fails. Without this budget a capacity incident still burns full
# 90-second interaction windows. Genuine transient recovery completes
# well inside it (create ~3 s + polls/backoffs ~12 s).
GEMINI_503_DEGRADED_MAX_SECONDS = 20.0


# ------------------------------------------------------------
# Cross-encoder reranker configuration (Phase 3C)
# ------------------------------------------------------------

# The validated experiment (Phase 3B) measured, on the golden dataset
# (n=100), baseline Hit@1 62% / Hit@6 91% / MRR 0.725 versus depth-70
# reranking Hit@1 85% / Hit@6 100% / MRR 0.909. Integration is OFF by
# default: production behavior is byte-identical to pre-Phase-3C until
# explicitly enabled. One-line activation (rollback = unset / "false"):
#
#     ALGOFORGE_RERANK_ENABLED=true
#     ALGOFORGE_RERANK_DEPTH=70
#
# Any value other than a case-insensitive "true" / "1" disables the
# reranker. A MALFORMED DEPTH leaves the reranker DISABLED (config
# error is logged once at startup) — never a crash, never a partial
# rerank. The depth default when enabled is the validated 70.
RERANK_ENABLED_ENV = "ALGOFORGE_RERANK_ENABLED"
RERANK_DEPTH_ENV = "ALGOFORGE_RERANK_DEPTH"
RERANK_DEPTH_DEFAULT = 70


def _parse_rerank_config():
    """Resolve the reranker feature flags from the environment.

    Returns (enabled, depth). Safe defaulting on every malformed input:
    non-boolean enabled values and non-positive/non-integer depths both
    degrade to (False, RERANK_DEPTH_DEFAULT) with a startup warning, so
    a typo can never turn a heavy model load on by accident and can
    never disable the rollback switch.
    """
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

# Per-request retrieval scores, keyed by Qdrant point ID.
# A ContextVar keeps concurrent FastAPI requests isolated: each
# request reads only the scores produced by its own retrieve_chunks
# call. There is no module-level mutable global and no lock.
_retrieval_scores_var = contextvars.ContextVar(
    "retrieval_scores", default={}
)

# Request-local retrieval diagnostics (Backend/eval harness). Same
# ContextVar isolation rules as the score storage above: replaced
# atomically per request, never shared across concurrent FastAPI tasks.
_retrieval_diagnostics_var = contextvars.ContextVar(
    "retrieval_diagnostics", default=None
)


def _set_retrieval_diagnostics(diagnostics):
    """Atomically replace this request's retrieval diagnostics."""
    _retrieval_diagnostics_var.set(diagnostics)


def get_retrieval_diagnostics():
    """Return the current request's diagnostics snapshot (or None).

    Shape: {"candidates": [{point_id, video_id, video_title,
    hybrid_score, title_score, semantic_score, text_score, rank,
    selected, cull_reason}...], "config": {...}} ordered best-first by
    hybrid score. Purely observational: populated by retrieve_chunks,
    never read by production retrieval logic.
    """
    return _retrieval_diagnostics_var.get()


def _set_retrieval_scores(scores):
    """Atomically replace this request's retrieval-score snapshot."""
    _retrieval_scores_var.set(dict(scores))


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

MAX_QUESTION_LENGTH = 500


# ============================================================
# ERROR HANDLER
# ============================================================

def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# ============================================================
# STEP 0 — VALIDATE ENVIRONMENT
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or not api_key.strip():
    fail(
        "GEMINI_API_KEY environment variable is not set or empty.\n"
        "Set it with:\n"
        'setx GEMINI_API_KEY "your-key"\n'
        "Then restart the terminal."
    )


if not os.path.isdir(QDRANT_DIR):
    fail(
        f"Qdrant storage not found at:\n{QDRANT_DIR}\n\n"
        "Run load_to_qdrant.py first."
    )


# ============================================================
# STEP 1 — CONNECT TO QDRANT
# ============================================================

try:

    qdrant_client = QdrantClient(
        path=QDRANT_DIR
    )

except Exception as e:

    fail(
        f"Failed to connect to Qdrant: {e}"
    )


try:

    collections = [
        c.name
        for c in qdrant_client
        .get_collections()
        .collections
    ]

except Exception as e:

    fail(
        f"Failed to list Qdrant collections: {e}"
    )


if COLLECTION_NAME not in collections:

    fail(
        f"Collection '{COLLECTION_NAME}' does not exist.\n"
        "Run load_to_qdrant.py first."
    )


try:

    collection_info = qdrant_client.get_collection(
        COLLECTION_NAME
    )

    if collection_info.points_count == 0:

        fail(
            f"Collection '{COLLECTION_NAME}' exists "
            "but contains zero points.\n"
            "Run load_to_qdrant.py."
        )

except Exception as e:

    fail(
        f"Failed to inspect Qdrant collection: {e}"
    )


# ============================================================
# STEP 2 — LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

try:

    embed_model = SentenceTransformer(
        EMBED_MODEL_NAME
    )

except Exception as e:

    fail(
        f"Failed to load embedding model: {e}"
    )


# Validate embedding dimension.
try:

    test_vector = embed_model.encode(
        "test"
    ).tolist()

    if len(test_vector) != VECTOR_SIZE:

        fail(
            "Embedding dimension mismatch.\n"
            f"Expected: {VECTOR_SIZE}\n"
            f"Got: {len(test_vector)}"
        )

except Exception as e:

    fail(
        f"Failed to validate embedding model: {e}"
    )


# ============================================================
# STEP 3 — GEMINI CLIENT
# ============================================================

try:

    genai_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_HTTP_TIMEOUT_MS
        )
    )

except Exception as e:

    fail(
        f"Failed to initialize Gemini client: {e}"
    )


# ============================================================
# SYSTEM PROMPT — AGENTIC GENERATION LAYER
# ============================================================

SYSTEM_PROMPT = """\
You are the AI DSA teacher for the Striver A2Z DSA knowledge base.

Your job is NOT to merely summarize retrieved transcript chunks.

Your job is to understand the student's question, reason over the
retrieved knowledge, select the relevant information, and then teach
the student clearly using that information.

============================================================
CORE RULE
============================================================

The retrieved transcript context is your knowledge source.

Use the retrieved context as the primary and authoritative source.

Do not silently replace the retrieved material with outside knowledge.

Do not invent facts, examples, algorithms, complexities, code details,
or claims that are not supported by the retrieved context.

If the retrieved context is insufficient, say so honestly.

============================================================
INTERNAL REASONING PROCESS
============================================================

Before producing the final answer, internally determine:

1. What exactly is the student asking?
2. What DSA topic/problem is involved?
3. What type of question is this?
   - concept
   - problem explanation
   - intuition
   - brute force
   - better approach
   - optimal approach
   - dry run
   - complexity
   - code
   - code explanation
   - why/how clarification
   - comparison
4. Which retrieved source is the exact topic?
5. Which retrieved information is actually relevant?
6. What level of explanation is appropriate?
7. Does the retrieved context contain enough information to answer?

Do this reasoning internally.

Do NOT expose hidden chain-of-thought or internal reasoning to the student.

Only provide the useful final explanation.

============================================================
SOURCE PRIORITY
============================================================

Retrieved sources may contain:

- an exact topic match
- closely related topics
- unrelated semantic matches

When an exact-topic source exists, treat it as PRIMARY.

Related sources are SECONDARY and should only be used when they
actually help answer the student's question and do not conflict
with the exact source.

Do not mix different DSA problems together.

For example:

Question:
"Explain 3 Sum"

If the retrieved context contains:

- 3 Sum
- 4 Sum
- Two Sum
- Target Sum

the 3 Sum source is the primary source.

Do NOT explain 3 Sum by combining unrelated details from 4 Sum,
Target Sum, or another problem unless the retrieved material itself
clearly uses them for explanation.

============================================================
QUESTION-ADAPTIVE TEACHING
============================================================

Do NOT use one rigid answer template for every question.

Adapt the response to what the student actually asked.

For a simple conceptual question:
- explain the concept directly
- use a small example only if supported

For a problem explanation:
- explain what the problem asks
- explain intuition
- explain the progression of approaches present in the source
- explain why the optimal approach works
- include important implementation details
- include complexity when supported

For an intuition question:
- focus primarily on WHY the approach works

For a brute-force question:
- explain the brute-force idea
- explain its complexity if supported

For an optimal-approach question:
- focus on the optimal approach
- explain the pointer/state movement
- explain why it works
- include complexity if supported

For a dry-run question:
- walk through the provided/example input step by step
- do not invent an example if the source does not support one

For a complexity question:
- directly answer the requested time/space complexity
- explain where it comes from

For a code question:
- provide code only when code is present or sufficiently specified
  by the retrieved material
- explain the important parts

For a "why" or "how" follow-up:
- answer that specific point first
- do not unnecessarily repeat the entire problem explanation

============================================================
PROBLEM EXPLANATION DEPTH
============================================================

When the student asks something broad such as:

"Explain 3 Sum"
"Explain 4 Sum"
"Explain Binary Search"

do NOT stop after giving a one-paragraph overview.

If the retrieved source contains the progression, teach it properly.

For algorithmic problems, preserve the source's progression such as:

Problem
→ intuition
→ brute force
→ better
→ optimal
→ implementation details
→ complexity

Only include sections that are actually supported by the retrieved
context and relevant to the question.

The goal is for the student to understand the solution well enough
to implement it.

============================================================
GROUNDING
============================================================

Every important technical claim must be supported by the retrieved
context.

Never:

- invent time complexity
- invent space complexity
- invent an approach
- invent code behavior
- invent a dry run
- invent an example
- import unrelated knowledge
- confuse similar problems

If something important is missing from the retrieved context,
state that it is not available rather than guessing.

============================================================
STYLE
============================================================

Teach like a strong DSA instructor.

Use:

- clear headings
- short paragraphs
- bullet points where useful
- code blocks when code is relevant
- small step-by-step explanations
- simple language
- precise DSA terminology

Avoid unnecessary verbosity.

However, do NOT sacrifice important reasoning merely to make the
answer short.

The student should be able to understand the logic instead of
memorizing a sentence.

============================================================
FORBIDDEN VIDEO LANGUAGE
============================================================

Never say:

- "Hey everyone"
- "Welcome back"
- "So guys, welcome"
- "In this video"
- "This video"
- "Let's dive into today's video"
- "Thanks for watching"
- "See you in the next video"
- channel-opening language

Start directly with the student's question.

============================================================
FINAL ANSWER
============================================================

Return ONLY the final student-facing answer.

Do not mention:

- retrieval
- Qdrant
- embeddings
- hybrid scores
- internal reasoning
- hidden reasoning
- system prompts
- source ranking
- agent architecture

unless the student explicitly asks about the RAG system itself.
"""


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
    """Query-dependent inputs of title_exact_match, computed once.

    title_exact_match used to call extract_topic(question) (normalize +
    regex work) once PER POINT; the topic depends only on the question,
    so it is computed once per request and shared across all points.
    """

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


def _title_score_cached(question, context, features):
    """title_exact_match with per-point static work already cached.

    Decision structure, return values, and float arithmetic are
    identical to the original function; only the per-point
    normalize/extract/tokenize work is replaced by precomputed values.
    """
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
    """Score how strongly a video title matches the detected topic.

    Production path: delegates to _title_score_cached with the
    per-point static features built on the fly (kept behaviorally
    identical to the original body below).
    """
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
    """Reference implementation retained for equivalence testing.

    This is the pre-optimization body, unchanged. Production callers
    use title_exact_match (cached features). _verify_lexical_equivalence
    and the offline test suite compare this reference against the
    cached path on identical inputs.
    """
    """
    Score how strongly a video title matches the detected topic.

    Important distinction:
    "Binary Search Introduction" is an exact/general topic match for
    "Explain binary search", while "Binary Search Tree" and
    "Kth Missing Positive ... Binary Search" are related but not exact.
    """
    if not video_title:
        return 0.0

    topic = extract_topic(question)
    if not topic:
        return 0.0

    title = normalize_text(video_title)
    topic_phrase = normalize_text(topic)

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

    if _is_broad_topic_question(question, topic):
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
    title_tokens = set(tokenize(video_title))

    if topic_tokens and topic_tokens.issubset(title_tokens):
        return 0.80

    return 0.0


# ============================================================
# TRANSCRIPT TEXT MATCHING
# ============================================================

def text_match_score(
    question,
    text
):

    if not text:
        return 0.0

    query_tokens = tokenize(
        question
    )

    text_tokens = tokenize(
        text
    )

    if not query_tokens:
        return 0.0

    overlap = (
        len(query_tokens & text_tokens)
        / len(query_tokens)
    )

    return min(
        overlap,
        1.0
    )


# ============================================================
# STATIC LEXICAL FEATURE CACHE
# ------------------------------------------------------------
# Profiling (2026-09-25) showed lexical_retrieve spending ~3.5 s per
# request recomputing query-independent work for all ~4,850 static
# points: normalize_text/tokenize of every title and transcript, plus
# regex work. The collection is static after indexing, so the points
# and their query-independent lexical features are built once (lazy,
# synchronized) and reused read-only by every request.
#
# Behavior preservation: the cached values are EXACTLY what the
# per-request code used to compute (same functions, same inputs); only
# the computation moved from request time to startup. Query-dependent
# scoring still runs per request with the original formulas.
# ============================================================

# Introductory/general title markers used by title_exact_match when
# the question is a broad topic query. Hoisted verbatim from the
# per-call literal (same members, no behavior change).
_TITLE_INTRO_MARKERS = frozenset(
    {
        "introduction", "intro", "basics", "basic", "fundamentals",
        "overview", "concept", "concepts", "real", "life"
    }
)


@dataclass(frozen=True)
class __PointLexicalFeatures:
    """Query-independent lexical features of one static point.

    title_normalized / title_token_set reproduce exactly what
    title_exact_match computed per point per request;
    text_token_set reproduces exactly what text_match_score computed
    per point per request. Immutable and therefore safe to share
    across concurrent requests without locks.
    """

    title_normalized: str
    title_token_set: frozenset
    text_token_set: frozenset


def __build_point_lexical_features(payload):
    """Precompute the static lexical features of one point payload.

    Uses the SAME normalization/tokenization functions as the original
    per-request path, so cached values are bit-identical to freshly
    computed ones.
    """
    title = payload.get("video_title", "")
    text = payload.get("text", "")

    return __PointLexicalFeatures(
        title_normalized=normalize_text(title),
        title_token_set=frozenset(tokenize(title)),
        text_token_set=frozenset(tokenize(text)),
    )


# One-time initialization of the static lexical cache. The double-
# checked lock guards against concurrent first requests; after init
# the cache is treated as read-only.
_LEXICAL_CACHE_LOCK = threading.Lock()
_LEXICAL_CACHE = None


def _get_lexical_cache():
    """Return {"points", "features"} for the static collection.

    Initialized once per process from a single full Qdrant scroll
    (~0.2 s for 4,850 points, measured). On initialization failure the
    cache stays an empty structure, mirroring the previous behavior of
    a failed per-request scan (retrieval then runs on semantic results
    only). A failed init is retried on the next request until it
    succeeds; after a successful init the cache is never rebuilt.
    """
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
            # Do NOT cache the failure: the pre-optimization code
            # re-scanned on every request, so a transient Qdrant
            # outage degraded to semantic-only for that request and
            # recovered on the next one. Preserve exactly that.
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
# CROSS-ENCODER RERANKER — LAZY PRODUCTION SINGLETON (Phase 3C)
# ------------------------------------------------------------
# ONE instance per process, created on the first enabled request and
# reused for every request afterwards (the model object is shared;
# per-request candidate data never is — rerank happens on local lists
# inside each retrieve_chunks call).
#
# Failure policy:
# - Initialization failure: recorded and retried on the NEXT enabled
#   request (the request itself falls back to the existing ranking, so
#   /ask never fails because of the optional reranker).
# - Inference failure: logged per request, fallback flag observable via
#   obs.set_reranker_fallback, and the existing fused ranking is used.
# ============================================================

_RERANKER_LOCK = threading.Lock()
_RERANKER_SINGLETON = None
_RERANKER_INIT_FAILED = False


def reset_reranker_singleton():
    """Drop the cached reranker (rollback/tests); next request reloads."""
    global _RERANKER_SINGLETON, _RERANKER_INIT_FAILED
    with _RERANKER_LOCK:
        _RERANKER_SINGLETON = None
        _RERANKER_INIT_FAILED = False


def get_reranker_meta():
    """Metadata for observability (model, device, depth, batch size)."""
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
    """Return the shared CrossReranker, or None (fallback to fused order).

    Lazily built ONCE per process from the validated configuration
    (BAAI/bge-reranker-base, CUDA auto-detect with CPU fallback, batch
    16). Never raises; never loads per request.
    """
    global _RERANKER_SINGLETON, _RERANKER_INIT_FAILED

    if _RERANKER_SINGLETON is not None:
        return _RERANKER_SINGLETON

    with _RERANKER_LOCK:
        if _RERANKER_SINGLETON is not None:
            return _RERANKER_SINGLETON

        if _RERANKER_INIT_FAILED:
            # Previous construction failed (e.g. no CUDA/torch problem).
            # Keep serving the fused ranking; a restart or a successful
            # reset_reranker_singleton() clears this.
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
    """Reorder the fused pool with the cross-encoder (validated 3B rules).

    Applies EXACTLY the Phase 3B semantics: score the top-`depth` of the
    fused ordering, sort that head by reranker score (stable — equal
    scores keep fused order), and keep the tail in fused order. The
    returned list carries the same item dicts plus a diagnostic
    "reranker_score" key. NEVER mutates the input; NEVER raises.
    """
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
        # Defensive: a scorer returning the wrong arity cannot be mapped
        # back onto positions — fall back rather than guess.
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
# LOAD ALL PAYLOADS FOR LEXICAL SEARCH
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

        all_points.extend(
            points
        )

        if next_offset is None:
            break

        offset = next_offset

    _perf("qdrant_scroll", _t, extra=f"points_loaded={len(all_points)}")
    return all_points


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(
    question
):

    _t = time.monotonic()
    try:

        query_vector = embed_model.encode(
            question
        ).tolist()

    except Exception as e:

        print(
            f"ERROR: Failed to embed question: {e}"
        )

        return []

    _perf("embedding", _t)
    _t = time.monotonic()
    if len(query_vector) != VECTOR_SIZE:

        print(
            "ERROR: Query vector dimension mismatch:"
        )

        print(
            f"  Expected: {VECTOR_SIZE}"
        )

        print(
            f"  Got: {len(query_vector)}"
        )

        return []

    try:

        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=SEMANTIC_CANDIDATES,
        ).points

    except Exception as e:

        print(
            f"ERROR: Qdrant semantic search failed: {e}"
        )

        return []

    _perf("qdrant_query", _t)
    return results


# ============================================================
# LEXICAL RETRIEVAL
# ============================================================

def lexical_retrieve(
    question,
    all_points=None
):
    """Lexical candidates for a question, best-first.

    Default path (all_points=None): scores the static point cache with
    precomputed per-point features -- identical scores to the original
    per-request implementation, without re-tokenizing ~4,850 points.
    Explicit all_points (tests, callers with injected chunk lists) are
    scored exactly as before, with features built for the call.
    """

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
    """Offline equivalence harness: cached path vs original formulas.

    Runs the production lexical_retrieve and an inline re-implementation
    of the ORIGINAL per-request formulas over the same points, comparing
    (lexical_score, point_id) lists exactly. Returns a list of
    (query, ok, detail) tuples for the offline test suite.
    """
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


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    top_k=TOP_K
):

    # Score storage for this request starts empty; STEP 10 fills it
    # via the request-local ContextVar (see _set_retrieval_scores).
    _t = time.monotonic()
    _set_retrieval_scores({})

    # Observability: this request uses the hybrid strategy (semantic +
    # lexical are both always executed); topic/exact-match/count fields
    # are attached below where they are actually computed.
    obs.set_retrieval_meta(strategy="hybrid")

    # Observability: reranker flag state for this request. When enabled,
    # _get_reranker() enriches this with model/device and any fallback.
    obs.set_reranker_meta(enabled=RERANK_ENABLED, depth=RERANK_DEPTH)

    # ========================================================
    # STEP 1 — Extract topic
    # ========================================================

    topic = extract_topic(
        question
    )

    if topic:

        print(
            f"  Detected topic: {topic}"
        )

    else:

        print(
            "  Detected topic: general DSA query"
        )

    obs.set_retrieval_meta(topic=topic)

    # ========================================================
    # STEP 2 — Semantic search
    # ========================================================

    _perf("topic_detection", _t)
    _t = time.monotonic()
    semantic_results = semantic_retrieve(
        question
    )
    _perf("semantic_retrieve_total", _t)
    _t = time.monotonic()

    candidates = {}

    for rank, hit in enumerate(
        semantic_results
    ):

        point_id = str(
            hit.id
        )

        semantic_score = float(
            hit.score
        )

        candidates[
            point_id
        ] = {

            "hit": hit,

            "semantic_score":
                max(
                    0.0,
                    min(
                        semantic_score,
                        1.0
                    )
                ),

            "title_score":
                0.0,

            "text_score":
                0.0,

            "source":
                "semantic",

            "semantic_rank":
                rank,
        }

    # ========================================================
    # STEP 3 — Load local records
    # ========================================================

    # The static point cache replaces the per-request full scroll
    # (profiling: 0.2 s/request re-reading an immutable collection).
    # all_points=None makes lexical_retrieve use the cached points and
    # their precomputed features; explicit overrides are still honored.
    all_points = None
    _perf("load_all_points_total", _t, extra="cached")
    _t = time.monotonic()

    # ========================================================
    # STEP 4 — Lexical/title search
    # ========================================================

    lexical_results = lexical_retrieve(
        question,
        all_points
    )
    _perf("lexical_retrieve", _t)
    _t = time.monotonic()

    # ========================================================
    # STEP 5 — Merge lexical candidates
    # ========================================================

    for lexical_score, point in lexical_results:

        point_id = str(
            point.id
        )

        payload = point.payload or {}

        title = payload.get(
            "video_title",
            ""
        )

        text = payload.get(
            "text",
            ""
        )

        title_score = title_exact_match(
            question,
            title
        )

        text_score = text_match_score(
            question,
            text
        )

        if point_id not in candidates:

            candidates[
                point_id
            ] = {

                "hit":
                    point,

                "semantic_score":
                    0.0,

                "title_score":
                    title_score,

                "text_score":
                    text_score,

                "source":
                    "lexical",

                "semantic_rank":
                    999999,
            }

        else:

            candidates[
                point_id
            ][
                "title_score"
            ] = max(
                candidates[
                    point_id
                ][
                    "title_score"
                ],
                title_score
            )

            candidates[
                point_id
            ][
                "text_score"
            ] = max(
                candidates[
                    point_id
                ][
                    "text_score"
                ],
                text_score
            )

    # ========================================================
    # STEP 6 — Calculate hybrid score
    # ========================================================

    _perf("merge_lexical", _t)
    _t = time.monotonic()
    ranked = []

    for data in candidates.values():

        hit = data[
            "hit"
        ]

        payload = hit.payload or {}

        title = payload.get(
            "video_title",
            ""
        )

        text = payload.get(
            "text",
            ""
        )

        title_score = max(
            data["title_score"],
            title_exact_match(
                question,
                title
            )
        )

        text_score = max(
            data["text_score"],
            text_match_score(
                question,
                text
            )
        )

        semantic_score = data[
            "semantic_score"
        ]

        hybrid_score = (
            TITLE_WEIGHT
            * title_score
            +
            SEMANTIC_WEIGHT
            * semantic_score
            +
            TEXT_WEIGHT
            * text_score
        )

        # Exact title boost.
        if title_score >= 0.95:

            hybrid_score += 0.20

        hybrid_score = min(
            hybrid_score,
            1.0
        )

        ranked.append(
            {
                "hit":
                    hit,

                "hybrid_score":
                    hybrid_score,

                "title_score":
                    title_score,

                "semantic_score":
                    semantic_score,

                "text_score":
                    text_score,
            }
        )

    # ========================================================
    # STEP 7 — Sort
    # ========================================================

    _perf("hybrid_scoring", _t)
    _t = time.monotonic()
    ranked.sort(
        key=lambda item:
            item["hybrid_score"],
        reverse=True
    )

    # Observability: full candidate-pool size before selection culls it.
    obs.set_retrieval_meta(candidate_count=len(ranked))

    # ========================================================
    _perf("ranking_sort", _t)
    _t = time.monotonic()
    # STEP 7B — Cross-encoder rerank (Phase 3C, optional)
    # ========================================================

    # Validated cross-encoder reordering of the fused pool, EXACTLY the
    # Phase 3B experiment semantics: top-RERANK_DEPTH of the fused
    # order is re-sorted by reranker score (stable ties), the tail keeps
    # fused order. OFF by default; on any failure falls back to the
    # fused ranking below (never fails the request).
    if RERANK_ENABLED:

        reranker = _get_reranker()

        if reranker is not None:

            ranked = _rerank_candidates(
                question, ranked, RERANK_DEPTH, reranker_instance=reranker
            )

        _perf("reranker_stage", _t)

    # ========================================================
    _t = time.monotonic()
    # STEP 8 — Exact-topic detection
    # ========================================================

    exact_topic_candidates = [

        item

        for item in ranked

        if item[
            "title_score"
        ] >= 0.95

    ]

    exact_topic_found = (
        len(exact_topic_candidates)
        > 0
    )

    if topic:

        if exact_topic_found:

            print(
                f"  Exact topic match found: {topic}"
            )

        else:

            print(
                f"  WARNING: No exact title match found for: {topic}"
            )

            print(
                "  Semantic results will be treated cautiously."
            )

    # Observability: whether an exact-topic source exists for this query.
    obs.set_retrieval_meta(exact_topic=exact_topic_found)

    # ========================================================
    # STEP 9 — Final selection (apply_production_selection)
    # ========================================================

    # The verbatim production selection rules (threshold gate, one chunk
    # per normalized title, top_k), extracted in Phase 3 and proven
    # behaviorally identical to the previous inline copy by the eval
    # suite. Reusing it here guarantees the reranked pool is selected
    # with the SAME rules as the disabled path.
    selected = apply_production_selection(ranked, top_k=top_k)

    # seen_videos is needed below only for the diagnostics cull_reason;
    # recompute it from the selection so the snapshot is unchanged.
    seen_videos = {
        normalize_text(
            (hit.payload or {}).get("video_title", "Unknown")
        )
        for hit in selected
    }

    # ========================================================
    _perf("source_selection", _t)
    _t = time.monotonic()
    # STEP 10 — Store final retrieval scores
    # ========================================================

    for item in ranked:

        point_id = str(
            item["hit"].id
        )

        _set_retrieval_scores(
            {**_retrieval_scores_var.get(), point_id: item["hybrid_score"]}
        )

    # ========================================================
    # STEP 11 — Debug output
    # ========================================================

    if selected:

        print(
            "\n  Hybrid retrieval:"
        )

        for index, hit in enumerate(
            selected,
            start=1
        ):

            payload = hit.payload or {}

            title = payload.get(
                "video_title",
                "Unknown"
            )

            score = get_retrieval_score(
                str(hit.id)
            )

            print(
                f"    {index}. {title}"
            )

            print(
                f"       hybrid score: {score:.4f}"
            )

    _perf("score_store_and_debug", _t)

    # Diagnostics snapshot for the evaluation harness (Backend/eval).
    # Captures the ranked candidate pool and why each entry did or did
    # not reach the final selection, WITHOUT changing any scoring,
    # ordering, or selection behavior. Mirrors the ContextVar pattern:
    # the stored dict is replaced atomically once per request and is
    # treated as read-only by consumers.
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


def apply_production_selection(ranked, top_k=TOP_K):
    """Apply the EXACT production selection rules to a pre-ranked pool.

    Extracted verbatim from retrieve_chunks' STEP 8+9 so the evaluation
    harness can re-run production selection over a re-ranked pool without
    duplicating (and potentially diverging from) the rules:

      - one chunk per normalized video title (dedup, insertion order)
      - drop candidates with semantic_score < MIN_SCORE_THRESHOLD AND
        title_score < 0.70 (the production threshold gate)
      - stop at top_k

    "ranked" items must expose the same keys retrieve_chunks builds
    ("hit" with payload, "title_score", "semantic_score"). The input
    list is NOT mutated; iteration order defines dedup priority exactly
    as in production. retrieve_chunks itself still runs its own inline
    copy of this logic (untouched) — the two are kept aligned by the
    eval tests, which assert identical selection on shared inputs.
    """
    selected = []
    seen_videos = set()

    for item in ranked:

        hit = item["hit"]

        payload = hit.payload or {}

        title = payload.get(
            "video_title",
            "Unknown"
        )

        normalized_title = normalize_text(
            title
        )

        if normalized_title in seen_videos:
            continue

        semantic_score = item[
            "semantic_score"
        ]

        title_score = item[
            "title_score"
        ]

        if (

            semantic_score
            < MIN_SCORE_THRESHOLD

            and

            title_score
            < 0.70

        ):
            continue

        selected.append(hit)

        seen_videos.add(normalized_title)

        if len(selected) >= top_k:
            break

    return selected


def get_retrieval_score(point_id):
    """Return a retrieval score for the current request only.

    Request-local storage (ContextVar), so two concurrent FastAPI
    tasks never overwrite each other's scores.
    """
    return float(_retrieval_scores_var.get().get(str(point_id), 0.0))


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================

def classify_sources(
    question,
    chunks
):
    """
    Classify retrieved sources as PRIMARY or RELATED.

    This information is passed to Gemini so it can reason about
    source priority rather than treating every retrieved chunk
    equally.
    """

    sources = []

    topic = extract_topic(
        question
    )

    for index, hit in enumerate(
        chunks,
        start=1
    ):

        payload = hit.payload or {}

        title = payload.get(
            "video_title",
            "Unknown"
        )

        title_score = title_exact_match(
            question,
            title
        )

        if title_score >= 0.95:

            source_type = "PRIMARY — EXACT TOPIC"

        elif title_score >= 0.45:

            source_type = "RELATED — PARTIAL TOPIC MATCH"

        else:

            source_type = "RELATED — SEMANTIC SUPPORT"

        sources.append(
            {
                "index": index,
                "title": title,
                "type": source_type,
                "score": get_retrieval_score(hit.id)
            }
        )

    return sources


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_prompt(
    question,
    chunks
):

    if not chunks:
        return None

    context_blocks = []

    source_metadata = classify_sources(
        question,
        chunks
    )

    metadata_by_index = {
        item["index"]: item
        for item in source_metadata
    }

    for i, hit in enumerate(
        chunks,
        start=1
    ):

        payload = hit.payload or {}

        text = payload.get(
            "text",
            ""
        ).strip()

        title = payload.get(
            "video_title",
            "Unknown"
        )

        if not text:
            continue

        metadata = metadata_by_index.get(
            i,
            {}
        )

        source_type = metadata.get(
            "type",
            "RELATED"
        )

        score = metadata.get(
            "score",
            0.0
        )

        context_blocks.append(
            f"""
[Source {i}]
SOURCE TYPE: {source_type}
VIDEO TITLE: "{title}"
RETRIEVAL SCORE: {score:.4f}

TRANSCRIPT:
{text}
""".strip()
        )

    if not context_blocks:
        return None

    context_text = "\n\n".join(
        context_blocks
    )

    topic = extract_topic(
        question
    )

    detected_topic = (
        topic
        if topic
        else "general DSA"
    )

    return f"""
============================================================
STUDENT QUESTION
============================================================

{question}


============================================================
DETECTED TOPIC
============================================================

{detected_topic}


============================================================
RETRIEVED STRIVER KNOWLEDGE
============================================================

{context_text}


============================================================
YOUR TASK
============================================================

Answer the student's question using the retrieved knowledge above.

First, internally analyze the question and determine what the
student actually needs.

Then internally identify which source is the PRIMARY exact-topic
source and which information is relevant.

Then construct the best teaching response for this particular
question.

Do not expose your internal reasoning.

Important:

- Teach rather than merely summarize.
- Do not stop after stating the main idea.
- Explain WHY when the source supports the reasoning.
- Preserve brute → better → optimal progression when relevant.
- Include important implementation details when relevant.
- Use examples/dry runs only when supported by the context.
- Do not confuse related problems.
- Do not invent missing information.
- Do not mention the retrieval system.
- Answer directly.

============================================================
FINAL RESPONSE
============================================================

Return only the student-facing answer.
""".strip()


# ============================================================
# GEMINI BACKGROUND INTERACTION
# ============================================================

def _extract_api_error_code(error):
    """Best-effort extraction of an HTTP/API error code from SDK exceptions."""
    for attr in ("code", "status_code", "http_status"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    text = str(error)
    match = re.search(r"\b([45]\d{2})\b", text)
    return int(match.group(1)) if match else None


def _is_daily_quota_error(error):
    """Detect the free-tier DAILY quota exhaustion message.

    Observed: 429 with 'limit: 20 requests per day on Free Tier'.
    Unlike per-minute rate limits, this cannot clear within a request's
    lifetime, so retrying only delays the inevitable failure.
    """
    text = str(error).lower()
    return (
        "requests per day" in text
        or ("per day" in text and "limit" in text)
    )


def _is_transient_gemini_error(error):
    # A daily-quota 429 looks transient by code but is permanent for
    # the rest of the day; check it first.
    if _is_daily_quota_error(error):
        return False
    """Identify failures that are normally safe to retry."""
    code = _extract_api_error_code(error)
    if code in {408, 429, 500, 502, 503, 504}:
        return True

    text = str(error).lower()
    transient_phrases = (
        "high demand",
        "temporarily unavailable",
        "temporary",
        "overloaded",
        "rate limit",
        "rate-limit",
        "too many requests",
        "service unavailable",
        "internal server error",
        "deadline exceeded",
        "timeout",
        "timed out",
    )
    return any(phrase in text for phrase in transient_phrases)


def _retry_delay(attempt):
    """Exponential backoff with bounded jitter."""
    base = min(
        GEMINI_RETRY_MAX_SECONDS,
        GEMINI_RETRY_BASE_SECONDS * (2 ** attempt),
    )
    return min(
        GEMINI_RETRY_MAX_SECONDS,
        base + random.uniform(0.0, 0.75),
    )


class _GeminiPermanentError(Exception):
    """A Gemini failure that no amount of retrying can fix (e.g. bad key)."""


class _Gemini503UnavailableError(Exception):
    """Sustained Gemini 503 capacity failure -- fail this request fast.

    Raised once GEMINI_503_MAX_CONSECUTIVE consecutive 503s accumulate
    within a single ask_gemini call. ask_gemini deliberately does NOT
    start a fresh interaction for this: during a capacity incident
    every new interaction is poisoned identically (create OK, then
    503 on polls), so retrying only burns wall-clock time and quota.
    """


def _is_503_service_unavailable(error):
    """Detect the Gemini capacity 503 (service_unavailable / high demand)."""
    code = _extract_api_error_code(error)
    if code == 503:
        return True
    text = str(error).lower()
    return "service_unavailable" in text


def _register_503_error(error, streak):
    """Count a 503 toward the fast-fail streak; raise at the threshold.

    The streak is deliberately NOT reset on non-503 errors: the observed
    capacity signature alternates 503s with poisoned-interaction 400s,
    and those 400s must not reset the count. There is deliberately no
    reset at all: only a fresh ask_gemini request starts a new streak.
    """
    if not _is_503_service_unavailable(error):
        return
    streak["count"] += 1
    if streak["count"] >= GEMINI_503_MAX_CONSECUTIVE:
        raise _Gemini503UnavailableError(
            f"Gemini capacity unavailable: {streak['count']} consecutive "
            "503 service_unavailable responses. Failing fast instead of "
            "waiting out the outage."
        ) from error


def _create_gemini_interaction(user_message, deadline=None, error_503_streak=None):
    """Create a background interaction with bounded transient-error retries."""
    if error_503_streak is None:
        error_503_streak = {"count": 0}
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        # The total generation budget must also cover create attempts:
        # each create call can stall up to the client HTTP timeout before
        # raising, so retries without a deadline check can hang for many
        # minutes. The first attempt is always allowed through.
        if (
            deadline is not None
            and attempt > 0
            and time.monotonic() >= deadline
        ):
            print("  Gemini create retry budget exhausted.")
            return None

        _t_create = time.monotonic()
        try:
            _created = genai_client.interactions.create(
                model=GEMINI_MODEL,
                input=user_message,
                system_instruction=SYSTEM_PROMPT,
                background=True,
                # NOTE: do NOT add "thinking_level" here. Passing it inside
                # generation_config of a background interaction makes every
                # subsequent GET on that interaction fail with
                # HTTP 400 invalid_request (reproduced 2026-09-20 via both
                # SDK polling and raw REST; the identical request without
                # it completes normally).
                generation_config={
                    "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
                },
            )
            _perf("gemini_create_call", _t_create)
            return _created
        except Exception as error:
            if not _is_transient_gemini_error(error):
                raise _GeminiPermanentError(
                    f"Gemini create failed permanently: {error}"
                ) from error

            # Capacity 503s during create count toward the same
            # request-wide fast-fail streak as poll 503s.
            _register_503_error(error, error_503_streak)

            if attempt >= GEMINI_MAX_RETRIES:
                print(f"  Failed to start Gemini task: {error}")
                return None

            delay = _retry_delay(attempt)
            print(
                f"  Gemini start failed temporarily. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    return None

def _poll_gemini_interaction(interaction_id):
    """
    Poll a background interaction via the official google-genai SDK.

    Returns an Interaction object exposing ``.status`` and the
    client-computed ``.output_text`` property (built from
    steps[].content[].text). Raw REST polling was removed: the SDK path
    was verified working end to end, and raw REST responses have no
    top-level output_text field to read.

    Exceptions propagate to ask_gemini's retry loop, which classifies
    transient vs permanent failures.
    """
    return genai_client.interactions.get(id=interaction_id)

def _generate_with_single_interaction(
    user_message, deadline, error_503_streak=None, degraded_deadline=None
):
    """Run one create -> poll lifecycle. Return text, or None to allow retry.

    Raises _GeminiPermanentError only for failures where retrying cannot
    succeed (e.g. authentication). Everything else — including an
    interaction poisoned by a mid-flight 503, which then 400s on every
    poll — returns None so ask_gemini can start a fresh interaction.
    """
    if error_503_streak is None:
        error_503_streak = {"count": 0}

    interaction = _create_gemini_interaction(
        user_message,
        deadline=deadline,
        error_503_streak=error_503_streak,
    )
    if interaction is None:
        return None

    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        print("  Gemini returned no interaction ID.")
        return None

    print(f"  Gemini task started: {interaction_id}")
    print(
        "  Initial status: "
        f"{getattr(interaction, 'status', 'unknown')}"
    )

    start_time = time.monotonic()
    consecutive_poll_failures = 0
    poll_count = 0

    while True:
        # Degraded budget: once a capacity 503 has been seen, the
        # whole request must conclude quickly. Checked HERE (not only
        # in ask_gemini) because this poll loop can otherwise run for
        # the full 90-second interaction window.
        if (
            degraded_deadline is not None
            and error_503_streak["count"] > 0
            and time.monotonic() >= degraded_deadline
        ):
            print(
                "  Gemini degraded budget (post-503) exhausted; "
                "stopping this interaction."
            )
            return None

        elapsed = time.monotonic() - start_time
        if (
            elapsed >= GEMINI_MAX_WAIT_SECONDS
            or time.monotonic() >= deadline
        ):
            print(
                f"  Gemini task exceeded the local "
                f"{GEMINI_MAX_WAIT_SECONDS}s wait limit."
            )
            return None

        _t_poll = time.monotonic()
        try:
            result = _poll_gemini_interaction(interaction_id)

            poll_count += 1
            _perf(
                "gemini_poll_get",
                _t_poll,
                extra=(
                    f"poll={poll_count} "
                    f"status={getattr(result, 'status', 'unknown')}"
                ),
            )
            consecutive_poll_failures = 0

        except Exception as error:
            consecutive_poll_failures += 1
            code = _extract_api_error_code(error)

            print(
                f"  Gemini polling error "
                f"(attempt {consecutive_poll_failures}/"
                f"{GEMINI_MAX_RETRIES + 1}, code={code}): {error}"
            )

            if _is_daily_quota_error(error):
                # No client-side action can help until the daily reset.
                raise _GeminiPermanentError(
                    f"Gemini daily quota exhausted: {error}"
                ) from error

            # Capacity 503s count toward the request-wide fast-fail
            # streak; the threshold raise propagates to ask_gemini.
            _register_503_error(error, error_503_streak)

            if (
                not _is_transient_gemini_error(error)
                or consecutive_poll_failures > GEMINI_MAX_RETRIES
            ):
                # NOTE: non-transient poll failures (e.g. 400) return None
                # rather than raising: an interaction can be individually
                # poisoned (observed after mid-flight 503s) while a fresh
                # interaction succeeds, so ask_gemini retries with a new
                # interaction instead of giving up.
                print("  Non-retryable Gemini polling failure.")
                return None

            delay = _retry_delay(consecutive_poll_failures - 1)
            print(
                f"  Gemini appears temporarily unavailable. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)
            continue

        status = getattr(result, "status", "unknown")
        print(f"  Gemini status: {status}")

        # ----------------------------------------------------
        # Terminal states that carry generated text
        # ----------------------------------------------------

        if status in TERMINAL_TEXT_STATUSES:
            response_text = getattr(result, "output_text", None)

            if response_text and response_text.strip():
                if status != "completed":
                    print(
                        f"  Gemini stopped early with status '{status}'; "
                        "using the partial output produced."
                    )
                print("  Gemini task completed.")
                return response_text.strip()

            print(
                f"  Gemini finished with status '{status}' "
                "but returned no text."
            )
            return None

        if status == "failed":
            print(
                f"  Gemini task failed: "
                f"{getattr(result, 'error', None)}"
            )
            return None

        if status in ("cancelled", "canceled"):
            print("  Gemini task was cancelled.")
            return None

        if status == "requires_action":
            # Only relevant for tool-using interactions; this pipeline
            # registers no tools, so this state is a dead end here.
            print("  Gemini interaction requires action; not supported.")
            return None

        # queued / in_progress / any future status: keep polling until
        # the bounded wait in GEMINI_MAX_WAIT_SECONDS expires.
        time.sleep(GEMINI_POLL_INTERVAL_SECONDS)


def ask_gemini(user_message):
    """Generate a grounded answer, retrying with fresh interactions.

    A single interaction can be permanently poisoned by a transient
    server-side event (observed: 503 high-demand mid-generation, after
    which every poll returns 400 invalid_request forever). The only
    client-side recovery is a brand-new interaction, so failures inside
    one lifecycle are retried here within GEMINI_MAX_TOTAL_SECONDS.
    """
    deadline = time.monotonic() + GEMINI_MAX_TOTAL_SECONDS
    attempt = 0
    # Request-wide 503 fast-fail streak, shared across all
    # create/poll attempts so a sustained capacity incident is
    # recognized no matter where in the lifecycle it appears.
    error_503_streak = {"count": 0}
    # Armed for the whole request; enforced only after a 503 is seen.
    degraded_deadline = time.monotonic() + GEMINI_503_DEGRADED_MAX_SECONDS

    while True:
        # Never open a new interaction window once the degraded
        # budget is spent.
        if (
            error_503_streak["count"] > 0
            and time.monotonic() >= degraded_deadline
        ):
            print(
                "  Gemini degraded budget exhausted; not retrying "
                "after 503."
            )
            return None

        attempt += 1
        print("  Starting Gemini background task...")

        _t_gen = time.monotonic()
        try:
            answer = _generate_with_single_interaction(
                user_message,
                deadline,
                error_503_streak=error_503_streak,
                degraded_deadline=degraded_deadline,
            )
            _perf("gemini_interaction_success", _t_gen, extra=f"attempt={attempt}")
        except _Gemini503UnavailableError as error:
            print(f"  {error}")
            print("  Failing fast (Gemini capacity incident).")
            return None
        except _GeminiPermanentError as error:
            print(f"  {error}")
            return None

        if answer is not None:
            return answer

        if attempt > GEMINI_MAX_RETRIES or time.monotonic() >= deadline:
            print(f"  Gemini generation failed after {attempt} attempt(s).")
            return None

        delay = _retry_delay(attempt - 1)
        print(
            f"  Retrying generation with a fresh interaction "
            f"in {delay:.1f}s..."
        )
        time.sleep(min(delay, max(0.0, deadline - time.monotonic())))


# ============================================================
# SOURCE FORMATTING
# ============================================================

def format_sources(
    chunks
):

    if not chunks:
        return "  (no sources)"

    lines = []

    for i, hit in enumerate(
        chunks,
        start=1
    ):

        payload = hit.payload or {}

        title = payload.get(
            "video_title",
            "Unknown"
        )

        url = payload.get(
            "youtube_url",
            ""
        )

        try:

            start = int(
                float(
                    payload.get(
                        "start",
                        0
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            start = 0

        # ----------------------------------------------------
        # Correct YouTube timestamp link
        # ----------------------------------------------------

        if url:

            separator = (
                "&"
                if "?" in url
                else "?"
            )

            timestamp_link = (
                f"{url}"
                f"{separator}"
                f"t={start}"
            )

        else:

            timestamp_link = "N/A"

        point_id = str(
            hit.id
        )

        if url:
            score = get_retrieval_score(point_id)
        else:
            score = 0.0

        lines.append(
            f"  [{i}] {title} — {start}s\n"
            f"      {timestamp_link}\n"
            f"      (hybrid score: {score:.4f})"
        )

    return "\n".join(
        lines
    )


# ============================================================
# QUESTION VALIDATION
# ============================================================

def validate_question(
    raw_question
):

    question = raw_question.strip()

    if not question:

        return (
            None,
            "Please enter a non-empty question."
        )

    if len(question) > MAX_QUESTION_LENGTH:

        return (
            None,
            (
                f"Question too long "
                f"({len(question)} characters). "
                f"Keep it under "
                f"{MAX_QUESTION_LENGTH} characters."
            )
        )

    return (
        question,
        None
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "MISSION ANTHROPIC — STRIVER A2Z RAG"
    )

    print(
        "=" * 60
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME} "
        f"({collection_info.points_count} chunks indexed)"
    )

    print(
        "Ask a question about the DSA course. "
        "Type 'exit' to quit.\n"
    )

    while True:

        try:

            raw_question = input(
                "Your question: "
            )

        except EOFError:

            print(
                "\nInput closed. Goodbye, bro!"
            )

            break

        question, error = (
            validate_question(
                raw_question
            )
        )

        if error:

            print(
                f"{error}\n"
            )

            continue

        if question.lower() in (
            "exit",
            "quit"
        ):

            print(
                "Goodbye, bro!"
            )

            break

        # ====================================================
        # RETRIEVAL
        # ====================================================

        print(
            "\nSearching knowledge base..."
        )

        chunks = retrieve_chunks(
            question
        )

        if not chunks:

            print(
                "\nNo relevant content found.\n"
            )

            continue

        # ====================================================
        # BUILD AGENTIC PROMPT
        # ====================================================

        user_message = build_prompt(
            question,
            chunks
        )

        if user_message is None:

            print(
                "Retrieved chunks contained no usable text.\n"
            )

            continue

        # ====================================================
        # GEMINI
        # ====================================================

        print(
            "Generating answer...\n"
        )

        answer = ask_gemini(
            user_message
        )

        if answer is None:

            print(
                "\nCould not generate an answer "
                "because the Gemini request failed."
            )

            print(
                "Check the Gemini error above.\n"
            )

            continue

        # ====================================================
        # ANSWER
        # ====================================================

        print(
            "=" * 60
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 60
        )

        print(
            answer
        )

        # ====================================================
        # SOURCES
        # ====================================================

        print(
            "\n"
            +
            "-" * 60
        )

        print(
            "SOURCES"
        )

        print(
            "-" * 60
        )

        print(
            format_sources(
                chunks
            )
        )

        print(
            "\n"
            +
            "=" * 60
            +
            "\n"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nInterrupted. Goodbye, bro!"
        )

        sys.exit(0)

    except Exception as e:

        fail(
            f"Unexpected error: {e}"
        )