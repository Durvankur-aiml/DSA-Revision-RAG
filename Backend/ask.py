import os
import re
import sys
import time
import random
import threading

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


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

# Protects the in-process retrieval-score snapshot when FastAPI handles
# multiple requests concurrently.
RETRIEVAL_SCORE_LOCK = threading.Lock()


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

MAX_QUESTION_LENGTH = 500


# ============================================================
# RUNTIME RETRIEVAL SCORE STORAGE
# ============================================================

LAST_RETRIEVAL_SCORES = {}


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


def title_exact_match(question, video_title):
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
        intro_markers = {
            "introduction", "intro", "basics", "basic", "fundamentals",
            "overview", "concept", "concepts", "real", "life"
        }
        title_tokens = set(title.split())

        if title_tokens & intro_markers:
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
# LOAD ALL PAYLOADS FOR LEXICAL SEARCH
# ============================================================

def load_all_points():

    all_points = []

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

    return all_points


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_retrieve(
    question
):

    try:

        query_vector = embed_model.encode(
            question
        ).tolist()

    except Exception as e:

        print(
            f"ERROR: Failed to embed question: {e}"
        )

        return []

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

    return results


# ============================================================
# LEXICAL RETRIEVAL
# ============================================================

def lexical_retrieve(
    question,
    all_points
):

    candidates = []

    for point in all_points:

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


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    top_k=TOP_K
):

    global LAST_RETRIEVAL_SCORES

    with RETRIEVAL_SCORE_LOCK:
        LAST_RETRIEVAL_SCORES = {}

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

    # ========================================================
    # STEP 2 — Semantic search
    # ========================================================

    semantic_results = semantic_retrieve(
        question
    )

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

    try:

        all_points = load_all_points()

    except Exception as e:

        print(
            f"ERROR: Failed to scan Qdrant records: {e}"
        )

        all_points = []

    # ========================================================
    # STEP 4 — Lexical/title search
    # ========================================================

    lexical_results = lexical_retrieve(
        question,
        all_points
    )

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

    ranked.sort(
        key=lambda item:
            item["hybrid_score"],
        reverse=True
    )

    # ========================================================
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

    # ========================================================
    # STEP 9 — Final selection
    # ========================================================

    selected = []

    seen_videos = set()

    for item in ranked:

        hit = item[
            "hit"
        ]

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

        selected.append(
            hit
        )

        seen_videos.add(
            normalized_title
        )

        if len(selected) >= top_k:
            break

    # ========================================================
    # STEP 10 — Store final retrieval scores
    # ========================================================

    for item in ranked:

        point_id = str(
            item["hit"].id
        )

        LAST_RETRIEVAL_SCORES[
            point_id
        ] = item[
            "hybrid_score"
        ]

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

            point_id = str(
                hit.id
            )

            score = LAST_RETRIEVAL_SCORES.get(
                point_id,
                0.0
            )

            print(
                f"    {index}. {title}"
            )

            print(
                f"       hybrid score: {score:.4f}"
            )

    return selected


def get_retrieval_score(point_id):
    """Return a retrieval score without exposing mutable global state."""
    with RETRIEVAL_SCORE_LOCK:
        return float(LAST_RETRIEVAL_SCORES.get(str(point_id), 0.0))


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


def _create_gemini_interaction(user_message, deadline=None):
    """Create a background interaction with bounded transient-error retries."""
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

        try:
            return genai_client.interactions.create(
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
        except Exception as error:
            if not _is_transient_gemini_error(error):
                raise _GeminiPermanentError(
                    f"Gemini create failed permanently: {error}"
                ) from error

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

def _generate_with_single_interaction(user_message, deadline):
    """Run one create -> poll lifecycle. Return text, or None to allow retry.

    Raises _GeminiPermanentError only for failures where retrying cannot
    succeed (e.g. authentication). Everything else — including an
    interaction poisoned by a mid-flight 503, which then 400s on every
    poll — returns None so ask_gemini can start a fresh interaction.
    """
    interaction = _create_gemini_interaction(
        user_message,
        deadline=deadline,
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

    while True:
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

        try:
            result = _poll_gemini_interaction(interaction_id)

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

    while True:
        attempt += 1
        print("  Starting Gemini background task...")

        try:
            answer = _generate_with_single_interaction(user_message, deadline)
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

        score = get_retrieval_score(point_id)

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