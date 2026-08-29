import os
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

QDRANT_DIR = os.path.join(BASE_DIR, "qdrant_db")


# ==========================================
# CONFIG
# ==========================================

COLLECTION_NAME = "striver_a2z"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
TOP_K = 4
MIN_SCORE_THRESHOLD = 0.25   # below this, treat result as "not relevant"
GEMINI_MODEL = "gemini-3.6-flash"
MAX_QUESTION_LENGTH = 500
GEMINI_MAX_RETRIES = 2


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


# ==========================================
# STEP 0 — VALIDATE ENVIRONMENT
# ==========================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key or not api_key.strip():
    fail("GEMINI_API_KEY environment variable is not set or empty. "
         "Set it with: setx GEMINI_API_KEY \"your-key\" "
         "then restart your terminal.")

if not os.path.isdir(QDRANT_DIR):
    fail(f"Qdrant storage not found at: {QDRANT_DIR}. "
         f"Run load_to_qdrant.py first.")


# ==========================================
# STEP 1 — CONNECT TO QDRANT
# ==========================================

try:
    qdrant_client = QdrantClient(path=QDRANT_DIR)
except Exception as e:
    fail(f"Failed to connect to Qdrant: {e}")

try:
    collections = [c.name for c in qdrant_client.get_collections().collections]
except Exception as e:
    fail(f"Failed to list Qdrant collections: {e}")

if COLLECTION_NAME not in collections:
    fail(f"Collection '{COLLECTION_NAME}' does not exist. "
         f"Run load_to_qdrant.py first.")

try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        fail(f"Collection '{COLLECTION_NAME}' exists but has zero points. "
             f"Run load_to_qdrant.py to populate it.")
except Exception as e:
    fail(f"Failed to inspect collection: {e}")


# ==========================================
# STEP 2 — LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

try:
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
except Exception as e:
    fail(f"Failed to load embedding model: {e}")

# Sanity check the model actually produces the expected dimension
try:
    test_vec = embed_model.encode("test").tolist()
    if len(test_vec) != VECTOR_SIZE:
        fail(f"Embedding model produces dimension {len(test_vec)}, "
             f"but Qdrant collection expects {VECTOR_SIZE}. "
             f"Model/collection mismatch.")
except Exception as e:
    fail(f"Failed to validate embedding model output: {e}")


# ==========================================
# STEP 3 — CONNECT TO GEMINI
# ==========================================

try:
    genai_client = genai.Client(api_key=api_key)
except Exception as e:
    fail(f"Failed to initialize Gemini client: {e}")


# ==========================================
# CORE FUNCTIONS
# ==========================================

def retrieve_chunks(question, top_k=TOP_K):

    try:
        query_vector = embed_model.encode(question).tolist()
    except Exception as e:
        print(f"ERROR: Failed to embed question: {e}")
        return []

    if len(query_vector) != VECTOR_SIZE:
        print(f"ERROR: Query vector dimension mismatch: "
              f"got {len(query_vector)}, expected {VECTOR_SIZE}.")
        return []

    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points
    except Exception as e:
        print(f"ERROR: Qdrant search failed: {e}")
        return []

    # Filter out weak/irrelevant matches instead of trusting top_k blindly
    relevant = [hit for hit in results if hit.score >= MIN_SCORE_THRESHOLD]

    return relevant


def build_prompt(question, chunks):

    if not chunks:
        return None

    context_blocks = []

    for i, hit in enumerate(chunks, start=1):
        payload = hit.payload or {}
        text = payload.get("text", "").strip()

        if not text:
            continue

        context_blocks.append(f"[Source {i}]\n{text}")

    if not context_blocks:
        return None

    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful teaching assistant answering questions about a DSA (Data Structures & Algorithms) course based on video transcripts.

Use ONLY the context below to answer the question. If the context does not contain enough information to answer, say so clearly instead of guessing or using outside knowledge.

Context:
{context_text}

Question: {question}

Answer clearly and concisely, as if explaining to a student."""

    return prompt


def ask_gemini(prompt):

    last_error = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):

        try:
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        except Exception as e:
            last_error = e
            print(f"  Gemini call attempt {attempt} failed: {e}")
            continue

        response_text = getattr(response, "text", None)

        if response_text and response_text.strip():
            return response_text.strip()

        last_error = "Empty response from Gemini"
        print(f"  Gemini call attempt {attempt} returned empty response.")

    print(f"ERROR: Gemini API failed after {GEMINI_MAX_RETRIES} attempt(s): "
          f"{last_error}")
    return None


def format_sources(chunks):

    if not chunks:
        return "  (no sources)"

    lines = []

    for i, hit in enumerate(chunks, start=1):
        payload = hit.payload or {}
        title = payload.get("video_title", "Unknown")
        url = payload.get("youtube_url", "")

        try:
            start = int(float(payload.get("start", 0)))
        except (TypeError, ValueError):
            start = 0

        timestamp_link = f"{url}&t={start}" if url else "N/A"

        lines.append(
            f"  [{i}] {title} — {start}s\n"
            f"      {timestamp_link}\n"
            f"      (score: {hit.score:.4f})"
        )

    return "\n".join(lines)


def validate_question(raw_question):

    question = raw_question.strip()

    if not question:
        return None, "Please enter a non-empty question."

    if len(question) > MAX_QUESTION_LENGTH:
        return None, (f"Question too long ({len(question)} chars). "
                       f"Keep it under {MAX_QUESTION_LENGTH} characters.")

    return question, None


# ==========================================
# MAIN LOOP
# ==========================================

def main():

    print("\n" + "=" * 60)
    print("MISSION ANTHROPIC — STRIVER A2Z RAG")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME} "
          f"({collection_info.points_count} chunks indexed)")
    print("Ask a question about the DSA course. Type 'exit' to quit.\n")

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

        print("\nSearching knowledge base...")
        chunks = retrieve_chunks(question)

        if not chunks:
            print("No relevant content found for this question "
                  "(nothing scored above the relevance threshold).\n")
            continue

        prompt = build_prompt(question, chunks)

        if prompt is None:
            print("Retrieved chunks had no usable text. Skipping.\n")
            continue

        print("Generating answer...\n")
        answer = ask_gemini(prompt)

        if answer is None:
            print("Could not generate an answer due to a Gemini API error. "
                  "Try again in a moment.\n")
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