import os
import sys
import json
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
QDRANT_DIR = os.path.join(BASE_DIR, "qdrant_db")


COLLECTION_NAME = "striver_a2z"
VECTOR_SIZE = 384
MODEL_NAME = "all-MiniLM-L6-v2"
UPLOAD_BATCH_SIZE = 100
TEST_QUESTION = "What is a variable?"
TEST_QUERY_LIMIT = 3


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def make_point_id(video_id, chunk_index):
    """Deterministic unique ID per video+chunk, so multiple videos
    never collide, and re-running upload for the same video is idempotent
    (overwrites its own points instead of duplicating)."""
    unique_string = f"{video_id}_{chunk_index}"
    return int(hashlib.md5(unique_string.encode()).hexdigest()[:12], 16)


# ==========================================
# STEP 0 — VALIDATE PATHS
# ==========================================

if not os.path.isdir(EMBEDDINGS_DIR):
    fail(f"Embeddings directory does not exist: {EMBEDDINGS_DIR}")

try:
    os.makedirs(QDRANT_DIR, exist_ok=True)
except OSError as e:
    fail(f"Could not create Qdrant storage directory: {QDRANT_DIR}\n{e}")


# ==========================================
# STEP 1 — FIND EMBEDDINGS FILE(S)
# ==========================================

try:
    embedding_files = sorted([
        file for file in os.listdir(EMBEDDINGS_DIR)
        if file.lower().endswith("_embeddings.json")
    ])
except OSError as e:
    fail(f"Could not read embeddings directory: {e}")

if not embedding_files:
    fail(f"No embeddings JSON found inside {EMBEDDINGS_DIR}. "
         f"Run generate_embeddings.py first.")

print(f"Embedding file(s) to load: {len(embedding_files)} file(s)")


# ==========================================
# STEP 2 — LOAD AND VALIDATE RECORDS
# ==========================================

REQUIRED_KEYS = (
    "chunk_index", "video_id", "video_title",
    "youtube_url", "start", "end", "text", "embedding"
)

all_records = []
skipped_invalid = 0

for filename in embedding_files:

    path = os.path.join(EMBEDDINGS_DIR, filename)

    try:
        with open(path, "r", encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as e:
        print(f"WARNING: File '{filename}' is not valid JSON: {e}. Skipping file.")
        continue
    except OSError as e:
        print(f"WARNING: Could not open '{filename}': {e}. Skipping file.")
        continue

    if not isinstance(records, list) or len(records) == 0:
        print(f"WARNING: '{filename}' contains no records. Skipping file.")
        continue

    for i, record in enumerate(records):

        if not isinstance(record, dict):
            skipped_invalid += 1
            continue

        missing = [k for k in REQUIRED_KEYS if k not in record]
        if missing:
            skipped_invalid += 1
            continue

        embedding = record["embedding"]
        if not isinstance(embedding, list) or len(embedding) != VECTOR_SIZE:
            skipped_invalid += 1
            continue

        if not all(isinstance(x, (int, float)) for x in embedding):
            skipped_invalid += 1
            continue

        if not isinstance(record["text"], str) or not record["text"].strip():
            skipped_invalid += 1
            continue

        all_records.append(record)

if skipped_invalid > 0:
    print(f"\nSkipped {skipped_invalid} invalid record(s) total.")

if not all_records:
    fail("No valid records to upload after validation. Aborting.")

unique_videos = {r["video_id"] for r in all_records}
print(f"Total valid records loaded: {len(all_records)} "
      f"(across {len(unique_videos)} video(s))")


# ==========================================
# STEP 3 — CONNECT TO LOCAL PERSISTENT QDRANT
# ==========================================

print(f"\nConnecting to local Qdrant storage at: {QDRANT_DIR}")

try:
    client = QdrantClient(path=QDRANT_DIR)
except Exception as e:
    fail(f"Failed to connect to local Qdrant storage: {e}\n"
         f"Tip: make sure no other process is holding a lock on '{QDRANT_DIR}'.")


# ==========================================
# STEP 4 — CREATE COLLECTION IF IT DOESN'T EXIST
#          (additive — does not wipe existing data)
# ==========================================

try:
    existing_collections = [
        c.name for c in client.get_collections().collections
    ]
except Exception as e:
    fail(f"Failed to list existing Qdrant collections: {e}")

if COLLECTION_NAME not in existing_collections:
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"Collection '{COLLECTION_NAME}' created "
              f"(size={VECTOR_SIZE}, distance=COSINE).")
    except Exception as e:
        fail(f"Failed to create Qdrant collection: {e}")
else:
    try:
        existing_info = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists "
              f"with {existing_info.points_count} point(s). Adding to it.")
    except Exception as e:
        fail(f"Failed to inspect existing collection: {e}")


# ==========================================
# STEP 5 — BUILD POINTS (deterministic IDs, no collisions across videos)
# ==========================================

points = []

for i, record in enumerate(all_records):

    try:
        start = float(record["start"])
        end = float(record["end"])
        chunk_index = int(record["chunk_index"])
    except (TypeError, ValueError):
        continue

    point_id = make_point_id(record["video_id"], chunk_index)

    point = PointStruct(
        id=point_id,
        vector=record["embedding"],
        payload={
            "chunk_index": chunk_index,
            "video_id": record["video_id"],
            "video_title": record["video_title"],
            "youtube_url": record["youtube_url"],
            "start": start,
            "end": end,
            "text": record["text"],
        }
    )

    points.append(point)

if not points:
    fail("No points could be built from valid records. Aborting.")

print(f"\nPrepared {len(points)} point(s) for upload.")


# ==========================================
# STEP 6 — UPLOAD IN BATCHES
# ==========================================

print(f"Uploading in batches of {UPLOAD_BATCH_SIZE}...")

uploaded_count = 0
failed_batches = []

for batch_start in range(0, len(points), UPLOAD_BATCH_SIZE):

    batch = points[batch_start: batch_start + UPLOAD_BATCH_SIZE]
    batch_num = (batch_start // UPLOAD_BATCH_SIZE) + 1
    total_batches = (len(points) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE

    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        uploaded_count += len(batch)
        print(f"  Batch {batch_num}/{total_batches}: "
              f"{len(batch)} point(s) uploaded OK.")
    except Exception as e:
        print(f"  Batch {batch_num}/{total_batches}: FAILED ({e})")
        failed_batches.append(batch_num)

if failed_batches:
    print(f"\nWARNING: {len(failed_batches)} batch(es) failed to upload: "
          f"{failed_batches}")

if uploaded_count == 0:
    fail("Zero points were successfully uploaded. Aborting.")


# ==========================================
# STEP 7 — VERIFY
# ==========================================

try:
    collection_info = client.get_collection(COLLECTION_NAME)
    stored_count = collection_info.points_count
except Exception as e:
    fail(f"Failed to verify collection after upload: {e}")

print(f"Collection now has {stored_count} total point(s) stored.")


# ==========================================
# STEP 8 — TEST QUERY
# ==========================================

print("\n" + "=" * 50)
print("RUNNING TEST QUERY")
print("=" * 50)

try:
    embed_model = SentenceTransformer(MODEL_NAME)
    query_vector = embed_model.encode(TEST_QUESTION).tolist()

    if len(query_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Query vector dimension ({len(query_vector)}) does not "
            f"match collection dimension ({VECTOR_SIZE})."
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TEST_QUERY_LIMIT
    ).points

    if not results:
        print(f"\nWARNING: Query returned zero results.")
    else:
        print(f"\nQuery: \"{TEST_QUESTION}\"\n")

        for rank, hit in enumerate(results, start=1):
            payload = hit.payload or {}

            title = payload.get("video_title", "UNKNOWN")
            start = payload.get("start", "?")
            url = payload.get("youtube_url", "")
            text = payload.get("text", "")

            print(f"#{rank} (score={hit.score:.4f})")
            print(f"  Video: {title}")
            if url:
                print(f"  Link: {url}?t={int(float(start))}")
            print(f"  Text: {text[:150]}{'...' if len(text) > 150 else ''}")
            print()

except Exception as e:
    print(f"WARNING: Test query failed (upload itself was still "
          f"successful): {e}")


# ==========================================
# FINAL OUTPUT
# ==========================================

print("=" * 50)
print("QDRANT LOAD COMPLETED")
print("=" * 50)
print(f"Collection: {COLLECTION_NAME}")
print(f"Records validated this run: {len(all_records)} "
      f"(skipped {skipped_invalid} invalid)")
print(f"Points uploaded this run: {uploaded_count}")
print(f"Total points now in collection: {stored_count}")
if failed_batches:
    print(f"Failed batches: {failed_batches} -- INVESTIGATE THIS")
print(f"Storage location: {QDRANT_DIR}")