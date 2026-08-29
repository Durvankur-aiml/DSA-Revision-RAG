import os
import sys
import json
import torch
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_EMBEDDING_DIM = 384
BATCH_SIZE = 32


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_metadata(video_id):

    metadata_path = os.path.join(METADATA_DIR, f"{video_id}.json")

    if not os.path.exists(metadata_path):
        print(f"  WARNING: No metadata file for video_id '{video_id}'. "
              f"Using fallback values.")
        return {
            "video_id": video_id,
            "video_title": video_id,
            "youtube_url": f"https://youtu.be/{video_id}",
        }

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for key in ("video_id", "video_title", "youtube_url"):
                if key not in data:
                    raise ValueError(f"missing key '{key}'")
            return data
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"  WARNING: Failed to load metadata for '{video_id}': {e}. "
              f"Using fallback values.")
        return {
            "video_id": video_id,
            "video_title": video_id,
            "youtube_url": f"https://youtu.be/{video_id}",
        }


def main():

    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: GPU not detected. Using CPU.")

    if not os.path.isdir(TRANSCRIPT_DIR):
        fail(f"Transcript directory does not exist: {TRANSCRIPT_DIR}")

    transcript_files = [
        f for f in os.listdir(TRANSCRIPT_DIR) if f.lower().endswith(".json")
    ]

    if not transcript_files:
        fail(f"No transcript JSON found inside {TRANSCRIPT_DIR}")

    already_embedded = {
        f.replace("_embeddings.json", "")
        for f in os.listdir(EMBEDDINGS_DIR)
        if f.lower().endswith("_embeddings.json")
    }

    to_process = []
    skipped_files = []

    for f in transcript_files:
        video_id = os.path.splitext(f)[0]
        if video_id in already_embedded:
            skipped_files.append(f)
            continue
        to_process.append(f)

    if skipped_files:
        print(f"Skipping {len(skipped_files)} already-embedded transcript(s).")

    if not to_process:
        print("\nAll transcripts already embedded. Nothing to do.")
        return

    print(f"\n{len(to_process)} new transcript(s) to embed:")
    for f in to_process:
        print(f"  - {f}")

    print(f"\nLoading embedding model: {MODEL_NAME}...")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        fail(f"Failed to load embedding model: {e}")

    try:
        test_vec = model.encode("test").tolist()
        if len(test_vec) != EXPECTED_EMBEDDING_DIM:
            fail(f"Embedding model produces dimension {len(test_vec)}, "
                 f"expected {EXPECTED_EMBEDDING_DIM}.")
    except Exception as e:
        fail(f"Failed to validate embedding model output: {e}")

    success_count = 0
    failed_videos = []

    for transcript_file in to_process:

        video_id = os.path.splitext(transcript_file)[0]
        transcript_path = os.path.join(TRANSCRIPT_DIR, transcript_file)

        print(f"\n{'=' * 50}")
        print(f"Embedding: {transcript_file}")
        print("=" * 50)

        try:
            with open(transcript_path, "r", encoding="utf-8") as file:
                chunks = json.load(file)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ERROR: Could not load transcript: {e}. Skipping.")
            failed_videos.append(transcript_file)
            continue

        if not isinstance(chunks, list) or len(chunks) == 0:
            print(f"  WARNING: No valid chunks in this transcript. Skipping.")
            failed_videos.append(transcript_file)
            continue

        valid_chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            if not all(k in chunk for k in ("start", "end", "text")):
                continue
            if not isinstance(chunk["text"], str) or not chunk["text"].strip():
                continue
            try:
                float(chunk["start"])
                float(chunk["end"])
            except (TypeError, ValueError):
                continue
            valid_chunks.append(chunk)

        if not valid_chunks:
            print(f"  WARNING: No valid chunks after validation. Skipping.")
            failed_videos.append(transcript_file)
            continue

        video_metadata = load_metadata(video_id)

        texts = [c["text"] for c in valid_chunks]

        try:
            embeddings = model.encode(
                texts, batch_size=BATCH_SIZE,
                show_progress_bar=True, convert_to_numpy=True
            )
        except Exception as e:
            print(f"  ERROR: Embedding generation failed: {e}. Skipping.")
            failed_videos.append(transcript_file)
            continue

        if embeddings is None or len(embeddings) != len(valid_chunks):
            print(f"  ERROR: Embedding count mismatch. Skipping.")
            failed_videos.append(transcript_file)
            continue

        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            print(f"  WARNING: Unexpected embedding dimension "
                  f"{embeddings.shape[1]} (expected {EXPECTED_EMBEDDING_DIM}).")

        records = []
        for index, (chunk, vector) in enumerate(zip(valid_chunks, embeddings)):
            records.append({
                "chunk_index": index,
                "video_id": video_metadata["video_id"],
                "video_title": video_metadata["video_title"],
                "youtube_url": video_metadata["youtube_url"],
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "text": chunk["text"],
                "embedding": vector.tolist()
            })

        output_file = os.path.join(
            EMBEDDINGS_DIR, f"{video_id}_embeddings.json"
        )
        temp_file = output_file + ".tmp"

        try:
            with open(temp_file, "w", encoding="utf-8") as file:
                json.dump(records, file, indent=2, ensure_ascii=False)
            os.replace(temp_file, output_file)
        except OSError as e:
            print(f"  ERROR: Failed to save embeddings: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            failed_videos.append(transcript_file)
            continue

        print(f"  Saved {len(records)} embedded chunks to: {output_file}")
        success_count += 1

    print(f"\n{'=' * 50}")
    print(f"BATCH COMPLETE — {success_count}/{len(to_process)} video(s) "
          f"embedded successfully.")
    if failed_videos:
        print(f"Failed: {failed_videos}")
        print("Re-run this script to retry them.")
    print("=" * 50)


if __name__ == "__main__":
    main()