"""Golden dataset loading and validation.

The golden dataset is a JSONL file (Data/eval/golden.jsonl by default),
one evaluation case per line:

    {
      "id": "eval_001",
      "question": "Explain binary search.",
      "relevant_video_ids": ["MHf6awe89xw"],
      "relevant_chunk_ids": ["MHf6awe89xw_12"],
      "section": "Binary Search",
      "question_type": "exact_topic",
      "chunk_certainty": "exact",
      "notes": "..."
    }

Schema rules (enforced by validate_case / validate_dataset, all fatal):

- id            : required, non-empty string, unique across the dataset,
                  pattern ^[A-Za-z0-9_-]+$
- question      : required, non-empty string (stripped), <= 500 chars
                  (the production API's own limit)
- relevant_video_ids : required, non-empty list of valid 11-char YouTube
                  ids that MUST exist in Data/Metadata
- relevant_chunk_ids : optional list of chunk keys "videoId_chunkIndex";
                  when present every key must exist in Data/embeddings
- section       : optional string
- question_type : optional string
- chunk_certainty : optional string ("exact" | "video") — when absent it
                  is inferred: "exact" if chunk ids are given else "video"
- notes         : optional string

Validation NEVER repairs or silently drops malformed records: it raises
GoldenDatasetError listing every problem found.
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_GOLDEN_PATH = os.path.join(REPO_ROOT, "Data", "eval", "golden.jsonl")
METADATA_DIR = os.path.join(REPO_ROOT, "Data", "Metadata")
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "Data", "embeddings")

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
EVAL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
CHUNK_KEY_RE = re.compile(r"^(?P<video_id>[A-Za-z0-9_-]{11})_(?P<index>\d+)$")

MAX_QUESTION_LENGTH = 500

ALLOWED_TOP_LEVEL_KEYS = frozenset(
    {
        "id",
        "question",
        "relevant_video_ids",
        "relevant_chunk_ids",
        "section",
        "question_type",
        "chunk_certainty",
        "notes",
    }
)


class GoldenDatasetError(Exception):
    """Raised when the golden dataset violates its schema."""


def load_metadata_video_ids(metadata_dir=METADATA_DIR):
    """The set of video ids with a metadata file (ground-truth universe)."""
    ids = set()
    for filename in os.listdir(metadata_dir):
        if filename.endswith(".json"):
            ids.add(os.path.splitext(filename)[0])
    return ids


def load_embedding_chunk_keys(embeddings_dir=EMBEDDINGS_DIR):
    """The set of valid chunk keys 'videoId_chunkIndex' on disk."""
    keys = set()
    for filename in os.listdir(embeddings_dir):
        if not filename.endswith("_embeddings.json"):
            continue
        video_id = filename[: -len("_embeddings.json")]
        path = os.path.join(embeddings_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue
        for record in records:
            index = record.get("chunk_index")
            if isinstance(index, int):
                keys.add(f"{video_id}_{index}")
    return keys


def validate_case(case, known_video_ids, known_chunk_keys, line_number=None):
    """Validate one golden case. Returns a normalized dict.

    Raises GoldenDatasetError (message includes the line number when
    given) on ANY violation. Never repairs silently.
    """
    prefix = f"line {line_number}: " if line_number is not None else ""

    def bad(message):
        raise GoldenDatasetError(f"{prefix}{message}")

    if not isinstance(case, dict):
        bad(f"record is not a JSON object (got {type(case).__name__})")

    unknown_keys = set(case) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_keys:
        bad(f"unknown field(s): {sorted(unknown_keys)}")

    eval_id = case.get("id")
    if not isinstance(eval_id, str) or not eval_id.strip():
        bad("missing or empty 'id'")
    if not EVAL_ID_RE.match(eval_id):
        bad(f"invalid 'id' {eval_id!r} (allowed: letters, digits, _, -)")

    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        bad(f"[{eval_id}] missing or empty 'question'")
    if len(question.strip()) > MAX_QUESTION_LENGTH:
        bad(
            f"[{eval_id}] question exceeds {MAX_QUESTION_LENGTH} chars "
            f"({len(question.strip())})"
        )

    video_ids = case.get("relevant_video_ids")
    if not isinstance(video_ids, list) or not video_ids:
        bad(f"[{eval_id}] 'relevant_video_ids' must be a non-empty list")
    for video_id in video_ids:
        if not isinstance(video_id, str) or not YOUTUBE_ID_RE.match(video_id):
            bad(f"[{eval_id}] invalid video id: {video_id!r}")
        if known_video_ids is not None and video_id not in known_video_ids:
            bad(
                f"[{eval_id}] video id {video_id!r} has no metadata file "
                "(ground truth must exist in Data/Metadata)"
            )

    chunk_ids = case.get("relevant_chunk_ids")
    if chunk_ids is not None:
        if not isinstance(chunk_ids, list):
            bad(f"[{eval_id}] 'relevant_chunk_ids' must be a list")
        for chunk_key in chunk_ids:
            if not isinstance(chunk_key, str):
                bad(f"[{eval_id}] chunk key is not a string: {chunk_key!r}")
            match = CHUNK_KEY_RE.match(chunk_key)
            if not match:
                bad(
                    f"[{eval_id}] malformed chunk key {chunk_key!r} "
                    "(expected 'videoId_chunkIndex')"
                )
            if match.group("video_id") not in video_ids:
                bad(
                    f"[{eval_id}] chunk key {chunk_key!r} does not belong "
                    "to this case's relevant_video_ids"
                )
            if known_chunk_keys is not None and chunk_key not in known_chunk_keys:
                bad(
                    f"[{eval_id}] chunk key {chunk_key!r} does not exist "
                    "in Data/embeddings"
                )

    certainty = case.get("chunk_certainty")
    if certainty is not None and certainty not in ("exact", "video"):
        bad(f"[{eval_id}] 'chunk_certainty' must be 'exact' or 'video'")

    for optional in ("section", "question_type", "notes"):
        value = case.get(optional)
        if value is not None and not isinstance(value, str):
            bad(f"[{eval_id}] '{optional}' must be a string")

    # Normalized view used by the runner (defaults filled deterministically).
    normalized = {
        "id": eval_id,
        "question": question.strip(),
        "relevant_video_ids": list(video_ids),
        "relevant_chunk_ids": list(chunk_ids) if chunk_ids else [],
        "chunk_certainty": certainty
        or ("exact" if chunk_ids else "video"),
    }
    for optional in ("section", "question_type", "notes"):
        if case.get(optional) is not None:
            normalized[optional] = case[optional]
    return normalized


def validate_dataset(cases, known_video_ids=None, known_chunk_keys=None):
    """Validate a sequence of parsed records; returns the normalized list.

    Checks dataset-level invariants (duplicate ids) in addition to
    per-record schema validation. Fails loudly on the FIRST problem,
    reporting its line number.
    """
    seen_ids = set()
    normalized = []
    for line_number, case in enumerate(cases, start=1):
        record = validate_case(
            case, known_video_ids, known_chunk_keys, line_number=line_number
        )
        if record["id"] in seen_ids:
            raise GoldenDatasetError(
                f"line {line_number}: duplicate evaluation id "
                f"{record['id']!r}"
            )
        seen_ids.add(record["id"])
        normalized.append(record)
    if not normalized:
        raise GoldenDatasetError("golden dataset is empty")
    return normalized


def load_golden_dataset(path=DEFAULT_GOLDEN_PATH, verify_files=True):
    """Load + validate the golden dataset from disk.

    When verify_files is True (production default) video/chunk ground
    truth is checked against Data/Metadata and Data/embeddings. Tests
    pass verify_files=False with synthetic datasets.
    """
    if not os.path.exists(path):
        raise GoldenDatasetError(f"golden dataset not found: {path}")

    known_video_ids = None
    known_chunk_keys = None
    if verify_files:
        known_video_ids = load_metadata_video_ids()
        known_chunk_keys = load_embedding_chunk_keys()

    cases = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                raise GoldenDatasetError(
                    f"line {line_number}: blank line (blank lines are not "
                    "allowed in the golden dataset — keep it dense)"
                )
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise GoldenDatasetError(
                    f"line {line_number}: invalid JSON ({error})"
                ) from error
            cases.append(case)

    return validate_dataset(cases, known_video_ids, known_chunk_keys)
