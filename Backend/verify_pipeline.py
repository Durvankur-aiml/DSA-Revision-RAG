import os
import re
import sys
import json
import math
import time
import hashlib
from collections import Counter, defaultdict

from qdrant_client import QdrantClient


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "transcripts")
METADATA_DIR = os.path.join(BASE_DIR, "data", "metadata")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
QDRANT_DIR = os.path.join(BASE_DIR, "qdrant_db")

COLLECTION_NAME = "striver_a2z"
EXPECTED_EMBEDDING_DIM = 384
SCROLL_BATCH = 512

# A legitimate deterministic point id is a 48-bit md5 slice, so it is always
# very large. An id below this bound came from an older loader that used a
# plain loop counter, and can never be overwritten by upsert.
MIN_LEGITIMATE_POINT_ID = 100000

YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Temp windows newer than this are assumed to belong to a transcription run
# that is still going, not to an interrupted one.
STALE_TEMP_SECONDS = 600

REQUIRED_PAYLOAD_KEYS = (
    "chunk_index", "video_id", "video_title",
    "youtube_url", "start", "end", "text",
)


# ==========================================================
# REPORTING
# ==========================================================

results = {"FAIL": [], "WARN": [], "INFO": []}


def fail(message):
    print(f"\nERROR: {message}")
    sys.exit(1)


def record(level, check, message):
    results[level].append((check, message))


def show(level, check, message):
    record(level, check, message)
    print(f"  [{level}] {check}: {message}")


def ok(check, message):
    print(f"  [ OK ] {check}: {message}")


def banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def make_point_id(video_id, chunk_index):
    """Must stay byte-identical to load_to_qdrant.make_point_id().

    Deliberately duplicated rather than imported: load_to_qdrant.py has no
    __main__ guard, so importing it would run the entire loader.
    """
    unique_string = f"{video_id}_{chunk_index}"
    return int(hashlib.md5(unique_string.encode()).hexdigest()[:12], 16)


# ==========================================================
# DISK READERS
# ==========================================================

def read_disk_state():

    for label, path in (("transcripts", TRANSCRIPT_DIR),
                        ("embeddings", EMBEDDINGS_DIR)):
        if not os.path.isdir(path):
            fail(f"{label} directory does not exist: {path}")

    state = {
        "audio": set(),
        "metadata": set(),
        "transcripts": {},     # video_id -> chunk count
        "embeddings": {},      # video_id -> list of records (embedding stripped)
        "debris": [],
        "unreadable": [],
        "active_run": None,
    }

    if os.path.isdir(AUDIO_DIR):
        for name in os.listdir(AUDIO_DIR):
            full = os.path.join(AUDIO_DIR, name)
            if name.lower().endswith(".mp3"):
                state["audio"].add(os.path.splitext(name)[0])
            elif name.lower().endswith((".part", ".ytdl", ".tmp", ".webm")):
                state["debris"].append(os.path.join("audio", name))
            elif os.path.isdir(full) and name == "_windows_temp":
                windows = os.listdir(full)
                if not windows:
                    continue
                newest = max(
                    os.path.getmtime(os.path.join(full, w)) for w in windows
                )
                age = time.time() - newest
                if age < STALE_TEMP_SECONDS:
                    state["active_run"] = (
                        f"_windows_temp/ holds {len(windows)} window(s) touched "
                        f"{int(age)}s ago — a transcription run looks like it is "
                        f"still going, so the counts below are a moving snapshot"
                    )
                else:
                    state["debris"].append(
                        f"_windows_temp/ holds {len(windows)} stale window(s), "
                        f"newest {int(age // 60)} min old — leftover from an "
                        f"interrupted run (os.rmdir cannot remove a non-empty "
                        f"directory, so it silently persists)"
                    )

    if os.path.isdir(METADATA_DIR):
        for name in os.listdir(METADATA_DIR):
            if name.lower().endswith(".json"):
                state["metadata"].add(os.path.splitext(name)[0])

    for name in os.listdir(TRANSCRIPT_DIR):
        if not name.lower().endswith(".json"):
            continue
        video_id = os.path.splitext(name)[0]
        path = os.path.join(TRANSCRIPT_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as file:
                chunks = json.load(file)
            if not isinstance(chunks, list):
                raise ValueError("top level is not a list")
            state["transcripts"][video_id] = len(chunks)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            state["unreadable"].append(f"transcripts/{name}: {e}")

    for name in os.listdir(EMBEDDINGS_DIR):
        if not name.lower().endswith(".json"):
            continue
        video_id = name[:-len("_embeddings.json")] \
            if name.endswith("_embeddings.json") else os.path.splitext(name)[0]
        path = os.path.join(EMBEDDINGS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError("top level is not a list")
            slim = []
            for r in records:
                slim.append({
                    "chunk_index": r.get("chunk_index"),
                    "video_id": r.get("video_id"),
                    "youtube_url": r.get("youtube_url"),
                    "text": r.get("text"),
                    "dim": len(r["embedding"]) if isinstance(
                        r.get("embedding"), list) else None,
                })
            state["embeddings"][video_id] = slim
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
            state["unreadable"].append(f"embeddings/{name}: {e}")

    return state


def read_qdrant_state():

    if not os.path.isdir(QDRANT_DIR):
        fail(f"Qdrant directory does not exist: {QDRANT_DIR}")

    try:
        client = QdrantClient(path=QDRANT_DIR)
    except Exception as e:
        fail(
            f"Could not open Qdrant at {QDRANT_DIR}: {e}\n"
            f"Local Qdrant allows only ONE process at a time. "
            f"Close ask.py / load_to_qdrant.py and retry."
        )

    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        fail(f"Could not list collections: {e}")

    if COLLECTION_NAME not in collections:
        fail(f"Collection '{COLLECTION_NAME}' not found. Found: {collections}")

    points = []
    offset = None
    while True:
        try:
            batch, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            fail(f"Failed while scrolling points: {e}")

        points.extend(batch)
        if offset is None:
            break

    return points


# ==========================================================
# CHECKS
# ==========================================================

def check_disk_integrity(disk):
    banner("1. DISK INTEGRITY")

    if disk["active_run"]:
        show("INFO", "run in progress", disk["active_run"])

    if disk["unreadable"]:
        for item in disk["unreadable"]:
            show("FAIL", "unreadable file", item)
    else:
        ok("file readability", "every transcript and embedding file parsed")

    if disk["debris"]:
        for item in disk["debris"]:
            show("WARN", "debris", item)
    else:
        ok("debris", "no .part / .tmp / leftover temp-window files")

    bad_ids = sorted(
        v for v in set(disk["transcripts"]) | set(disk["embeddings"])
        if not YOUTUBE_ID_RE.match(v)
    )
    if bad_ids:
        for v in bad_ids:
            show("FAIL", "video_id format", f"not an 11-char YouTube id: {v!r}")
    else:
        ok("video_id format", "all filenames are valid 11-char YouTube ids")

    mismatched = sorted(
        vid for vid, recs in disk["embeddings"].items()
        for r in recs if r["video_id"] != vid
    )
    if mismatched:
        for v in sorted(set(mismatched)):
            show("FAIL", "id/filename agreement",
                 f"{v}: records carry a different video_id")
    else:
        ok("id/filename agreement",
           "every embedding record's video_id matches its filename")

    wrong_dim = [
        (vid, r["chunk_index"], r["dim"])
        for vid, recs in disk["embeddings"].items()
        for r in recs if r["dim"] != EXPECTED_EMBEDDING_DIM
    ]
    if wrong_dim:
        for vid, ci, dim in wrong_dim[:10]:
            show("FAIL", "embedding dimension",
                 f"{vid} chunk {ci} has dim {dim}, expected "
                 f"{EXPECTED_EMBEDDING_DIM}")
    else:
        ok("embedding dimension",
           f"all vectors on disk are {EXPECTED_EMBEDDING_DIM}-dim")


def check_stage_parity(disk, points):
    banner("2. STAGE PARITY (the check that matters)")

    qdrant_counts = Counter(
        p.payload.get("video_id") for p in points if p.payload
    )
    transcripts = disk["transcripts"]
    embeddings = {k: len(v) for k, v in disk["embeddings"].items()}

    print(f"  audio mp3       : {len(disk['audio'])}")
    print(f"  metadata json   : {len(disk['metadata'])}")
    print(f"  transcript json : {len(transcripts)}  "
          f"({sum(transcripts.values())} chunks)")
    print(f"  embedding json  : {len(embeddings)}  "
          f"({sum(embeddings.values())} records)")
    print(f"  qdrant videos   : {len(qdrant_counts)}  "
          f"({sum(qdrant_counts.values())} points)")
    print()

    mismatches = []
    for vid in sorted(set(transcripts) | set(embeddings) | set(qdrant_counts)):
        t = transcripts.get(vid)
        e = embeddings.get(vid)
        q = qdrant_counts.get(vid)

        # A video mid-pipeline is not an error; a disagreement in counts is.
        if e is not None and t is not None and t != e:
            mismatches.append((vid, t, e, q, "transcript != embeddings"))
        elif q is not None and e is not None and e != q:
            mismatches.append((vid, t, e, q, "embeddings != qdrant"))
        elif q is not None and e is None:
            mismatches.append((vid, t, e, q, "in qdrant but no embeddings file"))

    if mismatches:
        show("FAIL", "per-video parity",
             f"{len(mismatches)} video(s) disagree across stages")
        print(f"       {'video_id':<14}{'trans':>6}{'embed':>7}{'qdrant':>8}  reason")
        for vid, t, e, q, why in mismatches[:20]:
            print(f"       {vid:<14}{str(t):>6}{str(e):>7}{str(q):>8}  {why}")
    else:
        ok("per-video parity",
           f"transcript == embeddings == qdrant for all "
           f"{len(qdrant_counts)} indexed video(s)")

    pending_transcribe = sorted(disk["audio"] - set(transcripts))
    pending_embed = sorted(set(transcripts) - set(embeddings))
    pending_load = sorted(set(embeddings) - set(qdrant_counts))
    missing_metadata = sorted(disk["audio"] - disk["metadata"])

    if missing_metadata:
        show("FAIL", "metadata coverage",
             f"{len(missing_metadata)} mp3 without metadata json "
             f"(triggers the fabricated-URL fallback): "
             f"{missing_metadata[:5]}")
    else:
        ok("metadata coverage", "every mp3 has a metadata json")

    for label, items in (("awaiting transcription", pending_transcribe),
                         ("awaiting embedding", pending_embed),
                         ("awaiting qdrant load", pending_load)):
        if items:
            show("INFO", "backlog",
                 f"{len(items)} video(s) {label}: {items[:5]}"
                 f"{' ...' if len(items) > 5 else ''}")

    if not (pending_transcribe or pending_embed or pending_load):
        ok("backlog", "no video is stuck mid-pipeline")


def check_distinctness(points):
    banner("3. DISTINCTNESS (what count checks cannot see)")

    legacy = [p for p in points
              if isinstance(p.id, int) and p.id < MIN_LEGITIMATE_POINT_ID]
    if legacy:
        show("FAIL", "legacy point ids",
             f"{len(legacy)} point(s) with id < {MIN_LEGITIMATE_POINT_ID} — "
             f"orphans from an older loader, unreachable by upsert. "
             f"Sample ids: {[p.id for p in legacy[:5]]}")
    else:
        ok("legacy point ids", "no sequential/counter-based orphans")

    nondeterministic = []
    for p in points:
        payload = p.payload or {}
        vid = payload.get("video_id")
        ci = payload.get("chunk_index")
        if vid is None or ci is None:
            continue
        if p.id != make_point_id(vid, ci):
            nondeterministic.append((p.id, vid, ci))

    if nondeterministic:
        show("FAIL", "point id determinism",
             f"{len(nondeterministic)} point(s) do not match "
             f"make_point_id(video_id, chunk_index) — re-running the loader "
             f"will duplicate rather than overwrite them. "
             f"Sample: {nondeterministic[:3]}")
    else:
        ok("point id determinism",
           f"all {len(points)} ids reproduce from (video_id, chunk_index)")

    identity = Counter(
        (p.payload.get("video_id"), p.payload.get("chunk_index"))
        for p in points if p.payload
    )
    dupes = {k: n for k, n in identity.items() if n > 1}
    if dupes:
        extra = sum(n - 1 for n in dupes.values())
        show("FAIL", "duplicate identity",
             f"{len(dupes)} (video_id, chunk_index) pair(s) appear more than "
             f"once — {extra} redundant point(s). Sample: "
             f"{list(dupes.items())[:3]}")
    else:
        ok("duplicate identity",
           "every (video_id, chunk_index) appears exactly once")

    gaps = []
    by_video = defaultdict(list)
    for p in points:
        if p.payload:
            by_video[p.payload.get("video_id")].append(
                p.payload.get("chunk_index"))
    for vid, indices in by_video.items():
        if sorted(indices) != list(range(len(indices))):
            gaps.append(vid)
    if gaps:
        show("FAIL", "chunk_index continuity",
             f"{len(gaps)} video(s) have gaps or repeats in chunk_index: "
             f"{gaps[:5]}")
    else:
        ok("chunk_index continuity",
           "every video runs 0..n-1 with no gaps")

    texts = Counter(
        p.payload.get("text") for p in points
        if p.payload and p.payload.get("text")
    )
    repeated = {t: n for t, n in texts.items() if n > 1}
    if repeated:
        # Short lines legitimately recur across videos (intro/outro, music,
        # catchphrases). Only worth investigating when the text is long.
        substantial = {t: n for t, n in repeated.items() if len(t) > 200}
        show("WARN", "duplicate text",
             f"{len(repeated)} text value(s) appear in more than one point "
             f"({len(substantial)} of them longer than 200 chars). "
             f"Short repeats are usually genuine.")
        for t, n in sorted(repeated.items(), key=lambda kv: -len(kv[0]))[:3]:
            print(f"       x{n}  ({len(t)} chars)  {t[:70]!r}")
    else:
        ok("duplicate text", f"all {len(points)} chunk texts are distinct")


def check_payload_and_vectors(points):
    banner("4. PAYLOAD & VECTOR HEALTH")

    schemas = Counter(tuple(sorted((p.payload or {}).keys())) for p in points)
    expected = tuple(sorted(REQUIRED_PAYLOAD_KEYS))
    if len(schemas) != 1:
        show("FAIL", "payload schema",
             f"{len(schemas)} different key sets across points: "
             f"{[list(s) for s in schemas]}")
    elif next(iter(schemas)) != expected:
        show("FAIL", "payload schema",
             f"uniform but unexpected key set: {list(next(iter(schemas)))}")
    else:
        ok("payload schema", f"all {len(points)} payloads carry exactly "
                             f"{len(expected)} expected keys")

    bad_url = []
    noncanonical = []
    for p in points:
        payload = p.payload or {}
        url = payload.get("youtube_url") or ""
        vid = payload.get("video_id")
        if not url or re.search(r"\s", url) or not url.startswith("https://"):
            bad_url.append((vid, url))
        elif url != f"https://youtu.be/{vid}":
            noncanonical.append((vid, url))

    if bad_url:
        show("FAIL", "url format",
             f"{len(bad_url)} malformed url(s) (empty, whitespace, or not "
             f"https). Sample: {bad_url[:3]}")
    else:
        ok("url format", f"all {len(points)} urls are well-formed https")

    if noncanonical:
        show("WARN", "url shape",
             f"{len(noncanonical)} url(s) are not the canonical "
             f"https://youtu.be/<id> form. Sample: {noncanonical[:3]}")

    dims = set()
    broken_floats = 0
    for p in points:
        vec = p.vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()), None)
        if not isinstance(vec, (list, tuple)):
            broken_floats += 1
            continue
        dims.add(len(vec))
        for value in vec:
            if not isinstance(value, (int, float)) or \
                    math.isnan(value) or math.isinf(value):
                broken_floats += 1
                break

    if dims != {EXPECTED_EMBEDDING_DIM}:
        show("FAIL", "vector dimension",
             f"found dimensions {sorted(dims)}, expected "
             f"{{{EXPECTED_EMBEDDING_DIM}}}")
    else:
        ok("vector dimension",
           f"all vectors are {EXPECTED_EMBEDDING_DIM}-dim")

    if broken_floats:
        show("FAIL", "vector values",
             f"{broken_floats} vector(s) contain NaN, Inf, or non-numeric "
             f"values — these silently poison cosine similarity")
    else:
        ok("vector values", "no NaN or Inf anywhere")


# ==========================================================
# MAIN
# ==========================================================

def main():

    banner("MISSION ANTHROPIC — PIPELINE VERIFICATION")
    print(f"Project root: {BASE_DIR}")
    print(f"Collection  : {COLLECTION_NAME}")

    disk = read_disk_state()
    points = read_qdrant_state()

    if not points:
        fail(f"Collection '{COLLECTION_NAME}' contains zero points.")

    check_disk_integrity(disk)
    check_stage_parity(disk, points)
    check_distinctness(points)
    check_payload_and_vectors(points)

    banner("SUMMARY")
    n_fail = len(results["FAIL"])
    n_warn = len(results["WARN"])
    n_info = len(results["INFO"])

    print(f"  FAIL : {n_fail}")
    print(f"  WARN : {n_warn}")
    print(f"  INFO : {n_info}")

    if n_fail:
        print("\n  Failing checks:")
        for check, message in results["FAIL"]:
            print(f"    - {check}: {message.splitlines()[0]}")
        print("\nRESULT: PIPELINE NOT CLEAN — fix the failures above before "
              "processing more videos.")
        print("=" * 60)
        return 1

    if n_warn:
        print("\nRESULT: PIPELINE CLEAN (with warnings worth a glance).")
    else:
        print("\nRESULT: PIPELINE CLEAN.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
