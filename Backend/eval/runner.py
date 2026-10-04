"""Retrieval evaluation runner.

Drives the PRODUCTION retrieval functions (ask.semantic_retrieve /
ask.lexical_retrieve / ask.retrieve_chunks — never a re-implementation)
over the golden dataset and measures quality at two levels:

- CANDIDATE level: the union of semantic top-40 + lexical top-40,
  i.e. everything ask.retrieve_chunks considered before fusion.
- FINAL level: the production result after fusion -> threshold ->
  dedup -> source selection (what the API actually returns).

Failure classification (per eval question, first match wins where a
total order is needed; a question can carry several tags but exactly one
primary category):

  no_results           - retrieve_chunks returned nothing at all
  wrong_topic          - nothing retrieved shares even a token of the
                         expected video's title/topic
  correct_topic_wrong_video - related videos retrieved, expected video
                         absent from candidates
  candidate_miss       - expected video present in NO candidate pool
                         (neither semantic nor lexical surfaced it)
  lexical_miss         - candidates present but the lexical channel
                         contributed nothing for the expected video
  semantic_miss        - candidates present but the semantic channel
                         contributed nothing for the expected video
  threshold_removed    - expected video was a candidate but every chunk
                         was culled by the score threshold
  dedup_removed        - expected video's best chunk was culled by the
                         one-chunk-per-title rule (another chunk of the
                         SAME video ranked higher)
  beyond_top_k         - relevant chunk survived selection pressure but
                         ranked outside the final top-k that was scored
  correct_video_wrong_chunk - expected video IS in the final result but
                         a different chunk of it (chunk-labeled only)
  ground_truth_ambiguity - multiple equally-plausible relevant videos
                         exist and none ranked first (tagged, not counted
                         as a hard failure)
  other                - anything that defies the above

Determinism: no randomness, no LLM, sorted iteration everywhere; the
only nondeterministic values are latencies, which are reported but
excluded from metric equality.
"""

import json
import os
import statistics
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ask  # noqa: E402  (production retrieval — the system under test)

from eval import metrics  # noqa: E402

REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_GOLDEN_PATH = os.path.join(REPO_ROOT, "Data", "eval", "golden.jsonl")
DEFAULT_REPORT_DIR = os.path.join(REPO_ROOT, "Data", "eval", "reports")

K_VALUES = (1, 3, 5, 6, 10)


# ---------------------------------------------------------------------------
# Identity projection
# ---------------------------------------------------------------------------


def candidate_video_ids(diag):
    """All video ids present in the candidate pool (any channel)."""
    return [
        entry.get("video_id")
        for entry in diag.get("candidates", [])
        if entry.get("video_id")
    ]


def candidate_chunk_keys(diag):
    """All chunk keys in the candidate pool."""
    keys = []
    for entry in diag.get("candidates", []):
        video_id = entry.get("video_id")
        index = entry.get("chunk_index")
        if video_id and isinstance(index, int):
            keys.append(f"{video_id}_{index}")
    return keys


def final_video_ids(selected):
    return [(hit.payload or {}).get("video_id") for hit in selected]


def final_chunk_keys(selected):
    keys = []
    for hit in selected:
        payload = hit.payload or {}
        video_id = payload.get("video_id")
        index = payload.get("chunk_index")
        if video_id and isinstance(index, int):
            keys.append(f"{video_id}_{index}")
    return keys


def _dedupe_preserving_order(ids):
    seen = set()
    ordered = []
    for value in ids:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


# ---------------------------------------------------------------------------
# Channel participation (for lexical/semantic miss classification)
# ---------------------------------------------------------------------------


def channel_participation(diag, relevant_video_ids, question):
    """Which retrieval channels surfaced the relevant videos.

    Runs the two production channel functions once more against the same
    question (deterministic, read-only) and reports, per relevant video,
    whether semantic/lexical surfaced it. Cost: one extra embedding +
    query per question — acceptable for offline evaluation.
    """
    semantic_hits = ask.semantic_retrieve(question)
    lexical_hits = ask.lexical_retrieve(question)

    semantic_videos = {
        (hit.payload or {}).get("video_id") for hit in semantic_hits
    }
    lexical_videos = {
        (point.payload or {}).get("video_id") for point, _ in lexical_hits
    } if lexical_hits and isinstance(lexical_hits[0], tuple) else {
        (hit.payload or {}).get("video_id") for hit in lexical_hits
    }

    participation = {}
    for video_id in relevant_video_ids:
        participation[video_id] = {
            "semantic": video_id in semantic_videos,
            "lexical": video_id in lexical_videos,
        }
    return participation


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def classify_failure(case, diag, selected, top_k):
    """Return the primary failure category for a missed question.

    Called only when the question missed at its ground-truth level.
    """
    relevant_videos = set(case["relevant_video_ids"])
    relevant_chunks = set(case["relevant_chunk_ids"])

    candidates = diag.get("candidates", []) if diag else []
    candidate_videos = {
        entry.get("video_id") for entry in candidates
    }

    selected_videos = {
        (hit.payload or {}).get("video_id") for hit in selected
    }

    if diag is None:
        return "other", "no diagnostics captured"
    if not candidates:
        return "no_results", "candidate pool was empty"

    if relevant_videos & selected_videos:
        # The right video made the final cut; the miss must be chunk-level.
        if relevant_chunks:
            return (
                "correct_video_wrong_chunk",
                "expected video selected but the labeled chunk is not",
            )
        return "other", "video-level hit should not reach classify_failure"

    if relevant_videos & candidate_videos:
        # Right video was a candidate but none of its chunks survived.
        best_for_video = [
            entry
            for entry in candidates
            if entry.get("video_id") in relevant_videos
        ]
        reasons = {entry.get("cull_reason") for entry in best_for_video}
        if reasons == {"duplicate_title"} or "duplicate_title" in reasons:
            return (
                "dedup_removed",
                "expected video's chunks were culled by the "
                "one-chunk-per-title rule",
            )
        if reasons <= {"below_threshold", "duplicate_title"}:
            return (
                "threshold_removed",
                "every chunk of the expected video fell below the "
                "selection threshold",
            )
        return (
            "beyond_top_k",
            "relevant chunks survived thresholding but lost the "
            "top-k race to other videos",
        )

    # Expected video never surfaced at all: which channel failed?
    participation = channel_participation(
        diag, sorted(relevant_videos), case["question"]
    )
    semantic_any = any(
        info["semantic"] for info in participation.values()
    )
    lexical_any = any(
        info["lexical"] for info in participation.values()
    )
    if semantic_any and lexical_any:
        return "other", "channels report the video but the pool lacks it"
    if semantic_any and not lexical_any:
        return "lexical_miss", "semantic surfaced it, lexical did not"
    if lexical_any and not semantic_any:
        return "semantic_miss", "lexical surfaced it, semantic did not"
    return (
        "candidate_miss",
        "neither semantic nor lexical surfaced the expected video",
    )


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def evaluate_case(case, top_k=None):
    """Run production retrieval for one case and compute its metrics.

    Returns a per-question record with metrics at both levels, the
    candidate/final identities, and the failure category when missed.
    """
    question = case["question"]
    relevant_videos = set(case["relevant_video_ids"])
    relevant_chunks = set(case["relevant_chunk_ids"])
    level = case["chunk_certainty"]  # "exact" | "video"

    t0 = time.perf_counter()
    selected = ask.retrieve_chunks(question)
    elapsed = time.perf_counter() - t0

    diag = ask.get_retrieval_diagnostics() or {}
    config = diag.get("config", {})

    # ---- identity lists (deduped, order preserved) --------------------
    cand_videos = _dedupe_preserving_order(candidate_video_ids(diag))
    cand_chunks = _dedupe_preserving_order(candidate_chunk_keys(diag))
    final_videos = _dedupe_preserving_order(final_video_ids(selected))
    final_chunks = _dedupe_preserving_order(final_chunk_keys(selected))

    top_k_effective = top_k or config.get("top_k") or ask.TOP_K

    # ---- metrics at the labeled level ---------------------------------
    if level == "exact":
        cand_metric_ids, final_metric_ids = cand_chunks, final_chunks
        relevant_metric = relevant_chunks or relevant_videos
    else:
        cand_metric_ids, final_metric_ids = cand_videos, final_videos
        relevant_metric = relevant_videos

    record_metrics = {
        "level": level,
        "candidate": {
            f"hit@{k}": metrics.hit_at_k(cand_metric_ids, relevant_metric, k)
            for k in K_VALUES
        },
        "final": {
            f"hit@{k}": metrics.hit_at_k(final_metric_ids, relevant_metric, k)
            for k in K_VALUES
        },
        "candidate_recall@20": metrics.recall_at_k(
            cand_metric_ids, relevant_metric, 20
        ),
        "final_recall@6": metrics.recall_at_k(
            final_metric_ids, relevant_metric, 6
        ),
        "final_recall@20": metrics.recall_at_k(
            final_metric_ids, relevant_metric, 20
        ),
        "mrr_candidate": metrics.mrr(cand_metric_ids, relevant_metric),
        "mrr_final": metrics.mrr(final_metric_ids, relevant_metric),
    }

    hit_final = record_metrics["final"][f"hit@{top_k_effective}"]
    failure = None
    failure_detail = None
    if not hit_final:
        failure, failure_detail = classify_failure(
            case, diag, selected, top_k_effective
        )

    exact_topic = bool(diag.get("candidates")) and any(
        entry.get("title_score", 0) >= 0.95 for entry in diag.get("candidates", [])
    )

    return {
        "eval_id": case["id"],
        "question": question,
        "section": case.get("section"),
        "question_type": case.get("question_type"),
        "level": level,
        "ground_truth": {
            "video_ids": sorted(relevant_videos),
            "chunk_ids": sorted(relevant_chunks),
        },
        "candidate_ids": {
            "videos": cand_videos,
            "chunks": cand_chunks,
        },
        "final_ids": {
            "videos": final_videos,
            "chunks": final_chunks,
        },
        "final_results": [
            {
                "rank": rank,
                "video_id": (hit.payload or {}).get("video_id"),
                "chunk_index": (hit.payload or {}).get("chunk_index"),
                "title": (hit.payload or {}).get("video_title"),
                "hybrid_score": ask.get_retrieval_score(str(hit.id)),
            }
            for rank, hit in enumerate(selected, start=1)
        ],
        "metrics": record_metrics,
        "retrieval_seconds": elapsed,
        "hit": bool(hit_final),
        "failure_category": failure,
        "failure_detail": failure_detail,
        "candidate_count": len(diag.get("candidates", [])),
        "exact_topic_candidate": exact_topic,
    }


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


def _aggregate(records, key):
    values = [r["metrics"][key] for r in records]
    return metrics.mean(values)


def run_evaluation(golden_cases, top_k=None):
    """Evaluate every case; return the aggregate report dict (in-memory)."""
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records = []
    retrieval_seconds = []

    for case in golden_cases:
        record = evaluate_case(case, top_k=top_k)
        records.append(record)
        retrieval_seconds.append(record["retrieval_seconds"])

    n = len(records)
    video_labeled = [r for r in records if r["metrics"]["level"] == "video"]
    chunk_labeled = [r for r in records if r["metrics"]["level"] == "exact"]

    def agg(records_subset, scope, key):
        """Mean of a per-record metric; dotted keys traverse nesting
        (e.g. "candidate.hit@1" -> record["metrics"]["candidate"]["hit@1"]).
        Flat keys ("candidate_recall@20") are used as-is."""
        values = []
        for record in records_subset:
            value = record["metrics"]
            for part in key.split("."):
                value = value[part]
            values.append(value)
        return {
            "scope": scope,
            "n": len(values),
            "mean": metrics.mean(values),
        }

    aggregate = {
        "n": n,
        "video_labeled_n": len(video_labeled),
        "chunk_labeled_n": len(chunk_labeled),
        "hit": {
            "candidate": {
                f"hit@{k}": agg(records, "all", f"candidate.hit@{k}")
                for k in K_VALUES
            },
            "final": {
                f"hit@{k}": agg(records, "all", f"final.hit@{k}")
                for k in K_VALUES
            },
        },
        "video_level": {
            f"hit@{k}": agg(video_labeled, "video", f"final.hit@{k}")
            for k in K_VALUES
        },
        "chunk_level": {
            f"hit@{k}": agg(chunk_labeled, "chunk", f"final.hit@{k}")
            for k in K_VALUES
        },
        "recall": {
            "candidate@20": agg(records, "all", "candidate_recall@20"),
            "final@6": agg(records, "all", "final_recall@6"),
            "final@20": agg(records, "all", "final_recall@20"),
        },
        "mrr": {
            "candidate": agg(records, "all", "mrr_candidate"),
            "final": agg(records, "all", "mrr_final"),
        },
        "avg_final_source_count": metrics.mean(
            [len(r["final_ids"]["videos"]) for r in records]
        ),
        "no_result_rate": (
            sum(1 for r in records if not r["final_ids"]["videos"]) / n
            if n
            else None
        ),
        "exact_topic_candidate_rate": (
            sum(1 for r in records if r["exact_topic_candidate"]) / n
            if n
            else None
        ),
        "candidate_pool_recall": agg(records, "all", "candidate_recall@20"),
        "latency_seconds": {
            "p50": metrics.percentile(retrieval_seconds, 0.50),
            "p95": metrics.percentile(retrieval_seconds, 0.95),
            "mean": metrics.mean(retrieval_seconds),
        },
        "failure_categories": dict(
            sorted(
                {
                    category: sum(
                        1
                        for r in records
                        if r["failure_category"] == category
                    )
                    for category in {
                        r["failure_category"]
                        for r in records
                        if r["failure_category"]
                    }
                }.items()
            )
        ),
    }

    return {
        "aggregate": aggregate,
        "records": records,
        "started_at": started,
        "retrieval_seconds": retrieval_seconds,
    }


# ---------------------------------------------------------------------------
# Configuration capture (baseline provenance)
# ---------------------------------------------------------------------------


def capture_configuration():
    """Record the retrieval configuration under test (read-only)."""
    try:
        import subprocess
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:
        commit = None

    return {
        "top_k": ask.TOP_K,
        "semantic_candidates": ask.SEMANTIC_CANDIDATES,
        "lexical_candidates": ask.LEXICAL_CANDIDATES,
        "min_score_threshold": ask.MIN_SCORE_THRESHOLD,
        "fusion_weights": {
            "title": ask.TITLE_WEIGHT,
            "semantic": ask.SEMANTIC_WEIGHT,
            "text": ask.TEXT_WEIGHT,
        },
        "exact_title_boost": 0.20,
        "embedding_model": ask.EMBED_MODEL_NAME,
        "vector_size": ask.VECTOR_SIZE,
        "qdrant_collection": ask.COLLECTION_NAME,
        "gemini_model": ask.GEMINI_MODEL,
        "git_commit": commit or None,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    golden_path = argv[0] if argv else DEFAULT_GOLDEN_PATH
    report_dir = argv[1] if len(argv) > 1 else DEFAULT_REPORT_DIR

    from eval.dataset import load_golden_dataset
    from eval.report import write_reports

    cases = load_golden_dataset(golden_path)
    print(f"Loaded {len(cases)} golden cases from {golden_path}")

    result = run_evaluation(cases)
    config = capture_configuration()

    paths = write_reports(
        result,
        config,
        report_dir=report_dir,
        dataset_path=golden_path,
        report_name="baseline",
    )
    print(f"Reports written: {paths}")
    try:
        # Close the local Qdrant store deterministically instead of
        # relying on interpreter shutdown (avoids __del__ noise).
        ask.qdrant_client.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
