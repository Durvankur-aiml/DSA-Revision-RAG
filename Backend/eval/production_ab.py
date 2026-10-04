"""Phase 3C A/B harness: run the golden dataset through the REAL
production path (ask.retrieve_chunks with the live Qdrant collection
and the real cross-encoder), for one reranker flag configuration per
process (the flag is parsed at import time).

Per case:
  - ask.retrieve_chunks(question)  (production function, real pipeline)
  - video identity from the selected chunks' payloads
  - metrics identical to eval.metrics (hit@k, recall@6, MRR)

The output JSON is consumed by the Phase 3C report. Latency comes from
obs.record_stage timers captured from the perf log writer (we hook
obs._emit is NOT touched; instead stages are read from the request
context snapshot per case).
"""

import json
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ask  # noqa: E402  (production retrieval — the system under test)
import obs  # noqa: E402

from eval import metrics  # noqa: E402
from eval.dataset import load_golden_dataset, DEFAULT_GOLDEN_PATH  # noqa: E402

REPO_ROOT = os.path.dirname(BASE_DIR)
# One output file per flag state so A and B results coexist for the report.
OUT_PATH_OFF = os.path.join(
    REPO_ROOT, "Data", "eval", "reranker_production_ab_off.json"
)
OUT_PATH_ON = os.path.join(
    REPO_ROOT, "Data", "eval", "reranker_production_ab_on.json"
)

K_VALUES = (1, 3, 5, 6, 10)


def main():
    cases = load_golden_dataset(DEFAULT_GOLDEN_PATH)
    print(f"Loaded {len(cases)} golden cases. RERANK_ENABLED={ask.RERANK_ENABLED} "
          f"RERANK_DEPTH={ask.RERANK_DEPTH}")

    records = []
    for i, case in enumerate(cases, start=1):
        question = case["question"]
        relevant_videos = set(case["relevant_video_ids"])
        relevant_chunks = set(case["relevant_chunk_ids"])
        level = case["chunk_certainty"]

        ctx = obs.begin_request(f"abcase-{i}", "POST", "/ask-eval")

        try:
            selected = ask.retrieve_chunks(question)
        finally:
            obs.end_request()

        final_videos = []
        final_chunks = []
        for hit in selected:
            payload = hit.payload or {}
            video_id = payload.get("video_id")
            if video_id and video_id not in final_videos:
                final_videos.append(video_id)
            index = payload.get("chunk_index")
            if video_id and isinstance(index, int):
                key = f"{video_id}_{index}"
                if key not in final_chunks:
                    final_chunks.append(key)

        if level == "exact":
            relevant_metric = relevant_chunks or relevant_videos
        else:
            relevant_metric = relevant_videos

        record = {
            "eval_id": case["id"],
            "question": question,
            "level": level,
            "mrr": metrics.mrr(final_videos, relevant_videos),
            "recall@6": metrics.recall_at_k(final_videos, relevant_videos, 6),
            "stages": dict(ctx.stages),
            "candidate_count": ctx.candidate_count,
        }
        for k in (1, 3, 5, 6):
            record[f"hit@{k}"] = bool(
                metrics.hit_at_k(final_videos, relevant_videos, k)
            )
        records.append(record)

        if i % 10 == 0:
            print(f"  {i}/{len(cases)} done")

    video_labeled = [r for r in records if r["level"] == "video"]

    def mean_of(key, subset=None):
        values = []
        for record in subset if subset is not None else records:
            values.append(record[key])
        return {"n": len(values), "mean": metrics.mean(values)}

    rerank_times = [r["stages"]["reranker_stage"] for r in records
                    if "reranker_stage" in r["stages"]]
    retrieval_times = [
        sum(v for k, v in r["stages"].items() if k != "reranker_stage")
        for r in records
    ]
    total_times = [
        retrieval + (rerank or 0.0)
        for retrieval, rerank in zip(
            retrieval_times,
            [r["stages"].get("reranker_stage", 0.0) for r in records],
        )
    ]

    aggregate = {
        "n": len(records),
        "video_labeled_n": len(video_labeled),
        "hit": {f"hit@{k}": mean_of(f"hit@{k}") for k in (1, 3, 5, 6)},
        "video_level_hit": {
            f"hit@{k}": mean_of(f"hit@{k}", video_labeled)
            for k in (1, 3, 5, 6)
        },
        "recall@6": mean_of("recall@6"),
        "mrr": mean_of("mrr"),
        "latency_seconds": {
            "retrieval_p50": metrics.percentile(retrieval_times, 0.50),
            "retrieval_p95": metrics.percentile(retrieval_times, 0.95),
            "rerank_p50": metrics.percentile(rerank_times, 0.50),
            "rerank_p95": metrics.percentile(rerank_times, 0.95),
            "total_p50": metrics.percentile(total_times, 0.50),
            "total_p95": metrics.percentile(total_times, 0.95),
        },
    }

    # hit@3 / hit@5 were not stored per-record above for brevity; compute
    # them from the stored hit booleans where possible.
    payload = {
        "rerank_enabled": ask.RERANK_ENABLED,
        "rerank_depth": ask.RERANK_DEPTH,
        "aggregate": aggregate,
        "records": records,
    }

    out_path = OUT_PATH_ON if ask.RERANK_ENABLED else OUT_PATH_OFF
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(json.dumps(aggregate["hit"], indent=2))
    print("MRR:", aggregate["mrr"]["mean"])
    print("Recall@6:", aggregate["recall@6"]["mean"])
    lat = aggregate["latency_seconds"]
    print(
        f"Latency: retrieval p50={lat['retrieval_p50']*1000:.0f}ms "
        f"rerank p50={(lat['rerank_p50'] or 0)*1000:.0f}ms "
        f"total p50={lat['total_p50']*1000:.0f}ms "
        f"total p95={lat['total_p95']*1000:.0f}ms"
    )

    try:
        ask.qdrant_client.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
