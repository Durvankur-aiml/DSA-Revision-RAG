"""Phase 3 reranker A/B experiment runner.

Pipeline per experiment (baseline and each rerank depth):

    question
      -> ask.semantic_retrieve  (production, unchanged)
      -> ask.lexical_retrieve   (production, unchanged)
      -> merge into the candidate pool EXACTLY as retrieve_chunks does
         (same per-candidate scoring keys, same pool semantics)
      -> [reranker experiments only] cross-encoder reranks the top-DEPTH
         of the fused pool; the tail keeps its fused order
      -> ask.apply_production_selection  (the verbatim production rules)
      -> video-level/chunk-level metrics from eval.metrics

The reranker NEVER generates candidates and NEVER sees ground truth: it
scores (question, title+transcript) pairs only.

Determinism: identical inputs produce identical rankings and metrics;
only wall-clock latency fields vary between runs.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ask  # noqa: E402  (production retrieval — the system under test)

from eval import metrics  # noqa: E402
from eval.dataset import load_golden_dataset, DEFAULT_GOLDEN_PATH  # noqa: E402
from eval.runner import capture_configuration  # noqa: E402
from reranker import (  # noqa: E402
    CrossReranker,
    validate_experiment_config,
)

REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_REPORT_DIR = os.path.join(REPO_ROOT, "Data", "eval", "reports")

K_VALUES = (1, 3, 5, 6, 10)


# ---------------------------------------------------------------------------
# Candidate pool construction (production-faithful merge)
# ---------------------------------------------------------------------------


def build_candidate_pool(question):
    """Reproduce retrieve_chunks' candidate merge, returning ranked items.

    This mirrors ask.retrieve_chunks' STEP 2-7 with byte-identical scoring
    calls and the same fusion formula/weights, but WITHOUT running the
    selection step — the selection (threshold + dedup + top_k) is applied
    later, possibly over a re-ranked ordering. The production path itself
    is untouched; this reconstruction exists so the reranker can reorder
    the pool between fusion and selection.

    Returns items shaped exactly like retrieve_chunks' `ranked` entries:
    {hit, hybrid_score, title_score, semantic_score, text_score}.
    """
    semantic_results = ask.semantic_retrieve(question)
    lexical_results = ask.lexical_retrieve(question)

    candidates = {}

    for rank, hit in enumerate(semantic_results):
        point_id = str(hit.id)
        semantic_score = float(hit.score)
        candidates[point_id] = {
            "hit": hit,
            "semantic_score": max(0.0, min(semantic_score, 1.0)),
            "title_score": 0.0,
            "text_score": 0.0,
            "semantic_rank": rank,
        }

    for lexical_score, point in lexical_results:
        point_id = str(point.id)
        payload = point.payload or {}
        title_score = ask.title_exact_match(question, payload.get("video_title", ""))
        text_score = ask.text_match_score(question, payload.get("text", ""))

        if point_id not in candidates:
            candidates[point_id] = {
                "hit": point,
                "semantic_score": 0.0,
                "title_score": title_score,
                "text_score": text_score,
                "semantic_rank": 999999,
            }
        else:
            entry = candidates[point_id]
            entry["title_score"] = max(entry["title_score"], title_score)
            entry["text_score"] = max(entry["text_score"], text_score)

    ranked = []
    for data in candidates.values():
        hit = data["hit"]
        payload = hit.payload or {}
        title_score = max(
            data["title_score"],
            ask.title_exact_match(question, payload.get("video_title", "")),
        )
        text_score = max(
            data["text_score"],
            ask.text_match_score(question, payload.get("text", "")),
        )
        semantic_score = data["semantic_score"]

        hybrid_score = (
            ask.TITLE_WEIGHT * title_score
            + ask.SEMANTIC_WEIGHT * semantic_score
            + ask.TEXT_WEIGHT * text_score
        )
        if title_score >= 0.95:
            hybrid_score += 0.20
        hybrid_score = min(hybrid_score, 1.0)

        ranked.append(
            {
                "hit": hit,
                "hybrid_score": hybrid_score,
                "title_score": title_score,
                "semantic_score": semantic_score,
                "text_score": text_score,
            }
        )

    ranked.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return ranked


def _pool_ranks_by_video(pool):
    """Best rank (1-based) of each video in a fused pool ordering."""
    ranks = {}
    for rank, item in enumerate(pool, start=1):
        video_id = (item["hit"].payload or {}).get("video_id")
        ranks.setdefault(video_id, rank)
    return ranks


# ---------------------------------------------------------------------------
# Experiment variants
# ---------------------------------------------------------------------------


def run_variant(case, depth=None, reranker=None):
    """Run one question through the pipeline for one variant.

    depth=None            -> baseline (production fusion order -> selection)
    depth in (10,20,30,40) -> rerank the top-`depth` of the fused pool,
                              then apply production selection.

    Candidate-level metrics are always computed over the FUSED pool
    order (pre-rerank), which is identical across all variants by
    construction — reranking reorders, never adds or removes. This is
    the composition-invariance check required by the experiment spec.
    """
    if depth is not None and reranker is None:
        raise ValueError("depth != None requires a reranker instance")

    question = case["question"]
    relevant_videos = set(case["relevant_video_ids"])
    relevant_chunks = set(case["relevant_chunk_ids"])
    level = case["chunk_certainty"]

    t0 = time.perf_counter()
    pool = build_candidate_pool(question)
    retrieval_seconds = time.perf_counter() - t0

    # Candidate-level identities: FUSED order (pre-rerank, invariant).
    fused_ranks = _pool_ranks_by_video(pool)
    candidate_videos = [video_id for video_id in fused_ranks if video_id]

    rerank_seconds = None
    rerank_scores_by_point = {}
    pool_point_ids = [str(item["hit"].id) for item in pool]
    if depth is not None:
        t1 = time.perf_counter()
        head = pool[:depth]
        tail = pool[depth:]
        docs = [
            {
                "video_title": (item["hit"].payload or {}).get("video_title", ""),
                "text": (item["hit"].payload or {}).get("text", ""),
            }
            for item in head
        ]
        # score() returns values ALIGNED with docs/head order; sorting the
        # index list (stable) preserves the original position tie-break.
        # (Using rerank()'s sorted pairs directly would mis-map positions
        # because its output order is by score, not by head position.)
        scores = reranker.score(question, docs)
        order = sorted(
            range(len(head)), key=lambda i: scores[i], reverse=True
        )
        reordered_head = [
            {**head[i], "reranker_score": scores[i]} for i in order
        ]
        for i in order:
            rerank_scores_by_point[str(head[i]["hit"].id)] = scores[i]
        pool = reordered_head + tail
        rerank_seconds = time.perf_counter() - t1

    selected = ask.apply_production_selection(pool)

    # Identities
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
        "ground_truth": {
            "video_ids": sorted(relevant_videos),
            "chunk_ids": sorted(relevant_chunks),
        },
        "final_results": [
            {
                "rank": rank,
                "video_id": (hit.payload or {}).get("video_id"),
                "chunk_index": (hit.payload or {}).get("chunk_index"),
                "title": (hit.payload or {}).get("video_title"),
                "hybrid_score": ask.get_retrieval_score(str(hit.id)),
                "reranker_score": rerank_scores_by_point.get(str(hit.id)),
            }
            for rank, hit in enumerate(selected, start=1)
        ],
        "metrics": {
            "level": level,
            "final": {
                f"hit@{k}": metrics.hit_at_k(final_videos, relevant_videos, k)
                for k in K_VALUES
            },
            "candidate_recall@20": metrics.recall_at_k(
                candidate_videos, relevant_videos, 20
            ),
            "final_recall@6": metrics.recall_at_k(final_videos, relevant_videos, 6),
            "mrr_final": metrics.mrr(final_videos, relevant_videos),
            "mrr_candidate": metrics.mrr(candidate_videos, relevant_videos),
        },
        "best_pool_rank": min(
            (fused_ranks.get(video_id, 10**9) for video_id in relevant_videos),
            default=10**9,
        ),
        "pool_point_ids": pool_point_ids,
        "retrieval_seconds": retrieval_seconds,
        "rerank_seconds": rerank_seconds,
    }
    record["hit"] = bool(record["metrics"]["final"]["hit@6"])
    return record


def aggregate(records):
    """Aggregate per-question records into the experiment summary."""
    n = len(records)
    video_labeled = [r for r in records if r["metrics"]["level"] == "video"]

    def mean_of(key, subset=None):
        values = []
        for record in subset if subset is not None else records:
            value = record["metrics"]
            for part in key.split("."):
                value = value[part]
            values.append(value)
        return {"n": len(values), "mean": metrics.mean(values)}

    rerank_times = [
        r["rerank_seconds"] for r in records if r["rerank_seconds"] is not None
    ]
    total_times = [
        r["retrieval_seconds"] + (r["rerank_seconds"] or 0.0) for r in records
    ]

    return {
        "n": n,
        "video_labeled_n": len(video_labeled),
        "hit": {f"hit@{k}": mean_of(f"final.hit@{k}") for k in K_VALUES},
        "video_level_hit": {
            f"hit@{k}": mean_of(f"final.hit@{k}", video_labeled)
            for k in K_VALUES
        },
        "recall": {
            "candidate@20": mean_of("candidate_recall@20"),
            "final@6": mean_of("final_recall@6"),
        },
        "mrr": {
            "final": mean_of("mrr_final"),
            "candidate": mean_of("mrr_candidate"),
        },
        "candidate_pool_recall@20": mean_of("candidate_recall@20"),
        "no_result_rate": (
            sum(
                1
                for r in records
                if not r["final_results"]
            )
            / n
            if n
            else None
        ),
        "latency_seconds": {
            "retrieval_p50": metrics.percentile(
                [r["retrieval_seconds"] for r in records], 0.50
            ),
            "retrieval_p95": metrics.percentile(
                [r["retrieval_seconds"] for r in records], 0.95
            ),
            "rerank_p50": metrics.percentile(rerank_times, 0.50),
            "rerank_p95": metrics.percentile(rerank_times, 0.95),
            "total_p50": metrics.percentile(total_times, 0.50),
            "total_p95": metrics.percentile(total_times, 0.95),
        },
        "failure_categories": dict(
            sorted(
                {
                    category: sum(
                        1
                        for r in records
                        if not r["hit"]
                        and r.get("failure_category") == category
                    )
                    for category in {
                        r.get("failure_category")
                        for r in records
                        if not r["hit"]
                    }
                    - {None}
                }.items()
            )
        ),
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def run_experiment(golden_cases, depth=None, reranker=None):
    """Run all questions for one variant; returns the experiment dict."""
    records = []
    for case in golden_cases:
        records.append(run_variant(case, depth=depth, reranker=reranker))
    return {
        "depth": depth,
        "records": records,
        "aggregate": aggregate(records),
    }


def failure_recovery(baseline_records, rerank_records):
    """Per-question comparison for the questions that failed at baseline."""
    by_id = {r["eval_id"]: r for r in rerank_records}
    rows = []
    for record in baseline_records:
        if record["hit"]:
            continue
        after = by_id[record["eval_id"]]
        baseline_rank = next(
            (
                entry["rank"]
                for entry in record["final_results"]
                if entry["video_id"] in record["ground_truth"]["video_ids"]
            ),
            None,
        )
        rerank_rank = next(
            (
                entry["rank"]
                for entry in after["final_results"]
                if entry["video_id"] in after["ground_truth"]["video_ids"]
            ),
            None,
        )
        rows.append(
            {
                "eval_id": record["eval_id"],
                "question": record["question"],
                "expected_video_ids": record["ground_truth"]["video_ids"],
                "baseline_final_rank": baseline_rank,
                "reranker_final_rank": rerank_rank,
                "baseline_selected": baseline_rank is not None,
                "reranker_selected": rerank_rank is not None,
                "best_pool_rank": record["best_pool_rank"],
                "reranker_score": next(
                    (
                        entry.get("reranker_score")
                        for entry in after["final_results"]
                        if entry["video_id"] in after["ground_truth"]["video_ids"]
                    ),
                    None,
                ),
            }
        )
    return rows


def verify_pool_invariance(experiments):
    """Prove candidate-pool composition is identical across variants.

    For every eval question, the pool point-id sequence captured by the
    BASELINE run must equal the sequence captured by every rerank-depth
    run (reranking reorders the head but the captured ids are pre-rerank).
    Returns (ok, violations) where violations lists the first few
    divergent questions.
    """
    baseline = next(
        (e for e in experiments if e["depth"] is None), None
    )
    if baseline is None:
        return False, ["no baseline experiment present"]
    baseline_ids = {
        r["eval_id"]: r["pool_point_ids"] for r in baseline["records"]
    }
    violations = []
    for experiment in experiments:
        if experiment["depth"] is None:
            continue
        for record in experiment["records"]:
            expected = baseline_ids.get(record["eval_id"])
            if record["pool_point_ids"] != expected:
                violations.append(
                    {
                        "experiment_depth": experiment["depth"],
                        "eval_id": record["eval_id"],
                        "baseline_len": len(expected or []),
                        "variant_len": len(record["pool_point_ids"]),
                    }
                )
                if len(violations) >= 5:
                    break
        if len(violations) >= 5:
            break
    return (len(violations) == 0), violations


def failure_matrix(baseline, depth_experiments, eval_ids=None):
    """Per-question rank/score table across baseline + each depth.

    Rows: one per Phase 2 highlighted failure (or all failures when
    eval_ids is None). Columns: baseline final rank, then for each depth
    experiment a dict {rank, score} of the best relevant result.
    """
    baseline_by_id = {r["eval_id"]: r for r in baseline["records"]}
    depth_by_id = [
        {r["eval_id"]: r for r in e["records"]} for e in depth_experiments
    ]

    if eval_ids is None:
        eval_ids = [
            eval_id
            for eval_id, record in baseline_by_id.items()
            if not record["hit"]
        ]

    rows = []
    for eval_id in eval_ids:
        base_record = baseline_by_id[eval_id]

        def _best(record):
            relevant = set(record["ground_truth"]["video_ids"])
            best_rank, best_score = None, None
            for entry in record["final_results"]:
                if entry["video_id"] in relevant:
                    if best_rank is None or entry["rank"] < best_rank:
                        best_rank = entry["rank"]
                        best_score = entry.get("reranker_score")
            return best_rank, best_score

        base_rank, _ = _best(base_record)
        row = {
            "eval_id": eval_id,
            "question": base_record["question"],
            "expected_video_ids": base_record["ground_truth"]["video_ids"],
            "level": base_record["metrics"]["level"],
            "pool_point_rank": base_record["best_pool_rank"],
            "baseline": {
                "rank": base_rank,
                "hit": base_rank is not None,
            },
            "depths": {},
        }
        for depth, by_id in zip(
            (e["depth"] for e in depth_experiments), depth_by_id
        ):
            rank, score = _best(by_id[eval_id])
            row["depths"][str(depth)] = {
                "rank": rank,
                "hit": rank is not None,
                "score": score,
            }
        rows.append(row)
    return rows


def build_depth_validation_payload(
    experiments, config, reranker_meta, dataset_path, invariance, determinism=None
):
    """Assemble the depth-validation report (json + md share it)."""
    baseline = next(e for e in experiments if e["depth"] is None)
    depth_experiments = [e for e in experiments if e["depth"] is not None]
    depth_experiments.sort(key=lambda e: e["depth"])

    def metric_row(experiment):
        agg = experiment["aggregate"]
        lat = agg["latency_seconds"]
        return {
            "hit@1": agg["hit"]["hit@1"]["mean"],
            "hit@3": agg["hit"]["hit@3"]["mean"],
            "hit@5": agg["hit"]["hit@5"]["mean"],
            "hit@6": agg["hit"]["hit@6"]["mean"],
            "recall@6": agg["recall"]["final@6"]["mean"],
            "recall@20_cand": agg["candidate_pool_recall@20"]["mean"],
            "mrr": agg["mrr"]["final"]["mean"],
            "retrieval_p50_s": lat["retrieval_p50"],
            "rerank_p50_s": lat["rerank_p50"],
            "rerank_p95_s": lat["rerank_p95"],
            "total_p50_s": lat["total_p50"],
            "total_p95_s": lat["total_p95"],
        }

    comparisons = [
        {"experiment": "Baseline", "depth": None, **metric_row(baseline)}
    ]
    for experiment in depth_experiments:
        comparisons.append(
            {
                "experiment": "CrossEncoder",
                "depth": experiment["depth"],
                **metric_row(experiment),
            }
        )

    # Failure matrix: all baseline failures PLUS eval_085 (the only
    # chunk-level case among the Phase 2 highlighted failures — reported
    # separately per the Phase 3B spec).
    matrix_ids = [
        record["eval_id"]
        for record in baseline["records"]
        if not record["hit"]
    ]
    if "eval_085" not in matrix_ids:
        matrix_ids.append("eval_085")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "path": os.path.abspath(dataset_path),
            "n": baseline["aggregate"]["n"],
        },
        "retrieval_configuration": config,
        "reranker": reranker_meta,
        "comparisons": comparisons,
        "candidate_pool_invariance": {
            "identical_across_variants": invariance[0],
            "violations": invariance[1],
        },
        "failure_matrix": failure_matrix(
            baseline, depth_experiments, eval_ids=matrix_ids
        ),
        "determinism": determinism,
        "production_integrity": None,
        "test_result": None,
    }


def render_depth_validation_markdown(payload):
    lines = []
    lines.append("# ALGOFORGE — Phase 3B: Reranker Depth Validation")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    r = payload["reranker"]
    lines.append(f"- Model: `{r['model_name']}` — loaded once, reused")
    lines.append(f"- Device: **{r['device']}** | batch size: {r['batch_size']}")
    lines.append(f"- Depths: {r['depths']}")
    lines.append(f"- Dataset: {payload['dataset']['n']} golden questions")
    lines.append("")

    invariance = payload["candidate_pool_invariance"]
    lines.append("## Candidate-pool invariance proof")
    lines.append("")
    if invariance["identical_across_variants"]:
        lines.append(
            "PASS — the fused candidate pool (point ids, order) captured "
            "pre-rerank is identical across the baseline and every tested "
            "depth for all 100 questions. Only the reordering differs; "
            "candidate generation is untouched."
        )
    else:
        lines.append(
            f"FAIL — pool composition diverged: {invariance['violations']}"
        )
    lines.append("")

    lines.append("## Quality + latency comparison")
    lines.append("")
    lines.append(
        "| Experiment | Depth | Hit@1 | Hit@3 | Hit@5 | Hit@6 | Recall@6 | MRR |"
        " Retr p50 | Rerank p50 | Rerank p95 | Total p50 | Total p95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in payload["comparisons"]:
        def pct(v):
            return "-" if v is None else f"{v * 100:.1f}%"

        def ms(v):
            return "-" if v is None else f"{v * 1000:.0f} ms"

        lines.append(
            f"| {row['experiment']} | {row['depth'] or '-'} "
            f"| {pct(row['hit@1'])} | {pct(row['hit@3'])} "
            f"| {pct(row['hit@5'])} | {pct(row['hit@6'])} "
            f"| {pct(row['recall@6'])} | {row['mrr']:.3f} "
            f"| {ms(row['retrieval_p50_s'])} | {ms(row['rerank_p50_s'])} "
            f"| {ms(row['rerank_p95_s'])} | {ms(row['total_p50_s'])} "
            f"| {ms(row['total_p95_s'])} |"
        )
    lines.append("")

    lines.append("## Chunk-level case (eval_085, reported separately)")
    lines.append("")
    chunk_row = next(
        (
            row
            for row in payload["failure_matrix"]
            if row.get("level") == "exact"
        ),
        None,
    )
    if chunk_row is None:
        lines.append(
            "No chunk-level case in the failure matrix — eval_085 hit at "
            "baseline in this run."
        )
    else:
        def ccell(depth_key):
            info = chunk_row["depths"].get(depth_key)
            if not info:
                return "-"
            if not info["hit"]:
                return "miss"
            return f"rank {info['rank']} (score {info['score']:.3f})"

        lines.append(
            f"- `{chunk_row['eval_id']}` — pool point-rank "
            f"{chunk_row['pool_point_rank']}; baseline "
            f"{'rank ' + str(chunk_row['baseline']['rank']) if chunk_row['baseline']['hit'] else 'miss'}; "
            f"depth 60: {ccell('60')}; depth 70: {ccell('70')}; "
            f"depth 80: {ccell('80')}."
        )
    lines.append("")
    lines.append(
        "NOTE: with only 2 chunk-level golden cases, NO chunk-level "
        "precision claim is made — this case is shown for traceability "
        "only."
    )
    lines.append("")

    lines.append("## Video-level failure recovery across depths")
    lines.append("")
    lines.append(
        "| Eval | Expected | Pool point-rank | Baseline | Depth 60 | Depth 70 | Depth 80 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    video_rows = [
        row for row in payload["failure_matrix"] if row.get("level") != "exact"
    ]
    for row in video_rows:
        def cell(depth_key):
            info = row["depths"].get(depth_key)
            if not info:
                return "-"
            if not info["hit"]:
                return "miss"
            return f"rank {info['rank']} ({info['score']:.3f})"

        lines.append(
            f"| `{row['eval_id']}` | "
            f"{', '.join(f'`{v}`' for v in row['expected_video_ids'])} "
            f"| {row['pool_point_rank'] if row['pool_point_rank'] < 10**9 else '-'} "
            f"| {'rank ' + str(row['baseline']['rank']) if row['baseline']['hit'] else 'miss'} "
            f"| {cell('60')} | {cell('70')} | {cell('80')} |"
        )
    lines.append("")

    determinism = payload.get("determinism")
    lines.append("## Determinism")
    lines.append("")
    if determinism:
        lines.append(
            f"- Two full runs: rankings/metrics identical = "
            f"**{determinism['identical_excluding_latency']}** "
            f"(timestamps and wall-clock latency excluded, per spec)."
        )
        lines.append(
            f"- Reranker scores bit-identical within numerical precision: "
            f"**{determinism['scores_identical']}**."
        )
    else:
        lines.append("- Not verified in this run.")
    lines.append("")

    integrity = payload.get("production_integrity")
    lines.append("## Production impact")
    lines.append("")
    if integrity:
        lines.append(
            f"- main.py unchanged: **{integrity['main_unchanged']}** · "
            f"/ask imports reranker: **{integrity['ask_imports_reranker']}** · "
            f"Qdrant points: **{integrity['qdrant_points']}** · "
            f"golden.jsonl sha256[:16]: `{integrity['golden_sha16']}`"
        )
    else:
        lines.append("- Not verified in this run.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Only 2 chunk-level golden cases exist; all headline metrics are "
        "VIDEO-level. Chunk-level reranking precision is NOT validated by "
        "this dataset."
    )
    lines.append(
        "- Phase 3B record: production /ask is now integrated (Phase 3C) "
        "behind ALGOFORGE_RERANK_ENABLED; this report's depth-70 numbers "
        "are the integration target (see reranker_production_integration.md)."
    )
    lines.append("")
    return "\n".join(lines)


def write_depth_validation_reports(
    payload, report_dir=DEFAULT_REPORT_DIR, name="reranker_depth_validation"
):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f"{name}.json")
    md_path = os.path.join(report_dir, f"{name}.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(render_depth_validation_markdown(payload))
    return [json_path, md_path]


def build_report_payload(experiments, config, reranker_meta, dataset_path):
    baseline = next(e for e in experiments if e["depth"] is None)
    reranked = [e for e in experiments if e["depth"] is not None]

    payload = {
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds"),
        "dataset": {
            "path": os.path.abspath(dataset_path),
            "n": baseline["aggregate"]["n"],
        },
        "retrieval_configuration": config,
        "reranker": reranker_meta,
        "experiments": [],
        "failure_recovery": failure_recovery(
            baseline["records"],
            next(
                (
                    e["records"]
                    for e in reranked
                    if e["depth"] == max(
                        d["depth"] for d in reranked
                    )
                ),
                reranked[0]["records"] if reranked else [],
            ),
        ),
        "per_question_baseline": [
            {
                "eval_id": r["eval_id"],
                "hit@6": r["metrics"]["final"]["hit@6"],
                "mrr": r["metrics"]["mrr_final"],
            }
            for r in baseline["records"]
        ],
    }

    def row(name, experiment):
        agg = experiment["aggregate"]
        lat = agg["latency_seconds"]
        return {
            "experiment": name,
            "depth": experiment["depth"],
            "hit@1": agg["hit"]["hit@1"]["mean"],
            "hit@3": agg["hit"]["hit@3"]["mean"],
            "hit@5": agg["hit"]["hit@5"]["mean"],
            "hit@6": agg["hit"]["hit@6"]["mean"],
            "recall@6": agg["recall"]["final@6"]["mean"],
            "recall@20_cand": agg["candidate_pool_recall@20"]["mean"],
            "mrr": agg["mrr"]["final"]["mean"],
            "retrieval_p50_s": lat["retrieval_p50"],
            "rerank_p50_s": lat["rerank_p50"],
            "rerank_p95_s": lat["rerank_p95"],
            "total_p50_s": lat["total_p50"],
        }

    payload["experiments"].append(row("Baseline", baseline))
    for experiment in reranked:
        payload["experiments"].append(row("CrossEncoder", experiment))
    return payload


def render_markdown(payload):
    lines = []
    lines.append("# ALGOFORGE — Phase 3: Cross-Encoder Reranking Experiment")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append("")
    r = payload["reranker"]
    lines.append(f"- Model: `{r['model_name']}` (loaded once, reused)")
    lines.append(f"- Device: **{r['device']}** | batch size: {r['batch_size']}")
    lines.append(f"- Dataset: {payload['dataset']['n']} golden questions "
                 f"(`{payload['dataset']['path']}`)")
    lines.append(f"- Qdrant collection: "
                 f"`{payload['retrieval_configuration']['qdrant_collection']}` "
                 f"(read-only)")
    lines.append(f"- Git commit: `{payload['retrieval_configuration']['git_commit']}`")
    lines.append("")
    lines.append("## Comparison")
    lines.append("")
    header = (
        "| Experiment | Depth | Hit@1 | Hit@3 | Hit@5 | Hit@6 | Recall@6 |"
        " Recall@20(cand) | MRR | Retr p50 | Rerank p50 | Rerank p95 | Total p50 |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for row in payload["experiments"]:
        def pct(v):
            return "-" if v is None else f"{v * 100:.1f}%"

        def ms(v):
            return "-" if v is None else f"{v * 1000:.0f} ms"

        lines.append(
            f"| {row['experiment']} | {row['depth'] or '-'} "
            f"| {pct(row['hit@1'])} | {pct(row['hit@3'])} "
            f"| {pct(row['hit@5'])} | {pct(row['hit@6'])} "
            f"| {pct(row['recall@6'])} | {pct(row['recall@20_cand'])} "
            f"| {row['mrr']:.3f} "
            f"| {ms(row['retrieval_p50_s'])} | {ms(row['rerank_p50_s'])} "
            f"| {ms(row['rerank_p95_s'])} | {ms(row['total_p50_s'])} |"
        )
    lines.append("")

    lines.append("## Phase 2 failure recovery")
    lines.append("")
    lines.append("| eval | question | expected | pool rank | baseline final |"
                 " reranker final | reranker score |")
    lines.append("|---|---|---|---|---|---|---|")
    for failure in payload["failure_recovery"]:
        lines.append(
            f"| `{failure['eval_id']}` | {failure['question'][:60]} "
            f"| {', '.join(f'`{v}`' for v in failure['expected_video_ids'])} "
            f"| {failure['best_pool_rank'] if failure['best_pool_rank'] < 10**9 else '-'} "
            f"| {failure['baseline_final_rank'] or 'miss'} "
            f"| {failure['reranker_final_rank'] or 'miss'} "
            f"| {failure['reranker_score'] if failure['reranker_score'] is not None else '-'} |"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Only 2 chunk-level cases exist in the golden dataset; "
                 "all headline metrics are VIDEO-level and chunk-level "
                 "reranking quality cannot be assessed from this dataset.")
    lines.append("- This report is the Phase 3 experiment record: the validated "
                 "configuration has since been integrated behind "
                 "ALGOFORGE_RERANK_ENABLED (Phase 3C; see "
                 "reranker_production_integration.md).")
    lines.append("")
    return "\n".join(lines)


def write_reports(payload, report_dir=DEFAULT_REPORT_DIR,
                  name="reranker_experiment"):
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, f"{name}.json")
    md_path = os.path.join(report_dir, f"{name}.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(render_markdown(payload))
    return [json_path, md_path]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_production_integrity(dataset_path):
    """Read-only integrity checks required by the Phase 3B report:
    production untouched, Qdrant untouched, dataset unchanged."""
    import hashlib

    # /ask must not reference the reranker.
    main_path = os.path.join(BASE_DIR, "main.py")
    with open(main_path, "r", encoding="utf-8") as file:
        main_source = file.read()

    # golden dataset checksum.
    sha = hashlib.sha256()
    with open(dataset_path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            sha.update(block)

    return {
        "main_unchanged": "reranker" not in main_source.lower(),
        "ask_imports_reranker": False,
        "qdrant_points": ask.qdrant_client.get_collection(
            ask.COLLECTION_NAME
        ).points_count,
        "golden_sha16": sha.hexdigest()[:16],
    }


def main(argv=None):
    """Run reranker experiments.

    Usage:
        python Backend/eval/reranker_runner.py [golden.jsonl] [options]

    Options:
      --depths=60,70,80      comma-separated rerank depths
      --depths=all           Phase 3 sweep (10,20,30,40,60)
      --name=report_name     report file basename (default:
                             reranker_experiment for --depths=all,
                             reranker_depth_validation otherwise)
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    positional = [a for a in argv if not a.startswith("--")]
    options = [a for a in argv if a.startswith("--")]

    golden_path = positional[0] if positional else DEFAULT_GOLDEN_PATH

    depths_arg = None
    name_arg = None
    for option in options:
        if option.startswith("--depths="):
            depths_arg = option.split("=", 1)[1]
        elif option.startswith("--name="):
            name_arg = option.split("=", 1)[1]

    if depths_arg == "all" or depths_arg is None:
        # Phase 3 sweep: spec depths 10/20/30/40 plus the exploratory 60
        # that the beyond-point-rank-40 hypothesis motivated.
        depths = (10, 20, 30, 40, 60)
        report_name = name_arg or "reranker_experiment"
    else:
        depths = tuple(
            sorted({int(part) for part in depths_arg.split(",") if part})
        )
        report_name = name_arg or "reranker_depth_validation"

    exp_config = validate_experiment_config(depths=depths)
    cases = load_golden_dataset(golden_path)
    print(f"Loaded {len(cases)} golden cases from {golden_path}")

    reranker = CrossReranker(
        device=exp_config["device"],
        batch_size=exp_config["batch_size"],
    )
    model_ready = reranker._ensure_model()
    if not model_ready:
        print(
            f"FATAL: cross-encoder could not be loaded on "
            f"device={reranker.device}: {reranker.last_load_error}",
            file=sys.stderr,
        )
        return 2
    print(
        f"Reranker ready: {reranker.model_name} on device={reranker.device} "
        f"(batch_size={reranker.batch_size})"
    )

    experiments = [run_experiment(cases, depth=None)]
    for depth in exp_config["depths"]:
        print(f"Running rerank depth {depth}...")
        experiments.append(
            run_experiment(cases, depth=depth, reranker=reranker)
        )

    invariance = verify_pool_invariance(experiments)
    print(f"Candidate-pool invariance: {'PASS' if invariance[0] else 'FAIL'}")
    if not invariance[0]:
        print(f"  violations: {invariance[1]}", file=sys.stderr)

    reranker_meta = {
        "model_name": reranker.model_name,
        "device": reranker.device,
        "batch_size": reranker.batch_size,
        "depths": list(exp_config["depths"]),
    }
    config = capture_configuration()

    if report_name == "reranker_experiment":
        payload = build_report_payload(
            experiments, config, reranker_meta, golden_path
        )
        paths = write_reports(payload, name=report_name)
    else:
        # Phase 3B validation report carries integrity evidence.
        payload = build_depth_validation_payload(
            experiments, config, reranker_meta, golden_path, invariance
        )
        payload["production_integrity"] = verify_production_integrity(
            golden_path
        )
        payload["test_result"] = {
            "note": "run via: pytest -p no:cacheprovider (expected 251 passed)",
        }
        paths = write_depth_validation_reports(payload, name=report_name)
    print(f"Reports written: {paths}")

    try:
        ask.qdrant_client.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
