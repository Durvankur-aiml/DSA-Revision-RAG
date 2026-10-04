"""Baseline report generation: baseline.json + baseline.md.

The reports are fully self-contained snapshots: dataset provenance,
retrieval configuration, aggregate metrics at both levels, latency
statistics, failure-category breakdown, and per-question results
including every failure with its category and detail. Transcript text is
never duplicated into reports (titles + ids only); no prompts, no keys.
"""

import json
import os
from datetime import datetime, timezone

from eval import metrics


def _pct(value):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _ms(seconds):
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.1f} ms"


def build_report_payload(result, config, dataset_path):
    """Assemble the full report dict (shared by JSON and MD writers)."""
    aggregate = result["aggregate"]
    records = result["records"]

    dataset_size = os.path.getsize(dataset_path) if os.path.exists(dataset_path) else None
    sections = sorted(
        {r["section"] for r in records if r.get("section")}
    )
    question_types = sorted(
        {r["question_type"] for r in records if r.get("question_type")}
    )

    failures = [r for r in records if not r["hit"]]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "evaluation_started_at": result["started_at"],
        "dataset": {
            "path": os.path.abspath(dataset_path),
            "size_bytes": dataset_size,
            "n": aggregate["n"],
            "video_labeled_n": aggregate["video_labeled_n"],
            "chunk_labeled_n": aggregate["chunk_labeled_n"],
            "sections": sections,
            "question_types": question_types,
            "section_count": len(sections),
        },
        "retrieval_configuration": config,
        "metrics": aggregate,
        "per_question": [
            {
                "eval_id": r["eval_id"],
                "question": r["question"],
                "section": r.get("section"),
                "question_type": r.get("question_type"),
                "level": r["metrics"]["level"],
                "ground_truth": r["ground_truth"],
                "hit": r["hit"],
                "failure_category": r["failure_category"],
                "failure_detail": r["failure_detail"],
                "final_hit@6": r["metrics"]["final"]["hit@6"],
                "candidate_hit@10": r["metrics"]["candidate"]["hit@10"],
                "candidate_recall@20": r["metrics"]["candidate_recall@20"],
                "final_recall@6": r["metrics"]["final_recall@6"],
                "mrr_final": r["metrics"]["mrr_final"],
                "mrr_candidate": r["metrics"]["mrr_candidate"],
                "candidate_count": r["candidate_count"],
                "final_results": r["final_results"],
                "retrieval_seconds": r["retrieval_seconds"],
            }
            for r in records
        ],
        "failures": [
            {
                "eval_id": r["eval_id"],
                "question": r["question"],
                "expected_video_ids": r["ground_truth"]["video_ids"],
                "expected_chunk_ids": r["ground_truth"]["chunk_ids"],
                "retrieved_video_ids": r["final_ids"]["videos"],
                "retrieved_chunk_ids": r["final_ids"]["chunks"],
                "candidate_video_ids": r["candidate_ids"]["videos"],
                "failure_category": r["failure_category"],
                "failure_detail": r["failure_detail"],
                "best_candidate_rank": _best_rank_of(r),
                "retrieval_seconds": r["retrieval_seconds"],
            }
            for r in failures
        ],
    }


def _best_rank_of(record):
    """Rank of the best relevant video inside the candidate pool."""
    relevant = set(record["ground_truth"]["video_ids"])
    for rank, video_id in enumerate(record["candidate_ids"]["videos"], 1):
        if video_id in relevant:
            return rank
    return None


def write_json_report(payload, report_dir, name="baseline"):
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return path


def render_markdown(payload):
    """Render the human-readable baseline.md."""
    config = payload["retrieval_configuration"]
    dataset = payload["dataset"]
    m = payload["metrics"]
    lines = []

    lines.append("# ALGOFORGE — Retrieval Baseline Report")
    lines.append("")
    lines.append(f"Generated: {payload['generated_at']}  ")
    lines.append(f"Git commit: `{config.get('git_commit')}`  ")
    lines.append(f"Dataset: `{payload['dataset']['path']}` "
                 f"({dataset['n']} questions)")
    lines.append("")

    lines.append("## 1. Retrieval configuration under test")
    lines.append("")
    fw = config["fusion_weights"]
    lines.append(f"- TOP_K: **{config['top_k']}**")
    lines.append(f"- Fusion weights: title **{fw['title']}** · "
                 f"semantic **{fw['semantic']}** · text **{fw['text']}**")
    lines.append(f"- Exact-title boost: **+{config['exact_title_boost']}** "
                 f"when title_score >= 0.95")
    lines.append(f"- MIN_SCORE_THRESHOLD: **{config['min_score_threshold']}**"
                 " (semantic) / 0.70 (title)")
    lines.append(f"- Candidate pools: semantic top-{config['semantic_candidates']}"
                 f" + lexical top-{config['lexical_candidates']}")
    lines.append(f"- Embeddings: {config['embedding_model']} "
                 f"({config['vector_size']}d, cosine)")
    lines.append(f"- Qdrant collection: `{config['qdrant_collection']}`")
    lines.append("")

    lines.append("## 2. Dataset coverage")
    lines.append("")
    lines.append(f"- {dataset['n']} questions: "
                 f"{dataset['video_labeled_n']} video-level · "
                 f"{dataset['chunk_labeled_n']} chunk-level")
    lines.append(f"- {dataset['section_count']} distinct sections: "
                 + ", ".join(dataset["sections"]))
    lines.append(f"- Question types: " + ", ".join(dataset["question_types"]))
    lines.append("")

    lines.append("## 3. Aggregate metrics")
    lines.append("")
    lines.append("### Final retrieval (production top-6 selection)")
    lines.append("")
    lines.append("| metric | value | n |")
    lines.append("|---|---|---|")
    for k in (1, 3, 5, 6, 10):
        entry = m["hit"]["final"][f"hit@{k}"]
        lines.append(f"| Final Hit@{k} | {_pct(entry['mean'])} | {entry['n']} |")
    lines.append(f"| Final Recall@6 | {_pct(m['recall']['final@6']['mean'])} "
                 f"| {m['recall']['final@6']['n']} |")
    lines.append(f"| Final Recall@20 | {_pct(m['recall']['final@20']['mean'])} "
                 f"| {m['recall']['final@20']['n']} |")
    lines.append(f"| Final MRR | {m['mrr']['final']['mean']:.3f} "
                 f"| {m['mrr']['final']['n']} |")
    lines.append("")

    lines.append("### Candidate retrieval (semantic 40 ∪ lexical 40, pre-selection)")
    lines.append("")
    lines.append("| metric | value | n |")
    lines.append("|---|---|---|")
    for k in (1, 3, 5, 6, 10):
        entry = m["hit"]["candidate"][f"hit@{k}"]
        lines.append(f"| Candidate Hit@{k} | {_pct(entry['mean'])} | {entry['n']} |")
    lines.append(f"| Candidate Recall@20 | "
                 f"{_pct(m['recall']['candidate@20']['mean'])} "
                 f"| {m['recall']['candidate@20']['n']} |")
    lines.append(f"| Candidate MRR | {m['mrr']['candidate']['mean']:.3f} "
                 f"| {m['mrr']['candidate']['n']} |")
    lines.append("")

    lines.append("### Ground-truth levels (final Hit@6, never mixed)")
    lines.append("")
    lines.append("| ground truth | Hit@1 | Hit@3 | Hit@6 | n |")
    lines.append("|---|---|---|---|---|")
    for label, block in (("video-level", m["video_level"]),
                         ("chunk-level", m["chunk_level"])):
        lines.append(
            f"| {label} | {_pct(block['hit@1']['mean'])} "
            f"| {_pct(block['hit@3']['mean'])} "
            f"| {_pct(block['hit@6']['mean'])} | {block['hit@6']['n']} |"
        )
    lines.append("")

    lines.append("### Health indicators")
    lines.append("")
    lines.append(f"- Average final source count: "
                 f"{m['avg_final_source_count']:.2f}")
    lines.append(f"- No-result rate: {_pct(m['no_result_rate'])}")
    lines.append(f"- Exact-topic candidate rate: "
                 f"{_pct(m['exact_topic_candidate_rate'])}")
    lines.append(f"- Retrieval latency: p50 {_ms(m['latency_seconds']['p50'])} · "
                 f"p95 {_ms(m['latency_seconds']['p95'])} · "
                 f"mean {_ms(m['latency_seconds']['mean'])} "
                 "(retrieval only — no Gemini)")
    lines.append("")

    lines.append("## 4. Failure categories")
    lines.append("")
    total_failures = sum(m["failure_categories"].values())
    if total_failures == 0:
        lines.append("No failures. (Suspicious for a real dataset — "
                     "check ground truth quality.)")
    else:
        lines.append("| category | count | share of failures |")
        lines.append("|---|---|---|")
        for category, count in m["failure_categories"].items():
            lines.append(
                f"| `{category}` | {count} | "
                f"{count / total_failures * 100:.0f}% |"
            )
    lines.append("")

    failures = payload["failures"]
    if failures:
        lines.append("## 5. Failed questions (all)")
        lines.append("")
        for failure in failures:
            lines.append(
                f"### `{failure['eval_id']}` — {failure['failure_category']}"
            )
            lines.append("")
            lines.append(f"> {failure['question']}")
            lines.append("")
            lines.append(f"- Expected video(s): "
                         f"{', '.join(f'`{v}`' for v in failure['expected_video_ids'])}")
            if failure["expected_chunk_ids"]:
                lines.append(
                    f"- Expected chunk(s): "
                    f"{', '.join(f'`{c}`' for c in failure['expected_chunk_ids'])}"
                )
            lines.append(f"- Retrieved: "
                         f"{', '.join(f'`{v}`' for v in failure['retrieved_video_ids']) or '(none)'}")
            lines.append(f"- Best candidate rank: {failure['best_candidate_rank']}")
            lines.append(f"- Detail: {failure['failure_detail']}")
            lines.append("")

    per_question = payload["per_question"]
    lines.append("## 6. Per-question results")
    lines.append("")
    lines.append("| id | level | hit@6 | cand@20 recall | MRR | failure |")
    lines.append("|---|---|---|---|---|---|")
    for row in per_question:
        lines.append(
            f"| `{row['eval_id']}` | {row['level']} "
            f"| {_pct(row['final_hit@6'])} "
            f"| {_pct(row['candidate_recall@20'])} "
            f"| {row['mrr_final']:.2f} "
            f"| {row['failure_category'] or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(payload, report_dir, name="baseline"):
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"{name}.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write(render_markdown(payload))
    return path


def write_reports(result, config, report_dir, dataset_path, report_name):
    payload = build_report_payload(result, config, dataset_path)
    json_path = write_json_report(payload, report_dir, report_name)
    md_path = write_markdown_report(payload, report_dir, report_name)
    return [json_path, md_path]
