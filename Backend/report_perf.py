"""Aggregate ALGOFORGE performance logs into a latency report.

Reads the rotating JSONL perf log written by Backend/obs.py (logs/perf.log
by default, or the path given as the first argument) and prints:

- request count, successful / failed requests, failure breakdown
- p50 / p95 for total, retrieval, and generation latency
- per-stage p50 / p95 for every recorded [PERF] stage

This utility is fully independent from the production request path: it
only reads a log file. It never imports ask.py, FastAPI, or any model,
so it is safe to run while the server is up (subject to Windows file
locking, in which case copy the file first).

Usage:
    python Backend/report_perf.py [path/to/perf.log]
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict


def percentile(sorted_values, fraction):
    """Nearest-rank percentile over an already-sorted list. Never empty.

    Nearest-rank definition: p_f is the value at index ceil(f * n) - 1
    (1-based rank ceil(f * n)). For n=4, p50 -> rank 2 -> index 1.
    """
    if not sorted_values:
        return None
    index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(fraction * len(sorted_values)) - 1),
    )
    return sorted_values[index]


def _fmt_ms(seconds):
    if seconds is None:
        return "-"
    return f"{seconds * 1000:.1f}ms"


def load_events(paths):
    """Yield parsed JSON objects from one or more JSONL log files.

    Malformed lines are skipped (rotating-file boundaries can split a
    line); a missing file raises FileNotFoundError so the caller can
    report a clean usage error.
    """
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


def summarize(events):
    """Compute the report rows from parsed events."""
    starts = 0
    ended = []
    failures = Counter()

    stage_values = defaultdict(list)

    for event in events:
        name = event.get("event")

        if name == "request_start":
            starts += 1
        elif name in ("request_end", "request_failed"):
            ended.append(event)
            if name == "request_failed":
                failures[event.get("failure_category") or "unknown"] += 1

        stages = event.get("stages")
        if isinstance(stages, dict):
            for stage_name, seconds in stages.items():
                if isinstance(seconds, (int, float)):
                    stage_values[stage_name].append(float(seconds))

    successful = sum(
        1
        for event in ended
        if event.get("failure_category") is None
        and (event.get("http_status") or 0) < 400
    )

    totals = sorted(
        event.get("total_latency")
        for event in ended
        if isinstance(event.get("total_latency"), (int, float))
    )
    retrievals = sorted(
        event.get("retrieval_latency")
        for event in ended
        if isinstance(event.get("retrieval_latency"), (int, float))
    )
    generations = sorted(
        event.get("generation_latency")
        for event in ended
        if isinstance(event.get("generation_latency"), (int, float))
    )

    stage_rows = {
        stage: sorted(values)
        for stage, values in stage_values.items()
    }

    return {
        "starts": starts,
        "ended": len(ended),
        "successful": successful,
        "failed": len(ended) - successful,
        "failures": failures,
        "totals": totals,
        "retrievals": retrievals,
        "generations": generations,
        "stage_rows": stage_rows,
    }


def render(report):
    lines = []
    lines.append("=" * 62)
    lines.append("ALGOFORGE PERFORMANCE REPORT")
    lines.append("=" * 62)
    lines.append(f"Requests started : {report['starts']}")
    lines.append(f"Requests ended   : {report['ended']}")
    lines.append(f"Successful       : {report['successful']}")
    lines.append(f"Failed           : {report['failed']}")

    if report["failures"]:
        lines.append("Failure breakdown:")
        for category, count in report["failures"].most_common():
            lines.append(f"  {category:<28} {count}")

    lines.append("-" * 62)
    lines.append(f"{'phase':<22}{'p50':>12}{'p95':>12}{'n':>8}")
    lines.append("-" * 62)

    for label, values in (
        ("total request", report["totals"]),
        ("retrieval", report["retrievals"]),
        ("generation", report["generations"]),
    ):
        lines.append(
            f"{label:<22}"
            f"{_fmt_ms(percentile(values, 0.50)):>12}"
            f"{_fmt_ms(percentile(values, 0.95)):>12}"
            f"{len(values):>8}"
        )

    if report["stage_rows"]:
        lines.append("-" * 62)
        lines.append("Per-stage ([PERF] instrumentation):")
        lines.append(f"{'stage':<22}{'p50':>12}{'p95':>12}{'n':>8}")
        lines.append("-" * 62)
        for stage in sorted(report["stage_rows"]):
            values = report["stage_rows"][stage]
            lines.append(
                f"{stage:<22}"
                f"{_fmt_ms(percentile(values, 0.50)):>12}"
                f"{_fmt_ms(percentile(values, 0.95)):>12}"
                f"{len(values):>8}"
            )

    lines.append("=" * 62)
    return "\n".join(lines)


def main(argv):
    if len(argv) > 1:
        paths = argv[1:]
    else:
        # Default: the same location obs.configure() writes to.
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = [os.path.join(base_dir, "logs", "perf.log")]

    for path in paths:
        if not os.path.exists(path):
            print(f"ERROR: perf log not found: {path}", file=sys.stderr)
            return 1

    report = summarize(load_events(paths))
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
