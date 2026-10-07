"""ALGOFORGE Citation and Source Formatting.

Provides deterministic timestamp link generation and formatted source text.
"""

_get_retrieval_score_fn = None


def register_score_getter(fn):
    """Register retrieval score lookup function (injected from retrieval layer)."""
    global _get_retrieval_score_fn
    _get_retrieval_score_fn = fn


def format_sources(chunks, get_score_fn=None):
    """Format retrieved chunks into human-readable citation strings for CLI/display."""
    if not chunks:
        return "  (no sources)"

    score_fn = get_score_fn or _get_retrieval_score_fn
    lines = []

    for i, hit in enumerate(chunks, start=1):
        payload = hit.payload or {}
        title = payload.get("video_title", "Unknown")
        url = payload.get("youtube_url", "")

        try:
            start = int(float(payload.get("start", 0)))
        except (TypeError, ValueError):
            start = 0

        if url:
            separator = "&" if "?" in url else "?"
            timestamp_link = f"{url}{separator}t={start}"
        else:
            timestamp_link = "N/A"

        point_id = str(hit.id)
        if url and score_fn:
            score = score_fn(point_id)
        else:
            score = 0.0

        lines.append(
            f"  [{i}] {title} — {start}s\n"
            f"      {timestamp_link}\n"
            f"      (hybrid score: {score:.4f})"
        )

    return "\n".join(lines)
