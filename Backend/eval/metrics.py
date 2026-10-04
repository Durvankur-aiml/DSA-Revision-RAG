"""Retrieval metric definitions (documented, level-agnostic).

All metric functions operate on an ORDERED list of retrieved identities
(best first) and a set of relevant identities. The caller decides what an
"identity" is:

- VIDEO level: identities are video ids. Used when ground truth is the
  relevant video (chunk_certainty == "video").
- CHUNK level: identities are chunk keys "videoId_chunkIndex". Used when
  chunk-level ground truth exists (chunk_certainty == "exact").

The two levels are NEVER mixed by the runner: video-level metrics are
reported over video-labeled questions and chunk-level metrics over
chunk-labeled questions, and every reported number is labeled with its
level. This module contains no I/O and no randomness.

Definitions
-----------
hit_at_k(retrieved, relevant, k)
    1 if ANY relevant identity appears in the top-k of `retrieved`,
    else 0. ("Did we get something right in the first k?")

recall_at_k(retrieved, relevant, k)
    |relevant ∩ top-k| / |relevant|. With one relevant identity this
    equals hit_at_k. ("How much of the known evidence did we surface?")

mrr(retrieved, relevant)
    1 / rank of the FIRST relevant identity (1-based), 0 if none present
    anywhere in `retrieved`. ("How high does the first right answer sit?")

All three are standard IR definitions (see Manning et al., Introduction
to Information Retrieval, ch. 8).
"""

import math


def _validate_inputs(retrieved, relevant, k=None):
    if k is not None:
        if not isinstance(k, int) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
    if not isinstance(retrieved, (list, tuple)):
        raise TypeError("retrieved must be an ordered list (best first)")
    if not isinstance(relevant, (set, frozenset, list, tuple)):
        raise TypeError("relevant must be a set or list of identities")
    return set(relevant)


def hit_at_k(retrieved, relevant, k):
    """1 if any relevant identity is within the top-k, else 0."""
    relevant_set = _validate_inputs(retrieved, relevant, k)
    top = retrieved[:k]
    return 1 if any(identity in relevant_set for identity in top) else 0


def recall_at_k(retrieved, relevant, k):
    """|relevant ∩ top-k| / |relevant| (0.0 when relevant is empty)."""
    relevant_set = _validate_inputs(retrieved, relevant, k)
    if not relevant_set:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for identity in top if identity in relevant_set)
    return found / len(relevant_set)


def mrr(retrieved, relevant):
    """1 / first-relevant rank (1-based); 0.0 when nothing relevant."""
    relevant_set = _validate_inputs(retrieved, relevant)
    for rank, identity in enumerate(retrieved, start=1):
        if identity in relevant_set:
            return 1.0 / rank
    return 0.0


def mean(values):
    """Arithmetic mean; None for an empty list (no observations)."""
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values, fraction):
    """Nearest-rank percentile, same definition as Backend/report_perf.py:
    value at 1-based rank ceil(fraction * n)."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1),
    )
    return ordered[index]
