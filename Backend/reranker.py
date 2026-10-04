"""Cross-encoder reranker module (ALGOFORGE Phase 3/3C).

Validated in Phase 3/3B and integrated into the production /ask path in
Phase 3C: retrieve_chunks uses one shared CrossReranker (created here,
lazily, once per process) to reorder the fused candidate pool behind
ALGOFORGE_RERANK_ENABLED (default off). The experiment harness
(Backend/eval/reranker_runner.py) uses the same class for its A/B
evaluations.

Design:
- The model is loaded ONCE per CrossReranker instance and reused.
- Device selection: CUDA when torch reports it available, else CPU
  (never hardcoded to GPU).
- Inference is batched (one forward pass per batch of pairs, not per
  candidate).
- The document representation is deterministic:
      Title:\n<video title>\n\nTranscript:\n<chunk text>
  No metadata beyond title+transcript is used, so no ground truth can
  leak into scoring.
- rerank() never mutates the caller's candidate list; it returns new
  (candidate, score) pairs sorted best-first, preserving the candidate
  objects themselves.
- An injectable score_fn makes the class fully testable offline (tests
  never download or load a model).
"""

import os


# ---------------------------------------------------------------------------
# Document formatting (deterministic representation for the reranker)
# ---------------------------------------------------------------------------


def format_candidate_document(candidate):
    """Build the deterministic (query, doc) document for one candidate.

    Accepts either the diagnostics dict shape (video_title/text keys) or
    any mapping with those keys. Missing pieces degrade to empty strings
    rather than raising — a chunk with no text is still rankable by its
    title alone.
    """
    if not isinstance(candidate, dict):
        raise TypeError(
            f"candidate must be a dict, got {type(candidate).__name__}"
        )
    title = str(candidate.get("video_title") or "").strip()
    text = str(candidate.get("text") or "").strip()
    return f"Title:\n{title}\n\nTranscript:\n{text}"


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def select_device(prefer_cuda=True):
    """'cuda' when torch+CUDA are available (and preferred), else 'cpu'.

    Never raises: any torch import/inspection failure degrades to CPU.
    """
    if not prefer_cuda:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# The reranker
# ---------------------------------------------------------------------------


class CrossReranker:
    """Cross-encoder scorer + reranker over an existing candidate pool.

    Also the production singleton type used by ask.retrieve_chunks
    (Phase 3C integration).

    Parameters
    ----------
    model_name : str
        HuggingFace model id (default BAAI/bge-reranker-base).
    device : str | None
        "cuda" / "cpu"; None auto-selects (CUDA when available).
    batch_size : int
        Pairs per forward pass.
    score_fn : callable | None
        Injected scoring function used instead of the model. Signature:
        score_fn(pairs: list[tuple[str, str]]) -> list[float]. For tests
        and for the deterministic fake ranking path.
    max_length : int
        Tokenizer max sequence length for the real model.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        device=None,
        batch_size=16,
        score_fn=None,
        max_length=512,
        prefer_cuda=True,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._score_fn = score_fn
        self.device = device or select_device(prefer_cuda)
        self._model = None  # loaded lazily, ONCE
        self._last_load_error = None

        # Validate the configuration eagerly so misconfiguration surfaces
        # at construction, not deep inside an experiment run.
        self._validate_config()

    def _validate_config(self):
        if self.device not in ("cuda", "cpu"):
            raise ValueError(
                f"device must be 'cuda' or 'cpu', got {self.device!r}"
            )
        if self._score_fn is not None and not callable(self._score_fn):
            raise TypeError("score_fn must be callable")

    # -- model lifecycle -------------------------------------------------

    def _ensure_model(self):
        """Load the real cross-encoder once; reuse afterwards.

        Returns True when the model is usable. On failure, records the
        error and returns False (the caller decides whether that is
        fatal for its context). Tests with an injected score_fn never
        reach this path.
        """
        if self._model is not None:
            return True
        if self._score_fn is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            return True
        except Exception as error:  # pragma: no cover - exercised live only
            self._last_load_error = error
            return False

    @property
    def last_load_error(self):
        return self._last_load_error

    # -- scoring ----------------------------------------------------------

    def score(self, query, candidates):
        """Score every candidate against the query. Higher = better.

        Returns a list of floats aligned with the input order. Batches
        the pairs; never mutates the candidates.
        """
        if not candidates:
            return []
        pairs = [
            (query, format_candidate_document(candidate))
            for candidate in candidates
        ]

        if self._score_fn is not None:
            return [float(value) for value in self._score_fn(pairs)]

        if not self._ensure_model():
            raise RuntimeError(
                f"Cross-encoder '{self.model_name}' unavailable on "
                f"device={self.device}: {self._last_load_error}"
            )

        scores = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            batch_scores = self._model.predict(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            scores.extend(float(value) for value in batch_scores)
        return scores

    # -- reranking ---------------------------------------------------------

    def rerank(self, query, candidates, top_k=None):
        """Return [(candidate, reranker_score)] sorted best-first.

        - Does NOT mutate the input list or the candidate dicts.
        - Ties are broken by the candidate's original position (stable,
          deterministic).
        - top_k=None keeps all candidates; otherwise the first top_k
          pairs are returned.
        - Duplicate candidates (equal dicts) are kept and score
          identically; they occupy separate positions in the output.
        """
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"top_k must be a positive int, got {top_k!r}")

        candidates = list(candidates)  # defensive copy of the list itself
        if not candidates:
            return []

        scores = self.score(query, candidates)
        pairs = list(zip(candidates, scores))
        # Stable sort: equal scores keep original candidate order.
        pairs.sort(key=lambda pair: pair[1], reverse=True)

        if top_k is not None:
            pairs = pairs[:top_k]
        return pairs


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

EXPERIMENT_DEPTHS = (10, 20, 30, 40)


def validate_experiment_config(depths=None, batch_size=16, device=None):
    """Validate the reranker experiment configuration; returns normalized.

    Raises ValueError on invalid depth/batch/device values so the runner
    fails before touching any data.
    """
    depths = tuple(depths) if depths is not None else EXPERIMENT_DEPTHS
    if not depths:
        raise ValueError("at least one rerank depth is required")
    for depth in depths:
        if not isinstance(depth, int) or depth < 1:
            raise ValueError(f"rerank depth must be a positive int: {depth!r}")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int: {batch_size!r}")
    if device is not None and device not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda', 'cpu' or None: {device!r}")
    return {
        "depths": depths,
        "batch_size": batch_size,
        "device": device or select_device(),
    }
