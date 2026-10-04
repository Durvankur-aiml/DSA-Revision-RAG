"""Offline tests for the Phase 3 reranker experiment.

The Cross-Encoder is ALWAYS mocked (injectable score_fn) — no test
downloads a model, touches Qdrant, or calls Gemini. The real model runs
only in the explicit experiment (Backend/eval/reranker_runner.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import reranker as reranker_mod  # noqa: E402
from reranker import (  # noqa: E402
    CrossReranker,
    format_candidate_document,
    select_device,
    validate_experiment_config,
)

# The experiment runner imports ask.py (heavy, offline-stubbed by
# conftest) and the eval package — import it here so the runner-variant
# tests can patch its namespace directly.
import importlib

experiment_runner = importlib.import_module("eval.reranker_runner")
reranker_runner = experiment_runner  # alias used by variant fixtures


# ---------------------------------------------------------------------------
# 1. Model initialization abstraction
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_score_fn_model_never_loaded(self):
        calls = []

        def fake_score(pairs):
            calls.append(len(pairs))
            return [0.0] * len(pairs)

        reranker = CrossReranker(score_fn=fake_score)
        reranker.score("q", [{"video_title": "t", "text": "x"}])
        assert calls == [1]
        assert reranker._model is None  # real model never touched

    def test_lazy_model_not_loaded_at_construction(self):
        reranker = CrossReranker(score_fn=lambda pairs: [0.0] * len(pairs))
        assert reranker._model is None

    def test_invalid_batch_size_rejected(self):
        with pytest.raises(ValueError):
            CrossReranker(batch_size=0)

    def test_invalid_device_rejected(self):
        with pytest.raises(ValueError):
            CrossReranker(device="tpu")

    def test_non_callable_score_fn_rejected(self):
        with pytest.raises(TypeError):
            CrossReranker(score_fn="not-callable")


# ---------------------------------------------------------------------------
# 2. Document formatting
# ---------------------------------------------------------------------------


class TestDocumentFormatting:
    def test_deterministic_shape(self):
        doc = format_candidate_document(
            {"video_title": "BS-1. Binary Search", "text": "lo op"}  # no comma
        )
        assert doc == "Title:\nBS-1. Binary Search\n\nTranscript:\nlo op"
        again = format_candidate_document(
            {"video_title": "BS-1. Binary Search", "text": "lo op"}
        )
        assert doc == again

    def test_leakage_fields_never_used(self):
        # Extra fields that could carry ground truth are simply ignored.
        doc = format_candidate_document(
            {
                "video_title": "T",
                "text": "X",
                "expected_video_id": "GOLDENID123",
                "is_relevant": True,
                "answer": "cheat",
            }
        )
        assert "GOLDENID123" not in doc
        assert "cheat" not in doc
        assert doc == "Title:\nT\n\nTranscript:\nX"

    def test_missing_fields_degrade_to_empty(self):
        doc = format_candidate_document({})
        assert doc == "Title:\n\n\nTranscript:\n"

    def test_non_dict_rejected(self):
        with pytest.raises(TypeError):
            format_candidate_document("not a dict")


# ---------------------------------------------------------------------------
# 3. Deterministic ordering + depth + edge cases
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_reranker():
    """Deterministic fake model: score = length of the transcript text."""
    return CrossReranker(
        score_fn=lambda pairs: [float(len(doc.split())) for _, doc in pairs]
    )


def _cands(*texts):
    return [{"video_title": f"T{i}", "text": t} for i, t in enumerate(texts)]


class TestRerankOrdering:
    def test_sorted_best_first(self, fake_reranker):
        cands = _cands("a b c", "a b c d e f", "a b")
        result = fake_reranker.rerank("q", cands)
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0][0]["text"] == "a b c d e f"

    def test_ties_keep_original_order(self, fake_reranker):
        cands = _cands("same length", "same length", "same length")
        result = fake_reranker.rerank("q", cands)
        assert [c["video_title"] for c, _ in result] == ["T0", "T1", "T2"]

    def test_top_n_depth(self, fake_reranker):
        cands = _cands(*[f"word{i} " * (i + 1) for i in range(20)])
        result = fake_reranker.rerank("q", cands, top_k=10)
        assert len(result) == 10
        # The kept ones must be the globally best.
        assert result[0][0]["text"].startswith("word19")

    def test_invalid_top_k_rejected(self, fake_reranker):
        with pytest.raises(ValueError):
            fake_reranker.rerank("q", _cands("x"), top_k=0)

    def test_empty_candidates(self, fake_reranker):
        assert fake_reranker.rerank("q", []) == []
        assert fake_reranker.score("q", []) == []

    def test_single_candidate(self, fake_reranker):
        result = fake_reranker.rerank("q", _cands("only one here"))
        assert len(result) == 1

    def test_duplicate_candidates_score_identically(self, fake_reranker):
        cands = _cands("same text", "same text", "different")
        result = fake_reranker.rerank("q", cands)
        scores = [score for _, score in result]
        assert scores[0] == scores[1]  # duplicates equal
        assert len(result) == 3        # and both kept

    def test_no_mutation_of_original_list(self, fake_reranker):
        cands = _cands("a b", "a b c")
        snapshot = [dict(c) for c in cands]
        fake_reranker.rerank("q", cands, top_k=1)
        assert cands == snapshot                      # list + dicts intact
        assert [c["video_title"] for c in cands] == ["T0", "T1"]

    def test_scores_align_with_input_order(self, fake_reranker):
        # The fake scores by total word count across title + transcript.
        cands = _cands("one two", "one two three four")
        scores = fake_reranker.score("q", cands)
        assert scores == [5.0, 7.0]  # input order, not ranked order


# ---------------------------------------------------------------------------
# 4. Device selection + CPU fallback
# ---------------------------------------------------------------------------


class TestDeviceSelection:
    def test_prefer_cuda_false_forces_cpu(self):
        assert select_device(prefer_cuda=False) == "cpu"

    def test_device_is_valid_string(self):
        assert select_device() in ("cuda", "cpu")

    def test_reranker_records_device(self):
        reranker = CrossReranker(score_fn=lambda p: [0.0], prefer_cuda=False)
        assert reranker.device == "cpu"

    def test_cuda_selection_when_available(self, monkeypatch):
        # Simulate a CUDA-capable torch without importing real GPU state.
        fake_torch = type(sys)("torch")
        fake_torch.cuda = type(sys)("cuda")
        fake_torch.cuda.is_available = lambda: True
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert select_device() == "cuda"

    def test_cpu_fallback_when_cuda_absent(self, monkeypatch):
        fake_torch = type(sys)("torch")
        fake_torch.cuda = type(sys)("cuda")
        fake_torch.cuda.is_available = lambda: False
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert select_device() == "cpu"

    def test_cpu_fallback_when_torch_broken(self, monkeypatch):
        class _Broken:
            def __getattr__(self, name):
                raise RuntimeError("torch import side effect")

        monkeypatch.setitem(sys.modules, "torch", _Broken())
        assert select_device() == "cpu"


# ---------------------------------------------------------------------------
# 5. Experiment configuration validation
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_default_depths(self):
        config = validate_experiment_config()
        assert config["depths"] == (10, 20, 30, 40)
        assert config["device"] in ("cuda", "cpu")

    def test_custom_depths_normalized(self):
        config = validate_experiment_config(depths=[5, 25])
        assert config["depths"] == (5, 25)

    def test_empty_depths_rejected(self):
        with pytest.raises(ValueError):
            validate_experiment_config(depths=[])

    def test_bad_depth_rejected(self):
        for bad in (0, -3, "ten", 2.5):
            with pytest.raises(ValueError):
                validate_experiment_config(depths=[bad])

    def test_bad_batch_size_rejected(self):
        with pytest.raises(ValueError):
            validate_experiment_config(batch_size=0)

    def test_bad_device_rejected(self):
        with pytest.raises(ValueError):
            validate_experiment_config(device="npu")


# ---------------------------------------------------------------------------
# 6. Runner variant logic (offline, fake retrieval + fake reranker)
# ---------------------------------------------------------------------------


class _Point:
    def __init__(self, video_id, index, title):
        self.id = f"{video_id}_{index}"
        self.score = 0.5
        self.payload = {
            "video_id": video_id,
            "chunk_index": index,
            "video_title": title,
            "start": 0.0,
            "end": 90.0,
            "text": f"transcript of {video_id} chunk {index}",
        }


class TestRunnerVariants:
    @pytest.fixture()
    def patched_ask(self, monkeypatch):
        """Deterministic fake production channels + selection."""
        evals = reranker_runner

        def fake_semantic(question):
            # The distractor always ranks first; the target is present but
            # second (so the reranker has something at a low rank to
            # recover). "binary" in the question lets the target's TITLE
            # earn hybrid points via the real title scorer in tests that
            # exercise fusion effects.
            return [
                _Point("OTHERVID111", 0, "Other Topic Intro"),
                _Point("TARGETVID1", 0, "Binary Search Intro Real"),
            ]

        def fake_lexical(question, all_points=None):
            return []

        monkeypatch.setattr(evals.ask, "semantic_retrieve", fake_semantic)
        monkeypatch.setattr(evals.ask, "lexical_retrieve", fake_lexical)
        monkeypatch.setattr(
            evals.ask, "TOP_K", 6, raising=False
        )
        return evals

    def _reranker_preferring(self, keyword):
        """Fake reranker scoring docs whose text contains `keyword` highest."""
        def score(pairs):
            out = []
            for _, doc in pairs:
                out.append(1.0 if keyword in doc.lower() else 0.0)
            return out

        return CrossReranker(score_fn=score)

    def test_baseline_variant(self, patched_ask):
        # The fake semantic channel ranks OTHERVID111 first and TARGETVID1
        # second; both pass the threshold gate, so the baseline selection
        # keeps BOTH (order preserved) — target at final rank 2.
        case = {
            "id": "v1",
            "question": "explain binary things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        record = patched_ask.run_variant(case, depth=None)
        assert record["hit"] is True
        assert record["final_results"][1]["video_id"] == "TARGETVID1"
        assert record["metrics"]["final"]["hit@1"] == 0
        assert record["metrics"]["final"]["hit@3"] == 1

    def test_reranker_recovers_later_candidate(self, patched_ask):
        # Question WITHOUT the magic word: semantic puts OTHER first and
        # the fake lexical is empty, so the target sits behind in the pool.
        # A reranker preferring the keyword promotes it.
        case = {
            "id": "v2",
            "question": "explain things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        reranker = self._reranker_preferring("targetvid1")
        record = patched_ask.run_variant(
            case, depth=10, reranker=reranker
        )
        # The reranker moved the relevant chunk ahead of the distractor;
        # production selection then keeps it at rank 1 (rank swap vs the
        # baseline variant's rank 2).
        assert record["final_results"][0]["video_id"] == "TARGETVID1"
        assert record["metrics"]["final"]["hit@1"] == 1
        assert record["hit"] is True
        # Reranker score is propagated into the final results.
        assert record["final_results"][0]["reranker_score"] == 1.0

    def test_depth_requires_reranker(self, patched_ask):
        case = {
            "id": "v3",
            "question": "explain things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        with pytest.raises(ValueError):
            patched_ask.run_variant(case, depth=10, reranker=None)

    def test_candidate_composition_invariant_across_variants(
        self, patched_ask
    ):
        """Pool recall must be identical with and without reranking."""
        case = {
            "id": "v4",
            "question": "explain binary things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        baseline = patched_ask.run_variant(case, depth=None)
        reranked = patched_ask.run_variant(
            case, depth=20, reranker=self._reranker_preferring("nothing")
        )
        assert (
            baseline["metrics"]["candidate_recall@20"]
            == reranked["metrics"]["candidate_recall@20"]
        )

    def test_aggregate_shapes(self, patched_ask):
        case = {
            "id": "v5",
            "question": "explain binary things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        experiment = patched_ask.run_experiment([case], depth=None)
        agg = experiment["aggregate"]
        assert agg["n"] == 1
        assert agg["hit"]["hit@6"]["mean"] == 1.0
        assert "retrieval_p50" in agg["latency_seconds"]
        assert "rerank_p50" in agg["latency_seconds"]
        assert "total_p50" in agg["latency_seconds"]

    def test_deterministic_variants(self, patched_ask):
        case = {
            "id": "v6",
            "question": "explain binary things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        reranker = self._reranker_preferring("targetvid1")
        runs = [
            patched_ask.run_variant(case, depth=10, reranker=reranker)
            for _ in range(2)
        ]
        for run in runs:
            run.pop("retrieval_seconds", None)
            run.pop("rerank_seconds", None)
        assert runs[0] == runs[1]

    def test_pool_point_ids_captured_pre_rerank(self, patched_ask):
        """pool_point_ids is captured BEFORE reranking, so it must be
        identical between the baseline and any rerank-depth variant —
        the Phase 3B invariance contract at record level."""
        case = {
            "id": "v7",
            "question": "explain binary things",
            "relevant_video_ids": ["TARGETVID1"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        baseline = patched_ask.run_variant(case, depth=None)
        reranked = patched_ask.run_variant(
            case,
            depth=10,
            reranker=self._reranker_preferring("targetvid1"),
        )
        assert baseline["pool_point_ids"] == reranked["pool_point_ids"]
        # And it is a non-empty identity list for this fixture.
        assert len(baseline["pool_point_ids"]) == 2


class TestPoolInvariance:
    def _experiment(self, records, depth):
        return {"depth": depth, "records": records, "aggregate": {}}

    def _record(self, eval_id, point_ids):
        return {
            "eval_id": eval_id,
            "pool_point_ids": point_ids,
            "ground_truth": {"video_ids": ["X"], "chunk_ids": []},
            "final_results": [],
            "metrics": {"level": "video"},
            "hit": True,
        }

    def test_pass_when_identical(self):
        baseline = self._experiment(
            [self._record("a", ["p1", "p2"])], None
        )
        d60 = self._experiment([self._record("a", ["p1", "p2"])], 60)
        d70 = self._experiment([self._record("a", ["p1", "p2"])], 70)
        ok, violations = experiment_runner.verify_pool_invariance(
            [baseline, d60, d70]
        )
        assert ok and violations == []

    def test_fail_on_composition_change(self):
        baseline = self._experiment(
            [self._record("a", ["p1", "p2"])], None
        )
        d60 = self._experiment([self._record("a", ["p1", "p2", "p3"])], 60)
        ok, violations = experiment_runner.verify_pool_invariance(
            [baseline, d60]
        )
        assert not ok
        assert violations[0]["eval_id"] == "a"
        assert violations[0]["experiment_depth"] == 60

    def test_no_baseline_is_failure(self):
        d60 = self._experiment([self._record("a", ["p1"])], 60)
        ok, violations = experiment_runner.verify_pool_invariance([d60])
        assert not ok


class TestFailureMatrix:
    def _record(self, eval_id, final_results, level="video"):
        return {
            "eval_id": eval_id,
            "question": "q",
            "ground_truth": {
                "video_ids": ["TARGET1"],
                "chunk_ids": [],
            },
            "final_results": final_results,
            "metrics": {"level": level},
            "hit": any(
                entry["video_id"] == "TARGET1" for entry in final_results
            ),
            "best_pool_rank": 63,
        }

    def test_matrix_rows_and_depth_columns(self):
        baseline = {
            "depth": None,
            "aggregate": {},
            "records": [
                self._record(
                    "e1",
                    [
                        {"rank": 1, "video_id": "OTHER9", "reranker_score": None},
                    ],
                )
            ],
        }
        d60 = {
            "depth": 60,
            "aggregate": {},
            "records": [
                self._record(
                    "e1",
                    [
                        {"rank": 2, "video_id": "TARGET1", "reranker_score": 0.9},
                        {"rank": 1, "video_id": "OTHER9", "reranker_score": 0.95},
                    ],
                )
            ],
        }
        d70 = {
            "depth": 70,
            "aggregate": {},
            "records": [
                self._record(
                    "e1",
                    [
                        {"rank": 1, "video_id": "TARGET1", "reranker_score": 0.9},
                    ],
                )
            ],
        }
        matrix = experiment_runner.failure_matrix(baseline, [d60, d70])
        assert len(matrix) == 1  # only baseline failures listed
        row = matrix[0]
        assert row["eval_id"] == "e1"
        assert row["baseline"]["hit"] is False
        # Best (lowest) relevant rank wins per depth.
        assert row["depths"]["60"] == {"rank": 2, "hit": True, "score": 0.9}
        assert row["depths"]["70"]["rank"] == 1

    def test_explicit_eval_ids_respected(self):
        baseline = {
            "depth": None,
            "aggregate": {},
            "records": [
                self._record("e1", [{"rank": 1, "video_id": "TARGET1"}])
            ],
        }
        matrix = experiment_runner.failure_matrix(baseline, [], eval_ids=["e1"])
        assert [row["eval_id"] for row in matrix] == ["e1"]
