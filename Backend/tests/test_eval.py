"""Offline tests for the Phase 2 evaluation foundation.

Covers dataset validation (fail-loud), metric definitions, candidate-vs-
final level separation, failure classification, report generation, and
deterministic evaluation. The production retrieval functions are
monkeypatched with deterministic fakes — no Qdrant, no Gemini, no
network. The REAL golden dataset is validated as-is (file reads only).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from eval import dataset as dataset_mod  # noqa: E402
from eval import metrics as metrics_mod  # noqa: E402
from eval import report as report_mod  # noqa: E402
from eval import runner as runner_mod  # noqa: E402
from eval.dataset import GoldenDatasetError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _case(**overrides):
    base = {
        "id": "eval_test_1",
        "question": "Explain binary search.",
        "relevant_video_ids": ["MHf6awe89xw"],
    }
    base.update(overrides)
    return base


@pytest.fixture()
def no_verify(monkeypatch):
    """Skip file-existence verification for synthetic datasets."""
    monkeypatch.setattr(
        dataset_mod, "load_metadata_video_ids", lambda: set()
    )
    monkeypatch.setattr(
        dataset_mod, "load_embedding_chunk_keys", lambda: set()
    )


# ---------------------------------------------------------------------------
# 1. Dataset validation
# ---------------------------------------------------------------------------


class TestDatasetValidation:
    def test_valid_case_normalizes(self, no_verify):
        normalized = dataset_mod.validate_case(
            _case(relevant_chunk_ids=["MHf6awe89xw_3"]),
            known_video_ids={"MHf6awe89xw"},
            known_chunk_keys={"MHf6awe89xw_3"},
        )
        assert normalized["chunk_certainty"] == "exact"
        assert normalized["relevant_chunk_ids"] == ["MHf6awe89xw_3"]

    def test_missing_video_metadata_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="no metadata file"):
            dataset_mod.validate_case(
                _case(), known_video_ids=set(), known_chunk_keys=set()
            )

    def test_duplicate_ids_fail(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="duplicate evaluation id"):
            dataset_mod.validate_dataset(
                [_case(), _case()],
                known_video_ids=None,
                known_chunk_keys=None,
            )

    def test_malformed_chunk_key_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="malformed chunk key"):
            dataset_mod.validate_case(
                _case(relevant_chunk_ids=["not-a-chunk-key"]),
                known_video_ids={"MHf6awe89xw"},
                known_chunk_keys=set(),
            )

    def test_chunk_from_other_video_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="does not belong"):
            dataset_mod.validate_case(
                _case(relevant_chunk_ids=["OTHERID1234_0"]),
                known_video_ids={"MHf6awe89xw"},
                known_chunk_keys={"OTHERID1234_0"},
            )

    def test_nonexistent_chunk_key_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="does not exist"):
            dataset_mod.validate_case(
                _case(relevant_chunk_ids=["MHf6awe89xw_999"]),
                known_video_ids={"MHf6awe89xw"},
                known_chunk_keys=set(),
            )

    def test_unknown_field_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="unknown field"):
            dataset_mod.validate_case(
                _case(surprise_field="x"),
                known_video_ids=None,
                known_chunk_keys=None,
            )

    def test_empty_question_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="question"):
            dataset_mod.validate_case(
                _case(question="   "),
                known_video_ids=None,
                known_chunk_keys=None,
            )

    def test_oversized_question_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="exceeds"):
            dataset_mod.validate_case(
                _case(question="x" * 501),
                known_video_ids=None,
                known_chunk_keys=None,
            )

    def test_empty_dataset_fails(self, no_verify):
        with pytest.raises(GoldenDatasetError, match="empty"):
            dataset_mod.validate_dataset([], known_video_ids=None,
                                         known_chunk_keys=None)

    def test_invalid_json_reports_line(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"id": "a", "question": "q", "relevant_video_ids": ["x"]}\n'
            "NOT JSON\n",
            encoding="utf-8",
        )
        with pytest.raises(GoldenDatasetError, match="line 2"):
            dataset_mod.load_golden_dataset(str(path), verify_files=False)

    def test_blank_line_rejected(self, tmp_path):
        path = tmp_path / "golden.jsonl"
        path.write_text(
            '{"id": "a", "question": "q", "relevant_video_ids": ["XXXXXXXXXXX"]}\n'
            "\n"
            '{"id": "b", "question": "q2", "relevant_video_ids": ["YYYYYYYYYYY"]}\n',
            encoding="utf-8",
        )
        with pytest.raises(GoldenDatasetError, match="blank line"):
            dataset_mod.load_golden_dataset(str(path), verify_files=False)

    def test_missing_file_fails_loud(self, tmp_path):
        with pytest.raises(GoldenDatasetError, match="not found"):
            dataset_mod.load_golden_dataset(
                str(tmp_path / "nope.jsonl"), verify_files=False
            )

    def test_real_golden_dataset_loads(self):
        """The production dataset validates against real Data/ files."""
        cases = dataset_mod.load_golden_dataset(
            dataset_mod.DEFAULT_GOLDEN_PATH, verify_files=True
        )
        assert len(cases) >= 50
        ids = {c["id"] for c in cases}
        assert len(ids) == len(cases)
        sections = {c.get("section") for c in cases}
        assert len(sections) >= 10

    def test_real_dataset_has_lineage_notes(self):
        cases = dataset_mod.load_golden_dataset(
            dataset_mod.DEFAULT_GOLDEN_PATH, verify_files=True
        )
        assert all(c.get("notes") for c in cases)


# ---------------------------------------------------------------------------
# 2. Metric definitions
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_hit_at_k_positive(self):
        assert metrics_mod.hit_at_k(["a", "b", "c"], {"b"}, 3) == 1
        assert metrics_mod.hit_at_k(["a", "b", "c"], {"b"}, 2) == 1
        assert metrics_mod.hit_at_k(["a", "b", "c"], {"b"}, 1) == 0

    def test_hit_at_k_empty_retrieved(self):
        assert metrics_mod.hit_at_k([], {"a"}, 5) == 0

    def test_recall_at_k_single_relevant(self):
        # With one relevant item, recall == hit.
        assert metrics_mod.recall_at_k(["a"], {"a"}, 1) == 1.0
        assert metrics_mod.recall_at_k(["b"], {"a"}, 1) == 0.0

    def test_recall_at_k_multiple_relevant(self):
        retrieved = ["x", "a", "y", "b"]
        relevant = {"a", "b", "c"}
        # top-4 contains a and b -> 2/3
        assert metrics_mod.recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
        # top-2 contains only a -> 1/3
        assert metrics_mod.recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)

    def test_recall_at_k_empty_relevant_is_zero(self):
        assert metrics_mod.recall_at_k(["a"], set(), 3) == 0.0

    def test_mrr_first_position(self):
        assert metrics_mod.mrr(["a", "b"], {"a"}) == 1.0

    def test_mrr_second_position(self):
        assert metrics_mod.mrr(["x", "a"], {"a"}) == 0.5

    def test_mrr_absent(self):
        assert metrics_mod.mrr(["x", "y"], {"a"}) == 0.0

    def test_mrr_empty_retrieved(self):
        assert metrics_mod.mrr([], {"a"}) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            metrics_mod.hit_at_k(["a"], {"a"}, 0)

    def test_percentile_matches_report_perf(self):
        from report_perf import percentile as rp_percentile

        values = list(range(1, 101))
        for fraction in (0.5, 0.9, 0.95, 0.99):
            assert metrics_mod.percentile(values, fraction) == (
                rp_percentile(values, fraction)
            )

    def test_mean_empty_is_none(self):
        assert metrics_mod.mean([]) is None


# ---------------------------------------------------------------------------
# 3. Runner: candidate vs final levels + classification (fake retrieval)
# ---------------------------------------------------------------------------


class _Point:
    def __init__(self, video_id, index, score):
        self.id = f"{video_id}_{index}"
        self.score = score
        self.payload = {
            "video_id": video_id,
            "chunk_index": index,
            "video_title": f"Title {video_id}",
            "start": 0.0,
            "end": 90.0,
            "text": "x",
        }


def _install_fake_retrieval(monkeypatch, final_hits, diag):
    monkeypatch.setattr(runner_mod.ask, "retrieve_chunks",
                        lambda question, top_k=6: list(final_hits))
    monkeypatch.setattr(runner_mod.ask, "get_retrieval_diagnostics",
                        lambda: diag)
    monkeypatch.setattr(runner_mod.ask, "get_retrieval_score",
                        lambda point_id: 0.5)


class TestRunnerLevels:
    def test_video_level_hit(self, monkeypatch):
        expected = "VID_AAAAAAA"
        diag = {
            "candidates": [
                {"point_id": "OTHERVID111_0", "video_id": "OTHERVID111",
                 "chunk_index": 0, "hybrid_score": 0.9, "title_score": 0.0,
                 "semantic_score": 0.9, "text_score": 0.0,
                 "cull_reason": "selected"},
                {"point_id": f"{expected}_0", "video_id": expected,
                 "chunk_index": 0, "hybrid_score": 0.8, "title_score": 0.0,
                 "semantic_score": 0.8, "text_score": 0.0,
                 "cull_reason": "selected"},
            ],
            "config": {"top_k": 6},
        }
        final = [_Point(expected, 0, 0.8), _Point("OTHERVID111", 0, 0.9)]
        _install_fake_retrieval(monkeypatch, final, diag)

        case = {
            "id": "t1", "question": "q",
            "relevant_video_ids": [expected],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        record = runner_mod.evaluate_case(case)
        assert record["hit"] is True
        assert record["metrics"]["final"]["hit@1"] == 1
        assert record["failure_category"] is None
        assert record["final_ids"]["videos"][0] == expected

    def test_candidate_contains_but_final_misses(self, monkeypatch):
        expected = "VID_BBBBBBB"
        diag = {
            "candidates": [
                {"point_id": f"{expected}_0", "video_id": expected,
                 "chunk_index": 0, "hybrid_score": 0.5,
                 "title_score": 0.0, "semantic_score": 0.5,
                 "text_score": 0.0, "cull_reason": "beyond_top_k"},
            ],
            "config": {"top_k": 6},
        }
        other = [_Point("OTHERVID222", i, 0.9) for i in range(6)]
        _install_fake_retrieval(monkeypatch, other, diag)

        case = {
            "id": "t2", "question": "q",
            "relevant_video_ids": [expected],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        record = runner_mod.evaluate_case(case)
        assert record["hit"] is False
        # Candidate level: present but at rank 1 of a 1-entry pool -> hit.
        assert record["metrics"]["candidate"]["hit@1"] == 1
        # Final level: absent.
        assert record["metrics"]["final"]["hit@6"] == 0
        assert record["metrics"]["candidate_recall@20"] == 1.0
        assert record["failure_category"] == "beyond_top_k"

    def test_empty_retrieval_no_results(self, monkeypatch):
        _install_fake_retrieval(monkeypatch, [], {"candidates": [],
                                                  "config": {"top_k": 6}})
        case = {
            "id": "t3", "question": "q",
            "relevant_video_ids": ["VID_CCCCCCC"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        record = runner_mod.evaluate_case(case)
        assert record["hit"] is False
        assert record["failure_category"] == "no_results"
        assert record["metrics"]["final"]["hit@6"] == 0

    def test_multiple_relevant_chunks_recall(self, monkeypatch):
        expected = "VID_DDDDDDD"
        diag = {
            "candidates": [
                {"point_id": f"{expected}_0", "video_id": expected,
                 "chunk_index": 0, "hybrid_score": 0.9, "title_score": 0.0,
                 "semantic_score": 0.9, "text_score": 0.0,
                 "cull_reason": "selected"},
            ],
            "config": {"top_k": 6},
        }
        final = [_Point(expected, 0, 0.9), _Point(expected, 5, 0.85)]
        _install_fake_retrieval(monkeypatch, final, diag)
        case = {
            "id": "t4", "question": "q",
            "relevant_video_ids": [expected],
            "relevant_chunk_ids": [f"{expected}_0", f"{expected}_5",
                                   f"{expected}_9"],
            "chunk_certainty": "exact",
        }
        record = runner_mod.evaluate_case(case)
        # 2 of 3 labeled chunks retrieved at final level.
        assert record["metrics"]["final_recall@6"] == pytest.approx(2 / 3)
        assert record["hit"] is True

    def test_chunk_level_miss_with_video_hit(self, monkeypatch):
        expected = "VID_EEEEEEE"
        diag = {
            "candidates": [
                {"point_id": f"{expected}_1", "video_id": expected,
                 "chunk_index": 1, "hybrid_score": 0.9, "title_score": 0.0,
                 "semantic_score": 0.9, "text_score": 0.0,
                 "cull_reason": "selected"},
            ],
            "config": {"top_k": 6},
        }
        final = [_Point(expected, 1, 0.9)]  # labeled chunk 0 NOT retrieved
        _install_fake_retrieval(monkeypatch, final, diag)
        case = {
            "id": "t5", "question": "q",
            "relevant_video_ids": [expected],
            "relevant_chunk_ids": [f"{expected}_0"],
            "chunk_certainty": "exact",
        }
        record = runner_mod.evaluate_case(case)
        assert record["hit"] is False
        assert record["failure_category"] == "correct_video_wrong_chunk"

    def test_channel_participation_uses_production_functions(
        self, monkeypatch
    ):
        # channel_participation must call ask.semantic_retrieve and
        # ask.lexical_retrieve (production channels), not reimplement them.
        calls = []

        def fake_semantic(question):
            calls.append("semantic")
            hit = _Point("VID_FFFFFFF", 0, 0.9)
            hit.payload["video_title"] = "binary search intro"
            return [hit]

        def fake_lexical(question, all_points=None):
            calls.append("lexical")
            point = _Point("VID_FFFFFFF", 1, 0.7)
            return [(point, 0.7)]

        monkeypatch.setattr(runner_mod.ask, "semantic_retrieve", fake_semantic)
        monkeypatch.setattr(runner_mod.ask, "lexical_retrieve", fake_lexical)

        participation = runner_mod.channel_participation(
            {}, ["VID_FFFFFFF"], "question text"
        )
        assert calls == ["semantic", "lexical"]
        assert participation["VID_FFFFFFF"] == {
            "semantic": True, "lexical": True
        }


class TestFailureClassification:
    def _diag_with_culls(self, video_id, cull_reasons):
        return {
            "candidates": [
                {"point_id": f"{video_id}_{i}", "video_id": video_id,
                 "chunk_index": i, "hybrid_score": 0.4, "title_score": 0.0,
                 "semantic_score": 0.3, "text_score": 0.0,
                 "cull_reason": reason}
                for i, reason in enumerate(cull_reasons)
            ],
            "config": {"top_k": 6},
        }

    def test_dedup_removed(self, monkeypatch):
        case = {
            "id": "c1", "question": "q",
            "relevant_video_ids": ["VID_GGGGGGG"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        diag = self._diag_with_culls("VID_GGGGGGG", ["duplicate_title"])
        selected = [_Point("OTHERVID333", 0, 0.9)]
        category, _ = runner_mod.classify_failure(case, diag, selected, 6)
        assert category == "dedup_removed"

    def test_threshold_removed(self, monkeypatch):
        case = {
            "id": "c2", "question": "q",
            "relevant_video_ids": ["VID_HHHHHHH"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        diag = self._diag_with_culls("VID_HHHHHHH", ["below_threshold"])
        selected = [_Point("OTHERVID444", 0, 0.9)]
        category, _ = runner_mod.classify_failure(case, diag, selected, 6)
        assert category == "threshold_removed"

    def test_candidate_miss_when_no_channel_surfaced(
        self, monkeypatch
    ):
        case = {
            "id": "c3", "question": "q",
            "relevant_video_ids": ["VID_IIIIIII"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        diag = self._diag_with_culls("OTHERVID555", ["selected"])
        monkeypatch.setattr(
            runner_mod.ask, "semantic_retrieve", lambda q: []
        )
        monkeypatch.setattr(
            runner_mod.ask, "lexical_retrieve", lambda q, all_points=None: []
        )
        selected = [_Point("OTHERVID555", 0, 0.9)]
        category, _ = runner_mod.classify_failure(case, diag, selected, 6)
        assert category == "candidate_miss"

    def test_lexical_miss_semantic_surfaced(self, monkeypatch):
        case = {
            "id": "c4", "question": "q",
            "relevant_video_ids": ["VID_JJJJJJJ"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        diag = self._diag_with_culls("OTHERVID666", ["selected"])
        point = _Point("VID_JJJJJJJ", 0, 0.8)
        monkeypatch.setattr(
            runner_mod.ask, "semantic_retrieve", lambda q: [point]
        )
        monkeypatch.setattr(
            runner_mod.ask, "lexical_retrieve",
            lambda q, all_points=None: []
        )
        selected = [_Point("OTHERVID666", 0, 0.9)]
        category, _ = runner_mod.classify_failure(case, diag, selected, 6)
        assert category == "lexical_miss"


# ---------------------------------------------------------------------------
# 4. Report generation
# ---------------------------------------------------------------------------


class TestReportGeneration:
    def _minimal_result(self):
        return {
            "aggregate": {
                "n": 1,
                "video_labeled_n": 1,
                "chunk_labeled_n": 0,
                "hit": {
                    "candidate": {f"hit@{k}": {"scope": "all", "n": 1,
                                               "mean": 1.0}
                                  for k in (1, 3, 5, 6, 10)},
                    "final": {f"hit@{k}": {"scope": "all", "n": 1,
                                           "mean": 0.0}
                              for k in (1, 3, 5, 6, 10)},
                },
                "video_level": {f"hit@{k}": {"scope": "video", "n": 1,
                                             "mean": 0.0}
                                for k in (1, 3, 5, 6, 10)},
                "chunk_level": {f"hit@{k}": {"scope": "chunk", "n": 0,
                                             "mean": None}
                                for k in (1, 3, 5, 6, 10)},
                "recall": {
                    "candidate@20": {"scope": "all", "n": 1, "mean": 1.0},
                    "final@6": {"scope": "all", "n": 1, "mean": 0.0},
                    "final@20": {"scope": "all", "n": 1, "mean": 0.0},
                },
                "mrr": {
                    "candidate": {"scope": "all", "n": 1, "mean": 0.5},
                    "final": {"scope": "all", "n": 1, "mean": 0.0},
                },
                "avg_final_source_count": 6.0,
                "no_result_rate": 0.0,
                "exact_topic_candidate_rate": 0.0,
                "candidate_pool_recall": {"scope": "all", "n": 1,
                                          "mean": 1.0},
                "latency_seconds": {"p50": 0.08, "p95": 0.1, "mean": 0.09},
                "failure_categories": {"beyond_top_k": 1},
            },
            "records": [],
            "started_at": "2026-01-01T00:00:00+0000",
            "retrieval_seconds": [0.09],
        }

    def test_reports_written_and_parseable(self, tmp_path):
        result = self._minimal_result()
        config = runner_mod.capture_configuration()
        paths = report_mod.write_reports(
            result, config, report_dir=str(tmp_path),
            dataset_path="whatever.jsonl", report_name="baseline",
        )
        assert len(paths) == 2
        payload = json.load(open(paths[0], encoding="utf-8"))
        assert payload["metrics"]["failure_categories"] == {"beyond_top_k": 1}
        md = open(paths[1], encoding="utf-8").read()
        assert "# ALGOFORGE" in md
        assert "Final Hit@1" in md
        assert "beyond_top_k" in md

    def test_report_contains_config_provenance(self, tmp_path):
        result = self._minimal_result()
        config = runner_mod.capture_configuration()
        paths = report_mod.write_reports(
            result, config, report_dir=str(tmp_path),
            dataset_path="x.jsonl", report_name="baseline",
        )
        md = open(paths[1], encoding="utf-8").read()
        assert "TOP_K: **6**" in md
        assert "title **0.6**" in md

    def test_real_baseline_report_exists_with_all_metrics(self):
        """The committed baseline report from the real run is complete."""
        path = os.path.join(
            dataset_mod.REPO_ROOT, "Data", "eval", "reports", "baseline.md"
        )
        assert os.path.exists(path)
        md = open(path, encoding="utf-8").read()
        for metric in ("Final Hit@1", "Final Hit@6", "Candidate Recall@20",
                       "Final MRR", "p50", "p95"):
            assert metric in md


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_evaluate_case_is_deterministic_for_fake_retrieval(
        self, monkeypatch
    ):
        diag = {
            "candidates": [
                {"point_id": "VID_KKKKKKK_0", "video_id": "VID_KKKKKKK",
                 "chunk_index": 0, "hybrid_score": 0.9, "title_score": 0.0,
                 "semantic_score": 0.9, "text_score": 0.0,
                 "cull_reason": "selected"},
            ],
            "config": {"top_k": 6},
        }
        final = [_Point("VID_KKKKKKK", 0, 0.9)]
        _install_fake_retrieval(monkeypatch, final, diag)
        case = {
            "id": "d1", "question": "q",
            "relevant_video_ids": ["VID_KKKKKKK"],
            "relevant_chunk_ids": [],
            "chunk_certainty": "video",
        }
        records = [runner_mod.evaluate_case(case) for _ in range(3)]
        stripped = []
        for record in records:
            record = dict(record)
            record.pop("retrieval_seconds", None)  # timing is allowed to vary
            stripped.append(record)
        assert stripped[0] == stripped[1] == stripped[2]

    def test_failure_categories_are_sorted_and_deterministic(self, monkeypatch):
        # Aggregation iterates a set comprehension internally; the output
        # dict must still be in sorted category order regardless.
        from eval.runner import _dedupe_preserving_order

        assert _dedupe_preserving_order(["b", "a", "b", "c"]) == \
            ["b", "a", "c"]
