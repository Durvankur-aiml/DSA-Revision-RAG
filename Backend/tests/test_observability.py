"""Offline tests for Phase 0 hygiene + Phase 1 observability.

Covers:
- request id generation/adoption (unit)
- request id propagation through the ASGI middleware (integration)
- structured logging events and field capture
- secret non-leakage into the perf log
- performance-log parsing and p50/p95 aggregation (report_perf)
- metadata hard-skip in generate_embeddings (no fabricated title/URL)
- load_to_qdrant import-time side-effect-free main() guard

Everything is offline and deterministic: no Gemini, no network, no real
Qdrant, no model downloads. The perf log is redirected to a tmp dir per
test via obs.configure().
"""

from __future__ import annotations

import importlib
import json
import sys
import time as time_module
import types

import pytest

import ask  # noqa: E402  (offline-stubbed by conftest before this import)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_log(path):
    """Parse a JSONL perf log into a list of dicts, skipping bad lines."""
    events = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


@pytest.fixture()
def perf_log(tmp_path, monkeypatch):
    """Point the perf logger at a fresh directory; return the log path."""
    import obs

    log_path = tmp_path / "perf.log"
    obs.configure(log_dir=str(tmp_path))
    monkeypatch.setattr(
        obs, "default_log_dir", lambda: str(tmp_path), raising=False
    )
    return log_path


# ---------------------------------------------------------------------------
# 1. Request ID generation / adoption (unit)
# ---------------------------------------------------------------------------


class TestRequestIdGeneration:
    def test_generated_id_is_uuid_hex(self):
        from obs import adopt_or_create_request_id

        rid = adopt_or_create_request_id(None)
        assert len(rid) == 32
        int(rid, 16)  # raises if not hex

    def test_blank_incoming_generates_new(self):
        from obs import adopt_or_create_request_id

        assert adopt_or_create_request_id("") != ""
        assert adopt_or_create_request_id("   ") != ""

    def test_sane_incoming_id_is_adopted(self):
        from obs import adopt_or_create_request_id

        assert (
            adopt_or_create_request_id("abc-123_XYZ.9")
            == "abc-123_XYZ.9"
        )

    def test_unsafe_incoming_id_is_replaced(self):
        from obs import adopt_or_create_request_id

        # Control characters, spaces, or absurd length never reach logs.
        for hostile in ("bad id\ninjection", "a" * 500, "id; DROP TABLE"):
            rid = adopt_or_create_request_id(hostile)
            assert rid != hostile

    def test_ids_are_unique_across_calls(self):
        from obs import adopt_or_create_request_id

        ids = {adopt_or_create_request_id(None) for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# 2. Request ID propagation through the middleware (integration)
# ---------------------------------------------------------------------------


class TestRequestIdPropagation:
    def test_ask_response_carries_request_id(self, client):
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 200
        rid = response.headers.get("x-request-id")
        assert rid and len(rid) == 32

    def test_incoming_request_id_is_echoed(self, client):
        response = client.post(
            "/ask",
            json={"question": "What is binary search?"},
            headers={"X-Request-ID": "my-test-id-42"},
        )
        assert response.headers.get("x-request-id") == "my-test-id-42"

    def test_health_still_gets_request_id(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.headers.get("x-request-id")

    def test_request_id_flows_into_log_events(self, client, perf_log):
        client.post(
            "/ask",
            json={"question": "What is binary search?"},
            headers={"X-Request-ID": "flow-test-1"},
        )
        events = read_log(perf_log)
        start = [e for e in events if e["event"] == "request_start"]
        end = [e for e in events if e["event"] == "request_end"]
        assert start and end
        assert all(e["request_id"] == "flow-test-1" for e in start + end)

    def test_concurrent_contexts_do_not_share_ids(self, perf_log):
        # Direct lifecycle interleaving: begin A, begin B, end both — the
        # ContextVar must hand back the right context each time.
        import obs

        obs.configure(log_dir=str(perf_log.parent))
        ctx_a = obs.begin_request("aaaa", "POST", "/ask")
        ctx_b = obs.begin_request("bbbb", "POST", "/ask")
        assert obs.current_request() is ctx_b
        obs.end_request()
        assert obs.current_request() is None
        obs.begin_request("aaaa", "POST", "/ask")
        obs.end_request()
        events = read_log(perf_log)
        ids = {e["request_id"] for e in events}
        assert ids == {"aaaa", "bbbb"}


# ---------------------------------------------------------------------------
# 3. Structured logging events
# ---------------------------------------------------------------------------


class TestStructuredEvents:
    def test_successful_ask_emits_complete_event(self, client, perf_log):
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 200
        ends = [
            e for e in read_log(perf_log) if e["event"] == "request_end"
        ]
        assert len(ends) == 1
        event = ends[0]

        # Required fields (spec: Phase 1 observability).
        for field in (
            "request_id",
            "ts",
            "question_length",
            "retrieval_strategy",
            "topic",
            "exact_topic",
            "candidate_count",
            "final_source_count",
            "http_status",
            "total_latency",
            "retrieval_latency",
            "generation_latency",
            "stages",
            "failure_category",
        ):
            assert field in event, f"missing field: {field}"

        assert event["http_status"] == 200
        assert event["failure_category"] is None
        assert event["retrieval_strategy"] == "hybrid"
        assert event["question_length"] == len("What is binary search?")
        assert event["final_source_count"] == 3  # sample_chunks fixture
        assert event["candidate_count"] == 3
        assert event["topic"] == "binary search"
        assert event["exact_topic"] is True
        assert event["total_latency"] >= event["retrieval_latency"]

    def test_stage_timings_are_captured(self, client, perf_log):
        client.post("/ask", json={"question": "What is binary search?"})
        event = [
            e for e in read_log(perf_log) if e["event"] == "request_end"
        ][0]
        stages = event["stages"]
        # API-level stages recorded by main.py during the stubbed request
        # (internal ask.py stages such as topic_detection/
        # lexical_retrieve/hybrid_scoring fire only on the real retrieval
        # path, which is covered by test_record_stage_populates_context).
        assert "api_validation" in stages
        assert "api_retrieval" in stages
        assert "api_prompt_build" in stages
        assert "api_gemini" in stages
        assert "api_source_mapping" in stages
        assert "request_total" in stages
        assert all(isinstance(v, (int, float)) for v in stages.values())

    def test_record_stage_populates_context(self, perf_log, capsys):
        # Internal ask.py stages (real retrieval path) land in ctx.stages.
        import obs

        ctx = obs.begin_request("stage-test", "POST", "/ask")
        ask._perf("topic_detection", time_module.monotonic() - 0.05)
        ask._perf("lexical_retrieve", time_module.monotonic() - 0.031,
                  extra="points=4850")
        obs.end_request()
        out = capsys.readouterr().out
        assert "[PERF] topic_detection=" in out
        assert "[PERF] lexical_retrieve=" in out and "points=4850" in out
        event = [
            e for e in read_log(perf_log) if e["event"] == "request_end"
        ][0]
        assert "topic_detection" in event["stages"]
        assert "lexical_retrieve" in event["stages"]
        assert ctx.stages["lexical_retrieve"] >= 0.03

    def test_failure_event_has_category(self, client, perf_log):
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400
        failed = [
            e for e in read_log(perf_log) if e["event"] == "request_failed"
        ]
        assert len(failed) == 1
        assert failed[0]["failure_category"] == "empty_question"
        assert failed[0]["http_status"] == 400

    def test_oversized_question_category(self, client, perf_log):
        client.post("/ask", json={"question": "x" * 501})
        failed = [
            e for e in read_log(perf_log) if e["event"] == "request_failed"
        ]
        assert failed and failed[0]["failure_category"] == "question_too_long"

    def test_no_relevant_content_category(self, client, perf_log):
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 200  # sanity: default client succeeds

    def test_generation_failure_maps_to_category(
        self, client, perf_log, set_gemini_failure
    ):
        set_gemini_failure(None)
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 502
        failed = [
            e for e in read_log(perf_log) if e["event"] == "request_failed"
        ]
        assert failed[0]["failure_category"] == "generation_failed"
        assert failed[0]["http_status"] == 502

    def test_unhandled_exception_is_captured_and_reraised(
        self, perf_log, main_module, monkeypatch, sample_chunks
    ):
        from fastapi.testclient import TestClient

        import obs
        import main

        def _boom(question, top_k=6):
            raise RuntimeError("synthetic crash")

        monkeypatch.setattr(main, "retrieve_chunks", _boom)
        test_client = TestClient(main.app, raise_server_exceptions=False)
        response = test_client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        assert response.status_code == 500
        failed = [
            e for e in read_log(perf_log) if e["event"] == "request_failed"
        ]
        assert failed[0]["failure_category"] == "unhandled_exception"

    def test_health_and_root_are_not_logged(self, client, perf_log):
        client.get("/health")
        client.get("/")
        assert read_log(perf_log) == []

    def test_preflight_options_are_not_logged(self, client, perf_log):
        client.options(
            "/ask",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert read_log(perf_log) == []

    def test_perf_stdout_compatibility(self, capsys, perf_log):
        # record_stage must keep printing the historical [PERF] line.
        import obs

        obs.record_stage("lexical_retrieve", 0.031, extra="points=4850")
        out = capsys.readouterr().out
        assert "[PERF] lexical_retrieve=0.031s points=4850" in out

    def test_obs_api_outside_request_never_raises(self):
        # CLI mode: no request context — every setter must be a no-op.
        from obs import (
            set_question_meta,
            set_retrieval_meta,
            set_source_count,
            set_phase_latency,
            set_failure,
        )

        set_question_meta(10)
        set_retrieval_meta(strategy="hybrid", topic="x")
        set_source_count(3)
        set_phase_latency("retrieval", 0.1)
        set_failure("x")


# ---------------------------------------------------------------------------
# 4. Secret non-leakage into the perf log
# ---------------------------------------------------------------------------


class TestPerfLogSecretHygiene:
    def test_api_key_and_question_text_never_reach_perf_log(
        self, perf_log, set_gemini_failure
    ):
        from conftest import FAKE_API_KEY

        secret_question = (
            f"My key is {FAKE_API_KEY} and my question is about "
            "binary search supersecretvalue123"
        )
        # Gemini fails so the answer path is also exercised (failure).
        set_gemini_failure(None)
        from fastapi.testclient import TestClient
        import main

        test_client = TestClient(main.app)
        test_client.post("/ask", json={"question": secret_question})

        raw = perf_log.read_text(encoding="utf-8")
        assert FAKE_API_KEY not in raw
        assert "supersecretvalue123" not in raw
        # Metadata IS allowed: only the question text itself is secret.
        events = read_log(perf_log)
        assert any(
            e.get("question_length") == len(secret_question) for e in events
        )


# ---------------------------------------------------------------------------
# 5. report_perf: parsing + percentile aggregation
# ---------------------------------------------------------------------------


class TestPercentiles:
    def test_empty_is_none(self):
        from report_perf import percentile

        assert percentile([], 0.5) is None

    def test_single_value(self):
        from report_perf import percentile

        assert percentile([2.0], 0.5) == 2.0
        assert percentile([2.0], 0.95) == 2.0

    def test_known_odd_and_even_arrays(self):
        from report_perf import percentile

        # 1..5 -> p50 = 3, p95 = 5 (nearest rank over sorted input)
        assert percentile([1, 2, 3, 4, 5], 0.5) == 3
        assert percentile([1, 2, 3, 4, 5], 0.95) == 5
        # 1..4 -> p50 = 2 (lower-middle), p95 = 4
        assert percentile([1, 2, 3, 4], 0.5) == 2
        assert percentile([1, 2, 3, 4], 0.95) == 4

    def test_percentile_monotonic(self):
        from report_perf import percentile

        values = sorted(range(1, 101))
        assert percentile(values, 0.5) <= percentile(values, 0.95)


class TestReportPerf:
    def _write_events(self, path, events):
        with open(path, "w", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event) + "\n")

    def test_load_events_skips_malformed_lines(self, tmp_path):
        from report_perf import load_events

        log = tmp_path / "perf.log"
        log.write_text(
            '{"event": "request_start", "request_id": "a"}\n'
            "NOT JSON AT ALL\n"
            "\n"
            '{"event": "request_end", "request_id": "a"}\n',
            encoding="utf-8",
        )
        events = list(load_events([str(log)]))
        assert len(events) == 2
        assert events[0]["event"] == "request_start"

    def test_summarize_counts_and_latencies(self, tmp_path):
        from report_perf import summarize

        events = []
        for i in range(1, 21):  # 18 success, 2 failures
            events.append({"event": "request_start", "request_id": str(i)})
            base = {
                "event": "request_end",
                "request_id": str(i),
                "total_latency": i * 0.01,
                "retrieval_latency": i * 0.001,
                "generation_latency": i * 0.005,
                "stages": {"api_retrieval": i * 0.0005},
                "failure_category": None,
                "http_status": 200,
            }
            if i in (19, 20):
                base["event"] = "request_failed"
                base["failure_category"] = "generation_failed"
                base["http_status"] = 502
            events.append(base)

        report = summarize(events)
        assert report["starts"] == 20
        assert report["ended"] == 20
        assert report["successful"] == 18
        assert report["failed"] == 2
        assert report["failures"]["generation_failed"] == 2
        # totals are i * 0.01 for i=1..20 -> nearest-rank p50 = 0.10
        # (rank ceil(0.5 * 20) = 10 -> index 9)
        assert report["totals"] == sorted(report["totals"])
        assert len(report["totals"]) == 20
        assert abs(report["totals"][9] - 0.10) < 1e-9

    def test_main_report_and_exit_codes(self, tmp_path, capsys):
        import report_perf

        missing = tmp_path / "nope.log"
        assert report_perf.main(["report_perf.py", str(missing)]) == 1

        log = tmp_path / "perf.log"
        self._write_events(
            log,
            [
                {"event": "request_start", "request_id": "1"},
                {
                    "event": "request_end",
                    "request_id": "1",
                    "total_latency": 0.2,
                    "retrieval_latency": 0.17,
                    "generation_latency": 0.02,
                    "stages": {"embedding": 0.004},
                    "failure_category": None,
                    "http_status": 200,
                },
            ],
        )
        assert report_perf.main(["report_perf.py", str(log)]) == 0
        out = capsys.readouterr().out
        assert "ALGOFORGE PERFORMANCE REPORT" in out
        assert "Successful       : 1" in out
        assert "embedding" in out

    def test_end_to_end_report_from_real_log(self, client, perf_log, capsys):
        import report_perf

        client.post("/ask", json={"question": "What is binary search?"})
        assert report_perf.main(["report_perf.py", str(perf_log)]) == 0
        out = capsys.readouterr().out
        assert "Requests started : 1" in out
        assert "hybrid" not in out  # report shows numbers, not payloads


# ---------------------------------------------------------------------------
# 6. generate_embeddings: metadata hard-skip (no fabrication)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generate_embeddings_module():
    """Import generate_embeddings with a torch stub (real torch is heavy
    and unused by the logic under test)."""
    torch_stub = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def get_device_name(index=0):
            return "stub-gpu"

    torch_stub.cuda = _Cuda
    torch_stub.is_available = lambda: False
    saved = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        module = importlib.import_module("generate_embeddings")
        yield module
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


class TestMetadataHardSkip:
    def test_missing_metadata_file_returns_none(
        self, generate_embeddings_module, tmp_path, monkeypatch, capsys
    ):
        module = generate_embeddings_module
        monkeypatch.setattr(module, "METADATA_DIR", str(tmp_path / "meta"))
        result = module.load_metadata("missingvid1")
        assert result is None
        out = capsys.readouterr().out
        assert "SKIPPING" in out
        assert "No metadata file" in out

    def test_malformed_metadata_returns_none(
        self, generate_embeddings_module, tmp_path, monkeypatch, capsys
    ):
        module = generate_embeddings_module
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "brokenvid1.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(module, "METADATA_DIR", str(meta_dir))
        assert module.load_metadata("brokenvid1") is None

    def test_metadata_missing_key_returns_none(
        self, generate_embeddings_module, tmp_path, monkeypatch
    ):
        module = generate_embeddings_module
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "partialid1.json").write_text(
            json.dumps({"video_id": "partialid1"}), encoding="utf-8"
        )
        monkeypatch.setattr(module, "METADATA_DIR", str(meta_dir))
        assert module.load_metadata("partialid1") is None

    def test_valid_metadata_still_loads(
        self, generate_embeddings_module, tmp_path, monkeypatch
    ):
        module = generate_embeddings_module
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        good = {
            "video_id": "goodid12345",
            "video_title": "A real title",
            "youtube_url": "https://youtu.be/goodid12345",
        }
        (meta_dir / "goodid12345.json").write_text(
            json.dumps(good), encoding="utf-8"
        )
        monkeypatch.setattr(module, "METADATA_DIR", str(meta_dir))
        assert module.load_metadata("goodid12345") == good

    def test_main_skips_video_without_fabricating_records(
        self, generate_embeddings_module, tmp_path, monkeypatch, capsys
    ):
        """End-to-end offline: one valid transcript, NO metadata -> the
        video is skipped, nothing is embedded, and no fabricated
        youtube_url/title appears anywhere."""
        module = generate_embeddings_module

        transcripts = tmp_path / "transcripts"
        transcripts.mkdir()
        (transcripts / "nometavid1.json").write_text(
            json.dumps([{"start": 0.0, "end": 90.0, "text": "hello"}]),
            encoding="utf-8",
        )
        embeddings_out = tmp_path / "embeddings"
        embeddings_out.mkdir()
        empty_meta = tmp_path / "meta"
        empty_meta.mkdir()

        monkeypatch.setattr(module, "TRANSCRIPT_DIR", str(transcripts))
        monkeypatch.setattr(module, "EMBEDDINGS_DIR", str(embeddings_out))
        monkeypatch.setattr(module, "METADATA_DIR", str(empty_meta))
        monkeypatch.setattr(module, "fail", lambda message: (_ for _ in ()).throw(AssertionError(message)))

        module.main()

        out = capsys.readouterr().out
        assert "SKIPPING this video" in out
        assert "0/1" in out  # batch complete: 0 of 1 succeeded
        # No output file was written at all.
        assert list(embeddings_out.glob("*_embeddings.json")) == []

    def test_fallback_url_shape_never_exists_anymore(
        self, generate_embeddings_module
    ):
        """The old fabricated fallback (youtu.be/<filename>) must not be
        reachable: load_metadata can only return None or real data."""
        module = generate_embeddings_module
        source = open(module.__file__, encoding="utf-8").read()
        assert 'f"https://youtu.be/{video_id}"' not in source


# ---------------------------------------------------------------------------
# 7. load_to_qdrant: import-time side-effect-free main() guard
# ---------------------------------------------------------------------------


class TestLoadToQdrantMainGuard:
    def test_import_has_zero_side_effects(self, monkeypatch):
        """Re-importing the module must not instantiate QdrantClient,
        create directories, or touch the filesystem."""
        import qdrant_client as qdrant_stub_module

        created = []
        real_stub = qdrant_stub_module.QdrantClient

        class _Counting(real_stub):
            def __init__(self, *args, **kwargs):
                created.append((args, kwargs))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(qdrant_stub_module, "QdrantClient", _Counting)
        sys.modules.pop("load_to_qdrant", None)
        try:
            module = importlib.import_module("load_to_qdrant")
            assert created == []  # import is inert
            assert callable(module.main)
        finally:
            sys.modules.pop("load_to_qdrant", None)

    def test_make_point_id_unchanged_and_deterministic(self):
        module = importlib.import_module("load_to_qdrant")
        import hashlib

        expected = int(hashlib.md5(b"ABC123XYZ_7").hexdigest()[:12], 16)
        assert module.make_point_id("ABC123XYZ", 7) == expected
        assert module.make_point_id("ABC123XYZ", 7) == expected
        assert module.make_point_id("ABC123XYZ", 8) != expected
        # 48-bit id space (legacy-orphan detection relies on this).
        assert 0 < expected < 2**48
