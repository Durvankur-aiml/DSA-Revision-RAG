"""ALGOFORGE structured observability.

Minimal, stdlib-only, production-path-safe:

- A per-request context (request id, stage timings, counters) carried in a
  ContextVar, set ONCE by the ASGI middleware and afterwards mutated only
  through the shared object (safe across the threadpool boundary; never
  re-assigned inside worker threads).
- A rotating JSONL perf log (logs/perf.log) written by a dedicated logger
  that never propagates and never raises.
- Backwards-compatible [PERF] stdout lines (same format as before) are
  emitted by record_stage so existing diagnostics keep working.

Design rules (enforced by this module's structure):
- Observability failures can never fail a request: every public function
  swallows and reports exceptions to stderr at most.
- Secrets, API keys, prompts, and question text are NEVER logged — only
  metadata (lengths, counts, latencies, statuses).
- Request ids: an incoming X-Request-ID header is adopted when sane,
  otherwise a uuid4 hex is generated. The id is echoed back to clients.
"""

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


class RequestContext:
    """Mutable per-request observation state.

    The ContextVar holds this object; everything downstream mutates the
    object (never re-binds the var), so state set in the event loop is
    visible in threadpool handlers and vice versa.
    """

    __slots__ = (
        "request_id",
        "method",
        "path",
        "start_monotonic",
        "question_length",
        "retrieval_strategy",
        "topic",
        "exact_topic",
        "candidate_count",
        "source_count",
        "stages",
        "retrieval_latency",
        "generation_latency",
        "http_status",
        "failure_category",
        "reranker_enabled",
        "reranker_fallback",
        "reranker_model",
        "reranker_device",
        "reranker_depth",
        "llm_provider",
        "llm_model",
        "llm_fallback_used",
        "llm_fallback_provider",
        "llm_tokens",
        "agent_intent",
        "agent_response_mode",
        "agent_retrieval_required",
        "agent_reranking_required",
        "citation_status",
        "citation_count",
        "valid_citation_count",
        "invalid_citation_count",
        "unsupported_claim_count",
        "malformed_citation_count",
        "citation_latency",
        "agent_plan",
        "agent_step",
        "agent_step_status",
        "agent_step_latency",
        "agent_recovery_attempted",
        "agent_generation_attempts",
        "agent_final_status",
        "agent_total_latency",
    )

    def __init__(self, request_id, method="?", path="?"):
        self.request_id = request_id
        self.method = method
        self.path = path
        self.start_monotonic = time.monotonic()
        self.question_length = None
        self.retrieval_strategy = None
        self.topic = None
        self.exact_topic = None
        self.candidate_count = None
        self.source_count = None
        self.stages = {}
        self.retrieval_latency = None
        self.generation_latency = None
        self.http_status = None
        self.failure_category = None
        self.reranker_enabled = None
        self.reranker_fallback = None
        self.reranker_model = None
        self.reranker_device = None
        self.reranker_depth = None
        self.llm_provider = None
        self.llm_model = None
        self.llm_fallback_used = None
        self.llm_fallback_provider = None
        self.llm_tokens = None
        self.agent_intent = None
        self.agent_response_mode = None
        self.agent_retrieval_required = None
        self.agent_reranking_required = None
        self.citation_status = None
        self.citation_count = None
        self.valid_citation_count = None
        self.invalid_citation_count = None
        self.unsupported_claim_count = None
        self.malformed_citation_count = None
        self.citation_latency = None
        self.agent_plan = None
        self.agent_step = None
        self.agent_step_status = None
        self.agent_step_latency = None
        self.agent_recovery_attempted = None
        self.agent_generation_attempts = None
        self.agent_final_status = None
        self.agent_total_latency = None

    def snapshot(self):
        """Flat, JSON-ready dict of everything known about this request."""
        data = {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "total_latency": round(
                time.monotonic() - self.start_monotonic, 6
            ),
            "question_length": self.question_length,
            "retrieval_strategy": self.retrieval_strategy,
            "topic": self.topic,
            "exact_topic": self.exact_topic,
            "candidate_count": self.candidate_count,
            "final_source_count": self.source_count,
            "stages": {
                name: round(seconds, 6)
                for name, seconds in self.stages.items()
            },
            "retrieval_latency": self.retrieval_latency,
            "generation_latency": self.generation_latency,
            "http_status": self.http_status,
            "failure_category": self.failure_category,
            "reranker_enabled": self.reranker_enabled,
            "reranker_fallback": self.reranker_fallback,
            "reranker_model": self.reranker_model,
            "reranker_device": self.reranker_device,
            "reranker_depth": self.reranker_depth,
        }
        if self.llm_provider is not None:
            data["llm_provider"] = self.llm_provider
        if self.llm_model is not None:
            data["llm_model"] = self.llm_model
        if self.llm_fallback_used is not None:
            data["llm_fallback_used"] = self.llm_fallback_used
        if self.llm_fallback_provider is not None:
            data["llm_fallback_provider"] = self.llm_fallback_provider
        if self.llm_tokens is not None:
            data["llm_tokens"] = self.llm_tokens
        if self.agent_intent is not None:
            data["agent_intent"] = self.agent_intent
        if self.agent_response_mode is not None:
            data["agent_response_mode"] = self.agent_response_mode
        if self.agent_retrieval_required is not None:
            data["agent_retrieval_required"] = self.agent_retrieval_required
        if self.agent_reranking_required is not None:
            data["agent_reranking_required"] = self.agent_reranking_required
        if self.citation_status is not None:
            data["citation_status"] = self.citation_status
        if self.citation_count is not None:
            data["citation_count"] = self.citation_count
        if self.valid_citation_count is not None:
            data["valid_citation_count"] = self.valid_citation_count
        if self.invalid_citation_count is not None:
            data["invalid_citation_count"] = self.invalid_citation_count
        if self.unsupported_claim_count is not None:
            data["unsupported_claim_count"] = self.unsupported_claim_count
        if self.malformed_citation_count is not None:
            data["malformed_citation_count"] = self.malformed_citation_count
        if self.citation_latency is not None:
            data["citation_latency"] = self.citation_latency
        if self.agent_plan is not None:
            data["agent_plan"] = self.agent_plan
        if self.agent_step is not None:
            data["agent_step"] = self.agent_step
        if self.agent_step_status is not None:
            data["agent_step_status"] = self.agent_step_status
        if self.agent_step_latency is not None:
            data["agent_step_latency"] = self.agent_step_latency
        if self.agent_recovery_attempted is not None:
            data["agent_recovery_attempted"] = self.agent_recovery_attempted
        if self.agent_generation_attempts is not None:
            data["agent_generation_attempts"] = self.agent_generation_attempts
        if self.agent_final_status is not None:
            data["agent_final_status"] = self.agent_final_status
        if self.agent_total_latency is not None:
            data["agent_total_latency"] = self.agent_total_latency
        return data


_request_ctx = contextvars.ContextVar("algoforge_request_ctx", default=None)


def current_request():
    """The active RequestContext, or None outside a request."""
    return _request_ctx.get()


# ---------------------------------------------------------------------------
# Logging backend (rotating JSONL file, never raises)
# ---------------------------------------------------------------------------

_LOG_DIR_ENV = "ALGOFORGE_LOG_DIR"
_PERF_LOGGER_NAME = "algoforge.perf"
_MAX_LOG_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Paths that are health/noise checks and are not logged as request events.
_QUIET_PATHS = {"/", "/health"}

# Incoming request ids must be short printable tokens (header hygiene).
_SAFE_REQUEST_ID = re.compile(r"^[\w.-]{1,64}$")

perf_logger = logging.getLogger(_PERF_LOGGER_NAME)
perf_logger.propagate = False
perf_logger.setLevel(logging.INFO)
perf_logger.addHandler(logging.NullHandler())


def default_log_dir():
    """Repo-root logs/ directory (overridable via ALGOFORGE_LOG_DIR)."""
    env_dir = os.environ.get(_LOG_DIR_ENV, "").strip()
    if env_dir:
        return env_dir
    # Backend/obs.py -> repo root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "logs")


def configure(log_dir=None, log_filename="perf.log"):
    """(Re)point the perf logger at a rotating file. Never raises.

    Returns the log file path on success, None on failure. Tests use this
    to redirect output to a tmp dir; production calls it once at import.
    """
    try:
        target_dir = log_dir or default_log_dir()
        os.makedirs(target_dir, exist_ok=True)
        log_path = os.path.join(target_dir, log_filename)

        for handler in list(perf_logger.handlers):
            if isinstance(handler, RotatingFileHandler):
                handler.close()
                perf_logger.removeHandler(handler)

        handler = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        perf_logger.addHandler(handler)
        return log_path
    except Exception as error:
        print(f"WARNING: perf logging disabled ({error})", file=sys.stderr)
        return None


def _emit(record):
    """Serialize one structured event to the perf log. Never raises."""
    try:
        record["ts"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        perf_logger.info(json.dumps(record, default=str, ensure_ascii=True))
    except Exception as error:
        print(f"WARNING: perf log write failed ({error})", file=sys.stderr)


# ---------------------------------------------------------------------------
# Request lifecycle (called by the ASGI middleware in main.py)
# ---------------------------------------------------------------------------


def begin_request(request_id, method, path):
    """Create and install this request's context. Returns the context."""
    ctx = RequestContext(request_id, method=method, path=path)
    _request_ctx.set(ctx)
    if path not in _QUIET_PATHS:
        _emit(
            {
                "event": "request_start",
                "request_id": request_id,
                "method": method,
                "path": path,
            }
        )
    return ctx


def end_request():
    """Emit request_end (or request_failed) and clear the context."""
    ctx = _request_ctx.get()
    if ctx is None:
        return
    _request_ctx.set(None)
    if ctx.path in _QUIET_PATHS:
        return
    snapshot = ctx.snapshot()
    snapshot["event"] = (
        "request_end" if ctx.failure_category is None else "request_failed"
    )
    _emit(snapshot)


def adopt_or_create_request_id(incoming):
    """Use a sane incoming X-Request-ID, else generate a uuid4 hex."""
    if incoming:
        candidate = incoming.strip()
        if _SAFE_REQUEST_ID.match(candidate):
            return candidate
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Enrichment API (called from request handlers / ask.py)
# ---------------------------------------------------------------------------


def set_question_meta(question_length):
    try:
        ctx = _request_ctx.get()
        if ctx is not None:
            ctx.question_length = int(question_length)
    except Exception:
        pass


def set_retrieval_meta(strategy=None, topic=None, exact_topic=None,
                       candidate_count=None):
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if strategy is not None:
            ctx.retrieval_strategy = strategy
        if topic is not None:
            ctx.topic = topic
        if exact_topic is not None:
            ctx.exact_topic = bool(exact_topic)
        if candidate_count is not None:
            ctx.candidate_count = int(candidate_count)
    except Exception:
        pass


def set_source_count(count):
    try:
        ctx = _request_ctx.get()
        if ctx is not None:
            ctx.source_count = int(count)
    except Exception:
        pass


def set_reranker_meta(enabled=None, fallback=None, model=None, device=None,
                      depth=None):
    """Record the reranker state for this request (Phase 3C).

    Called once per request with enabled/depth from ask.py, then
    enriched by the singleton (model/device) and on any failure
    (fallback=True). Never raises; unknown fields stay None and are
    omitted from the JSON snapshot.
    """
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if enabled is not None:
            ctx.reranker_enabled = bool(enabled)
        if fallback is not None:
            ctx.reranker_fallback = bool(fallback)
        if model is not None:
            ctx.reranker_model = str(model)
        if device is not None:
            ctx.reranker_device = str(device)
        if depth is not None:
            ctx.reranker_depth = int(depth)
    except Exception:
        pass


def set_llm_meta(provider=None, model=None, fallback_used=None,
                 fallback_provider=None, tokens=None):
    """Record LLM provider, model, fallback, and usage metadata for this request."""
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if provider is not None:
            ctx.llm_provider = str(provider)
        if model is not None:
            ctx.llm_model = str(model)
        if fallback_used is not None:
            ctx.llm_fallback_used = bool(fallback_used)
        if fallback_provider is not None:
            ctx.llm_fallback_provider = str(fallback_provider)
        if tokens is not None:
            ctx.llm_tokens = int(tokens)
    except Exception:
        pass


def set_agent_meta(intent=None, response_mode=None, retrieval_required=None,
                   reranking_required=None):
    """Record Agent Router decision metadata for this request."""
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if intent is not None:
            ctx.agent_intent = str(intent)
        if response_mode is not None:
            ctx.agent_response_mode = str(response_mode)
        if retrieval_required is not None:
            ctx.agent_retrieval_required = bool(retrieval_required)
        if reranking_required is not None:
            ctx.agent_reranking_required = bool(reranking_required)
    except Exception:
        pass


def set_citation_meta(status=None, count=None, valid_count=None,
                      invalid_count=None, unsupported_count=None,
                      malformed_count=None, latency=None):
    """Record Grounding & Citation verification metadata for this request."""
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if status is not None:
            ctx.citation_status = str(status)
        if count is not None:
            ctx.citation_count = int(count)
        if valid_count is not None:
            ctx.valid_citation_count = int(valid_count)
        if invalid_count is not None:
            ctx.invalid_citation_count = int(invalid_count)
        if unsupported_count is not None:
            ctx.unsupported_claim_count = int(unsupported_count)
        if malformed_count is not None:
            ctx.malformed_citation_count = int(malformed_count)
        if latency is not None:
            ctx.citation_latency = round(float(latency), 6)
    except Exception:
        pass


def set_coordinator_meta(
    plan=None,
    step=None,
    step_status=None,
    step_latency=None,
    recovery_attempted=None,
    generation_attempts=None,
    final_status=None,
    total_latency=None,
):
    """Record Agent Coordinator orchestration metadata for this request."""
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if plan is not None:
            ctx.agent_plan = str(plan)
        if step is not None:
            ctx.agent_step = str(step)
        if step_status is not None:
            ctx.agent_step_status = str(step_status)
        if step_latency is not None:
            ctx.agent_step_latency = round(float(step_latency), 6)
        if recovery_attempted is not None:
            ctx.agent_recovery_attempted = bool(recovery_attempted)
        if generation_attempts is not None:
            ctx.agent_generation_attempts = int(generation_attempts)
        if final_status is not None:
            ctx.agent_final_status = str(final_status)
        if total_latency is not None:
            ctx.agent_total_latency = round(float(total_latency), 6)
    except Exception:
        pass


def set_phase_latency(phase, seconds):
    """Record the retrieval/generation phase totals for quick reporting."""
    try:
        ctx = _request_ctx.get()
        if ctx is None:
            return
        if phase == "retrieval":
            ctx.retrieval_latency = round(float(seconds), 6)
        elif phase == "generation":
            ctx.generation_latency = round(float(seconds), 6)
    except Exception:
        pass


def set_failure(category):
    """Tag the failure category; the final event becomes request_failed."""
    try:
        ctx = _request_ctx.get()
        if ctx is not None and category:
            ctx.failure_category = str(category)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stage timings (backwards-compatible with the [PERF] lines)
# ---------------------------------------------------------------------------


def record_stage(stage, seconds, extra=None):
    """Record one named stage into the request context.

    Preserves the historical stdout line: "[PERF] stage=0.172s" (+ extra),
    so existing diagnostics keep working unchanged.
    """
    try:
        print(
            f"[PERF] {stage}={seconds:.3f}s"
            + (f" {extra}" if extra else ""),
            flush=False,
        )
    except Exception:
        pass
    try:
        ctx = _request_ctx.get()
        if ctx is not None:
            ctx.stages[stage] = round(float(seconds), 6)
    except Exception:
        pass


# Configure the default destination once at import. Tests may re-call
# configure() with their own directory.
configure()
