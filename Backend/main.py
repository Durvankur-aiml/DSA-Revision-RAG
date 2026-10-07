import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import obs
from ask import (
    retrieve_chunks,
    build_prompt,
    ask_gemini,
    format_sources,
    validate_question,
    route_query,
    coordinate_query,
    AgentCoordinator,
    verify_citations,
    VerificationStatus,
    _perf,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ALGOFORGE API",
    description="AI DSA Tutor powered by Striver A2Z RAG",
    version="1.0.0",
)


# ------------------------------------------------------------
# CORS — development only
# ------------------------------------------------------------

# The React/Vite dev server runs on a different origin from this API.
# Vite's default port is 5173 (the frontend is built in Phase 4; no
# vite.config exists yet, so the default is the discovered convention,
# see Frontend/src/lib/api.js which targets http://localhost:8000).
# Restricted to loopback origins on purpose — widen only with a
# concrete deployment reason, never to "*".
CORS_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_DEV_ORIGINS,
    allow_credentials=False,  # no cookies/auth in this API
    allow_methods=["GET", "POST"],  # only what the API exposes
    allow_headers=["Content-Type"],
)


# ------------------------------------------------------------
# Observability middleware — request ID + structured lifecycle
# ------------------------------------------------------------

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Attach a request id, time the request, and emit the lifecycle events.

    The RequestContext object is created ONCE here and afterwards only
    mutated (never re-bound in the ContextVar), so values recorded inside
    threadpool endpoints are visible in this middleware afterwards.
    CORS preflights (OPTIONS) and health paths are not logged as events.
    """
    if request.method == "OPTIONS":
        return await call_next(request)

    request_id = obs.adopt_or_create_request_id(
        request.headers.get("x-request-id")
    )
    ctx = obs.begin_request(request_id, request.method, request.url.path)

    try:
        response = await call_next(request)
    except Exception:
        obs.set_failure("unhandled_exception")
        obs.end_request()
        raise

    ctx.http_status = response.status_code
    if ctx.failure_category is None and response.status_code >= 500:
        obs.set_failure(f"http_{response.status_code}")

    response.headers["X-Request-ID"] = request_id
    obs.end_request()
    return response


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    title: str
    timestamp: int
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "ALGOFORGE API",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    _t0 = time.monotonic()
    _t = time.monotonic()

    # --------------------------------------------------------
    # 1. Validate question
    # --------------------------------------------------------

    question, error = validate_question(request.question)
    obs.set_question_meta(len(request.question))
    _perf("api_validation", _t)
    _t = time.monotonic()

    if error:
        obs.set_failure(
            "empty_question" if "non-empty" in error else "question_too_long"
        )
        raise HTTPException(
            status_code=400,
            detail=error,
        )

    # --------------------------------------------------------
    # 1.5. Agent Router — Intent Classification & Strategy
    # --------------------------------------------------------
    decision = route_query(question)
    _perf("agent_routing", _t, extra=f"intent={decision.intent.value} mode={decision.response_mode}")
    _t = time.monotonic()

    # --------------------------------------------------------
    # 2. Agent Coordinator — Controlled Multi-Step Execution
    # --------------------------------------------------------
    coord_result = coordinate_query(question, decision)
    _perf("agent_coordination", _t, extra=f"plan={coord_result.plan.summary()} status={coord_result.final_status}")

    if not decision.retrieval_required or coord_result.final_status == "OUT_OF_SCOPE":
        obs.set_failure("out_of_scope")
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the A2Z knowledge base. This query appears outside of the DSA course scope.",
        )

    chunks = coord_result.sources
    obs.set_phase_latency("retrieval", time.monotonic() - _t)
    _perf("api_retrieval", _t)
    _t = time.monotonic()

    if not chunks:
        if coord_result.final_status == "FAILED" and coord_result.error and coord_result.error.startswith("Retrieval failed:"):
            raise RuntimeError(coord_result.error)

        obs.set_failure("no_relevant_content")
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the A2Z knowledge base.",
        )

    # 3. Prompt Build Stage Timing
    _perf("api_prompt_build", _t)
    _t = time.monotonic()

    # 4. Generation Result
    answer = coord_result.final_answer
    obs.set_phase_latency("generation", time.monotonic() - _t)
    _perf("api_gemini", _t)
    _t = time.monotonic()

    if coord_result.final_status == "FAILED" or not answer:
        obs.set_failure("generation_failed")
        raise HTTPException(
            status_code=502,
            detail=coord_result.error or "Failed to generate an answer.",
        )

    # 4.5. Citation Verification Stage Timing
    _perf(
        "api_citation_verification",
        _t,
        extra=f"status={coord_result.verification_status or 'PASS'}",
    )
    _t = time.monotonic()

    # --------------------------------------------------------
    # 5. Convert sources
    # --------------------------------------------------------

    sources = []

    for hit in chunks:

        payload = hit.payload or {}

        title = payload.get(
            "video_title",
            "Unknown",
        )

        youtube_url = payload.get(
            "youtube_url",
            "",
        )

        try:
            timestamp = int(
                float(
                    payload.get(
                        "start",
                        0,
                    )
                )
            )
        except (TypeError, ValueError):
            timestamp = 0

        # Correct YouTube timestamp URL
        if youtube_url:

            separator = (
                "&"
                if "?" in youtube_url
                else "?"
            )

            timestamp_url = (
                f"{youtube_url}"
                f"{separator}"
                f"t={timestamp}"
            )

        else:
            timestamp_url = ""

        # Request-local hybrid score (ContextVar in ask.py — never shared
        # across concurrent FastAPI tasks).
        from ask import get_retrieval_score

        score = get_retrieval_score(hit.id)

        sources.append(
            Source(
                title=title,
                timestamp=timestamp,
                url=timestamp_url,
                score=round(score, 4),
            )
        )

    # Request metadata for the observability layer. Retrieval strategy
    # and topic are recorded inside retrieve_chunks (ask.py) where they
    # are actually computed.
    obs.set_source_count(len(sources))
    _perf("api_source_mapping", _t)
    _perf("request_total", _t0)

    return AskResponse(
        answer=answer,
        sources=sources,
    )