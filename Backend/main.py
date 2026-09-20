from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ask import (
    retrieve_chunks,
    build_prompt,
    ask_gemini,
    format_sources,
    validate_question,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Mission Anthropic API",
    description="AI DSA Tutor powered by Striver A2Z RAG",
    version="1.0.0",
)


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
        "service": "Mission Anthropic API",
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

    # --------------------------------------------------------
    # 1. Validate question
    # --------------------------------------------------------

    question, error = validate_question(request.question)

    if error:
        raise HTTPException(
            status_code=400,
            detail=error,
        )

    # --------------------------------------------------------
    # 2. Retrieve relevant chunks
    # --------------------------------------------------------

    chunks = retrieve_chunks(question)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the A2Z knowledge base.",
        )

    # --------------------------------------------------------
    # 3. Build grounded prompt
    # --------------------------------------------------------

    user_message = build_prompt(
        question,
        chunks,
    )

    if user_message is None:
        raise HTTPException(
            status_code=404,
            detail="Retrieved content contained no usable text.",
        )

    # --------------------------------------------------------
    # 4. Generate answer with Gemini
    # --------------------------------------------------------

    answer = ask_gemini(
        user_message,
    )

    if answer is None:
        raise HTTPException(
            status_code=502,
            detail="Failed to generate an answer.",
        )

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

        # Get hybrid score
        from ask import LAST_RETRIEVAL_SCORES

        score = LAST_RETRIEVAL_SCORES.get(
            str(hit.id),
            0.0,
        )

        sources.append(
            Source(
                title=title,
                timestamp=timestamp,
                url=timestamp_url,
                score=round(score, 4),
            )
        )

    # --------------------------------------------------------
    # 6. Return structured API response
    # --------------------------------------------------------

    return AskResponse(
        answer=answer,
        sources=sources,
    )