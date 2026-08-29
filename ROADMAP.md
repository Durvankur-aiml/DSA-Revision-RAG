# MISSION ANTHROPIC — STRIVER A2Z RAG
## Project Roadmap

Last updated: current session

---

## CURRENT STATUS (Demo Version — Working)

Completed and validated:

- [x] Project folder structure
- [x] Python 3.11.9 + venv
- [x] FFmpeg, yt-dlp installed
- [x] Playlist-based batch audio download (resumable, batch/start position control)
- [x] Whisper SMALL transcription (windowed, GPU-accelerated, gap-validated, sponsor-filtered)
- [x] ~90-second time-based chunking
- [x] Embeddings via `all-MiniLM-L6-v2` (384-dim, GPU-accelerated)
- [x] Qdrant local persistent vector DB (additive multi-video upload, no data wipe)
- [x] Gemini API integration (free tier) for answer generation
- [x] CLI-based end-to-end RAG loop: question → embed → retrieve → generate → cite sources + timestamps
- [x] Per-video metadata system (`data/metadata/{video_id}.json`)
- [x] Multi-video scaling in progress (batch processing, 10 videos at a time)

**This is a genuinely working, single-and-multi-video RAG pipeline.** Everything below is upgrade work on top of a solid foundation — not a rebuild.

---

## UPGRADE ROADMAP

### Tier 1 — High value, achievable soon

1. **Persona-grounded answers ("explain it like Striver would")**
   - Modify the system prompt in `ask.py` so Gemini doesn't just use retrieved transcript as facts, but mimics Striver's actual teaching style — his analogies, pacing, and phrasing patterns.
   - Low effort (prompt engineering only), high impact on how "authentic" the tool feels.
   - Status: proposed by user, not yet implemented.

2. **Topic/category tagging per chunk**
   - Tag each chunk at embedding time with a DSA topic (arrays, recursion, DP, graphs, trees, sorting, etc.)
   - Enables topic-filtered search in Qdrant (e.g. restrict search to "sorting" chunks when the question is clearly about sorting) — significantly improves retrieval precision over pure semantic search.
   - Could be done via keyword rules initially, or an LLM tagging pass per chunk.
   - Medium effort, high value for retrieval quality.

3. **Multi-turn conversation memory**
   - Currently every question in `ask.py` is independent — no memory of prior exchanges.
   - Add conversation history so follow-ups like "explain that simpler" or "give me an example" work naturally.
   - Medium effort (needs prompt + retrieval context design), high value for usability.

4. **Semantic / smarter chunking**
   - Current chunking is time-based (~90 sec), which can cut mid-thought.
   - Upgrade to sentence-boundary-aware or topic-aware chunking, ideally combined with a token limit rather than a fixed duration.
   - Original architecture doc flagged this as a known future improvement.
   - Medium effort, meaningfully improves answer quality and retrieval accuracy.

### Tier 2 — Medium value, more effort

5. **Reranking retrieved chunks**
   - After Qdrant returns top-K by embedding similarity, apply a cross-encoder reranker to reorder by true relevance to the question.
   - Well-established RAG accuracy technique.
   - Medium-high effort (new model, new pipeline stage).

6. **Frontend UI**
   - Currently CLI-only (`data/frontend/` folder still empty).
   - Build a simple web chat interface with an embedded YouTube player that jumps to the cited timestamp automatically.
   - High value for demo-ability (portfolio, interviews) — makes the project tangible to non-technical viewers.
   - Medium-high effort depending on framework choice.

7. **Difficulty-aware answers**
   - Detect intent/depth of question ("explain simply" vs "explain time complexity in depth") and adjust answer accordingly.
   - Medium effort (prompt logic + possibly a lightweight classifier).

### Tier 3 — Advanced / stretch goals

8. **Code extraction from video frames (OCR)**
   - Striver writes code on-screen while explaining. Currently only audio is used.
   - Extracting frames + OCR'ing code snippets would let the system show actual code alongside explanations.
   - High effort (computer vision + OCR pipeline), high "wow factor" if done well.

9. **Practice question generation**
   - Since chunks will eventually be topic-tagged, generate quiz/practice questions per topic using the LLM.
   - Turns the tool from pure Q&A into an active-recall study aid.
   - Medium effort, builds on Tier 1 topic tagging.

10. **Cross-video linking / topic graph**
    - Link related concepts across videos ("this is explained in more depth in video X").
    - Requires some topic relationship modeling.
    - High effort, ties the whole knowledge base together intelligently.

---

## SUGGESTED SEQUENCING

1. **Finish current multi-video batch scaling** (in progress — don't context-switch mid-build)
2. Once ~20-30 videos are indexed and stable → **Persona-grounded answers** (quick win, user's idea, high impact)
3. **Topic tagging + filtered retrieval** (single highest-value technical upgrade for retrieval quality)
4. Then choose based on goal:
   - Optimizing for **"show it off"** → Frontend UI next
   - Optimizing for **"make it genuinely more accurate"** → Semantic chunking next

---

## NOTES

- This roadmap is a living document — update it as priorities shift or new ideas come up mid-session.
- Nothing here should block or delay the current batch processing work already in progress.
