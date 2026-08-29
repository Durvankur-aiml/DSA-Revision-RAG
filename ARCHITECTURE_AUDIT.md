# MISSION ANTHROPIC — ARCHITECTURE AUDIT

Audit date: 2026-08-29
Scope: read-only inspection. **No project files were modified.**
Method: read all 5 backend scripts, audited all 30 transcript + 30 embedding files, decoded all 827 Qdrant points directly from `storage.sqlite`.

Everything below is measured from the actual repo. Where something could not be verified, it is labelled **UNVERIFIED**.

---

## HEADLINE

Two things you should know before anything else:

1. **`Backend/transcribe_audio.py` does not run.** It has a hard `SyntaxError` on line 18. It was corrupted by a copy-paste out of a rendered markdown view. Your ingestion pipeline is currently broken at the transcription stage. There is no backup and no git history for this project.

2. **Your Qdrant collection contains 118 redundant points out of 827 (14.3%).** All of them are copies of one video (C++ Basics L1), which is present **three times** under two different identities. This is the confirmed cause of both the "duplicate retrieval results" and the "malformed timestamp URL" you had already noticed.

The good news, and it is real: **transcript → embedding integrity is perfect (768 = 768, every single video matches),** and 768 of 827 Qdrant points have correct deterministic IDs that verify exactly against `make_point_id()`. The core pipeline design is sound. The problems are debris and one bad paste, not architecture.

---

## A. CURRENT ARCHITECTURE

```
download_audio.py     playlist → per-video mp3 + metadata json
        ↓
transcribe_audio.py   mp3 → windowed Whisper → ~90s chunks → transcript json   [BROKEN]
        ↓
generate_embeddings.py transcript json + metadata → 384-dim vectors → embeddings json
        ↓
load_to_qdrant.py     embeddings json → deterministic-ID points → Qdrant (upsert, additive)
        ↓
ask.py                question → embed → search → threshold filter → Gemini → answer + sources
```

Phase 1 (ingestion) and Phase 2 (RAG answering) are **both already built**. `ask.py` exists and is a complete, working CLI RAG loop. Your own notes describe Phase 2 as "next" — that is out of date; it is done.

Actual folder layout (differs from your conceptual doc):

```
Mission Anthropic - RAG/
├── Backend/            (capital B; 5 scripts, 2285 lines)
├── Data/
│   ├── Audio/          36 mp3 + 3 incomplete .webm.part
│   ├── Metadata/       35 json          <- extra stage not in your original doc
│   ├── Transcripts/    30 json
│   └── embeddings/     30 json          (lowercase 'e')
├── Frontend/           EMPTY
├── qdrant_db/          collection 'striver_a2z', 827 points, 4.3 MB
├── venv/
└── ROADMAP.md
```

Note `Data/Metadata/` — a per-video metadata layer that is not in your architecture doc but is load-bearing. It is the single source of truth for `video_title` and `youtube_url`.

---

## B. DATA FLOW & MEASURED COUNTS

| Stage | Count | Status |
|---|---|---|
| mp3 files | 36 | 35 real videos + 1 duplicate keyed by title |
| metadata json | 35 | matches the 35 real videos |
| transcript json | 30 | 29 real videos + 1 title-keyed dup; **6 mp3 still untranscribed** |
| transcript chunks | 768 | — |
| embedding records | 768 | **exact match, zero loss** |
| Qdrant points | 827 | 768 correct + **59 legacy orphans** |
| distinct text values | 709 | → 118 points are redundant copies |

The identity chain `VIDEO_ID → mp3 → transcript → embeddings → Qdrant payload` holds for every video **except** the one title-keyed intruder.

Untranscribed mp3s (audio downloaded, no transcript yet): `3Zv-s9UUrFM`, `N0MgLvceX7M`, `Z0R2u6gd3GU`, `bR7mQgwQ_o8`, `vwZj1K0e9U8`, `xvNwoz-ufXA`.

Incomplete downloads: `DhFh8Kw7ymk.webm.part`, `eD95WRfh81c.webm.part`, `eZr-6p0B7ME.webm.part`.

---

## C. FILE RESPONSIBILITIES

**`download_audio.py`** (236 lines) — Clean. Interactive batch download with `batch_size` + `start_position`. Always names files `{video_id}.mp3` / `{video_id}.json`; never uses titles. Writes metadata only after the mp3 is confirmed on disk. `already_downloaded()` requires **both** mp3 and metadata before skipping — correctly conservative. This file did **not** cause the title-keyed duplicate.

**`transcribe_audio.py`** (1168 lines) — **BROKEN, cannot execute.** Design (readable through the corruption): 300s windows, Whisper `small`, per-window offset restoration to global timestamps, 5s gap warnings, per-video sponsor skip ranges (`EAR7De6Goz4: [(170,250)]`), ~90s chunking, atomic temp-file save, `is_valid_transcript_file()` for resumable skipping. The `_windows_temp/` directory is absent, which confirms temp cleanup works correctly.

**`generate_embeddings.py`** (218 lines) — Clean and well-guarded. `all-MiniLM-L6-v2`, 384-dim, CUDA with CPU fallback, batch 32, per-chunk validation, atomic save, skip-if-already-embedded. **Critical behaviour:** it derives `video_id` from the *transcript filename* (`os.path.splitext(f)[0]`), then looks up `Data/Metadata/{video_id}.json`. If that metadata file is missing it silently falls back to `video_title = video_id` and `youtube_url = f"https://youtu.be/{video_id}"`. That fallback is the mechanism that produced the malformed URL.

**`load_to_qdrant.py`** (337 lines) — Strong. Validates 8 required keys, vector type/length, non-empty text. Additive upsert, never wipes. `make_point_id = int(md5(f"{video_id}_{chunk_index}")[:12], 16)` — 48 bits, collision probability at this scale ~1e-9, effectively zero. Verified: **all 768 hash-ID points match this function exactly.** Batched uploads of 100 with per-batch error capture, post-upload count verification, built-in smoke-test query.

**`ask.py`** (326 lines) — Complete RAG loop, genuinely well-defended: validates API key, Qdrant dir, collection existence, non-zero point count, embedding dimension vs collection dimension, question length, empty Gemini responses, and retries twice. `TOP_K=4`, `MIN_SCORE_THRESHOLD=0.25`, prompt instructs context-only answering. Two defects (below).

---

## D. DATA SCHEMAS (verified from real files)

Metadata — `Data/Metadata/{video_id}.json`:
```json
{ "video_id": "EAR7De6Goz4",
  "video_title": "C++ Basics in One Shot - Strivers A2Z DSA Course - L1",
  "youtube_url": "https://youtu.be/EAR7De6Goz4",
  "audio_filename": "EAR7De6Goz4.mp3" }
```

Transcript — `Data/Transcripts/{video_id}.json`: `[{ "start": float, "end": float, "text": str }, ...]`

Embeddings — `Data/embeddings/{video_id}_embeddings.json`: array of
`{ chunk_index:int, video_id:str, video_title:str, youtube_url:str, start:float, end:float, text:str, embedding:float[384] }`

Qdrant point: `id` = 48-bit md5 int; payload = the same fields **minus** `embedding`. Collection `striver_a2z`, size 384, distance Cosine.

---

## E. STRENGTHS (keep these, do not refactor them)

- **Verified zero data loss** transcript → embedding across all 30 files.
- **Deterministic, idempotent point IDs** — re-running the loader overwrites rather than duplicates. All 768 verified correct.
- **Additive loading** — no wipe-and-reload, so multi-video scaling is safe.
- **Atomic writes** (`.tmp` + `os.replace`) in both transcript and embedding stages — no half-written JSON.
- **Resumability everywhere** — skip-if-exists at download, transcribe, and embed stages.
- **Genuinely good error handling in `ask.py`,** including the dimension-vs-collection sanity check, which is a mistake most people only find in production.
- **Timestamp offset restoration** for windowed audio — conceptually the trickiest part of the pipeline, and it is right.
- **Temp window cleanup works** — `_windows_temp/` is absent after runs.

---

## F. CONFIRMED PROBLEMS

### F1. `transcribe_audio.py` is syntactically invalid — CRITICAL

`python -c "import ast; ast.parse(open('Backend/transcribe_audio.py').read())"` → `SyntaxError: invalid syntax` at line 18.

Three distinct markdown-paste artifacts:

| Artifact | Count | Where | Should be |
|---|---|---|---|
| Bold-eaten dunders | 2 | L18 `**file**`, L1167 `if **name** == "**main**":` | `__file__`, `__name__`/`__main__` |
| Literal ``` fences | 16 | L87, 95, 103, 124, 128, 156, 166, 201, 211, 238, 247, 312, 327, 648, 658, 1159 | deleted |
| Flattened indentation | 3 top-level defs | bodies sitting at column 0 | re-indented |

The mechanism is unambiguous: `__file__` became `**file**` because markdown read the double underscores as bold. This came out of a chat window, not an editor.

Encouraging detail: indentation *inside* the fenced regions survived (46.5% of lines are still indented). The fences wrap correctly-indented blocks. So this is a mechanical repair of ~21 specific locations, not a rewrite of 1168 lines. **The logic is recoverable.**

### F2. Every source link `ask.py` prints is malformed — HIGH

`ask.py:230` — `timestamp_link = f"{url}&t={start}"`, and metadata stores `youtube_url` as `https://youtu.be/{id}`.

Result: `https://youtu.be/EAR7De6Goz4&t=170`

There is no `?` in that URL, so `&t=170` is parsed as part of the *path*, not a query parameter. YouTube sees a video ID of `EAR7De6Goz4&t=170`, which does not exist. **This breaks all source links, not just the title-keyed one.** You had noticed "at least one" malformed link — it is actually all of them.

Correct forms: `https://youtu.be/{id}?t={s}` or `https://www.youtube.com/watch?v={id}&t={s}s`.

Same defect at `load_to_qdrant.py:315` in the smoke-test output.

### F3. 118 of 827 Qdrant points are redundant copies of one video — HIGH

C++ Basics L1 exists **three times**:

| # | video_id in payload | Point IDs | Count | Source |
|---|---|---|---|---|
| 1 | `EAR7De6Goz4` | **sequential 0–58** | 59 | legacy loader run — orphaned |
| 2 | `EAR7De6Goz4` | md5 hashes | 59 | correct, from `EAR7De6Goz4_embeddings.json` |
| 3 | `C++ Basics in One Shot - Strivers A2Z DSA Course - L1` | md5 hashes | 59 | title-keyed file, **malformed URL** |

177 points where there should be 59.

Two independent causes:

**(a) Legacy sequential IDs.** 59 points carry IDs `0`–`58`. Your current `make_point_id()` produces 48-bit hashes. So an *earlier version* of `load_to_qdrant.py` used a plain loop counter as the ID. Because upsert matches on ID, those old points can never be overwritten by the current loader — they are permanent orphans, unreachable from any file on disk. This is the exact reason `827 − 768 = 59`.

**(b) Title-keyed duplicate identity.** `Data/Transcripts/C++ Basics in One Shot - Strivers A2Z DSA Course - L1.json` and `Data/embeddings/C++ Basics in One Shot...*_embeddings.json` exist alongside the correct `EAR7De6Goz4` versions. Verified byte-identical: same 59 chunks, `identical text lists? True`. Because no `Data/Metadata/C++ Basics...json` exists, `generate_embeddings.py` hit its fallback and stamped `youtube_url = "https://youtu.be/C++ Basics in One Shot - Strivers A2Z DSA Course - L1"`.

`download_audio.py` cannot produce title-named files, so this is **legacy debris from an early manual/single-video experiment, not a live bug.** It will not recur — but it will keep polluting retrieval until removed.

**Why this actually hurts:** `TOP_K = 4`. A C++ basics question can burn 3 of 4 context slots on identical text, and can cite the record with the broken URL. This is precisely the "duplicate-looking retrieval results" you observed. Your instinct not to assume corruption was right — the DB is not corrupt, it has debris.

### F4. Your 827-point verification was a false positive — process lesson

`673 + 154 = 827` matched, and you reasonably concluded the pipeline was healthy. It was healthy *with respect to record loss* — that conclusion holds. But count arithmetic cannot detect duplicates, and 59 orphans + 59 title-keyed dupes were already sitting in the collection at the time.

The lesson worth keeping: **count checks prove nothing was lost; they do not prove nothing extra is present.** Distinctness checks are a separate assertion. Both are needed.

### F5. No version control — CRITICAL (process)

The parent `.git` belongs to an unrelated repo (`Day 2: First Python Program`). `git ls-files` returns nothing for this project. No `.bak`, `~`, `.orig`, or `.tmp` copies exist anywhere.

This is *why* F1 is critical rather than a five-second undo. A 1168-line file was destroyed by one paste with no recovery path.

---

## G. POTENTIAL RISKS (not currently breaking)

**G1. Path case mismatch.** All scripts use lowercase `"data"`, `"audio"`, `"transcripts"`, `"metadata"`; disk has `Data/`, `Audio/`, `Transcripts/`, `Metadata/` (only `embeddings/` matches). Windows is case-insensitive so this works today. It will break instantly on Linux, Docker, WSL, or any cloud deploy.

**G2. Silent metadata fallback.** `generate_embeddings.py:30-37` prints a warning then continues with a fabricated URL. It already caused F3(b). A missing metadata file should arguably be a hard skip, not a silent fallback, because the bad data is indistinguishable downstream.

**G3. Qdrant local mode is single-process.** `QdrantClient(path=...)` takes an exclusive lock. `ask.py` opens Qdrant at import time, so it and `load_to_qdrant.py` can never run concurrently. Fine now; a real constraint the moment you add the frontend from ROADMAP Tier 2.6.

**G4. `ask.py` does no deduplication.** No filtering on repeated `(video_id, chunk_index)` or near-identical text. Cleaning the DB fixes today's symptom; a dedup pass in retrieval would make it structurally impossible.

**G5. `GEMINI_MODEL = "gemini-3.6-flash"` — UNVERIFIED.** I could not reach the network to confirm this model ID, and it postdates my training. It does not match the naming pattern I know (1.5/2.0/2.5 flash/pro), but you report the RAG loop works end-to-end, which suggests it resolves fine. **Not listed as a bug** — just verify it if you ever see API errors.

**G6. 3 stalled `.webm.part` files.** No mp3 was produced, so `already_downloaded()` returns False and those videos will be retried correctly. Harmless, but they are dead bytes and could confuse a future audit.

**G7. Untranscribed backlog.** 6 mp3s have no transcript. Expected given F1 — transcription cannot run. Worth confirming they are pending-by-interruption and not pending-by-failure once the script is fixed.

---

## H. UNNECESSARY COMPLEXITY

Very little — this is a lean codebase and you should not let anyone talk you into a framework rewrite.

- **`load_to_qdrant.py` Steps 7–8** load a second `SentenceTransformer` purely to run one smoke-test query, roughly doubling the script's memory footprint for a debug convenience. Reasonable while scaling; a `--test` flag would be cleaner.
- **`ask.py` does all setup at module import** rather than inside `main()`. It makes the script un-importable and un-testable, and grabs the Qdrant lock on import. Refactor only when you build the frontend.
- **No LangChain / agent framework anywhere.** Correct call. Do not add one. You currently understand every line of your retrieval path, which is worth more than any abstraction — and it is exactly what makes this defensible in an interview.

---

## I. SAFEST NEXT STEP

Strict order. One change, one test, one verification.

**Step 0 — `git init` before touching anything.** Non-negotiable. F1 cost you a 1168-line file because this did not exist. Add a `.gitignore` for `venv/`, `Data/Audio/*.mp3`, `qdrant_db/`, then commit the current state *including the broken file* so the corruption itself is recoverable while we repair it.

**Step 1 — repair `transcribe_audio.py`.** ~21 mechanical fixes: delete 16 fence lines, restore 2 dunders, re-indent 3 def bodies. Validate with `ast.parse` first, then re-transcribe one *already-completed* video to a scratch directory and diff the chunk count against its existing transcript. If a known video reproduces its existing chunk count, the logic survived the corruption intact. Do not run the 6-video backlog until that diff is clean.

**Step 2 — fix the URL bug.** One-line change in `ask.py:230` (`&t=` → `?t=`), same at `load_to_qdrant.py:315`. Verify by clicking a printed link. Independent of everything else; safe to do anytime.

**Step 3 — clean the 118 redundant points.** Only after Steps 0–2, and back up `qdrant_db/` first. Delete the 59 sequential-ID points (`id < 100000` is a clean, safe selector — no legitimate hash ID is that small) and the 59 points whose `video_id` contains a space. Then delete the two title-keyed files from `Transcripts/` and `embeddings/`. Verify: 827 → 709, and 709 should equal your distinct-text count. That is a real distinctness assertion, not just arithmetic.

**Step 4 — add a `verify_pipeline.py`.** Assert per-video: transcript chunks == embedding records == Qdrant points, plus zero duplicate `(video_id, chunk_index)`, zero duplicate text, zero URLs containing a space, and every `video_id` matching `^[A-Za-z0-9_-]{11}$`. Any of those checks would have caught F3 months ago. This is the highest-leverage file you do not yet have.

**Only then** resume ROADMAP work (batch scaling, then persona prompt, then topic tagging).

Do not start on the frontend, reranking, or topic tagging until Step 4 passes. Everything in ROADMAP Tier 1+ assumes a trustworthy index, and right now 14.3% of your index is noise.

---

## WHAT TO CORRECT IN YOUR OWN NOTES

- Phase 2 (RAG answering) is **built**, not pending. `ask.py` is a complete working loop.
- Folder names are `Backend/`, `Data/Audio/`, `Data/Transcripts/`, `Data/Metadata/`, `Data/embeddings/` — capitalised, and there is a metadata stage your architecture doc omits.
- "Malformed timestamp URL" is **all** links, not one.
- "Duplicate retrieval results" is confirmed, quantified at 118 points, root-caused, and fixable — the database is not corrupt.
- The 827-point integrity check was sound about *loss* and blind to *duplication*.
