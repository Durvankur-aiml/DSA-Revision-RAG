# MISSION ANTHROPIC — ARCHITECTURE AUDIT

Original audit: 2026-08-29 (read-only)
**Revised: 2026-08-30** — re-verified independently against disk and the vector DB. Numbers and status below are current as of this revision; the 2026-08-29 findings are preserved in section J as resolved history.

Method: read all 6 backend scripts, `ast.parse` on each, audited every transcript and embedding file, decoded all 1936 Qdrant points, enumerated the full git object history.

Where something could not be verified, it is labelled **UNVERIFIED**.

---

## HEADLINE

**The pipeline is healthy and the index is clean.** Both critical problems from the 2026-08-29 audit are resolved:

1. `Backend/transcribe_audio.py` **runs**. It was a 1168-line markdown-paste wreck with a hard `SyntaxError`; it is now 303 lines and compiles. All six backend scripts pass `ast.parse`.
2. The **118 redundant Qdrant points are gone and have stayed gone** across 130 videos of subsequent processing. Zero legacy orphans, zero duplicate identities, zero title-keyed records.

Integrity is exact: **1936 Qdrant points == 1936 embedding records == 1936 transcript chunks**, with per-video parity holding for all 130 indexed videos.

Remaining work is hygiene, not architecture: git line-ending noise, a stale lock file, and a path-capitalisation decision that matters only when this leaves Windows.

---

## A. CURRENT ARCHITECTURE

```
download_audio.py      playlist → per-video mp3 + metadata json
        ↓
transcribe_audio.py    mp3 → windowed Whisper → ~90s chunks → transcript json
        ↓
generate_embeddings.py transcript + metadata → 384-dim vectors → embeddings json
        ↓
load_to_qdrant.py      embeddings json → deterministic-ID points → Qdrant (upsert, additive)
        ↓
ask.py                 question → embed → search → threshold filter → Gemini → answer + sources

verify_pipeline.py     [out of band] asserts integrity across every stage above
```

Phase 1 (ingestion) and Phase 2 (RAG answering) are both built and working.

Actual folder layout:

```
Mission Anthropic - RAG/
├── Backend/            6 scripts, 2166 lines
├── Data/
│   ├── Audio/          160 mp3 (4.3 GB, git-ignored)
│   ├── Metadata/       160 json   <- source of truth for title + url
│   ├── Transcripts/    143 json
│   └── embeddings/     130 json   (lowercase 'e')
├── Frontend/           EMPTY
├── qdrant_db/          collection 'striver_a2z', 1936 points
├── venv/
├── ROADMAP.md
├── README.md
├── AUDIT_LOG.md
└── ARCHITECTURE_AUDIT.md
```

`Data/Metadata/` is load-bearing and absent from the original design doc. It is the single source of truth for `video_title` and `youtube_url`.

---

## B. DATA FLOW & MEASURED COUNTS (2026-08-30)

| Stage | Count | Status |
|---|---|---|
| mp3 files | 160 | all named `{video_id}.mp3` |
| metadata json | 160 | every mp3 has one — no fallback risk |
| transcript json | 143 | climbing — a transcription run was live during the audit |
| embedding json | 130 | |
| transcript chunks (of the 130 embedded) | 1936 | |
| embedding records | 1936 | **exact match** |
| Qdrant points | 1936 | **exact match, zero orphans** |
| indexed videos | 130 | per-video parity exact for all |

Playlist target is 316 videos. 17 mp3 await transcription, 13 transcripts await embedding. Counts in this table are a moving snapshot — re-run `verify_pipeline.py` for live figures.

The identity chain `VIDEO_ID → mp3 → transcript → embeddings → Qdrant payload → source link` holds for every record with no exceptions.

---

## C. FILE RESPONSIBILITIES

**`download_audio.py`** (421 lines) — Clean. Interactive batch download with `batch_size` + `start_position`. Always names files `{video_id}.mp3` / `{video_id}.json`, never titles. Writes metadata only after the mp3 is confirmed present. `already_downloaded()` requires **both** mp3 and metadata before skipping.

**`transcribe_audio.py`** (303 lines) — **Working.** 300s windows, Whisper `small`, per-window offset restoration to global timestamps, 5s gap warnings (diagnostic only), per-video sponsor skip ranges (`EAR7De6Goz4: [(170, 250)]`), ~90s chunking, atomic temp-then-replace save, resumable via already-transcribed set.

**`generate_embeddings.py`** (225 lines) — Clean. `all-MiniLM-L6-v2`, 384-dim, CUDA with CPU fallback, batch 32, per-chunk validation, atomic save, skip-if-exists. Derives `video_id` from the transcript *filename*, then reads `Data/Metadata/{video_id}.json`. If metadata is missing it warns and falls back to a fabricated `https://youtu.be/{filename}` — see G2.

**`load_to_qdrant.py`** (337 lines) — Strong validation: 8 required keys, vector type and length, non-empty text. Additive upsert, never wipes. `make_point_id = int(md5(f"{video_id}_{chunk_index}")[:12], 16)` — 48 bits, collision probability ~1e-9 at this scale. All 1936 points verified against it. Batches of 100 with per-batch error capture, post-upload count check, built-in smoke query. **Structural outlier — see H1.**

**`ask.py`** (326 lines) — Complete RAG loop, well defended: validates API key, Qdrant dir, collection existence, non-zero point count, embedding dim vs collection dim, question length, empty Gemini responses, retries twice. `TOP_K=4`, `MIN_SCORE_THRESHOLD=0.25`, prompt instructs context-only answering.

**`verify_pipeline.py`** (554 lines) — **New 2026-08-30.** The post-batch integrity gate. Asserts distinctness, not just counts. See section K.

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

Qdrant point: `id` = 48-bit md5 int; payload = the same 7 fields **minus** `embedding`. Collection `striver_a2z`, size 384, distance Cosine. Verified uniform across all 1936 points.

---

## E. STRENGTHS (keep these, do not refactor them)

- **Verified zero data loss and zero duplication** — 1936 = 1936 = 1936, per-video, with distinctness separately asserted.
- **Deterministic, idempotent point IDs** — re-running the loader overwrites rather than duplicates. All 1936 verified.
- **Additive loading** — no wipe-and-reload, so scaling is safe.
- **Atomic writes** (`.tmp` + `os.replace`) in the transcript and embedding stages — no half-written JSON.
- **Resumability at every stage** — skip-if-exists at download, transcribe, and embed.
- **Timestamp offset restoration** for windowed audio — the trickiest part of the pipeline, and it is correct.
- **Good error handling in `ask.py`**, including the dimension-vs-collection check most people only discover in production.
- **No LangChain or agent framework anywhere.** Correct call. Every line of the retrieval path is understood, which is worth more than any abstraction.

---

## F. CONFIRMED PROBLEMS (open)

### F1. Stale `.git/index.lock` — blocks all commits

A 0-byte `.git/index.lock` dated 17:14 was left behind by a git process that died. Every `git add` and `git commit` fails until it is removed. Fix: confirm no git or IDE operation is running, then delete it.

### F2. No `.gitattributes` — 390 files show as modified with zero real changes

Files on disk are CRLF, HEAD stores LF, `core.autocrlf` is unset. Verified per-file with `git diff --ignore-cr-at-eol`: **394 files with a raw diff, of which only 4 are real — the other 390 differ by line endings alone.** This actively hides genuine changes in `git status`.

Fix: add `.gitattributes` (`* text=auto`, `*.py text eol=lf`, `*.json text eol=lf`), then `git add --renormalize .` — but only while no transcription run is active, since it rewrites all 390 tracked JSON files.

### F3. Stale remote URL after the GitHub rename

`origin` still points at `https://github.com/Durvankur-aiml/DSA-Revision---RAG.git`. GitHub redirects renamed repositories, so pushes still succeed, but the URL should be corrected to the current slug with `git remote set-url origin <new-url>`.

---

## G. POTENTIAL RISKS (not currently breaking)

**G1. Path case will misbehave on Linux — and it will not crash.** All six scripts use lowercase literals (`os.path.join(BASE_DIR, "data", "audio")`); disk is `Data/Audio/`, `Data/Transcripts/`, `Data/Metadata/`, `Data/embeddings/`. Windows is case-insensitive so this works today.

The failure mode on a case-sensitive filesystem is **silent duplicate work, not an error.** `download_audio.py` and `transcribe_audio.py` call `os.makedirs(..., exist_ok=True)` at module level, so they would *create* an empty lowercase `data/audio` tree alongside the real one. `already_downloaded()` would then return False for all 160 videos and re-download 4.3 GB, and transcription would start from zero into the parallel tree. Qdrant would not duplicate — deterministic IDs overwrite — so the cost is GPU-hours and bandwidth, not corruption.

Recommended fix: rename the disk directories to lowercase, so the scripts need no changes and `.gitignore` becomes literally correct. Requires a two-step `git mv --force` because `core.ignorecase=true` hides pure case renames.

**G2. Silent metadata fallback.** `generate_embeddings.py` warns then continues with a fabricated URL when metadata is missing. This is the mechanism that produced the historical title-keyed debris (J3). Currently harmless — all 160 mp3s have metadata — but a missing file should arguably be a hard skip, since the bad data is indistinguishable downstream.

**G3. Qdrant local mode is single-process.** `QdrantClient(path=...)` takes an exclusive lock. `ask.py`, `load_to_qdrant.py` and `verify_pipeline.py` can never run concurrently. `verify_pipeline.py` reports this explicitly rather than emitting a confusing stack trace. Becomes a real constraint at the frontend stage (ROADMAP Tier 2.6).

**G4. `ask.py` does no deduplication** on repeated `(video_id, chunk_index)` or near-identical text. The index is clean today, so nothing is wrong; retrieval-side dedup would make the historical failure structurally impossible.

**G5. `GEMINI_MODEL = "gemini-3.6-flash"` — UNVERIFIED.** No network egress available to confirm this model ID. It does not match the naming pattern of known releases, but the loop reportedly works end to end, which suggests it resolves. **Not a bug** — check only if API errors appear.

**G6. `_windows_temp/` survives an interrupted run.** `os.rmdir` cannot remove a non-empty directory and the failure is swallowed, so leftover window files persist silently. `verify_pipeline.py` distinguishes stale leftovers from a live run by file age.

---

## H. UNNECESSARY COMPLEXITY / INCONSISTENCY

This is a lean codebase and it should not be handed to a framework rewrite. Three specific deviations:

**H1. `load_to_qdrant.py` has no `def main()` and no `__main__` guard.** Roughly 290 lines execute at import. The other five scripts all follow `fail()` + `def main()` + guard. Concrete consequence: `verify_pipeline.py` must re-implement `make_point_id()` rather than import it, because importing would run the entire loader.

**H2. `ask.py` does all setup at module top level** (lines 45–115) rather than inside `main()`: Qdrant connect, model load, Gemini client. It has a `__main__` guard but grabs the Qdrant lock on import anyway, which makes it un-importable and un-testable. Worth fixing when the frontend arrives.

**H3. `load_to_qdrant.py` steps 7–8** load a second `SentenceTransformer` purely to run one smoke query, roughly doubling the script's memory footprint for a debug convenience. Reasonable while scaling; a `--test` flag would be cleaner.

---

## I. SAFEST NEXT STEP

Strict order. One change, one test, one verification.

1. **Let the running transcription batch finish.** Counts move while it runs.
2. **Clear the stale `.git/index.lock`,** then commit the pending changes (URL fix, `verify_pipeline.py`, `.gitignore`, these docs).
3. **Add `.gitattributes` and `git add --renormalize .`** to kill the 390-file CRLF noise. Only while nothing is writing to `Data/`.
4. **Decide the path-case fix** (G1). Lowercase rename recommended.
5. **Resume batches toward 316,** running `python Backend/verify_pipeline.py` after each one.
6. **Only then** resume ROADMAP work: persona-grounded prompt, then per-chunk topic tagging.

Do not start the frontend, reranking, or topic tagging before step 5 is habitual. Everything in ROADMAP Tier 1+ assumes a trustworthy index.

---

## J. RESOLVED HISTORY (from the 2026-08-29 audit)

Preserved because the failure modes are instructive, not because they are still live.

**J1. `transcribe_audio.py` was syntactically invalid — RESOLVED.** `SyntaxError` at line 18. Corrupted by pasting code out of a *rendered markdown view*, which produced three artifact types: bold-eaten dunders (`os.path.abspath(**file**)` instead of `__file__`, `if **name** == "**main**":`), 16 literal ``` fence lines, and 3 top-level `def` bodies flattened to column 0. The mechanism was unambiguous — markdown read `__file__`'s double underscores as bold emphasis. Now rewritten compactly at 303 lines and verified with `ast.parse`. **Root lesson: never paste code out of a rendered view; use a raw/copy button or a file.**

**J2. Every source link was malformed — RESOLVED.** `ask.py:230` built `f"{url}&t={start}"` while metadata stores `youtube_url` as `https://youtu.be/{id}`. With no `?` in the URL, `&t=170` parsed as part of the path, so YouTube saw a nonexistent video ID. This broke *all* links, not one. Fixed to `?t=` in `ask.py:230` and `load_to_qdrant.py:315`.

**J3. 118 of 827 Qdrant points were redundant — RESOLVED.** C++ Basics L1 existed three times (177 points where 59 belonged): 59 points with **sequential ids 0–58** from a legacy loader that used a loop counter — permanently orphaned, since upsert matches on id; 59 correct hash-ID points; and 59 hash-ID points keyed by *title* rather than video_id, carrying a fabricated malformed URL from the G2 fallback. `download_audio.py` cannot produce title-named files, so the third set was legacy debris from an early manual experiment, not a live bug. Cleaned; verified still clean 130 videos later.

**J4. The 827-point verification was a false positive — the key process lesson.** `673 + 154 = 827` matched, and the pipeline was declared healthy. That conclusion was sound about *record loss* and blind to *duplication* — 118 redundant points were already present. **Count checks prove nothing was lost; they do not prove nothing extra is present.** Distinctness is a separate assertion. This is the entire reason `verify_pipeline.py` exists.

**J5. No version control — RESOLVED.** The project had no git, which is why J1 cost a 1168-line file with no recovery path. Now 4 commits on `main` tracking `origin/main`.

**J6. `.gitignore` case bug — RESOLVED.** Patterns were lowercase (`data/audio/`) while the folders are capitalised (`Data/Audio/`). `git check-ignore -v` proved the match depended entirely on `core.ignorecase=true`, a Windows default: on a case-sensitive clone or CI runner, 4.3 GB of audio would have been staged. Now `[Dd]ata/[Aa]udio/`, verified to match with `core.ignorecase=false`.

**J7. Audio leak check — CLEAN.** Full history enumerated: zero commits have ever touched `.mp3`, `.webm`, `.m4a`, `.wav`, `.opus`, `.ogg`, `.flac`, `.aac`, `.mp4`, `.mkv`, `.part` or `.ytdl`, and no path under any `audio/`, `venv/`, `qdrant_db/`, `_windows_temp/` or `__pycache__/` directory has ever existed in any commit. Repo is 10.96 MiB across 405 historical paths. One 711.9 KB piece of J3 debris (`Data/embeddings/C++ Basics in One Shot - ...json`) does remain in history, added in `4825095` and removed in `5fe1e8a`; **deliberately not purged** — a history rewrite and force-push is a poor trade for 0.7 MB on a published repo.

---

## K. `verify_pipeline.py` — WHAT IT GUARANTEES

Run after every batch: `python Backend/verify_pipeline.py`. Exit code 0 = clean, 1 = failure, so it can gate a batch. Requires the Qdrant lock, so nothing else may hold it.

Three severity levels: **FAIL** (integrity broken), **WARN** (suspicious but legitimate cases exist), **INFO** (progress and backlog).

Checks performed:

1. Every transcript and embedding file parses.
2. Debris: `.part` / `.tmp` / stale `_windows_temp` contents, distinguished from a live run by file age.
3. Every `video_id` matches `^[A-Za-z0-9_-]{11}$` — catches title-keyed records (J3).
4. Every embedding record's `video_id` matches its filename.
5. All on-disk vectors are 384-dim.
6. **Per-video parity:** transcript chunks == embedding records == Qdrant points, individually for every video.
7. Every mp3 has a metadata json — pre-empts the G2 fallback.
8. Backlog reporting: videos awaiting transcription, embedding, or loading.
9. No point id below 100000 — catches legacy counter-based orphans (J3a).
10. Every point id reproduces from `make_point_id(video_id, chunk_index)`.
11. Zero duplicate `(video_id, chunk_index)` pairs.
12. `chunk_index` runs 0..n-1 contiguously per video.
13. Duplicate text — **WARN, escalating above 200 characters.** Short repeats across videos are legitimate; see the note below.
14. Payload schema uniform across all points, exactly the 7 expected keys.
15. URLs are well-formed https with no whitespace; non-canonical shapes warn.
16. All vectors 384-dim with no NaN or Inf, which would silently poison cosine similarity.

Validated two ways: against all 1936 live points, and against **18 injected synthetic faults, every one caught, with no false positives on a clean baseline.**

**Known benign warning:** the text `'I will find a light'` appears twice — `RlUu72JrOCQ` chunk 5 at 446.7s and `cHrH9CQ8pmY` chunk 8 at 710.8s. Two different videos, ~2 seconds each: Whisper transcribing background music. This is why check 13 is a warning rather than a failure.
