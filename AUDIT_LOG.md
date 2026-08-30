# AUDIT LOG

A dated record of verification work and changes made to this repository.

`ARCHITECTURE_AUDIT.md` describes the **current state** of the system.
This file records **what was checked, what was found, and what was changed** — including the things that turned out to be fine, because knowing a check was run and came back clean is worth as much as knowing a bug was fixed.

Newest entry first.

---

## 2026-08-30 — Independent re-audit, integrity gate, git hygiene

**Ground rule for this pass:** verify everything against disk and the live database rather than trusting the previous audit's write-up. That rule paid for itself twice — the transcript counts and the mp3 counts had both moved since 2026-08-29, and one previously reported figure (132 mp3) was simply stale rather than wrong.

### What was checked

**1. Vector index integrity — the main event.**
All 1936 points were decoded out of `qdrant_db/collection/striver_a2z/storage.sqlite` and compared against every embedding record and every transcript chunk on disk.

| Assertion | Result |
|---|---|
| Qdrant points == embedding records == transcript chunks | 1936 == 1936 == 1936 |
| Per-video parity across all three stages | exact for all 130 videos, both directions |
| `chunk_index` contiguous 0..n-1 | holds for every video |
| Points with `id < 100000` (legacy counter orphans) | **0** (was 59) |
| Point ids failing `make_point_id(video_id, chunk_index)` | **0** |
| Duplicate `(video_id, chunk_index)` pairs | **0** |
| Payload key sets | uniform 7 keys across all 1936 |
| Vector dimensions | 1936/1936 at 384-dim, no NaN, no Inf |
| Malformed `youtube_url` values | **0** |

**Conclusion: the index is genuinely clean.** The earlier 118-point cleanup held, and nothing crept back in over 130 videos of subsequent processing.

**2. Backend consistency.**
All six scripts pass `ast.parse`. Zero markdown-paste artifacts (no literal fences, no bold-eaten dunders). Zero references to the retired hardcoded `VIDEO_METADATA` dict. Zero hardcoded absolute paths — an initial regex hit turned out to be `https:` matching an `[a-z]:[\\/]` pattern, i.e. a false positive, and was discarded rather than reported. Zero TODO/FIXME markers.

One genuine outlier found: `load_to_qdrant.py` has neither `def main()` nor an `if __name__ == "__main__":` guard, so ~290 lines execute on import. Every other script has both. Recorded as H1 in `ARCHITECTURE_AUDIT.md`, not fixed in this pass — one architectural change at a time.

**3. The path-capitalisation question, answered properly.**
Scripts use lowercase path literals (`"data", "audio"`); disk is `Data/Audio/`. The interesting part is that on Linux **this would not crash.** Because `os.makedirs(..., exist_ok=True)` runs at module scope, the scripts would silently *create* an empty lowercase tree beside the real one, `already_downloaded()` would return False for all 160 videos, and 4.3 GB would be re-downloaded and re-transcribed into the wrong directory. Qdrant would not duplicate, since deterministic ids overwrite. So the cost is wasted GPU-hours, not corruption — a much less alarming but much harder-to-notice failure than an exception. Recorded as G1; the fix is a decision, not yet applied.

**4. Git history — audio leak check.**

`git log --all --full-history -- "*.mp3"` returned **nothing**. Rather than stop at one glob, the full object graph was enumerated with `git rev-list --objects --all`:

- **405 distinct paths across all of history**: 391 json, 5 py, 3 md, 1 gitignore, 5 tree entries.
- **0 commits** touching any of `.mp3 .webm .m4a .wav .opus .ogg .flac .aac .mp4 .mkv .part .ytdl`.
- **0 paths** ever existing under `audio/`, `venv/`, `qdrant_db/`, `_windows_temp/` or `__pycache__/` in any capitalisation.
- Repo size `10.96 MiB` (427 loose objects, 0 packs — never gc'd). Largest blobs are all legitimate embeddings JSONs at 300–700 KB, which is the expected weight of 384 floats serialised as text.

**No audio has ever been committed.** One non-audio artifact does remain in history: `Data/embeddings/C++ Basics in One Shot - Strivers A2Z DSA Course - L1_embeddings.json`, 711.9 KB, the old title-keyed debris file, added in `4825095` and deleted in `5fe1e8a`. It is the only committed path containing a space. **Deliberately left in place** — rewriting published history and force-pushing to reclaim 0.7 MB is a poor trade.

### What was changed

| File | Change | Why |
|---|---|---|
| `Backend/ask.py` | line 230: `f"{url}&t={start}"` → `f"{url}?t={start}"` | `youtube_url` is `https://youtu.be/{id}` with no `?`, so `&t=170` parsed as part of the path and YouTube saw a nonexistent video id. This broke **every** source link, not some of them. |
| `Backend/load_to_qdrant.py` | line 315: same `&t=` → `?t=` | Same bug in the loader's smoke-test output. |
| `.gitignore` | lowercase patterns → `[Dd]ata/[Aa]udio/` character ranges, with a comment explaining why | The old lowercase `data/audio/` matched the capitalised `Data/Audio/` **only** because Windows git defaults `core.ignorecase=true`. Proven with `git check-ignore -v` under `core.ignorecase=false`: on a Linux clone or CI runner, 4.3 GB of audio would have been staged. The ignore was working by accident. |
| `Backend/verify_pipeline.py` | **new file, 554 lines** | Permanent post-batch integrity gate. See below. |
| `Backend/*.py`, `.gitignore` | CRLF → LF on the 6 touched files | HEAD stores LF, disk was CRLF, so committing as-is would have shown `ask.py` as an entire-file rewrite instead of a one-line change. All six still compile. |
| `ARCHITECTURE_AUDIT.md` | rewritten to current state; 2026-08-29 findings moved to a "resolved history" section | It still claimed the transcriber was broken and quoted 827 points / 30 transcripts. A stale audit is worse than no audit. |
| `AUDIT_LOG.md` | this file | Durable record of the above. |

Net diff versus HEAD: **4 files changed, 135 insertions, 128 deletions** — of which 123/123 is the `ARCHITECTURE_AUDIT.md` rewrite and only **2 lines are functional code**. Plus 2 new files. Small, which is the point.

A side observation worth keeping: `git status` reports **397** modified files, `git diff` reports **394**, and only **4** are real. The 394-vs-4 gap is line endings. The 397-vs-394 gap is subtler — `git status` normally refreshes the index's stat cache as it runs, and it cannot do that while `index.lock` exists, so three unchanged files stay flagged from stale mtime data. **A blocked lock does not just stop writes, it degrades the accuracy of reads.** Worth knowing before trusting a `git status` taken under a lock.

### `verify_pipeline.py` — why it exists

The 2026-08-29 audit had passed a verification that read `673 + 154 = 827`, matched, and declared the pipeline healthy. It was wrong — 118 redundant points were sitting in the database at that moment.

The lesson is precise and worth restating: **a count check proves nothing was lost. It does not prove nothing extra is present.** Duplication and loss are different failures, and the arithmetic that catches one is blind to the other. Distinctness has to be asserted separately.

So `verify_pipeline.py` checks identity, not totals: per-video parity across all three stages, point ids reproducing from `make_point_id`, zero duplicate `(video_id, chunk_index)`, `chunk_index` continuity, legacy-id detection, payload schema uniformity, URL well-formedness, and vector dim plus NaN/Inf. It exits 1 on any FAIL so it can gate a batch.

Validated two ways:

1. Against all 1936 live points → 0 FAIL, 1 WARN, 3 INFO, exit 0.
2. Against **18 injected synthetic faults → all 18 caught, clean baseline, no false positives.** The injected faults deliberately included the three real failures from this project's own history: sequential legacy ids, title-keyed `video_id`, and fabricated URLs containing spaces.

It also earned one fix during development. Its first run flagged four in-use `_windows_temp` window files as debris — a false positive, because a transcription run was live. An mtime age check now distinguishes a run in progress (INFO) from an interrupted one (WARN). **A verifier that cries wolf trains you to ignore it**, which makes a noisy check worse than no check at all.

### Findings handed over rather than acted on

Three things were found that need a human decision or a Windows-side action:

- **A stale 0-byte `.git/index.lock` from 17:14** is blocking every `git add` and `git commit`. Not deleted here — you never remove another process's lock file on its behalf.
- **390 files show as modified with zero real content changes** — pure CRLF churn, verified per-file with `git diff --ignore-cr-at-eol`. Needs `.gitattributes` plus `git add --renormalize .`, which rewrites all 390 tracked JSONs, so it must wait until no transcription run is writing to `Data/`.
- **`origin` still points at `DSA-Revision---RAG.git`** after the GitHub rename. Pushes still work via GitHub's redirect, but the URL should be corrected.
- **46 untracked data files** (30 `Data/Metadata/`, 16 `Data/Transcripts/`) were sitting in the working tree from an in-flight batch. They are legitimate output, but they are **not** part of this audit — a `git add -A` would silently fold a data batch into the audit commit. Stage by explicit path instead.

Nothing was committed. Git identity is not visible from the audit environment, so any commit made there would be misattributed; commits are the repository owner's to author.

---

## 2026-08-29 — First read-only architecture audit

Full read of the five backend scripts then in existence, all transcript and embedding files, and the vector database, with no modifications. Produced `ARCHITECTURE_AUDIT.md`.

Headline findings, all since resolved and preserved in section J of that document: `transcribe_audio.py` was syntactically invalid (1168 lines, corrupted by pasting code out of a rendered markdown view); every source link was malformed via `&t=`; 118 of 827 Qdrant points were redundant, including 59 permanently orphaned points from a loader that used a loop counter as the point id; the project had no version control at all; and the 827-point verification that had declared everything healthy was a false positive.
