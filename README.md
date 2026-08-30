# DSA Revision — RAG

A Retrieval-Augmented Generation system built on top of Striver's A2Z DSA Course (YouTube playlist, 316 videos). Ask any DSA question in plain English and get an answer grounded in the actual lecture content — complete with the source video and a clickable timestamp.

## What it does

1. Downloads audio from a YouTube playlist in resumable batches
2. Transcribes each video using OpenAI Whisper (GPU-accelerated, windowed to avoid long-form transcription drift)
3. Chunks transcripts (~90s segments) and filters out sponsor/promo content
4. Generates embeddings for each chunk (`all-MiniLM-L6-v2`, 384-dim)
5. Stores vectors in a local Qdrant database with deterministic per-video IDs
6. On a question: embeds the query, retrieves the most relevant chunks, and asks Gemini to answer using *only* that retrieved context
7. Returns a grounded answer with citations — video title, timestamp, and a direct YouTube link

```
YouTube Playlist
      ↓
   yt-dlp (audio download)
      ↓
   Whisper (GPU transcription, windowed)
      ↓
   Chunking + gap/sponsor filtering
      ↓
   Sentence-Transformers (embeddings)
      ↓
   Qdrant (vector database)
      ↓
Question → embed → retrieve → Gemini → grounded answer + source + timestamp
```

## Status

- **130 / 316 videos** processed and indexed (in progress, batch pipeline is fully resumable)
- Core pipeline validated end-to-end: retrieval correctly surfaces relevant chunks across multiple videos, answers are grounded (not hallucinated) and cite real sources

## Stack

- **Audio ingestion:** `yt-dlp`
- **Transcription:** OpenAI Whisper (`small`), CUDA-accelerated
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** Qdrant (local, persistent)
- **LLM:** Google Gemini API (free tier)
- **Language:** Python

## Why this exists

Most RAG tutorials stop at "index one PDF, ask one question." This project pushes further:
- Handles a real 300+ video playlist, not a toy dataset
- Solves an actual production-style bug (Whisper silently dropping ~75 seconds of speech during long-form transcription — root-caused via isolated audio testing, fixed with windowed transcription)
- Deterministic, collision-safe vector IDs so re-running the pipeline never duplicates data
- Resumable batch processing — safe to interrupt and restart at any point in a 300+ video pipeline

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for planned upgrades: persona-grounded answers, topic tagging, semantic chunking, reranking, and a frontend UI.

## Running it locally

```bash
# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # or install packages listed in the scripts

# Set your Gemini API key
setx GEMINI_API_KEY "your-key-here"

# Pipeline (run in order)
python backend/download_audio.py       # paste playlist URL, choose batch size/start
python backend/transcribe_audio.py     # processes new videos only
python backend/generate_embeddings.py  # processes new videos only
python backend/load_to_qdrant.py       # additive upload, safe to re-run

# Ask questions
python backend/ask.py
```

## Notes

This is a personal learning/portfolio project built as part of a self-directed "Mission Anthropic" AI/ML learning track. Not affiliated with Striver or his course — built purely for personal DSA revision using publicly available lecture content.
