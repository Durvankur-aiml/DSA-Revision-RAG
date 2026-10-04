"""ALGOFORGE retrieval evaluation package (offline, retrieval-only).

Layout:
    dataset.py    - golden dataset loading + fail-loud validation
    metrics.py    - Hit@K / Recall@K / MRR definitions (documented)
    runner.py     - drives the PRODUCTION retrieval functions, candidate
                    and final levels, failure classification, latency
    report.py     - baseline.json / baseline.md generation
    build_golden.py - one-time human-curated builder that resolves
                    title keys against Data/Metadata into golden.jsonl

This package NEVER calls Gemini and NEVER writes to Qdrant.
"""
