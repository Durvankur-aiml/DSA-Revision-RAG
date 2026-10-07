"""ALGOFORGE Deterministic Grounding & Citation Verifier.

Validates LLM-generated answers against retrieved source sets:
1. Citation syntax, brackets, and indices.
2. Source metadata, validity, bounds, and YouTube timestamp URLs.
3. Transcript evidence overlap and unsupported technical claims.
4. Tri-state status determination (PASS, WARNING, FAIL) with bounded safe recovery.

Zero LLM calls, zero network requests, zero Qdrant writes, fully deterministic.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import CitationItem, CitationVerificationResult, VerificationStatus

# Standard English stopwords to filter out when assessing evidence overlap
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves", "also", "using",
    "used", "given", "step", "steps",
}

# Technical DSA terms indicating factual algorithmic claims
_DSA_TECHNICAL_TERMS: Set[str] = {
    "binary", "search", "complexity", "pointer", "pointers", "sliding",
    "window", "dp", "dynamic", "programming", "tree", "node", "nodes",
    "graph", "queue", "stack", "recursion", "recursive", "divide", "conquer",
    "hash", "hashing", "linked", "list", "lists", "sort", "sorting", "merge",
    "quick", "heap", "trie", "matrix", "traversal", "dfs", "bfs", "backtrack",
    "backtracking", "greedy", "bit", "array", "arrays", "element", "elements",
    "linear", "logarithmic", "exponential", "quadratic", "asymptotic",
}

# Regex for fenced code blocks (```...```) or inline code (`...`)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

# Regex to detect bracketed citation candidates not preceded by an identifier (e.g. avoid arr[i])
_CITATION_BRACKET_RE = re.compile(r"(?<![a-zA-Z0-9_])\[([^\]\r\n]+)\]")

# Known valid citation number patterns inside brackets:
# e.g., "1", "Source 1", "source 2", "1, 2", "Source 1, Source 2", "1: 45s", "Source 1 at 30s"
_TIMESTAMP_CITATION_RE = re.compile(
    r"^(?:Source\s*)?(\d+)\s*(?::|\s*at\s*)\s*(\d+(?:\.\d+)?)\s*s?$",
    re.IGNORECASE,
)
_MULTI_NUMERIC_CITATION_RE = re.compile(
    r"^(?:(?:Source\s*)?\d+\s*,\s*)*(?:Source\s*)?\d+$",
    re.IGNORECASE,
)
_SINGLE_NUMERIC_RE = re.compile(r"\b(\d+)\b")

# Definite malformed indicator inside citation brackets
_MALFORMED_INDICATOR_RE = re.compile(
    r"^(?:Source|source|ref|Ref|citation)\s*[^0-9\s].*$",
    re.IGNORECASE,
)


def _tokenize_text(text: str) -> Set[str]:
    """Extract lowercase alpha tokens with stopwords removed."""
    words = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _find_containing_sentence(text: str, start_idx: int, end_idx: int) -> str:
    """Find the exact sentence boundary around a character span."""
    left = 0
    for m in re.finditer(r"(?<=[.!?\n])\s+", text[:start_idx]):
        left = m.end()

    right = len(text)
    m = re.search(r"[.!?\n]", text[end_idx:])
    if m:
        right = end_idx + m.end()

    return text[left:right].strip()


def _extract_source_meta(source: Any) -> Dict[str, Any]:
    """Extract standard source metadata from a ScoredPoint, dict, or object."""
    if hasattr(source, "payload") and isinstance(source.payload, dict):
        payload = source.payload
        point_id = getattr(source, "id", None)
    elif isinstance(source, dict):
        payload = source
        point_id = source.get("id") or source.get("point_id")
    else:
        payload = getattr(source, "__dict__", {})
        point_id = getattr(source, "id", None)

    video_id = payload.get("video_id") or ""
    video_title = payload.get("video_title") or getattr(source, "title", "")
    youtube_url = payload.get("youtube_url") or getattr(source, "url", "")
    text = payload.get("text") or ""
    
    raw_start = payload.get("start")
    if raw_start is None:
        raw_start = getattr(source, "timestamp", 0)
    
    raw_end = payload.get("end")
    raw_duration = payload.get("duration")

    try:
        start = float(raw_start) if raw_start is not None else 0.0
    except (TypeError, ValueError):
        start = -1.0

    try:
        end = float(raw_end) if raw_end is not None else None
    except (TypeError, ValueError):
        end = -1.0

    try:
        duration = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError):
        duration = None

    # Construct clean YouTube URL if missing or format timestamp
    constructed_url = ""
    if youtube_url:
        separator = "&" if "?" in youtube_url else "?"
        constructed_url = f"{youtube_url}{separator}t={int(max(0, start))}"
    elif video_id:
        constructed_url = f"https://youtu.be/{video_id}?t={int(max(0, start))}"

    return {
        "id": point_id,
        "video_id": video_id,
        "video_title": video_title,
        "youtube_url": constructed_url or youtube_url,
        "raw_youtube_url": youtube_url,
        "text": text,
        "start": start,
        "end": end,
        "duration": duration,
        "payload": payload,
    }


class CitationVerifier:
    """Deterministic grounding and citation verifier for ALGOFORGE."""

    def __init__(self, min_evidence_overlap: float = 0.05):
        self.min_evidence_overlap = min_evidence_overlap

    def verify(
        self,
        answer: str,
        sources: Optional[List[Any]] = None,
        citation_meta: Optional[Dict[str, Any]] = None,
    ) -> CitationVerificationResult:
        """Verify the generated answer against the retrieved sources."""
        t0 = time.monotonic()
        sources = sources or []
        warnings: List[str] = []
        errors: List[str] = []
        citations: List[CitationItem] = []
        invalid_tags_to_strip: List[str] = []

        # ----------------------------------------------------
        # 0. Basic Input Checks
        # ----------------------------------------------------
        if not answer or not answer.strip():
            warnings.append("Answer is empty.")
            return CitationVerificationResult(
                status=VerificationStatus.WARNING,
                valid=True,
                citation_count=0,
                valid_citation_count=0,
                invalid_citation_count=0,
                unsupported_claim_count=0,
                malformed_citation_count=0,
                warnings=warnings,
                cleaned_answer=answer,
                latency_seconds=time.monotonic() - t0,
            )

        # ----------------------------------------------------
        # 1. Parse Citations Outside Code Blocks
        # ----------------------------------------------------
        # Mask code blocks so array lookups like arr[i] or dp[0] are not parsed
        code_spans: List[Tuple[int, int]] = [
            m.span() for m in _CODE_BLOCK_RE.finditer(answer)
        ]

        def _is_inside_code(start_pos: int, end_pos: int) -> bool:
            return any(s <= start_pos and end_pos <= e for s, e in code_spans)

        # Extract sentences for claim context
        sentence_delimiters = re.compile(r"(?<=[.!?])\s+|\n\n+")
        sentences = [s.strip() for s in sentence_delimiters.split(answer) if s.strip()]

        raw_matches = list(_CITATION_BRACKET_RE.finditer(answer))
        seen_citations: Set[Tuple[int, str]] = set()

        malformed_count = 0
        valid_citation_count = 0
        invalid_citation_count = 0

        for match in raw_matches:
            span_start, span_end = match.span()
            if _is_inside_code(span_start, span_end):
                continue

            raw_tag = match.group(0)
            inner = match.group(1).strip()

            # Find the exact sentence containing this citation instance
            claim_sentence = _find_containing_sentence(answer, span_start, span_end)

            # Check if this bracket is malformed (e.g. "[Source ?]", "[abc]", "[-1]")
            is_malformed = False
            if _MALFORMED_INDICATOR_RE.match(inner):
                is_malformed = True
            elif inner.startswith("-") and inner[1:].strip().isdigit():
                is_malformed = True
            elif not any(char.isdigit() for char in inner):
                # E.g. [abc] or [#] or empty
                # Only flag as malformed citation if it looks like a citation attempt or standalone bracket
                if any(k in inner.lower() for k in ("source", "ref", "cite")):
                    is_malformed = True
                else:
                    # Ignore arbitrary non-citation bracket notes like [note] unless explicitly malformed
                    continue

            if is_malformed:
                malformed_count += 1
                invalid_citation_count += 1
                errors.append(f"Malformed citation format: '{raw_tag}'")
                invalid_tags_to_strip.append(raw_tag)
                citations.append(
                    CitationItem(
                        raw_text=raw_tag,
                        source_index=None,
                        is_valid=False,
                        error_reason=f"Malformed citation syntax: '{raw_tag}'",
                        claim_text=claim_sentence,
                    )
                )
                continue

            # Check for timestamped citation e.g. [1: 45s]
            ts_match = _TIMESTAMP_CITATION_RE.match(inner)
            if ts_match:
                s_idx = int(ts_match.group(1))
                cited_ts = float(ts_match.group(2))
                extracted_indices = [(s_idx, cited_ts)]
            else:
                # Extract all numeric indices in bracket (handles [1], [Source 1], [1, 2], [Source 1, Source 2])
                nums = _SINGLE_NUMERIC_RE.findall(inner)
                if not nums:
                    continue
                extracted_indices = [(int(n), None) for n in nums]

            # Validate each extracted index
            for s_idx, embedded_ts in extracted_indices:
                cit_key = (s_idx, claim_sentence)
                is_duplicate = cit_key in seen_citations
                seen_citations.add(cit_key)

                item = CitationItem(
                    raw_text=raw_tag,
                    source_index=s_idx,
                    claim_text=claim_sentence,
                    start_seconds=embedded_ts,
                )

                if is_duplicate:
                    warnings.append(
                        f"Duplicate citation [{s_idx}] for the same claim."
                    )
                    item.warning_reason = "Duplicate citation"

                # Check index bounds against retrieved sources
                if s_idx < 1 or s_idx > len(sources):
                    item.is_valid = False
                    item.error_reason = (
                        f"Source index {s_idx} is out of bounds. "
                        f"Only {len(sources)} sources were retrieved."
                    )
                    invalid_citation_count += 1
                    errors.append(item.error_reason)
                    invalid_tags_to_strip.append(raw_tag)
                    citations.append(item)
                    continue

                # Source exists — extract metadata
                src_meta = _extract_source_meta(sources[s_idx - 1])
                item.source_payload = src_meta["payload"]
                item.youtube_url = src_meta["youtube_url"]

                # ----------------------------------------------------
                # 2. Source Metadata & Timestamp Validation
                # ----------------------------------------------------
                source_error = None

                # Video ID
                if not src_meta["video_id"] and not src_meta["raw_youtube_url"]:
                    source_error = (
                        f"Source {s_idx} missing video_id and youtube_url."
                    )

                # Start timestamp validity
                if src_meta["start"] < 0:
                    source_error = (
                        f"Source {s_idx} has invalid negative start timestamp "
                        f"({src_meta['start']})."
                    )

                # End timestamp validity (if present)
                if src_meta["end"] is not None and src_meta["end"] < 0:
                    source_error = (
                        f"Source {s_idx} has invalid negative end timestamp "
                        f"({src_meta['end']})."
                    )

                # start < end constraint
                if (
                    src_meta["end"] is not None
                    and src_meta["start"] >= src_meta["end"]
                ):
                    source_error = (
                        f"Source {s_idx} has start timestamp ({src_meta['start']}s) "
                        f">= end timestamp ({src_meta['end']}s)."
                    )

                # Timestamp within known duration
                if (
                    src_meta["duration"] is not None
                    and src_meta["duration"] > 0
                    and src_meta["start"] > src_meta["duration"]
                ):
                    source_error = (
                        f"Source {s_idx} start timestamp ({src_meta['start']}s) "
                        f"exceeds known video duration ({src_meta['duration']}s)."
                    )

                # Embedded timestamp check (if citation contained [1: 500s])
                if embedded_ts is not None and src_meta["duration"] is not None:
                    if embedded_ts > src_meta["duration"]:
                        source_error = (
                            f"Cited timestamp ({embedded_ts}s) exceeds source "
                            f"{s_idx} duration ({src_meta['duration']}s)."
                        )

                if source_error:
                    item.is_valid = False
                    item.error_reason = source_error
                    invalid_citation_count += 1
                    errors.append(source_error)
                    invalid_tags_to_strip.append(raw_tag)
                    citations.append(item)
                    continue

                # ----------------------------------------------------
                # 3. Evidence Overlap Validation
                # ----------------------------------------------------
                claim_tokens = _tokenize_text(claim_sentence)
                source_tokens = _tokenize_text(src_meta["text"])
                title_tokens = _tokenize_text(src_meta["video_title"])
                evidence_pool = source_tokens | title_tokens

                if claim_tokens:
                    overlap = claim_tokens & evidence_pool
                    overlap_ratio = len(overlap) / len(claim_tokens)
                else:
                    overlap_ratio = 1.0

                item.overlap_score = overlap_ratio

                # If claim is substantive and overlap is zero -> unsupported
                if len(claim_tokens) >= 3 and overlap_ratio == 0.0:
                    item.is_valid = False
                    item.error_reason = (
                        f"Citation [{s_idx}] is not supported by source {s_idx} "
                        f"transcript evidence (zero keyword overlap)."
                    )
                    invalid_citation_count += 1
                    errors.append(item.error_reason)
                    invalid_tags_to_strip.append(raw_tag)
                elif overlap_ratio < self.min_evidence_overlap and len(claim_tokens) >= 3:
                    item.warning_reason = (
                        f"Citation [{s_idx}] has weak transcript overlap ({overlap_ratio:.2f})."
                    )
                    warnings.append(item.warning_reason)
                    valid_citation_count += 1
                else:
                    item.is_valid = True
                    valid_citation_count += 1

                citations.append(item)

        # ----------------------------------------------------
        # 4. Global Unsupported DSA Claims Check
        # ----------------------------------------------------
        total_citations = len(citations)
        unsupported_claim_count = 0

        if total_citations == 0:
            # Check if answer makes substantive DSA technical claims without any citations
            answer_tokens = _tokenize_text(answer)
            dsa_matches = answer_tokens & _DSA_TECHNICAL_TERMS
            if len(dsa_matches) >= 2:
                unsupported_claim_count = len(dsa_matches)
                warnings.append(
                    f"Answer contains technical DSA claims ({', '.join(sorted(list(dsa_matches))[:4])}) "
                    "without source citations."
                )

        # ----------------------------------------------------
        # 5. Tri-State Status Determination
        # ----------------------------------------------------
        if errors or invalid_citation_count > 0 or malformed_count > 0:
            status = VerificationStatus.FAIL
            valid = False
        elif warnings or unsupported_claim_count > 0:
            status = VerificationStatus.WARNING
            valid = True
        else:
            status = VerificationStatus.PASS
            valid = True

        # ----------------------------------------------------
        # 6. Bounded Safe Recovery (Cleaned Answer)
        # ----------------------------------------------------
        cleaned_answer = answer
        if invalid_tags_to_strip:
            # Safely strip invalid citation markers without throwing or modifying source text destructively
            for tag in set(invalid_tags_to_strip):
                cleaned_answer = cleaned_answer.replace(tag, "")
            # Clean up extra double spaces before punctuation
            cleaned_answer = re.sub(r" +([.,!?;])", r"\1", cleaned_answer)
            cleaned_answer = re.sub(r"  +", " ", cleaned_answer).strip()

        latency = time.monotonic() - t0

        return CitationVerificationResult(
            status=status,
            valid=valid,
            citation_count=total_citations,
            valid_citation_count=valid_citation_count,
            invalid_citation_count=invalid_citation_count,
            unsupported_claim_count=unsupported_claim_count,
            malformed_citation_count=malformed_count,
            citations=citations,
            warnings=warnings,
            errors=errors,
            cleaned_answer=cleaned_answer,
            latency_seconds=latency,
        )


# Singleton verifier instance for direct functional use
_DEFAULT_VERIFIER = CitationVerifier()


def verify_citations(
    answer: str,
    sources: Optional[List[Any]] = None,
    citation_meta: Optional[Dict[str, Any]] = None,
) -> CitationVerificationResult:
    """Convenience functional interface to verify citations."""
    return _DEFAULT_VERIFIER.verify(answer, sources=sources, citation_meta=citation_meta)
