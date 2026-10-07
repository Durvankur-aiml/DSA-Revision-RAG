"""Deterministic tests for the Grounding & Citation Verification layer.

Tests all 18 required scenarios:
 1. valid answer with valid citations
 2. answer with no citations
 3. malformed citation
 4. citation pointing to nonexistent source
 5. citation index out of bounds
 6. duplicate citation
 7. invalid timestamp
 8. timestamp where start >= end
 9. timestamp outside known duration
10. citation whose transcript does not support the cited text
11. multiple valid citations
12. mixed valid + invalid citations
13. empty answer
14. answer containing code
15. normal DSA explanation
16. YouTube URL generation
17. verifier performs zero network calls
18. verifier performs zero LLM calls
"""

from __future__ import annotations

import socket
import pytest

from citations.models import VerificationStatus, CitationVerificationResult
from citations.verifier import CitationVerifier, verify_citations


@pytest.fixture
def sample_sources():
    """Deterministic sample retrieved source chunks."""
    return [
        {
            "id": "chunk_001",
            "video_id": "vid_bs_01",
            "video_title": "Binary Search Algorithm Deep Dive",
            "youtube_url": "https://www.youtube.com/watch?v=vid_bs_01",
            "text": "Binary search halves the search space at each iteration. It requires a sorted array to operate properly.",
            "start": 30.0,
            "end": 90.0,
            "duration": 600.0,
        },
        {
            "id": "chunk_002",
            "video_id": "vid_tp_02",
            "video_title": "Two Pointer Technique Explained",
            "youtube_url": "https://www.youtube.com/watch?v=vid_tp_02",
            "text": "Place the left pointer at the beginning and the right pointer at the end. Move pointers inward based on sum.",
            "start": 120.0,
            "end": 180.0,
            "duration": 800.0,
        },
        {
            "id": "chunk_003",
            "video_id": "vid_dp_03",
            "video_title": "Dynamic Programming Intro",
            "youtube_url": "https://www.youtube.com/watch?v=vid_dp_03",
            "text": "Memoization stores intermediate subproblem solutions in a cache or table to avoid redundant recursion.",
            "start": 15.0,
            "end": 75.0,
            "duration": 450.0,
        },
    ]


class TestCitationVerifier:
    """Deterministic verification test suite."""

    # 1. Valid answer with valid citations
    def test_valid_answer_with_valid_citations(self, sample_sources):
        answer = "Binary search repeatedly halves the search space on a sorted array [1]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.PASS
        assert result.valid is True
        assert result.citation_count == 1
        assert result.valid_citation_count == 1
        assert result.invalid_citation_count == 0
        assert len(result.errors) == 0
        assert result.citations[0].source_index == 1
        assert result.citations[0].is_valid is True

    # 2. Answer with no citations
    def test_answer_with_no_citations(self, sample_sources):
        answer = "Binary search operates in logarithmic time complexity using left and right pointers on a sorted array."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.WARNING
        assert result.valid is True
        assert result.citation_count == 0
        assert result.unsupported_claim_count > 0
        assert any("without source citations" in w for w in result.warnings)

    # 3. Malformed citation
    def test_malformed_citation(self, sample_sources):
        answer = "Binary search is the optimal approach [Source ?]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert result.malformed_citation_count >= 1
        assert result.invalid_citation_count >= 1
        assert "[Source ?]" not in result.cleaned_answer

    # 4. Citation pointing to nonexistent source
    def test_citation_pointing_to_nonexistent_source(self, sample_sources):
        # Only 3 sources provided; citation points to source 5
        answer = "Binary search space reduces quickly [5]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert result.invalid_citation_count == 1
        assert any("out of bounds" in err for err in result.errors)

    # 5. Citation index out of bounds (zero or negative or excessive)
    def test_citation_index_out_of_bounds(self, sample_sources):
        answer = "We partition the array in logarithmic time [99]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert result.invalid_citation_count == 1
        assert any("out of bounds" in err for err in result.errors)

    # 6. Duplicate citation
    def test_duplicate_citation(self, sample_sources):
        answer = (
            "Binary search halves the search space [1]. "
            "Binary search halves the search space [1]."
        )
        result = verify_citations(answer, sample_sources)

        assert any("Duplicate citation" in w for w in result.warnings)

    # 7. Invalid timestamp (e.g. negative timestamp)
    def test_invalid_timestamp(self):
        sources = [
            {
                "id": "bad_ts",
                "video_id": "vid_bad",
                "video_title": "Bad Timestamp Video",
                "text": "Some text content for search algorithms.",
                "start": -25.0,
                "end": 50.0,
            }
        ]
        answer = "Some text content for search algorithms [1]."
        result = verify_citations(answer, sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert any("negative start timestamp" in err for err in result.errors)

    # 8. Timestamp where start >= end
    def test_timestamp_where_start_ge_end(self):
        sources = [
            {
                "id": "inverted_ts",
                "video_id": "vid_inv",
                "video_title": "Inverted Timestamp Video",
                "text": "Some text content for search algorithms.",
                "start": 100.0,
                "end": 50.0,
            }
        ]
        answer = "Some text content for search algorithms [1]."
        result = verify_citations(answer, sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert any("start timestamp" in err and ">= end timestamp" in err for err in result.errors)

    # 9. Timestamp outside known duration
    def test_timestamp_outside_known_duration(self):
        sources = [
            {
                "id": "exceed_duration",
                "video_id": "vid_exceed",
                "video_title": "Short Video",
                "text": "Some text content for search algorithms.",
                "start": 400.0,
                "end": 450.0,
                "duration": 300.0,
            }
        ]
        answer = "Some text content for search algorithms [1]."
        result = verify_citations(answer, sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert any("exceeds known video duration" in err for err in result.errors)

    # 10. Citation whose transcript does not support the cited text
    def test_citation_whose_transcript_does_not_support_the_cited_text(self, sample_sources):
        # Claim is about Dijkstra's shortest path graph algorithm, but source 1 is about Binary Search
        answer = "Dijkstra computes the shortest path in a weighted graph using a min priority queue [1]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.FAIL
        assert result.invalid_citation_count >= 1
        assert any("not supported by source 1 transcript evidence" in err for err in result.errors)

    # 11. Multiple valid citations
    def test_multiple_valid_citations(self, sample_sources):
        answer = (
            "Binary search halves the search space at each iteration [1]. "
            "The two pointer technique places pointers at the ends and moves inward [2]."
        )
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.PASS
        assert result.valid is True
        assert result.citation_count == 2
        assert result.valid_citation_count == 2
        assert result.invalid_citation_count == 0

    # 12. Mixed valid + invalid citations
    def test_mixed_valid_plus_invalid_citations(self, sample_sources):
        answer = (
            "Binary search halves the search space at each iteration [1]. "
            "Quantum annealing solves SAT in polynomial time [88]."
        )
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.FAIL
        assert result.valid is False
        assert result.valid_citation_count == 1
        assert result.invalid_citation_count == 1
        # Safe recovery removes the phantom citation tag
        assert "[88]" not in result.cleaned_answer
        assert "[1]" in result.cleaned_answer

    # 13. Empty answer
    def test_empty_answer(self, sample_sources):
        result = verify_citations("", sample_sources)
        assert result.valid is True
        assert result.status == VerificationStatus.WARNING
        assert result.citation_count == 0
        assert any("empty" in w.lower() for w in result.warnings)

    # 14. Answer containing code
    def test_answer_containing_code(self, sample_sources):
        answer = (
            "Here is the standard binary search implementation:\n"
            "```python\n"
            "def binary_search(arr, target):\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
            "```\n"
            "The array must be sorted to halve the search space [1]."
        )
        result = verify_citations(answer, sample_sources)

        # Code brackets `arr[mid]` should NOT be counted as citations
        assert result.status == VerificationStatus.PASS
        assert result.citation_count == 1
        assert result.malformed_citation_count == 0
        assert result.citations[0].source_index == 1

    # 15. Normal DSA explanation
    def test_normal_dsa_explanation(self, sample_sources):
        answer = (
            "### Binary Search Overview\n\n"
            "Binary search is an efficient search algorithm designed for sorted collections [1]. "
            "By repeatedly halving the search space, it achieves logarithmic runtime [1].\n\n"
            "### Two Pointers\n"
            "For two-sum variants, placing pointers at the beginning and end allows linear scans [2]."
        )
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.PASS
        assert result.valid is True
        assert result.citation_count == 3
        assert result.valid_citation_count == 3
        assert result.invalid_citation_count == 0

    # 16. YouTube URL generation
    def test_youtube_url_generation(self):
        sources = [
            {
                "id": "c1",
                "video_id": "M3_p_test",
                "video_title": "Test Title",
                "text": "Algorithm searches through sorted elements.",
                "start": 45.0,
                "end": 90.0,
            }
        ]
        answer = "Algorithm searches through sorted elements [1]."
        result = verify_citations(answer, sources)

        assert result.status == VerificationStatus.PASS
        assert result.citations[0].youtube_url == "https://youtu.be/M3_p_test?t=45"

    # 17. Verifier performs zero network calls
    def test_verifier_zero_network_calls(self, monkeypatch, sample_sources):
        def no_network(*args, **kwargs):
            raise AssertionError("Network access attempted in deterministic verifier!")

        monkeypatch.setattr(socket.socket, "connect", no_network)
        
        answer = "Binary search repeatedly halves the search space on a sorted array [1]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.PASS
        assert result.valid is True

    # 18. Verifier performs zero LLM calls
    def test_verifier_zero_llm_calls(self, monkeypatch, sample_sources):
        import ask

        def no_llm(*args, **kwargs):
            raise AssertionError("LLM call attempted in deterministic verifier!")

        monkeypatch.setattr(ask, "ask_gemini", no_llm)

        answer = "Binary search repeatedly halves the search space on a sorted array [1]."
        result = verify_citations(answer, sample_sources)

        assert result.status == VerificationStatus.PASS
        assert result.valid is True
