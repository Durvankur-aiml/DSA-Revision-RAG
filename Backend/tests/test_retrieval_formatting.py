"""Retrieval formatting and pure-function regression tests.

Covers:
- prompt construction from retrieved chunks (metadata survival)
- source formatting + YouTube timestamp URL building (? vs &)
- question validation
- normalization/tokenization
- topic extraction and the BST-vs-binary-search title distinction
  (the regression that must never come back)

Only pure functions and formatting are exercised here; no Qdrant,
no embeddings, no network.
"""

from __future__ import annotations

import pytest

import ask


# ---------------------------------------------------------------------------
# TC-4: retrieved chunks -> prompt for Gemini (metadata survival)
# ---------------------------------------------------------------------------

class TestPromptConstruction:
    def test_prompt_contains_question_and_transcript_text(self, sample_chunks):
        prompt = ask.build_prompt("What is binary search?", sample_chunks)

        assert prompt is not None
        assert "What is binary search?" in prompt
        # Transcript text of every chunk survives into the prompt.
        assert "repeatedly halving" in prompt
        assert "kth" in prompt.lower()
        assert "answer space" in prompt

    def test_prompt_contains_source_titles_and_scores(self, sample_chunks):
        prompt = ask.build_prompt("What is binary search?", sample_chunks)

        assert "BS-1. Binary Search Introduction" in prompt
        assert "BS-16. Kth Missing Positive Number" in prompt
        assert "RETRIEVAL SCORE" in prompt

    def test_prompt_classifies_exact_topic_as_primary(self, sample_chunks):
        prompt = ask.build_prompt("What is binary search?", sample_chunks)

        assert "PRIMARY — EXACT TOPIC" in prompt
        assert "RELATED" in prompt

    def test_prompt_includes_detected_topic(self, sample_chunks):
        prompt = ask.build_prompt("What is binary search?", sample_chunks)

        assert "binary search" in prompt

    def test_prompt_returns_none_without_chunks(self):
        assert ask.build_prompt("Anything at all?", []) is None

    def test_prompt_skips_chunks_without_text(self, sample_chunks):
        stripped = list(sample_chunks)
        stripped[1].payload = dict(stripped[1].payload)
        stripped[1].payload["text"] = ""

        prompt = ask.build_prompt("What is binary search?", stripped)

        assert prompt is not None
        assert "kth" not in prompt.lower()
        # The other chunks are still present.
        assert "repeatedly halving" in prompt


class TestSourceClassification:
    def test_primary_vs_related(self, sample_chunks):
        classified = ask.classify_sources(
            "What is binary search?", sample_chunks
        )

        assert classified[0]["type"] == "PRIMARY — EXACT TOPIC"
        assert classified[1]["type"].startswith("RELATED")
        assert classified[2]["type"].startswith("RELATED")

    def test_classification_indices_are_one_based(self, sample_chunks):
        classified = ask.classify_sources(
            "What is binary search?", sample_chunks
        )
        assert [c["index"] for c in classified] == [1, 2, 3]


# ---------------------------------------------------------------------------
# TC-6: YouTube timestamp URLs
# ---------------------------------------------------------------------------

class TestTimestampUrls:
    def test_api_source_url_appends_timestamp_with_question_mark(
        self, client, retrieval_mock
    ):
        """youtu.be/<id> has no query string -> ?t=<seconds>."""
        response = client.post(
            "/ask", json={"question": "What is binary search?"}
        )
        sources = response.json()["sources"]

        assert sources[0]["url"] == "https://youtu.be/BS1VIDEO?t=86"
        assert sources[1]["url"] == "https://youtu.be/BS16VIDEO?t=261"

    def test_api_source_url_appends_timestamp_with_ampersand(
        self, client, retrieval_mock
    ):
        """A URL that already has a query string must use &t=..."""
        chunks = retrieval_mock  # fixture returns the call list; adjust data
        # Reach the underlying chunk data through a fresh retrieval call.
        import main

        # The first chunk's URL is rewritten for this test by monkey-
        # patching inside the fake retrieval via ask-side state: instead,
        # exercise the same production separator logic in ask.format_sources.
        url_with_query = "https://www.youtube.com/watch?v=BS1VIDEO"
        chunk = type(retrieval_mock)  # placeholder to keep structure clear

        result = ask.format_sources([])
        assert result  # no crash on empty input

        # Build a minimal chunk-like object for the direct unit test.
        class _Hit:
            id = "BS1VIDEO"
            payload = {
                "video_title": "BS-1. Binary Search Introduction",
                "youtube_url": url_with_query,
                "start": 86.0,
            }

        with ask.RETRIEVAL_SCORE_LOCK:
            ask.LAST_RETRIEVAL_SCORES["BS1VIDEO"] = 1.0

        formatted = ask.format_sources([_Hit()])
        assert f"{url_with_query}&t=86" in formatted
        assert "?t=86" not in formatted.split("watch?v")[1].split("&t")[0]

    def test_format_sources_handles_missing_url(self):
        class _Hit:
            id = "X"
            payload = {"video_title": "T", "youtube_url": "", "start": 5.0}

        formatted = ask.format_sources([_Hit()])
        assert "N/A" in formatted

    def test_format_sources_handles_bad_timestamp(self):
        class _Hit:
            id = "X"
            payload = {
                "video_title": "T",
                "youtube_url": "https://youtu.be/X",
                "start": None,
            }

        formatted = ask.format_sources([_Hit()])
        assert "?t=0" in formatted


# ---------------------------------------------------------------------------
# Question validation (API input boundary)
# ---------------------------------------------------------------------------

class TestValidateQuestion:
    def test_valid_question_passes(self):
        question, error = ask.validate_question("  What is binary search?  ")
        assert error is None
        assert question == "What is binary search?"

    def test_empty_question_rejected(self):
        question, error = ask.validate_question("   ")
        assert question is None
        assert error
        assert "non-empty" in error.lower()

    def test_oversized_question_rejected(self):
        question, error = ask.validate_question("x" * 501)
        assert question is None
        assert "too long" in error.lower()
        assert "500" in error

    def test_boundary_length_question_passes(self):
        question, error = ask.validate_question("x" * 500)
        assert error is None
        assert len(question) == 500


# ---------------------------------------------------------------------------
# Normalization / tokenization
# ---------------------------------------------------------------------------

class TestNormalizeText:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Three Sum", "3 sum"),
            ("3-Sum", "3 sum"),
            ("3_sum", "3 sum"),
            ("  Binary   SEARCH! ", "binary search"),
            ("Kth Missing Positive Number.", "kth missing positive number"),
        ],
    )
    def test_normalization(self, raw, expected):
        assert ask.normalize_text(raw) == expected

    def test_empty_inputs(self):
        assert ask.normalize_text("") == ""
        assert ask.normalize_text(None) == ""


class TestTokenize:
    def test_stopwords_removed(self):
        tokens = ask.tokenize("What is the binary search?")
        assert "what" not in tokens
        assert "is" not in tokens
        assert "the" not in tokens
        assert "binary" in tokens
        assert "search" in tokens

    def test_short_tokens_removed(self):
        tokens = ask.tokenize("a an I go to")
        assert all(len(t) > 1 for t in tokens)

    def test_multi_digit_numbers_survive(self):
        # Single-character tokens are dropped by the len > 1 rule; this
        # is intentional. One-digit sums ("3 Sum") are handled by the
        # dedicated number-sum regex in extract_topic instead.
        tokens = ask.tokenize("Explain 12 Sum.")
        assert "12" in tokens
        assert "sum" in tokens
        assert "3" not in ask.tokenize("Explain 3 Sum.")


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

class TestExtractTopic:
    @pytest.mark.parametrize(
        "question,expected",
        [
            ("What is binary search?", "binary search"),
            ("What is a binary search tree?", "binary search tree"),
            ("Explain 3 Sum.", "3 sum"),
            ("Explain recursion.", "recursion"),
            ("Explain BFS.", "bfs"),
            ("Explain Dijkstra.", "dijkstra"),
            ("How does Kadane's algorithm work?", "kadane"),
        ],
    )
    def test_known_topics(self, question, expected):
        assert ask.extract_topic(question) == expected

    def test_number_word_normalization(self):
        assert ask.extract_topic("Explain Three Sum") == "3 sum"


# ---------------------------------------------------------------------------
# The BST-vs-binary-search distinction (must never regress)
# ---------------------------------------------------------------------------

class TestTitleExactMatchDistinction:
    INTRO_TITLE = "BS-1. Binary Search Introduction | Real Life Example"
    BST_TITLE = "L39. Introduction to Binary Search Tree | BST"
    PROBLEM_TITLE = "BS-16. Kth Missing Positive Number | Maths + Binary Search"

    def test_binary_search_matches_intro_video(self):
        score = ask.title_exact_match("What is binary search?", self.INTRO_TITLE)
        assert score >= 0.95

    def test_binary_search_rejects_binary_search_tree(self):
        score = ask.title_exact_match("What is binary search?", self.BST_TITLE)
        assert score == 0.0

    def test_binary_search_downweights_problem_videos(self):
        score = ask.title_exact_match(
            "What is binary search?", self.PROBLEM_TITLE
        )
        assert score < 0.95

    def test_bst_question_matches_bst_video(self):
        score = ask.title_exact_match(
            "What is a binary search tree?", self.BST_TITLE
        )
        assert score >= 0.95

    def test_specific_problem_question_topic_extraction(self):
        """Documented known limitation (do not 'fix' by changing weights).

        For problem-specific questions outside the known-topic list,
        the fallback topic sorts tokens alphabetically, so
        title_exact_match returns 0.0. Retrieval still surfaces the
        correct source via the SEMANTIC path — verified live in Phase 1
        (BS-16 top-1, hybrid 0.2731). A future retrieval-quality
        improvement may raise this score; that work must go through the
        evaluation dataset, not ad-hoc weight changes.
        """
        topic = ask.extract_topic("Explain Kth Missing Positive Number.")
        # The fallback topic preserves all content tokens.
        assert set(topic.split()) == {"kth", "missing", "positive", "number"}

        score = ask.title_exact_match(
            "Explain Kth Missing Positive Number.", self.PROBLEM_TITLE
        )
        assert score == 0.0  # semantic path carries this query today

    def test_broad_vs_specific_intro_preference(self):
        """A broad topic question prefers introductory videos."""
        broad_score = ask.title_exact_match(
            "What is binary search?", self.INTRO_TITLE
        )
        problem_score = ask.title_exact_match(
            "What is binary search?",
            "BS-18. Allocate Books or Book Allocation | Hard Binary Search",
        )
        assert broad_score >= 0.95
        assert problem_score < 0.95

    def test_unrelated_title_scores_zero(self):
        score = ask.title_exact_match(
            "What is binary search?",
            "G-32. Dijkstra's Algorithm - Using Priority Queue",
        )
        assert score == 0.0
