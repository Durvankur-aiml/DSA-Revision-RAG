"""Gemini lifecycle regression tests.

These protect the Phase 1 stabilization behavior at the
``ask.ask_gemini`` level (full retry policy) and at the
``ask._generate_with_single_interaction`` level (single create->poll
lifecycle), using the scriptable FakeInteractionsAPI from conftest.
No real network access occurs.
"""

from __future__ import annotations

import pytest

import ask

from conftest import (
    FakeInteraction,
    TransientError,
    daily_quota_429,
    invalid_request_400,
    transient_503,
)

POLL = ask.GEMINI_POLL_INTERVAL_SECONDS


def total_sleep_wait(clock):
    return sum(clock.sleeps)


# ---------------------------------------------------------------------------
# TC-7: normal completed interaction
# ---------------------------------------------------------------------------

class TestCompletedPath:
    def test_completed_returns_answer(self, gemini_mock, fake_clock):
        gemini_mock.get_script = [
            FakeInteraction(status="in_progress"),
            FakeInteraction(status="in_progress"),
            FakeInteraction(
                status="completed",
                output_text="Binary search halves the search space.",
            ),
        ]

        answer = ask.ask_gemini("test message")

        assert answer == "Binary search halves the search space."
        assert gemini_mock.create_calls == 1
        assert gemini_mock.get_calls == 3

    def test_create_uses_sdk_background_and_no_thinking_level(
        self, gemini_mock, fake_clock
    ):
        """Regression guard: thinking_level must never come back."""
        gemini_mock.get_script = [
            FakeInteraction(status="completed", output_text="ok"),
        ]

        ask.ask_gemini("test message")

        assert gemini_mock.create_calls == 1
        kwargs = gemini_mock.create_kwargs_seen[0]
        assert kwargs["background"] is True
        generation_config = kwargs["generation_config"]
        assert "thinking_level" not in generation_config
        assert (
            generation_config["max_output_tokens"]
            == ask.GEMINI_MAX_OUTPUT_TOKENS
        )


# ---------------------------------------------------------------------------
# TC-8: terminal statuses that carry partial text
# ---------------------------------------------------------------------------

class TestTerminalPartialText:
    @pytest.mark.parametrize("status", ["incomplete", "budget_exceeded"])
    def test_partial_text_statuses_stop_polling_and_return_text(
        self, gemini_mock, fake_clock, status
    ):
        gemini_mock.get_script = [
            FakeInteraction(status=status, output_text="Partial answer."),
        ]

        answer = ask.ask_gemini("test message")

        assert answer == "Partial answer."
        # Exits on the first poll; no infinite polling, no retry churn.
        assert gemini_mock.get_calls == 1
        assert fake_clock.sleeps == []

    def test_incomplete_without_text_fails_cleanly_in_one_lifecycle(
        self, gemini_mock, fake_clock
    ):
        """incomplete + no text ends that lifecycle with None (the outer
        retry policy is exercised separately by the deadline tests)."""
        gemini_mock.get_script = [
            FakeInteraction(status="incomplete", output_text=None),
        ]

        answer = ask._generate_with_single_interaction(
            "test message", deadline=fake_clock.now + 1000
        )

        assert answer is None
        assert gemini_mock.get_calls == 1


# ---------------------------------------------------------------------------
# TC-9: transient 503 recovery on the same interaction
# ---------------------------------------------------------------------------

class TestTransient503Recovery:
    def test_503s_then_success_on_same_interaction(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="in_progress"),
            transient_503(),
            transient_503(),
            FakeInteraction(
                status="completed",
                output_text="Answer after transient 503s.",
            ),
        ]

        answer = ask.ask_gemini("test message")

        assert answer == "Answer after transient 503s."
        # Transient errors retry the SAME interaction (no fresh create).
        assert gemini_mock.create_calls == 1
        # One backoff sleep per transient failure.
        assert len(fake_clock.sleeps) >= 2

    def test_persistent_transient_errors_exhaust_and_fail(
        self, gemini_mock, fake_clock
    ):
        # 503s exhaust the per-interaction retry budget, then the outer
        # fresh-interaction retries run until the total budget expires.
        gemini_mock.get_script = [
            transient_503(),
            transient_503(),
            transient_503(),
            transient_503(),
        ]

        answer = ask.ask_gemini("test message")

        assert answer is None
        assert 1 <= gemini_mock.create_calls <= 4
        assert total_sleep_wait(fake_clock) <= (
            ask.GEMINI_MAX_TOTAL_SECONDS
            + ask.GEMINI_POLL_INTERVAL_SECONDS
            + 10
        )


# ---------------------------------------------------------------------------
# TC-10: poisoned interaction (the Phase 1 centerpiece fix)
# ---------------------------------------------------------------------------

class TestPoisonedInteraction:
    def test_fresh_interaction_recovers_after_poisoning(
        self, gemini_mock, fake_clock
    ):
        """create -> in_progress -> 503 -> 400 invalid_request.

        The 400 after a mid-flight 503 means this interaction is
        poisoned server-side. The implementation must abandon it and
        create a FRESH interaction, which then completes.
        """
        gemini_mock.get_script = [
            FakeInteraction(status="in_progress"),  # interaction 1
            transient_503(),                        # mid-flight overload
            invalid_request_400(),                  # poisoned from now on
            FakeInteraction(                        # interaction 2, fresh
                status="completed",
                output_text="Answer from the fresh interaction.",
            ),
        ]

        answer = ask.ask_gemini("test message")

        assert answer == "Answer from the fresh interaction."
        # Exactly one fresh interaction was created after the poison.
        assert gemini_mock.create_calls == 2
        # The 400 was NOT retried on the broken interaction forever:
        # poll1 in_progress, poll2 503, poll3 400, poll4 = fresh success.
        assert gemini_mock.get_calls == 4

    def test_all_poisoned_interactions_fail_within_total_budget(
        self, gemini_mock, fake_clock
    ):
        # Every interaction gets poisoned: 503 -> 400 forever.
        gemini_mock.get_script = [
            transient_503(),
            invalid_request_400(),
            transient_503(),
            invalid_request_400(),
            transient_503(),
            invalid_request_400(),
        ]

        answer = ask.ask_gemini("test message")

        assert answer is None
        assert 2 <= gemini_mock.create_calls <= 4
        assert total_sleep_wait(fake_clock) <= (
            ask.GEMINI_MAX_TOTAL_SECONDS
            + ask.GEMINI_POLL_INTERVAL_SECONDS
            + 10
        )


# ---------------------------------------------------------------------------
# TC-11: daily quota 429 fails fast
# ---------------------------------------------------------------------------

class TestDailyQuota:
    def test_daily_quota_during_poll_fails_fast(self, gemini_mock, fake_clock):
        gemini_mock.get_script = [
            FakeInteraction(status="in_progress"),
            daily_quota_429(),
        ]

        answer = ask.ask_gemini("test message")

        assert answer is None
        # No fresh interaction is attempted after daily-quota detection.
        assert gemini_mock.create_calls == 1
        # The only sleep is the single poll interval before the 429;
        # there are NO retry backoffs after quota detection (fast-fail).
        assert fake_clock.sleeps == [POLL]

    def test_daily_quota_on_create_is_permanent(self, gemini_mock, fake_clock):
        gemini_mock.create_error = daily_quota_429()

        answer = ask.ask_gemini("test message")

        assert answer is None
        assert gemini_mock.create_calls == 1
        assert fake_clock.sleeps == []

    def test_per_minute_rate_limit_is_still_transient(
        self, gemini_mock, fake_clock
    ):
        """A non-daily 429 must keep the retry behavior."""
        per_minute_429 = TransientError(
            "Error code: 429 - Rate limit exceeded. Please retry in 21s."
        )
        gemini_mock.get_script = [
            FakeInteraction(status="in_progress"),
            per_minute_429,
            FakeInteraction(status="completed", output_text="recovered"),
        ]

        answer = ask.ask_gemini("test message")

        assert answer == "recovered"
        assert gemini_mock.create_calls == 1
        assert len(fake_clock.sleeps) >= 2


# ---------------------------------------------------------------------------
# TC-12: deadlines bound everything
# ---------------------------------------------------------------------------

class TestDeadlines:
    def test_stuck_interactions_stop_within_total_budget(
        self, gemini_mock, fake_clock
    ):
        # Every poll returns stuck in_progress (fake fallback behavior).
        answer = ask.ask_gemini("test message")

        assert answer is None
        assert total_sleep_wait(fake_clock) <= (
            ask.GEMINI_MAX_TOTAL_SECONDS
            + ask.GEMINI_POLL_INTERVAL_SECONDS
            + 10
        )
        # A bounded number of fresh interactions is attempted.
        assert 2 <= gemini_mock.create_calls <= 4

    def test_slow_create_attempts_are_bounded(
        self, gemini_mock, fake_clock, monkeypatch
    ):
        """Each create stalls 40s (fake clock) then fails transiently.

        Total attempts must stay bounded and the call must return
        instead of hanging: the create-retry cap and the total deadline
        together guarantee termination.
        """
        attempts = {"count": 0}

        def slow_failing_create(**kwargs):
            attempts["count"] += 1
            ask.time.sleep(40)  # advances the fake clock
            raise transient_503()

        monkeypatch.setattr(gemini_mock, "create", slow_failing_create)

        answer = ask.ask_gemini("test message")

        assert answer is None
        # Bounded: at most (GEMINI_MAX_RETRIES + 1) creates per outer
        # attempt, and the outer loop stops at the total deadline.
        assert attempts["count"] <= 2 * (ask.GEMINI_MAX_RETRIES + 1)
        elapsed = fake_clock.now - 1000.0
        assert elapsed <= ask.GEMINI_MAX_TOTAL_SECONDS + 45

    def test_polling_stops_when_total_deadline_expires(
        self, gemini_mock, fake_clock, monkeypatch
    ):
        """A tight total deadline stops polling even though the
        per-interaction wait limit (90s) has not been reached."""
        monkeypatch.setattr(ask, "GEMINI_MAX_TOTAL_SECONDS", 30)

        answer = ask.ask_gemini("test message")

        assert answer is None
        assert total_sleep_wait(fake_clock) <= 30 + POLL + 5


# ---------------------------------------------------------------------------
# TC-13: malformed / unexpected Gemini responses
# ---------------------------------------------------------------------------

class TestMalformedResponses:
    def test_unexpected_status_is_bounded_and_clean(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="definitely_not_a_real_status"),
        ]

        answer = ask.ask_gemini("test message")

        # Unknown statuses are treated as still-running and polled
        # until the bounded wait limit expires; no exception escapes.
        assert answer is None
        assert gemini_mock.create_calls <= 4

    def test_completed_with_whitespace_only_text_is_not_accepted(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="completed", output_text="   "),
        ]

        answer = ask.ask_gemini("test message")

        # The lifecycle refuses blank text; the outer retry loop then
        # fails in bounded time instead of returning an empty answer.
        assert answer is None

    def test_completed_with_none_text_is_not_accepted(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="completed", output_text=None),
        ]

        answer = ask.ask_gemini("test message")
        assert answer is None

    def test_failed_status_ends_lifecycle_immediately(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="failed", error="safety block"),
        ]

        answer = ask._generate_with_single_interaction(
            "test message", deadline=fake_clock.now + 1000
        )

        assert answer is None
        assert gemini_mock.get_calls == 1

    def test_cancelled_status_ends_lifecycle_immediately(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="cancelled"),
        ]

        answer = ask._generate_with_single_interaction(
            "test message", deadline=fake_clock.now + 1000
        )

        assert answer is None
        assert gemini_mock.get_calls == 1

    def test_requires_action_is_not_supported_and_stops(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="requires_action"),
        ]

        answer = ask._generate_with_single_interaction(
            "test message", deadline=fake_clock.now + 1000
        )

        assert answer is None
        assert gemini_mock.get_calls == 1

    def test_failed_interaction_is_bounded_at_ask_gemini_level(
        self, gemini_mock, fake_clock
    ):
        gemini_mock.get_script = [
            FakeInteraction(status="failed", error="boom"),
        ]

        answer = ask.ask_gemini("test message")

        # The outer loop may retry with fresh interactions, but only a
        # bounded number of times.
        assert answer is None
        assert gemini_mock.create_calls <= 4
