"""Tests for src.utils.rate_limiter."""

from src.utils.rate_limiter import check_and_record


def test_allows_requests_up_to_the_limit():
    session_state = {}

    for _ in range(3):
        status = check_and_record(session_state, max_requests=3, window_seconds=3600, now=1000.0)
        assert status.allowed is True

    blocked = check_and_record(session_state, max_requests=3, window_seconds=3600, now=1000.0)

    assert blocked.allowed is False
    assert blocked.remaining == 0


def test_remaining_count_decreases():
    session_state = {}

    first = check_and_record(session_state, max_requests=5, window_seconds=3600, now=1000.0)
    second = check_and_record(session_state, max_requests=5, window_seconds=3600, now=1000.0)

    assert first.remaining == 4
    assert second.remaining == 3


def test_old_requests_outside_window_are_forgotten():
    session_state = {}

    for _ in range(5):
        check_and_record(session_state, max_requests=5, window_seconds=3600, now=1000.0)

    # An hour and one second later, the window has fully rolled over.
    status = check_and_record(session_state, max_requests=5, window_seconds=3600, now=1000.0 + 3601)

    assert status.allowed is True
    assert status.remaining == 4


def test_blocked_request_reports_a_positive_retry_after():
    session_state = {}
    check_and_record(session_state, max_requests=1, window_seconds=3600, now=1000.0)

    blocked = check_and_record(session_state, max_requests=1, window_seconds=3600, now=1000.0 + 100)

    assert blocked.allowed is False
    assert blocked.retry_after_seconds == 3500


def test_independent_sessions_have_independent_limits():
    session_a: dict = {}
    session_b: dict = {}

    for _ in range(3):
        check_and_record(session_a, max_requests=3, window_seconds=3600, now=1000.0)

    status_b = check_and_record(session_b, max_requests=3, window_seconds=3600, now=1000.0)

    assert status_b.allowed is True
