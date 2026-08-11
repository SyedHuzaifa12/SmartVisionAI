"""Simple session-scoped request rate limiting.

Streamlit Cloud gives no stable per-visitor identity without a backend/DB
(deliberately out of scope for this project - see README's AI System Design
Decisions). This limits requests per *browser session* (``st.session_state``)
instead, and is honest about that boundary rather than pretending to
per-device enforcement it can't actually provide: a new tab or a cleared
session resets the count. Its purpose is protecting a shared, free-tier API
key from rapid repeated clicks within one visit, not airtight abuse
prevention.

Deliberately independent of ``src/utils/response_cache.py``: this gates "how
many times can this session trigger the pipeline," while the cache
separately reduces "how many of those actually call Gemini." Coupling the
two (e.g. not counting cache hits against the limit) would save a little
bookkeeping and add real complexity for a portfolio-scale demo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import MutableMapping

_DEFAULT_KEY = "_rate_limit_timestamps"


@dataclass
class RateLimitStatus:
    """Outcome of a rate-limit check."""

    allowed: bool
    remaining: int
    retry_after_seconds: int


def check_and_record(
    session_state: MutableMapping[str, list[float]],
    *,
    max_requests: int,
    window_seconds: int,
    key: str = _DEFAULT_KEY,
    now: float | None = None,
) -> RateLimitStatus:
    """Check whether a new request is allowed under the limit, recording it if so.

    Args:
        session_state: A dict-like, per-session store (``st.session_state`` in
            the app; a plain dict in tests).
        max_requests: Maximum requests allowed within ``window_seconds``.
        window_seconds: The sliding window size, in seconds.
        key: Storage key within ``session_state`` (overridable for tests/multiple limiters).
        now: Injectable current time for deterministic tests; defaults to ``time.time()``.

    Returns:
        A ``RateLimitStatus`` - if ``allowed`` is False, no request is recorded.
    """
    current_time = time.time() if now is None else now
    timestamps = [t for t in session_state.get(key, []) if current_time - t < window_seconds]

    if len(timestamps) >= max_requests:
        session_state[key] = timestamps
        retry_after = int(window_seconds - (current_time - min(timestamps)))
        return RateLimitStatus(allowed=False, remaining=0, retry_after_seconds=max(retry_after, 1))

    timestamps.append(current_time)
    session_state[key] = timestamps
    return RateLimitStatus(allowed=True, remaining=max_requests - len(timestamps), retry_after_seconds=0)
