"""Bounded, in-process response cache for identical (feature, image) requests.

Deliberately a small explicit cache rather than Streamlit's built-in
``st.cache_data``:

- Hit/miss is a value we get back, not something hidden inside Streamlit's
  internals - the caller needs it to render Pipeline Metrics honestly
  (showing stale latency numbers as if freshly measured on a cache hit would
  be a real correctness bug in an observability feature).
- It's bounded (simple LRU eviction) - an unbounded cache on a long-running
  public demo process is a real memory-leak risk, not a hypothetical one.
- It has zero framework dependency, so it's trivially unit-testable and
  keeps ``src/`` decoupled from Streamlit, matching the rest of this codebase.

The cache is process-wide (not per-session), which is intentional: if two
different visitors upload the same image, the second one should also skip
the Gemini call - that's where the real API-cost savings come from.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_MAX_ENTRIES = 128
_cache: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()


@dataclass
class CacheOutcome(Generic[T]):
    """Result of a cache lookup/compute, with the hit/miss made explicit."""

    value: T
    was_cache_hit: bool


def _cache_key(feature_name: str, image_base64: str) -> str:
    """Hash (feature, image) to a short, fixed-size key rather than storing raw base64 as a dict key."""
    digest = hashlib.sha256(image_base64.encode("utf-8")).hexdigest()
    return f"{feature_name}:{digest}"


def get_or_compute(feature_name: str, image_base64: str, compute_fn: Callable[[], T]) -> CacheOutcome[T]:
    """Return the cached result for this (feature, image) pair, computing it once if absent.

    Thread-safe: Streamlit can serve multiple sessions concurrently within
    one process, and this cache is shared across all of them by design.
    """
    key = _cache_key(feature_name, image_base64)

    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            logger.info("Cache hit for %s", feature_name)
            return CacheOutcome(value=_cache[key], was_cache_hit=True)

    result = compute_fn()

    with _lock:
        _cache[key] = result
        _cache.move_to_end(key)
        if len(_cache) > _MAX_ENTRIES:
            _cache.popitem(last=False)

    return CacheOutcome(value=result, was_cache_hit=False)


def clear() -> None:
    """Clear all cached entries. Used by tests; also safe to call from an admin action."""
    with _lock:
        _cache.clear()
