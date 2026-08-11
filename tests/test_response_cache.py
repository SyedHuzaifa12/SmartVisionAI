"""Tests for src.utils.response_cache."""

import pytest

from src.utils import response_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    response_cache.clear()
    yield
    response_cache.clear()


def test_cache_miss_then_hit_avoids_recomputation():
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return "result"

    first = response_cache.get_or_compute("feature_a", "image_1", compute)
    second = response_cache.get_or_compute("feature_a", "image_1", compute)

    assert first.was_cache_hit is False
    assert second.was_cache_hit is True
    assert second.value == "result"
    assert calls["count"] == 1


def test_different_images_do_not_collide():
    outcome_a = response_cache.get_or_compute("feature_a", "image_1", lambda: "A")
    outcome_b = response_cache.get_or_compute("feature_a", "image_2", lambda: "B")

    assert outcome_a.value == "A"
    assert outcome_b.value == "B"
    assert outcome_a.was_cache_hit is False
    assert outcome_b.was_cache_hit is False


def test_different_features_on_same_image_do_not_collide():
    outcome_scene = response_cache.get_or_compute("describe_scene", "image_1", lambda: "scene result")
    outcome_objects = response_cache.get_or_compute("detect_objects", "image_1", lambda: "objects result")

    assert outcome_scene.value == "scene result"
    assert outcome_objects.value == "objects result"


def test_lru_eviction_drops_oldest_entry(monkeypatch):
    monkeypatch.setattr(response_cache, "_MAX_ENTRIES", 2)

    response_cache.get_or_compute("f", "image_1", lambda: "1")
    response_cache.get_or_compute("f", "image_2", lambda: "2")
    response_cache.get_or_compute("f", "image_3", lambda: "3")  # evicts image_1

    calls = {"count": 0}

    def recompute():
        calls["count"] += 1
        return "recomputed"

    outcome = response_cache.get_or_compute("f", "image_1", recompute)

    assert outcome.was_cache_hit is False
    assert calls["count"] == 1
