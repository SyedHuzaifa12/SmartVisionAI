"""Tests for src.guardrails (validation middleware + confidence gating)."""

from src.guardrails.confidence import apply_confidence_gate
from src.guardrails.validation import (
    fallback_ocr_classification,
    fallback_scene_analysis,
    run_with_guardrails,
    validate_ocr_classification,
    validate_scene_analysis,
)
from src.llm.schemas import OcrClassification, SceneAnalysis


def _valid_analysis(**overrides):
    base = dict(
        scene_summary="You are in a kitchen with a counter ahead of you.",
        environment_type="kitchen",
        important_objects=[],
        hazard_level="Safe",
        hazard_explanation="Nothing hazardous is visible.",
        suggested_action="Proceed as normal.",
        audio_summary="You are in a kitchen with nothing hazardous nearby.",
        scene_confidence=0.9,
        hazard_confidence=0.9,
    )
    base.update(overrides)
    return SceneAnalysis(**base)


def test_validate_scene_analysis_accepts_valid_safe_scene_with_no_objects():
    assert validate_scene_analysis(_valid_analysis()) is True


def test_validate_scene_analysis_rejects_none():
    assert validate_scene_analysis(None) is False


def test_validate_scene_analysis_rejects_dangerous_with_no_objects():
    analysis = _valid_analysis(hazard_level="Dangerous", hazard_explanation="A fire is visible.")
    assert validate_scene_analysis(analysis) is False


def test_validate_scene_analysis_accepts_dangerous_with_objects():
    analysis = _valid_analysis(
        hazard_level="Dangerous",
        hazard_explanation="A fire is visible.",
        important_objects=["Flames are visible ahead of you."],
    )
    assert validate_scene_analysis(analysis) is True


def test_validate_ocr_classification_rejects_empty_category():
    assert validate_ocr_classification(OcrClassification(category="", suggested_action="Do X", classification_confidence=0.5)) is False


def test_run_with_guardrails_passes_through_valid_result():
    outcome = run_with_guardrails(
        lambda: _valid_analysis(),
        validate_scene_analysis,
        fallback_scene_analysis,
        label="test",
    )

    assert outcome.passed_validation is True
    assert outcome.retried is False


def test_run_with_guardrails_retries_then_falls_back_on_persistent_failure():
    outcome = run_with_guardrails(
        lambda: None,
        validate_scene_analysis,
        fallback_scene_analysis,
        label="test",
    )

    assert outcome.passed_validation is False
    assert outcome.retried is True
    assert outcome.value.hazard_level == "Caution"  # fallback fails safe, not "Safe"


def test_run_with_guardrails_recovers_after_one_retry():
    attempts = {"count": 0}

    def flaky_compute():
        attempts["count"] += 1
        return None if attempts["count"] == 1 else _valid_analysis()

    outcome = run_with_guardrails(flaky_compute, validate_scene_analysis, fallback_scene_analysis, label="test")

    assert outcome.passed_validation is True
    assert outcome.retried is True
    assert attempts["count"] == 2


def test_run_with_guardrails_treats_exception_as_failure():
    def raises():
        raise RuntimeError("boom")

    outcome = run_with_guardrails(raises, validate_scene_analysis, fallback_scene_analysis, label="test")

    assert outcome.passed_validation is False
    assert outcome.value.hazard_level == "Caution"


def test_fallback_ocr_classification_has_zero_confidence():
    fallback = fallback_ocr_classification()
    assert fallback.classification_confidence == 0.0
    assert fallback.category == "Unknown"


def test_confidence_gate_replaces_response_below_scene_threshold():
    analysis = _valid_analysis(scene_confidence=0.1, hazard_confidence=0.9)

    gated = apply_confidence_gate(analysis, scene_threshold=0.5, hazard_threshold=0.5)

    assert "not confident" in gated.scene_summary.lower()
    assert gated.hazard_level == "Caution"


def test_confidence_gate_escalates_low_confidence_safe_to_caution():
    analysis = _valid_analysis(hazard_level="Safe", scene_confidence=0.9, hazard_confidence=0.2)

    gated = apply_confidence_gate(analysis, scene_threshold=0.5, hazard_threshold=0.5)

    assert gated.hazard_level == "Caution"
    assert "low" in gated.hazard_explanation.lower()


def test_confidence_gate_never_downgrades_dangerous():
    analysis = _valid_analysis(
        hazard_level="Dangerous",
        hazard_explanation="Fire visible.",
        important_objects=["Flames ahead."],
        scene_confidence=0.9,
        hazard_confidence=0.1,
    )

    gated = apply_confidence_gate(analysis, scene_threshold=0.5, hazard_threshold=0.5)

    assert gated.hazard_level == "Dangerous"


def test_confidence_gate_passes_through_high_confidence_response():
    analysis = _valid_analysis(scene_confidence=0.9, hazard_confidence=0.9)

    gated = apply_confidence_gate(analysis, scene_threshold=0.5, hazard_threshold=0.5)

    assert gated == analysis
