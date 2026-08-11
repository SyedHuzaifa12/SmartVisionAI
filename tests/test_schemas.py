"""Tests for src.llm.schemas."""

import pytest
from pydantic import ValidationError

from src.llm.schemas import ContentModerationResult, OcrClassification, ResponseEvaluation, SceneAnalysis


def _scene_kwargs(**overrides):
    base = dict(
        scene_summary="You are in a small kitchen with a counter ahead of you.",
        environment_type="kitchen",
        important_objects=["A kettle is on the counter ahead of you."],
        hazard_level="Caution",
        hazard_explanation="A kettle may be hot to the touch.",
        suggested_action="Approach the counter slowly with your hand out.",
        audio_summary="You are in a kitchen. There is a kettle ahead that may be hot, so approach slowly.",
        scene_confidence=0.8,
        hazard_confidence=0.7,
    )
    base.update(overrides)
    return base


def test_scene_analysis_accepts_valid_data():
    analysis = SceneAnalysis(**_scene_kwargs())

    assert analysis.hazard_level == "Caution"
    assert analysis.important_objects == ["A kettle is on the counter ahead of you."]
    assert analysis.scene_confidence == 0.8
    assert analysis.hazard_confidence == 0.7


def test_scene_analysis_defaults_empty_object_list():
    analysis = SceneAnalysis(
        scene_summary="An empty, open field with nothing notable nearby.",
        environment_type="park",
        hazard_level="Safe",
        hazard_explanation="No hazards are visible.",
        suggested_action="Proceed as normal.",
        audio_summary="You are in an open park with no notable hazards.",
        scene_confidence=0.9,
        hazard_confidence=0.9,
    )

    assert analysis.important_objects == []


def test_scene_analysis_rejects_invalid_hazard_level():
    with pytest.raises(ValidationError):
        SceneAnalysis(**_scene_kwargs(hazard_level="Extremely Dangerous"))


def test_scene_analysis_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        SceneAnalysis(**_scene_kwargs(scene_confidence=1.5))


def test_ocr_classification_accepts_valid_data():
    classification = OcrClassification(
        category="Medicine label",
        suggested_action="Check the expiry date before use.",
        classification_confidence=0.85,
    )

    assert classification.category == "Medicine label"
    assert classification.classification_confidence == 0.85


def test_response_evaluation_accepts_valid_data():
    evaluation = ResponseEvaluation(
        completeness_score=90,
        safety_score=95,
        navigation_usefulness_score=70,
        ocr_usefulness_score=None,
        overall_quality_score=85,
        evaluation_notes="Looks good.",
    )

    assert evaluation.ocr_usefulness_score is None
    assert evaluation.overall_quality_score == 85


def test_response_evaluation_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        ResponseEvaluation(
            completeness_score=150,
            safety_score=95,
            navigation_usefulness_score=70,
            overall_quality_score=85,
            evaluation_notes="Invalid.",
        )


def test_content_moderation_result_accepts_valid_data():
    result = ContentModerationResult(is_inappropriate=False, reason="Ordinary photo, no concerns.")

    assert result.is_inappropriate is False
    assert result.reason
