"""Integration-style tests for src.services.vision_assistant.

Gemini/OCR calls are monkeypatched so these run without any API key or
Tesseract install - the point is to verify the guardrails -> confidence-gate
-> evaluation wiring actually connects, not to test the model itself.
"""

from __future__ import annotations

from PIL import Image

import src.services.vision_assistant as vision_assistant
from src.llm.schemas import OcrClassification, SceneAnalysis


def _sample_analysis(**overrides) -> SceneAnalysis:
    base = dict(
        scene_summary="A hallway with a chair ahead on your left.",
        environment_type="office",
        important_objects=["A chair is ahead on your left."],
        hazard_level="Safe",
        hazard_explanation="Nothing hazardous is visible.",
        suggested_action="Proceed as normal.",
        audio_summary="You are in an office hallway with a chair ahead on your left.",
        scene_confidence=0.9,
        hazard_confidence=0.9,
    )
    base.update(overrides)
    return SceneAnalysis(**base)


def test_describe_scene_returns_feature_result_with_metrics_and_evaluation(monkeypatch):
    monkeypatch.setattr(vision_assistant, "analyze_scene", lambda *a, **k: _sample_analysis())

    result = vision_assistant.describe_scene("fake_base64")

    assert isinstance(result.data, SceneAnalysis)
    assert result.metrics.validation_passed is True
    assert result.metrics.stages  # at least one timed stage recorded
    assert result.evaluation is not None


def test_describe_scene_applies_confidence_gate(monkeypatch):
    low_confidence = _sample_analysis(scene_confidence=0.05)
    monkeypatch.setattr(vision_assistant, "analyze_scene", lambda *a, **k: low_confidence)

    result = vision_assistant.describe_scene("fake_base64")

    assert "not confident" in result.data.scene_summary.lower()


def test_describe_scene_falls_back_when_model_output_never_validates(monkeypatch):
    monkeypatch.setattr(vision_assistant, "analyze_scene", lambda *a, **k: None)

    result = vision_assistant.describe_scene("fake_base64")

    assert result.metrics.validation_passed is False
    assert result.metrics.retried is True
    assert result.data.hazard_level == "Caution"


def test_extract_text_from_image_skips_classification_when_no_text(monkeypatch):
    monkeypatch.setattr(
        vision_assistant, "extract_text_with_confidence", lambda image: ("", 0.0)
    )
    calls = {"count": 0}

    def _should_not_be_called(*args, **kwargs):
        calls["count"] += 1
        raise AssertionError("classify_extracted_text should not be called for empty OCR text")

    monkeypatch.setattr(vision_assistant, "classify_extracted_text", _should_not_be_called)

    result = vision_assistant.extract_text_from_image(Image.new("RGB", (2, 2)), "fake_base64")

    assert result.data.text == ""
    assert result.data.classification is None
    assert calls["count"] == 0


def test_extract_text_from_image_classifies_when_text_present(monkeypatch):
    monkeypatch.setattr(
        vision_assistant, "extract_text_with_confidence", lambda image: ("Ibuprofen 200mg", 0.85)
    )
    monkeypatch.setattr(
        vision_assistant,
        "classify_extracted_text",
        lambda image_base64, text: OcrClassification(
            category="Medicine label",
            suggested_action="Check the expiry date before use.",
            classification_confidence=0.8,
        ),
    )

    result = vision_assistant.extract_text_from_image(Image.new("RGB", (2, 2)), "fake_base64")

    assert result.data.classification.category == "Medicine label"
    assert result.data.low_confidence is False
    assert result.evaluation.ocr_usefulness_score is not None
