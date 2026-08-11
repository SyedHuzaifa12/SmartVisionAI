"""Tests for src.prompts.vision_prompts."""

from src.prompts.vision_prompts import (
    OBJECT_DETECTION_USER_PROMPT,
    OCR_CLASSIFICATION_SYSTEM_PROMPT,
    OCR_CLASSIFICATION_USER_PROMPT_TEMPLATE,
    PERSONAL_ASSISTANCE_PROMPT,
    SCENE_ANALYSIS_SYSTEM_PROMPT,
    SCENE_UNDERSTANDING_USER_PROMPT,
)


def test_prompts_are_non_empty_strings():
    for prompt in (
        SCENE_ANALYSIS_SYSTEM_PROMPT,
        SCENE_UNDERSTANDING_USER_PROMPT,
        OBJECT_DETECTION_USER_PROMPT,
        OCR_CLASSIFICATION_SYSTEM_PROMPT,
        PERSONAL_ASSISTANCE_PROMPT,
    ):
        assert isinstance(prompt, str)
        assert prompt.strip()


def test_scene_understanding_and_object_detection_prompts_differ():
    # These previously read almost identically; they must stay distinct.
    assert SCENE_UNDERSTANDING_USER_PROMPT != OBJECT_DETECTION_USER_PROMPT


def test_ocr_classification_prompt_template_formats():
    rendered = OCR_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(extracted_text="Ibuprofen 200mg")

    assert "Ibuprofen 200mg" in rendered
    assert "{extracted_text}" not in rendered
