"""Service layer for the four SmartVisionAI features.

This is where the AI-engineering pipeline actually composes: each feature
function generates a result, runs it through guardrail validation (retry
once, else a safe fallback), applies confidence-based gating, times each
stage, and computes a lightweight quality evaluation - then hands back one
``FeatureResult`` that bundles the user-facing data with the two dev-facing
signals (``metrics``, ``evaluation``). Keeping all of that here means
``app.py`` only ever renders; it never decides whether an answer was
trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from config import settings
from src.evaluation.response_evaluator import evaluate_ocr_result, evaluate_scene_analysis
from src.guardrails.confidence import apply_confidence_gate
from src.guardrails.validation import (
    fallback_content_moderation,
    fallback_ocr_classification,
    fallback_scene_analysis,
    run_with_guardrails,
    validate_content_moderation,
    validate_ocr_classification,
    validate_scene_analysis,
)
from src.llm.schemas import ContentModerationResult, OcrClassification, ResponseEvaluation, SceneAnalysis
from src.llm.vision_client import analyze_scene, check_content_moderation, classify_extracted_text, run_vision_prompt
from src.observability.metrics import PipelineMetrics, StageTiming, timed_stage
from src.ocr.extractor import extract_text_with_confidence
from src.prompts.vision_prompts import (
    OBJECT_DETECTION_USER_PROMPT,
    PERSONAL_ASSISTANCE_PROMPT,
    SCENE_ANALYSIS_SYSTEM_PROMPT,
    SCENE_UNDERSTANDING_USER_PROMPT,
)


@dataclass
class OcrResult:
    """OCR output plus its confidence, and (optional) content classification."""

    text: str
    confidence: float
    low_confidence: bool
    classification: OcrClassification | None


@dataclass
class FeatureResult:
    """Everything one feature invocation produces: the answer, plus dev-facing signals."""

    data: SceneAnalysis | OcrResult | str
    metrics: PipelineMetrics
    evaluation: ResponseEvaluation | None


def moderate_image(image_base64: str) -> ContentModerationResult:
    """Check whether an image contains explicit/severe content, gating all 4 features.

    Runs once per unique image (callers should cache on image_base64 alone,
    not per-feature) rather than being duplicated inside every feature
    function. Fails open: if the check itself can't be verified after a
    retry, the image is treated as not inappropriate - a technical glitch in
    this side-check should never block an ordinary image.
    """
    guardrail = run_with_guardrails(
        lambda: check_content_moderation(image_base64),
        validate_content_moderation,
        fallback_content_moderation,
        label="Content Moderation",
    )
    return guardrail.value


def describe_scene(image_base64: str) -> FeatureResult:
    """Generate a structured, navigation-focused scene analysis for the uploaded image."""
    stages: list[StageTiming] = []
    with timed_stage(stages, "vision_model"):
        guardrail = run_with_guardrails(
            lambda: analyze_scene(image_base64, SCENE_ANALYSIS_SYSTEM_PROMPT, SCENE_UNDERSTANDING_USER_PROMPT),
            validate_scene_analysis,
            fallback_scene_analysis,
            label="Scene Understanding",
        )

    analysis = apply_confidence_gate(
        guardrail.value,
        scene_threshold=settings.scene_confidence_threshold,
        hazard_threshold=settings.hazard_confidence_threshold,
    )
    metrics = PipelineMetrics(
        model_name=settings.gemini_model,
        stages=stages,
        validation_passed=guardrail.passed_validation,
        retried=guardrail.retried,
        error_detail=guardrail.error_detail,
    )
    return FeatureResult(data=analysis, metrics=metrics, evaluation=evaluate_scene_analysis(analysis))


def detect_objects(image_base64: str) -> FeatureResult:
    """Identify and prioritize the objects/hazards in the uploaded image for safe navigation."""
    stages: list[StageTiming] = []
    with timed_stage(stages, "vision_model"):
        guardrail = run_with_guardrails(
            lambda: analyze_scene(image_base64, SCENE_ANALYSIS_SYSTEM_PROMPT, OBJECT_DETECTION_USER_PROMPT),
            validate_scene_analysis,
            fallback_scene_analysis,
            label="Object Detection",
        )

    analysis = apply_confidence_gate(
        guardrail.value,
        scene_threshold=settings.scene_confidence_threshold,
        hazard_threshold=settings.hazard_confidence_threshold,
    )
    metrics = PipelineMetrics(
        model_name=settings.gemini_model,
        stages=stages,
        validation_passed=guardrail.passed_validation,
        retried=guardrail.retried,
        error_detail=guardrail.error_detail,
    )
    return FeatureResult(data=analysis, metrics=metrics, evaluation=evaluate_scene_analysis(analysis))


def extract_text_from_image(image: Image.Image, image_base64: str) -> FeatureResult:
    """Extract text via OCR, classify its likely category, and score OCR confidence.

    Classification is skipped when no text is detected, since there is
    nothing meaningful to categorize - and skipping avoids a wasted Gemini
    call on an empty extraction.
    """
    stages: list[StageTiming] = []
    with timed_stage(stages, "ocr_extraction"):
        text, ocr_confidence = extract_text_with_confidence(image)

    classification: OcrClassification | None = None
    validation_passed = True
    retried = False
    error_detail: str | None = None

    if text.strip():
        with timed_stage(stages, "ocr_classification_model"):
            guardrail = run_with_guardrails(
                lambda: classify_extracted_text(image_base64, text),
                validate_ocr_classification,
                fallback_ocr_classification,
                label="OCR Classification",
            )
        classification = guardrail.value
        validation_passed = guardrail.passed_validation
        retried = guardrail.retried
        error_detail = guardrail.error_detail

    result = OcrResult(
        text=text,
        confidence=ocr_confidence,
        low_confidence=ocr_confidence < settings.ocr_confidence_threshold,
        classification=classification,
    )
    metrics = PipelineMetrics(
        model_name=settings.gemini_model,
        stages=stages,
        validation_passed=validation_passed,
        retried=retried,
        error_detail=error_detail,
    )
    evaluation = evaluate_ocr_result(text, ocr_confidence, classification)
    return FeatureResult(data=result, metrics=metrics, evaluation=evaluation)


def provide_personal_assistance(image_base64: str) -> FeatureResult:
    """Provide context-specific assistive guidance for the uploaded image.

    Returns plain text (no structured schema), so no guardrails/evaluation
    apply here - only latency/model-name observability, for consistency with
    the other three features.
    """
    stages: list[StageTiming] = []
    with timed_stage(stages, "vision_model"):
        text = run_vision_prompt(image_base64, PERSONAL_ASSISTANCE_PROMPT)

    metrics = PipelineMetrics(model_name=settings.gemini_model, stages=stages)
    return FeatureResult(data=text, metrics=metrics, evaluation=None)
