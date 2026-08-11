"""Lightweight, rule-based evaluation of generated responses.

For development/demo purposes only: these scores are shown in a collapsed
"AI Evaluation" panel and never affect what the user is told.

Deliberately rule-based rather than a second Gemini "LLM-as-judge" call:
grading a model's output with the same model risks self-grading bias, and a
second call would double the latency and cost of every request. Deterministic
heuristics are also reproducible, which matters more than nuance for a
debugging/demo tool - the goal is a quick sanity signal, not a rigorous
benchmark.
"""

from __future__ import annotations

from src.llm.schemas import OcrClassification, ResponseEvaluation, SceneAnalysis

_NAVIGATION_KEYWORDS = (
    "left",
    "right",
    "ahead",
    "front",
    "behind",
    "near",
    "far",
    "above",
    "below",
    "beside",
)

_SAFETY_BASELINE = {"Safe": 95, "Caution": 65, "Dangerous": 30}


def evaluate_scene_analysis(analysis: SceneAnalysis) -> ResponseEvaluation:
    """Score a SceneAnalysis response for completeness, safety, and navigation usefulness."""
    completeness_checks = [
        bool(analysis.scene_summary.strip()),
        bool(analysis.environment_type.strip()) and analysis.environment_type.lower() != "unknown",
        bool(analysis.hazard_explanation.strip()),
        bool(analysis.suggested_action.strip()),
        bool(analysis.audio_summary.strip()),
    ]
    completeness = round(100 * sum(completeness_checks) / len(completeness_checks))

    safety_baseline = _SAFETY_BASELINE.get(analysis.hazard_level, 50)
    safety = round(safety_baseline * (0.5 + 0.5 * analysis.hazard_confidence))

    combined_text = (analysis.scene_summary + " " + " ".join(analysis.important_objects)).lower()
    keyword_hits = sum(1 for keyword in _NAVIGATION_KEYWORDS if keyword in combined_text)
    navigation_usefulness = min(100, 35 + keyword_hits * 15 + (10 if analysis.important_objects else 0))

    overall = round((completeness + safety + navigation_usefulness) / 3)

    return ResponseEvaluation(
        completeness_score=completeness,
        safety_score=safety,
        navigation_usefulness_score=navigation_usefulness,
        ocr_usefulness_score=None,
        overall_quality_score=overall,
        evaluation_notes=(
            f"Hazard level '{analysis.hazard_level}' (confidence {analysis.hazard_confidence:.0%}); "
            f"{keyword_hits} navigation-position cue(s) detected."
        ),
    )


def evaluate_ocr_result(
    text: str, ocr_confidence: float, classification: OcrClassification | None
) -> ResponseEvaluation:
    """Score an OCR feature response for completeness and OCR usefulness."""
    has_text = bool(text.strip())
    completeness_checks = [has_text, classification is not None]
    completeness = round(100 * sum(completeness_checks) / len(completeness_checks))

    length_component = min(60, len(text.strip()) // 2) if has_text else 0
    ocr_usefulness = round(min(100, length_component + ocr_confidence * 40)) if has_text else 0

    # OCR is not a navigation feature and doesn't assess physical hazards, so
    # those scores get a fixed neutral baseline rather than a computed value.
    safety = 90
    navigation_usefulness = 20

    overall = round((completeness + ocr_usefulness) / 2)

    classification_note = (
        f"classified as {classification.category}" if classification else "no classification available"
    )
    return ResponseEvaluation(
        completeness_score=completeness,
        safety_score=safety,
        navigation_usefulness_score=navigation_usefulness,
        ocr_usefulness_score=ocr_usefulness,
        overall_quality_score=overall,
        evaluation_notes=f"OCR confidence {ocr_confidence:.0%}; {classification_note}.",
    )
