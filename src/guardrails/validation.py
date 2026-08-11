"""Reusable validation middleware for structured LLM outputs.

Every structured LLM call in the app goes through the same pipeline:
generate -> validate -> retry once on failure -> graceful fallback. This is
the one place that logic lives, instead of each feature re-implementing its
own ad-hoc checks (which is how validation logic quietly rots and diverges
across a codebase).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from src.llm.schemas import ContentModerationResult, OcrClassification, SceneAnalysis

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Applied only before the retry (i.e. only on an already-failing path): a
# short pause meaningfully improves the odds of the retry succeeding if the
# root cause was transient - a rate-limit blip or a momentary API error -
# without adding latency to the common, successful case.
_RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class GuardrailOutcome(Generic[T]):
    """Result of running a compute function through validation guardrails."""

    value: T
    passed_validation: bool
    retried: bool
    error_detail: str | None = None
    """The underlying exception (type + message), if generation ever raised one.

    Surfaced all the way to the Pipeline Metrics dev panel - without this,
    a rate-limit error, a safety block, and a genuine bug all produce the
    *identical* generic fallback text, making the real cause undiagnosable
    from the UI alone.
    """


def run_with_guardrails(
    compute_fn: Callable[[], T],
    validate_fn: Callable[[T], bool],
    fallback_factory: Callable[[], T],
    *,
    label: str = "response",
    retry_backoff_seconds: float = _RETRY_BACKOFF_SECONDS,
) -> GuardrailOutcome[T]:
    """Call ``compute_fn``, validate its result, retry once, then fall back.

    A raised exception (e.g. a transient API error, or the model producing
    something so malformed even structured-output coercion fails) is treated
    the same as a failed validation, rather than crashing the whole feature.

    ``retry_backoff_seconds`` is overridable (tests pass 0) so the production
    default doesn't force every retry-path test to actually sleep.
    """

    def _attempt() -> tuple[T | None, str | None]:
        try:
            return compute_fn(), None
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s raised an exception during generation", label)
            return None, f"{type(exc).__name__}: {exc}"

    result, error_detail = _attempt()
    if validate_fn(result):
        return GuardrailOutcome(value=result, passed_validation=True, retried=False)

    logger.warning("%s failed guardrail validation - retrying once", label)
    if error_detail is None:
        error_detail = "Response failed validation (see src/guardrails/validation.py checks)."
    if retry_backoff_seconds > 0:
        time.sleep(retry_backoff_seconds)

    result, retry_error_detail = _attempt()
    if validate_fn(result):
        return GuardrailOutcome(value=result, passed_validation=True, retried=True)

    final_error_detail = retry_error_detail or error_detail
    logger.error("%s failed guardrail validation after retry - using fallback (%s)", label, final_error_detail)
    return GuardrailOutcome(
        value=fallback_factory(), passed_validation=False, retried=True, error_detail=final_error_detail
    )


def validate_scene_analysis(analysis: SceneAnalysis | None) -> bool:
    """Check that a SceneAnalysis has all required fields meaningfully populated."""
    if analysis is None:
        return False
    if analysis.hazard_level not in ("Safe", "Caution", "Dangerous"):
        return False
    if not analysis.scene_summary or len(analysis.scene_summary.strip()) < 10:
        return False
    if not analysis.environment_type or not analysis.environment_type.strip():
        return False
    if not analysis.hazard_explanation or not analysis.hazard_explanation.strip():
        return False
    if not analysis.suggested_action or not analysis.suggested_action.strip():
        return False
    # A "Dangerous" verdict with zero identified objects/hazards is internally
    # inconsistent, so require at least one in that specific case. A blanket
    # "object list must never be empty" rule would misfire on ordinary Safe
    # scenes that genuinely have nothing notable to point out.
    if analysis.hazard_level == "Dangerous" and not analysis.important_objects:
        return False
    if not (0.0 <= analysis.scene_confidence <= 1.0):
        return False
    if not (0.0 <= analysis.hazard_confidence <= 1.0):
        return False
    return True


def validate_ocr_classification(classification: OcrClassification | None) -> bool:
    """Check that an OcrClassification has meaningful required fields."""
    if classification is None:
        return False
    if not classification.category or not classification.category.strip():
        return False
    if not classification.suggested_action or not classification.suggested_action.strip():
        return False
    return True


def validate_content_moderation(result: ContentModerationResult | None) -> bool:
    """Check that a ContentModerationResult has a real reason, not just a bare boolean."""
    if result is None:
        return False
    if not result.reason or not result.reason.strip():
        return False
    return True


def fallback_scene_analysis() -> SceneAnalysis:
    """A safe, honest degraded response used when the model output can't be trusted.

    Defaults hazard_level to "Caution" rather than "Safe" - when we can't
    trust the analysis at all, failing toward caution is the only acceptable
    direction for an accessibility/safety tool.
    """
    return SceneAnalysis(
        scene_summary="I couldn't reliably analyze this image. It may be too dark, blurry, or unclear.",
        environment_type="Unknown",
        important_objects=[],
        hazard_level="Caution",
        hazard_explanation=(
            "The analysis could not be verified, so treat the surroundings with caution until you can "
            "confirm them another way."
        ),
        suggested_action="Please try capturing another image with better lighting and clearer framing.",
        audio_summary=(
            "I could not reliably analyze this image. Please try capturing another image with better "
            "lighting and framing, and proceed with caution in the meantime."
        ),
        scene_confidence=0.0,
        hazard_confidence=0.0,
    )


def fallback_ocr_classification() -> OcrClassification:
    """A safe, honest degraded response for OCR classification."""
    return OcrClassification(
        category="Unknown",
        suggested_action="Try capturing the text again with better lighting or a closer angle.",
        classification_confidence=0.0,
    )


def fallback_content_moderation() -> ContentModerationResult:
    """Fail *open* when the moderation check itself can't be verified.

    Deliberately the opposite direction from the scene/hazard fallbacks: a
    technical failure of this specific check (rate limit, transient error)
    should never block an ordinary image just because a side-check glitched.
    The main hazard-detection guardrails already handle physical safety
    conservatively; this check exists for a different, narrower purpose
    (explicit content), and its own failure mode shouldn't compound that.
    """
    return ContentModerationResult(is_inappropriate=False, reason="Moderation check could not be verified.")
