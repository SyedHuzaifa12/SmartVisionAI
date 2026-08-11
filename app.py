"""SmartVisionAI — Streamlit entry point.

Thin orchestration layer only: wires up the page, handles the image
upload/feature-selection widgets, and delegates all real work to
``src.services.vision_assistant`` and ``src.speech.tts``. This file never
decides whether a response is trustworthy - that happens in the service
layer (guardrails + confidence gating); here we only render whatever it
hands back, plus two dev-facing panels (Pipeline Metrics, AI Evaluation)
that never affect the primary answer above them.

Two production-hardening concerns also live here, at the thinnest possible
point (the button click), since both are Streamlit-session concepts:
- Rate limiting (``src.utils.rate_limiter``) - caps requests per browser
  session, protecting the shared free-tier API key from rapid repeat clicks.
- Response caching (``src.utils.response_cache``) - skips the Gemini call
  entirely for a (feature, image) pair already computed by *any* session.
These are independent by design: the limiter counts every click regardless
of whether it turns out to be a cache hit, keeping the two concerns simple
to reason about separately.

UI note: icons are used sparingly on purpose - one per major section/button,
never one per field. The hazard traffic light (green/yellow/red) is the one
icon that carries real information (instant severity signal); everything
else is plain typography so the app reads as a tool, not a chat toy.
"""

from __future__ import annotations

import logging

import streamlit as st

from config import configure_logging, settings
from src.llm.schemas import ResponseEvaluation, SceneAnalysis
from src.observability.metrics import PipelineMetrics
from src.services.vision_assistant import (
    OcrResult,
    describe_scene,
    detect_objects,
    extract_text_from_image,
    moderate_image,
    provide_personal_assistance,
)
from src.speech.tts import synthesize_speech
from src.ui.branding import render_footer, render_header, render_sidebar
from src.utils.image_utils import InvalidImageError, image_to_base64, validate_uploaded_image
from src.utils.rate_limiter import check_and_record
from src.utils.response_cache import get_or_compute

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="SmartVisionAI", page_icon="👁️", layout="centered")

HAZARD_ICONS = {"Safe": "🟢", "Caution": "🟡", "Dangerous": "🔴"}


def _present_scene_analysis(label: str, analysis: SceneAnalysis) -> None:
    """Render a structured scene analysis in the fixed, recruiter-facing order:
    Scene Summary -> Environment Type -> Important Objects -> Hazard Assessment
    -> Suggested Action -> Audio Summary.
    """
    st.markdown(f"### {label}")
    st.markdown(f"**Scene Summary:** {analysis.scene_summary}")
    st.markdown(f"**Environment Type:** {analysis.environment_type}")

    if analysis.important_objects:
        objects_list = "\n".join(f"- {obj}" for obj in analysis.important_objects)
        st.markdown(f"**Important Objects:**\n{objects_list}")
    else:
        st.markdown("**Important Objects:** None of particular note.")

    hazard_icon = HAZARD_ICONS.get(analysis.hazard_level, "⚪")
    st.markdown(
        f"**{hazard_icon} Hazard Assessment ({analysis.hazard_level}):** "
        f"{analysis.hazard_explanation}"
    )
    st.caption(
        f"Scene confidence: {analysis.scene_confidence:.0%} · "
        f"Hazard confidence: {analysis.hazard_confidence:.0%}"
    )
    st.markdown(f"**Suggested Action:** {analysis.suggested_action}")

    st.markdown("**Audio Summary:**")
    st.audio(synthesize_speech(analysis.audio_summary), format="audio/mp3")


def _present_ocr_result(result: OcrResult) -> None:
    """Render OCR output: Extracted Text -> Category -> Suggested Action -> Audio Summary."""
    st.markdown("### Extracted Text")
    if result.low_confidence and result.text.strip():
        st.warning(
            "OCR confidence is low - this text may be inaccurate. Consider retaking the photo "
            "with better lighting or a closer angle."
        )
    st.markdown(result.text if result.text.strip() else "_No text detected in the image._")
    st.caption(f"OCR confidence: {result.confidence:.0%}")

    audio_parts = [result.text.strip()] if result.text.strip() else ["No text was detected in the image."]
    if result.classification:
        st.markdown(f"**Category:** {result.classification.category}")
        st.markdown(f"**Suggested Action:** {result.classification.suggested_action}")
        audio_parts.append(f"This appears to be a {result.classification.category.lower()}.")
        audio_parts.append(result.classification.suggested_action)

    st.markdown("**Audio Summary:**")
    st.audio(synthesize_speech(" ".join(audio_parts)), format="audio/mp3")


def _present_text_result(label: str, result_text: str) -> None:
    """Render a plain-text feature result (used by Personalized Assistance)."""
    st.markdown(f"### {label}\n{result_text}")
    st.audio(synthesize_speech(result_text), format="audio/mp3")


def _present_pipeline_metrics(metrics: PipelineMetrics, served_from_cache: bool) -> None:
    """Dev-facing panel: model name, per-stage latency, total latency, validation status.

    ``served_from_cache`` is reported explicitly rather than silently reusing
    the cached (now stale) latency numbers as if they were freshly measured -
    that would be a real correctness bug in an observability feature.
    """
    with st.expander("Pipeline Metrics (dev)"):
        if served_from_cache:
            st.info(
                "Served from cache - an identical image/feature request was already computed; "
                "no new Gemini call was made. Latency below is from the original request."
            )
        st.markdown(f"**Model:** `{metrics.model_name}`")
        for stage in metrics.stages:
            st.markdown(f"- {stage.name}: {stage.duration_ms:.0f} ms")
        st.markdown(f"**Total latency:** {metrics.total_latency_ms:.0f} ms")
        status = "Passed" if metrics.validation_passed else "Used fallback"
        retried_note = " (retried once)" if metrics.retried else ""
        st.markdown(f"**Structured output validation:** {status}{retried_note}")
        if metrics.error_detail:
            st.markdown("**Underlying error** (why the fallback was used):")
            st.code(metrics.error_detail, language=None)


def _present_evaluation(evaluation: ResponseEvaluation) -> None:
    """Dev-facing panel: rule-based quality scores. Never shown as the primary answer."""
    with st.expander("AI Evaluation (dev)"):
        st.markdown(f"- **Completeness:** {evaluation.completeness_score}/100")
        st.markdown(f"- **Safety:** {evaluation.safety_score}/100")
        st.markdown(f"- **Navigation usefulness:** {evaluation.navigation_usefulness_score}/100")
        if evaluation.ocr_usefulness_score is not None:
            st.markdown(f"- **OCR usefulness:** {evaluation.ocr_usefulness_score}/100")
        st.markdown(f"- **Overall quality:** {evaluation.overall_quality_score}/100")
        st.caption(evaluation.evaluation_notes)


def _check_rate_limit() -> bool:
    """Return True if this session may make another request; otherwise show a message."""
    status = check_and_record(
        st.session_state,
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if not status.allowed:
        minutes = max(status.retry_after_seconds // 60, 1)
        st.warning(
            f"You've reached the limit of {settings.rate_limit_max_requests} requests per hour "
            f"for this session. Please try again in about {minutes} minute(s)."
        )
    return status.allowed


def _run_feature(label: str, image_base64: str, compute_fn, *args) -> None:
    """Run a feature's compute function (cached per feature+image) and present the result.

    Every call is gated by one content-moderation check first, cached on the
    image alone (not per-feature) so it costs one extra call per unique
    image - not one per button click. See src.services.vision_assistant.
    moderate_image for why this replaced relying on Gemini's declarative
    safety_settings.

    Dispatches presentation by result type so every feature shares one
    error-handling path regardless of whether it returns a structured
    ``SceneAnalysis``, an ``OcrResult``, or plain text - then renders the
    dev-facing metrics/evaluation panels below the main answer, all inside
    one bordered card so a result reads as a single unit, not a stream of
    loose markdown lines.
    """
    with st.container(border=True):
        # Broad except is intentional: any failure here is surfaced to the user via
        # st.error below, not silently swallowed.
        try:
            moderation = get_or_compute(
                "content_moderation", image_base64, lambda: moderate_image(image_base64)
            ).value
            if moderation.is_inappropriate:
                st.error(
                    "This image can't be processed: it appears to contain content this app doesn't "
                    "support (explicit, graphic, or hateful content). Please try a different image."
                )
                return

            outcome = get_or_compute(compute_fn.__name__, image_base64, lambda: compute_fn(*args))
            feature_result = outcome.value
            data = feature_result.data
            if isinstance(data, SceneAnalysis):
                _present_scene_analysis(label, data)
            elif isinstance(data, OcrResult):
                _present_ocr_result(data)
            else:
                _present_text_result(label, data)

            st.caption("AI engineering internals (latency, validation, quality scoring) below.")
            _present_pipeline_metrics(feature_result.metrics, outcome.was_cache_hit)
            if feature_result.evaluation is not None:
                _present_evaluation(feature_result.evaluation)
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", label)
            st.error(f"{label} failed: {exc}")


def main() -> None:
    """Render the SmartVisionAI Streamlit app."""
    if not settings.has_gemini_key:
        st.warning(
            "No Gemini API key configured. Set GEMINI_API_KEY in your `.env` "
            "file to enable Scene Understanding, Object Detection, and "
            "Personalized Assistance."
        )

    render_header()
    feature = render_sidebar()

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if not uploaded_image:
        st.info("Please upload an image to proceed.")
        render_footer()
        return

    try:
        image = validate_uploaded_image(
            uploaded_image,
            max_size_mb=settings.max_upload_size_mb,
            max_dimension_px=settings.max_image_dimension_px,
        )
    except InvalidImageError as exc:
        st.error(str(exc))
        render_footer()
        return

    st.image(image, caption="Uploaded image", use_container_width=True)
    image_base64 = image_to_base64(image)

    if feature == "Real-Time Scene Understanding" and st.button("Run Scene Understanding"):
        if _check_rate_limit():
            with st.spinner("Analyzing... this can take a few seconds."):
                _run_feature("Scene Understanding", image_base64, describe_scene, image_base64)

    elif feature == "Text-to-Speech Conversion" and st.button("Extract & Convert Text"):
        if _check_rate_limit():
            with st.spinner("Extracting text..."):
                _run_feature(
                    "Extracted Text", image_base64, extract_text_from_image, image, image_base64
                )

    elif feature == "Object Detection" and st.button("Run Object Detection"):
        if _check_rate_limit():
            with st.spinner("Detecting objects..."):
                _run_feature("Object Detection", image_base64, detect_objects, image_base64)

    elif feature == "Personalized Assistance" and st.button("Run Personalized Assistance"):
        if _check_rate_limit():
            with st.spinner("Providing personalized guidance..."):
                _run_feature(
                    "Personalized Assistance", image_base64, provide_personal_assistance, image_base64
                )

    render_footer()


if __name__ == "__main__":
    main()
