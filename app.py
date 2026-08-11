"""SmartVisionAI — Streamlit entry point.

Thin orchestration layer only: wires up the page, handles the image
upload/feature-selection widgets, and delegates all real work to
``src.services.vision_assistant`` and ``src.speech.tts``. This file never
decides whether a response is trustworthy - that happens in the service
layer (guardrails + confidence gating); here we only render whatever it
hands back, plus two dev-facing panels (Pipeline Metrics, AI Evaluation)
that never affect the primary answer above them.
"""

from __future__ import annotations

import logging

import streamlit as st
from PIL import Image

from config import configure_logging, settings
from src.llm.schemas import ResponseEvaluation, SceneAnalysis
from src.observability.metrics import PipelineMetrics
from src.services.vision_assistant import (
    OcrResult,
    describe_scene,
    detect_objects,
    extract_text_from_image,
    provide_personal_assistance,
)
from src.speech.tts import synthesize_speech
from src.ui.branding import render_footer, render_header, render_sidebar
from src.utils.image_utils import image_to_base64

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
    st.markdown(f"**🧭 Scene Summary:** {analysis.scene_summary}")
    st.markdown(f"**🏠 Environment Type:** {analysis.environment_type}")

    if analysis.important_objects:
        objects_list = "\n".join(f"- {obj}" for obj in analysis.important_objects)
        st.markdown(f"**📦 Important Objects:**\n{objects_list}")
    else:
        st.markdown("**📦 Important Objects:** None of particular note.")

    hazard_icon = HAZARD_ICONS.get(analysis.hazard_level, "⚪")
    st.markdown(
        f"**{hazard_icon} Hazard Assessment ({analysis.hazard_level}):** "
        f"{analysis.hazard_explanation}"
    )
    st.caption(
        f"Scene confidence: {analysis.scene_confidence:.0%} · "
        f"Hazard confidence: {analysis.hazard_confidence:.0%}"
    )
    st.markdown(f"**➡️ Suggested Action:** {analysis.suggested_action}")

    st.markdown("**🔊 Audio Summary:**")
    st.audio(synthesize_speech(analysis.audio_summary), format="audio/mp3")


def _present_ocr_result(result: OcrResult) -> None:
    """Render OCR output: Extracted Text -> Category -> Suggested Action -> Audio Summary."""
    st.markdown("### Extracted Text 📝")
    if result.low_confidence and result.text.strip():
        st.warning(
            "⚠️ OCR confidence is low - this text may be inaccurate. Consider retaking the photo "
            "with better lighting or a closer angle."
        )
    st.markdown(result.text if result.text.strip() else "_No text detected in the image._")
    st.caption(f"OCR confidence: {result.confidence:.0%}")

    audio_parts = [result.text.strip()] if result.text.strip() else ["No text was detected in the image."]
    if result.classification:
        st.markdown(f"**🏷️ Category:** {result.classification.category}")
        st.markdown(f"**➡️ Suggested Action:** {result.classification.suggested_action}")
        audio_parts.append(f"This appears to be a {result.classification.category.lower()}.")
        audio_parts.append(result.classification.suggested_action)

    st.markdown("**🔊 Audio Summary:**")
    st.audio(synthesize_speech(" ".join(audio_parts)), format="audio/mp3")


def _present_text_result(label: str, result_text: str) -> None:
    """Render a plain-text feature result (used by Personalized Assistance)."""
    st.markdown(f"### {label}:\n{result_text}")
    st.audio(synthesize_speech(result_text), format="audio/mp3")


def _present_pipeline_metrics(metrics: PipelineMetrics) -> None:
    """Dev-facing panel: model name, per-stage latency, total latency, validation status."""
    with st.expander("⏱️ Pipeline Metrics (dev)"):
        st.markdown(f"**Model:** `{metrics.model_name}`")
        for stage in metrics.stages:
            st.markdown(f"- {stage.name}: {stage.duration_ms:.0f} ms")
        st.markdown(f"**Total latency:** {metrics.total_latency_ms:.0f} ms")
        status = "✅ passed" if metrics.validation_passed else "⚠️ used fallback"
        retried_note = " (retried once)" if metrics.retried else ""
        st.markdown(f"**Structured output validation:** {status}{retried_note}")


def _present_evaluation(evaluation: ResponseEvaluation) -> None:
    """Dev-facing panel: rule-based quality scores. Never shown as the primary answer."""
    with st.expander("🧪 AI Evaluation (dev)"):
        st.markdown(f"- **Completeness:** {evaluation.completeness_score}/100")
        st.markdown(f"- **Safety:** {evaluation.safety_score}/100")
        st.markdown(f"- **Navigation usefulness:** {evaluation.navigation_usefulness_score}/100")
        if evaluation.ocr_usefulness_score is not None:
            st.markdown(f"- **OCR usefulness:** {evaluation.ocr_usefulness_score}/100")
        st.markdown(f"- **Overall quality:** {evaluation.overall_quality_score}/100")
        st.caption(evaluation.evaluation_notes)


def _run_feature(label: str, compute_fn, *args) -> None:
    """Run a feature's compute function and present the result, handling errors.

    Dispatches presentation by result type so every feature shares one
    error-handling path regardless of whether it returns a structured
    ``SceneAnalysis``, an ``OcrResult``, or plain text - then renders the
    dev-facing metrics/evaluation panels below the main answer.
    """
    # Broad except is intentional: any failure here is surfaced to the user via
    # st.error below, not silently swallowed.
    try:
        feature_result = compute_fn(*args)
        data = feature_result.data
        if isinstance(data, SceneAnalysis):
            _present_scene_analysis(label, data)
        elif isinstance(data, OcrResult):
            _present_ocr_result(data)
        else:
            _present_text_result(label, data)

        _present_pipeline_metrics(feature_result.metrics)
        if feature_result.evaluation is not None:
            _present_evaluation(feature_result.evaluation)
    except Exception as exc:  # noqa: BLE001
        logger.exception("%s failed", label)
        st.error(f"{label} failed: {exc}")


def main() -> None:
    """Render the SmartVisionAI Streamlit app."""
    if not settings.has_gemini_key:
        st.warning(
            "⚠️ No Gemini API key configured. Set GEMINI_API_KEY in your `.env` "
            "file to enable Scene Understanding, Object Detection, and "
            "Personalized Assistance."
        )

    render_header()
    feature = render_sidebar()

    uploaded_image = st.file_uploader(
        "📤 **Upload an Image**", type=["jpg", "jpeg", "png"]
    )

    if not uploaded_image:
        st.info("🚨 **Please upload an image to proceed.**")
        render_footer()
        return

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image 📷", use_container_width=True)
    image_base64 = image_to_base64(image)

    if feature == "Real-Time Scene Understanding" and st.button(
        "🔍 **Run Scene Understanding**"
    ):
        with st.spinner("Analyzing... Be Patient!"):
            _run_feature("Scene Understanding 📸", describe_scene, image_base64)

    elif feature == "Text-to-Speech Conversion" and st.button(
        "📝 **Extract & Convert Text**"
    ):
        with st.spinner("Extracting text..."):
            _run_feature(
                "Extracted Text 📝", extract_text_from_image, image, image_base64
            )

    elif feature == "Object Detection" and st.button("🕵️‍♂️ **Run Object Detection**"):
        with st.spinner("Detecting objects..."):
            _run_feature("Object Detection 🕵️‍♂️", detect_objects, image_base64)

    elif feature == "Personalized Assistance" and st.button(
        "💡 **Run Personalized Assistance**"
    ):
        with st.spinner("Providing personalized guidance..."):
            _run_feature(
                "Personalized Assistance 💡", provide_personal_assistance, image_base64
            )

    render_footer()


if __name__ == "__main__":
    main()
