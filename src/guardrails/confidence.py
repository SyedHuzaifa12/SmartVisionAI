"""Confidence-aware gating for scene analysis.

The model reports its own confidence per field (see ``SceneAnalysis``), but a
self-reported number is only useful if something actually acts on it. This
module is that action: it rewrites the response when confidence is too low
to trust, always failing toward the *safer* outcome rather than silently
showing (or silently hiding) an uncertain answer.

Two distinct failure modes are handled differently on purpose:

- Low ``scene_confidence`` means the model likely can't reliably describe
  the image at all (bad lighting, blur, framing) - the whole response is
  replaced with an honest "please recapture" message.
- Low ``hazard_confidence`` (but otherwise-reasonable scene confidence)
  means the *hazard verdict specifically* shouldn't be trusted. Rather than
  hiding the hazard field, an uncertain "Safe" is escalated to "Caution" -
  never downgraded - and a caveat is appended, so the user is never told
  "safe" on a guess.
"""

from __future__ import annotations

from src.llm.schemas import SceneAnalysis


def apply_confidence_gate(
    analysis: SceneAnalysis,
    *,
    scene_threshold: float,
    hazard_threshold: float,
) -> SceneAnalysis:
    """Return ``analysis`` unchanged, or a confidence-appropriate degraded version."""
    if analysis.scene_confidence < scene_threshold:
        return SceneAnalysis(
            scene_summary=(
                "I'm not confident enough in this image to describe it reliably - it may be too dark, "
                "blurry, or unclear."
            ),
            environment_type="Unknown",
            important_objects=[],
            hazard_level="Caution",
            hazard_explanation="Scene confidence is too low to make a reliable hazard judgment.",
            suggested_action="Please try capturing another image with better lighting and clearer framing.",
            audio_summary=(
                "I'm not confident enough in this image to guide you reliably. Please try capturing "
                "another image with better lighting and framing, and proceed with caution in the "
                "meantime."
            ),
            scene_confidence=analysis.scene_confidence,
            hazard_confidence=analysis.hazard_confidence,
        )

    if analysis.hazard_confidence < hazard_threshold:
        escalated_level = "Caution" if analysis.hazard_level == "Safe" else analysis.hazard_level
        return analysis.model_copy(
            update={
                "hazard_level": escalated_level,
                "hazard_explanation": (
                    f"{analysis.hazard_explanation} (Hazard confidence was low, so treat this "
                    "assessment with extra caution.)"
                ),
            }
        )

    return analysis
