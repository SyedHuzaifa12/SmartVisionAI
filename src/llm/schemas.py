"""Structured output schemas for Gemini vision analysis.

Using a typed schema (via LangChain's ``with_structured_output``) instead of
parsing hazard level / environment type / object list out of freeform prose
makes those fields reliable to render and test, instead of fragile
string/regex scraping of a paragraph. Confidence fields are part of the
schema itself (not bolted on afterwards) so the model must commit to a
calibration estimate for every response, and that estimate is validated
(``src/guardrails``) before it ever reaches the user.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HazardLevel = Literal["Safe", "Caution", "Dangerous"]


class SceneAnalysis(BaseModel):
    """A structured, navigation-focused analysis of a single image."""

    scene_summary: str = Field(
        description=(
            "A concise, navigation-focused description of the scene: where the user is, what is "
            "happening, and the general layout, written as one flowing spoken description - not a list."
        )
    )
    environment_type: str = Field(
        description=(
            "The type of environment shown, e.g. home, office, kitchen, hospital, classroom, street, "
            "mall, restaurant, park. Use 'Unknown' if it cannot be reasonably determined."
        )
    )
    important_objects: list[str] = Field(
        default_factory=list,
        description=(
            "The most important objects for navigation or safety, each as one short phrase naming the "
            "object, its position relative to the viewer (left/right/front/behind/near/far), and how it "
            "affects the user - e.g. 'A chair is ahead on your left, safe to sit on.' Omit insignificant "
            "background objects. Empty list if nothing relevant is present."
        ),
    )
    hazard_level: HazardLevel = Field(
        description="Overall safety classification for someone physically present in this scene."
    )
    hazard_explanation: str = Field(
        description=(
            "A concise, evidence-based explanation of why this hazard level was chosen, naming the "
            "specific observable hazard(s) if any (e.g. stairs, wet floor, exposed wires, moving "
            "vehicles, fire or smoke, glass doors, construction, animals in the path). State only the "
            "visible evidence that justifies the decision - for example 'Caution because a cow and "
            "goats occupy the walking path and pottery tools are on the ground.' Do not describe your "
            "reasoning process or think aloud - state the conclusion and its visible evidence only. If "
            "Safe, briefly say what you observed that supports that."
        )
    )
    suggested_action: str = Field(
        description=(
            "One short, practical instruction for what the user should do next, e.g. 'Move slightly to "
            "your right', 'Proceed carefully', or 'Use caution while approaching the stairs.'"
        )
    )
    audio_summary: str = Field(
        description=(
            "A short, natural, spoken-friendly narration (2-4 sentences) combining the scene, hazard "
            "level, and suggested action into one summary suitable for text-to-speech. Do not just "
            "concatenate the other fields verbatim."
        )
    )
    scene_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your honest confidence (0.0-1.0) that scene_summary, environment_type, and "
            "important_objects are accurate. Be conservative: lower this for blurry, dark, cluttered, "
            "partially framed, or ambiguous images. Do not default to a high value just because you "
            "produced an answer - overconfidence is more harmful than admitting uncertainty here."
        ),
    )
    hazard_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your honest confidence (0.0-1.0) specifically in the hazard_level judgment. Lower this "
            "when hazard-relevant details are unclear, occluded, distant, or ambiguous, even if you are "
            "confident about the rest of the scene."
        ),
    )


class OcrClassification(BaseModel):
    """Classification of OCR-extracted text, plus a suggested next step."""

    category: str = Field(
        description=(
            "The likely category of the extracted text: Medicine label, Business card, Signboard, "
            "Menu, Document, Book page, Product packaging, or Other if none fit clearly."
        )
    )
    suggested_action: str = Field(
        description=(
            "One short, practical instruction for what the user should do with this text, e.g. 'Check "
            "the expiry date before use' or 'This looks like a business card, consider saving the "
            "contact.'"
        )
    )
    classification_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your honest confidence (0.0-1.0) in this category classification, given the extracted "
            "text and image context. Lower this if the text is short, garbled, or the item type is "
            "genuinely ambiguous."
        ),
    )


class ResponseEvaluation(BaseModel):
    """Development/demo-only quality scoring for a generated response.

    Never shown as the primary answer and never gates what the user is told
    - purely an internal quality signal, rendered in a collapsed panel.
    """

    completeness_score: int = Field(
        ge=0, le=100, description="How complete the response is - are all expected fields present and substantive."
    )
    safety_score: int = Field(
        ge=0, le=100, description="How well the response protects user safety, based on hazard level and confidence."
    )
    navigation_usefulness_score: int = Field(
        ge=0,
        le=100,
        description="How useful the response is for physically navigating the space (position-aware, actionable detail).",
    )
    ocr_usefulness_score: int | None = Field(
        default=None, ge=0, le=100, description="How useful the OCR result is; None for non-OCR features."
    )
    overall_quality_score: int = Field(ge=0, le=100, description="Overall response quality, combining the above.")
    evaluation_notes: str = Field(description="A brief, one-sentence rationale for the scores given.")
