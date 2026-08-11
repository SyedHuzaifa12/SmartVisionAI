"""Thin wrapper around the Gemini multimodal chat model.

Replaces near-identical "build message -> invoke -> read .content" blocks
with reusable calls, including structured-output calls (``analyze_scene``,
``classify_extracted_text``) that return typed Pydantic objects instead of
freeform prose. Every structured call sends a ``SystemMessage`` (shared
persona/constraints) separately from the ``HumanMessage`` (per-feature task +
image) - see ``src/prompts/vision_prompts.py`` for why that split matters.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

from config import settings
from src.llm.schemas import ContentModerationResult, OcrClassification, SceneAnalysis
from src.prompts.vision_prompts import (
    CONTENT_MODERATION_SYSTEM_PROMPT,
    CONTENT_MODERATION_USER_PROMPT,
    OCR_CLASSIFICATION_SYSTEM_PROMPT,
    OCR_CLASSIFICATION_USER_PROMPT_TEMPLATE,
)
from src.utils.image_utils import to_image_data_url

logger = logging.getLogger(__name__)

# Explicit content-safety thresholds for this public-facing app, rather than
# relying on Gemini's un-configured defaults. Deliberately the *least* strict
# blocking level (BLOCK_ONLY_HIGH) across every category: this app's own
# hazard-detection mission needs to describe ordinary knives, vehicles, and
# people without every such photo getting flagged as "medium" severity by
# Gemini's classifiers. Only clearly severe/extreme content - real nudity,
# actual weapons/bombs, graphic violence, explicit hate symbols - gets
# blocked; everything else passes through untouched.
#
# Only the 4 generic categories are used - NOT the HARM_CATEGORY_IMAGE_*
# variants. Those exist as enum members in this Python library, but the
# actual Gemini REST API (v1beta) rejects them with a 400 INVALID_ARGUMENT
# ("Invalid value ... HarmCategory") - confirmed against the live API, not a
# guess. That single bug caused every request to fail regardless of image
# content, which is what looked like "guardrails rejecting everything." The
# generic categories already apply to multimodal (image+text) requests.
_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
}


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return a lazily-initialized, process-wide Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        safety_settings=_SAFETY_SETTINGS,
    )


def _extract_text(content: str | list) -> str:
    """Normalize a LangChain message's ``.content`` into plain text.

    Newer Gemini models return ``content`` as a list of blocks (e.g.
    ``[{"type": "text", "text": "...", "extras": {...}}]``) instead of a
    plain string. Only the text blocks are relevant for display/speech.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return str(content)


def run_vision_prompt(image_base64: str, prompt_text: str) -> str:
    """Send an image + instruction prompt to Gemini and return the reply text.

    Used only by Personalized Assistance, which returns plain text rather
    than a structured schema.
    """
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": to_image_data_url(image_base64)},
        ]
    )
    logger.info("Invoking Gemini model=%s", settings.gemini_model)
    response = get_chat_model().invoke([message])
    return _extract_text(response.content)


def analyze_scene(image_base64: str, system_prompt: str, user_prompt: str) -> SceneAnalysis:
    """Run a structured scene analysis (summary, environment, objects, hazard, action).

    Args:
        image_base64: Base64-encoded image data (no data URI prefix).
        system_prompt: Shared persona/constraints (see ``SCENE_ANALYSIS_SYSTEM_PROMPT``).
        user_prompt: The per-feature focus instruction (navigation-narrative
            vs. object-enumeration). The output shape itself comes from
            :class:`~src.llm.schemas.SceneAnalysis`, not from prompt text.

    Returns:
        A validated ``SceneAnalysis`` instance.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": to_image_data_url(image_base64)},
            ]
        ),
    ]
    logger.info("Invoking Gemini model=%s for structured scene analysis", settings.gemini_model)
    structured_model = get_chat_model().with_structured_output(SceneAnalysis)
    return structured_model.invoke(messages)


def check_content_moderation(image_base64: str) -> ContentModerationResult:
    """Classify whether an image contains explicit/severe content (see the prompt for the exact categories).

    A narrow, explicit check owned entirely by this app - not delegated to
    Gemini's declarative ``safety_settings``, which has already been caught
    both rejecting valid requests outright and not reliably flagging real
    explicit content at any threshold tested.

    Returns:
        A validated ``ContentModerationResult`` instance.
    """
    messages = [
        SystemMessage(content=CONTENT_MODERATION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": CONTENT_MODERATION_USER_PROMPT},
                {"type": "image_url", "image_url": to_image_data_url(image_base64)},
            ]
        ),
    ]
    logger.info("Invoking Gemini model=%s for content moderation", settings.gemini_model)
    structured_model = get_chat_model().with_structured_output(ContentModerationResult)
    return structured_model.invoke(messages)


def classify_extracted_text(image_base64: str, extracted_text: str) -> OcrClassification:
    """Classify OCR-extracted text (e.g. medicine label, menu, signboard) and suggest an action.

    Args:
        image_base64: Base64-encoded image the text was extracted from, given as
            visual context alongside the text itself.
        extracted_text: The raw text returned by OCR.

    Returns:
        A validated ``OcrClassification`` instance.
    """
    user_prompt = OCR_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(extracted_text=extracted_text)
    messages = [
        SystemMessage(content=OCR_CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": to_image_data_url(image_base64)},
            ]
        ),
    ]
    logger.info("Invoking Gemini model=%s for OCR classification", settings.gemini_model)
    structured_model = get_chat_model().with_structured_output(OcrClassification)
    return structured_model.invoke(messages)
