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
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from src.llm.schemas import OcrClassification, SceneAnalysis
from src.prompts.vision_prompts import (
    OCR_CLASSIFICATION_SYSTEM_PROMPT,
    OCR_CLASSIFICATION_USER_PROMPT_TEMPLATE,
)
from src.utils.image_utils import to_image_data_url

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return a lazily-initialized, process-wide Gemini chat model instance."""
    return ChatGoogleGenerativeAI(
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
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
