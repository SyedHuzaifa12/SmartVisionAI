"""Text extraction from images via Tesseract OCR."""

from __future__ import annotations

import logging

import pytesseract
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def extract_text(image: Image.Image) -> str:
    """Run OCR on an image and return the extracted text.

    Raises:
        pytesseract.TesseractError: If the Tesseract binary cannot be found
            or fails to process the image.
    """
    logger.info("Running OCR text extraction")
    return pytesseract.image_to_string(image)


def extract_text_with_confidence(image: Image.Image) -> tuple[str, float]:
    """Run OCR and return the extracted text plus a measured confidence score.

    Confidence here is Tesseract's own average word-level confidence
    (0-100, normalized to 0.0-1.0) - a real, measured signal from the OCR
    engine itself, rather than an LLM guessing how sure it is about text it
    didn't generate. Words Tesseract couldn't score are excluded (it reports
    -1 for non-text regions).
    """
    logger.info("Running OCR text extraction with confidence")
    text = pytesseract.image_to_string(image)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_confidences = [
        int(conf) for conf in data.get("conf", []) if str(conf).lstrip("-").isdigit() and int(conf) >= 0
    ]
    confidence = (sum(word_confidences) / len(word_confidences) / 100) if word_confidences else 0.0
    return text, confidence
