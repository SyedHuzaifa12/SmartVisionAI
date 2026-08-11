"""Centralized application configuration, loaded from environment variables.

Keeping every tunable/secret value in one place makes it obvious what the
app depends on and lets each deployment (local dev, Docker, CI) override
behavior without touching source code.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "vision_logo.webp"


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for SmartVisionAI."""

    gemini_api_key: str
    gemini_model: str
    tesseract_cmd: str | None
    log_level: str
    scene_confidence_threshold: float
    hazard_confidence_threshold: float
    ocr_confidence_threshold: float

    @property
    def has_gemini_key(self) -> bool:
        """Whether a (non-placeholder) Gemini API key is configured."""
        return bool(self.gemini_api_key)


def _resolve_tesseract_cmd() -> str | None:
    """Resolve the Tesseract OCR binary path.

    Prefers an explicit ``TESSERACT_CMD`` env var (needed on Windows, where
    Tesseract isn't automatically on PATH). Falls back to auto-detection via
    ``PATH`` so the app also works out of the box on Linux/macOS/Docker.
    """
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        return configured
    return shutil.which("tesseract")


def get_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        tesseract_cmd=_resolve_tesseract_cmd(),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        # Below these, a response is too unreliable to present as-is; see
        # src/guardrails/confidence.py for how each threshold is applied.
        scene_confidence_threshold=float(os.getenv("SCENE_CONFIDENCE_THRESHOLD", "0.5")),
        hazard_confidence_threshold=float(os.getenv("HAZARD_CONFIDENCE_THRESHOLD", "0.5")),
        ocr_confidence_threshold=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.4")),
    )


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for the whole application."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


settings = get_settings()
