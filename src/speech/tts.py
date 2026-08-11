"""Text-to-speech synthesis via gTTS.

Was previously pyttsx3 (fully offline, via the local espeak driver). Switched
after pyttsx3's espeak driver proved unreliable on headless cloud containers:
its callback races garbage collection there (`ReferenceError: weakly-
referenced object no longer exists`), producing an empty/corrupt MP3 with no
error surfaced to the app. gTTS is a lightweight, free, no-API-key wrapper
around Google Translate's TTS endpoint - it returns audio bytes directly over
HTTP with no local audio driver involved, so that failure mode can't occur.
It does require outbound internet access, which Streamlit Cloud has.
"""

from __future__ import annotations

import io
import logging

from gtts import gTTS

logger = logging.getLogger(__name__)


def synthesize_speech(text: str) -> bytes:
    """Convert text to spoken audio and return the resulting MP3 bytes."""
    logger.info("Synthesizing speech for %d characters of text", len(text))
    buffer = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buffer)
    return buffer.getvalue()
