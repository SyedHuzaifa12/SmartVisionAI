"""Offline text-to-speech synthesis via pyttsx3."""

from __future__ import annotations

import logging
import os
import tempfile

import pyttsx3

logger = logging.getLogger(__name__)

SPEECH_RATE = 150
SPEECH_VOLUME = 1.0


def synthesize_speech(text: str) -> bytes:
    """Convert text to spoken audio and return the resulting MP3 bytes.

    Each call renders to its own temporary file (instead of a single shared
    filename) so concurrent Streamlit sessions don't overwrite one another's
    audio output; the temp file is removed once its bytes are read.
    """
    logger.info("Synthesizing speech for %d characters of text", len(text))
    fd, audio_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", SPEECH_RATE)
        engine.setProperty("volume", SPEECH_VOLUME)
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        with open(audio_path, "rb") as audio_file:
            return audio_file.read()
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
