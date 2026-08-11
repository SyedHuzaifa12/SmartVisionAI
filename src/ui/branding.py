"""Static branding/markup blocks, kept separate from app logic and orchestration."""

from __future__ import annotations

import streamlit as st

from config import LOGO_PATH

FEATURE_OPTIONS = [
    "Real-Time Scene Understanding",
    "Text-to-Speech Conversion",
    "Object Detection",
    "Personalized Assistance",
]


def render_header() -> None:
    """Render the page title and tagline."""
    st.markdown(
        """
        <h1 style="color: #003366; text-align: center;">SmartVisionAI</h1>
        <h3 style="color:#4B8BBE;">AI-powered scene understanding for the visually impaired</h3>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the sidebar (logo, about section, instructions, feature picker).

    Returns:
        The currently selected feature name.
    """
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=250)

    st.sidebar.title("About SmartVisionAI")
    st.sidebar.markdown(
        """
        **Features**
        - **Scene Understanding** - navigation-focused scene summary, environment type, key objects, and a Safe/Caution/Dangerous hazard rating.
        - **Extract Text** - OCR text extraction with automatic category detection (medicine label, menu, signboard, etc.).
        - **Object Detection** - prioritized, position-aware objects and hazards for safe navigation.
        - **Personalized Assistance** - context-specific help understanding a document, label, or item.

        Every result includes a spoken audio summary.

        **How it helps**

        Assists visually impaired users by describing scenes, assessing hazards, extracting and classifying text, detecting objects, and providing spoken guidance throughout.

        **Powered by**
        - Google Gemini API for scene analysis
        - LangChain for AI integration
        - Tesseract OCR for text recognition
        - gTTS for speech synthesis
        - Streamlit for the UI
        """
    )
    st.sidebar.text_area(
        "Instructions (how it works)",
        "1. Select a feature. 2. Upload an image. 3. Click the button to generate a result.",
    )

    return st.sidebar.radio(
        "Select a feature",
        FEATURE_OPTIONS,
        index=0,
    )


def render_footer() -> None:
    """Render the page footer/credits."""
    st.markdown(
        """
        <hr>
        <footer style="text-align:left;">
            <p>Powered by <strong>Google Gemini API, LangChain, Streamlit</strong> &middot; Built by Syed Huzaifa</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )
