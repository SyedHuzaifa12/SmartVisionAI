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
        <h1 style="color: #003366; text-align: center;">SmartVisionAI👁️</h1>
        <h3 style="color:#4B8BBE;">Transforming Vision with AI 🤖✨</h3>
        <h3 style="color:#2F4F4F;">Empowering the visually impaired with real-time insights and audio guidance for a more accessible world! </h3>
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

    st.sidebar.title("💼 About SmartVisionAI")
    st.sidebar.markdown(
        """
        📌 **Features**:
        - 📸 **Scene Understanding**: Navigation-focused scene summary, environment type, key objects, and a Safe/Caution/Dangerous hazard rating.
        - 📝 **Extract Text**: OCR text extraction with automatic category detection (medicine label, menu, signboard, etc.).
        - 🔍 **Object Detection**: Prioritized, position-aware objects and hazards for safe navigation.
        - 💬 **Personalized Assistance**: Context-specific help understanding a document, label, or item.
        - 🔊 Every result includes a spoken audio summary.

        🌟 **How it helps**:
        Assists visually impaired users by describing scenes, assessing hazards, extracting and classifying text, detecting objects, and providing spoken guidance throughout.

        🤖 **Powered by**:
        - **Google Gemini API** for scene analysis.
        - **LangChain** for integrating AI.
        - **Tesseract OCR** for text recognition.
        - **pyttsx3** for speech synthesis.
        - **Streamlit** for enhanced UI.
        """
    )
    st.sidebar.text_area(
        "🎯 Instructions(how it works)",
        "1. Select a functionality. 2. Upload an image. 3. Click on the button to generate.",
    )

    return st.sidebar.radio(
        "### **Select a Feature ⚙️:**",
        FEATURE_OPTIONS,
        index=0,
    )


def render_footer() -> None:
    """Render the page footer/credits."""
    st.markdown(
        """
        <hr>
        <footer style="text-align:left;">
            <p>Powered by <strong>Google Gemini API,LangChain,Streamlit </strong> | Syed Huzaifa ❤️</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )
