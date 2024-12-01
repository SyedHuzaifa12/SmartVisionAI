import os
import base64
import io

from PIL import Image
import pyttsx3
import pytesseract
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Get API key
api_key = "AIzaSyBHlgoNaJy0fg89UtNrQFFOL1Prznd6URI"

# Initialize Chat Model
chat_model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-1.5-flash")

# SmartVisionAI Branding and App Title
st.markdown(
    """
    <h1 style="color: #003366; text-align: center;">SmartVisionAI👁️</h1>
    <h3 style="color:#4B8BBE;">Transforming Vision with AI 🤖✨</h3> 
    <h3 style="color:#2F4F4F;">Empowering the visually impaired with real-time insights and audio guidance for a more accessible world! </h3>
    """, unsafe_allow_html=True
)
st.sidebar.image(
    r"F:\photo\g.webp",
    width=250
)
st.sidebar.title("💼 About SmartVisionAI")
st.sidebar.markdown(
    """
    📌 **Features**:  
    - 📸 **Describe Scene**: Provides detailed image descriptions.  
    - 📝 **Extract Text**: Extracts text using OCR.  
    - 🔊 **Text-to-Speech**: Converts text to audio output.  
    - 🔍 **Object Detection**: Identify objects in image for safe navigation.
    - 💬 **Personalized Assistance**: Provides context-specific information.
    

    🌟 **How it helps**:  
    Assists visually impaired users by describing scenes, extracting text, generating speech, detecting objects, and assisting specifically.  

    🤖 **Powered by**:  
    - **Google Gemini API** for scene analysis.  
     - **LangChain** for integrating AI.  
    - **Tesseract OCR** for text recognition.  
    - **gtts** for speech synthesis.
    - **Streamlit** for enhanced UI.
    """
)
st.sidebar.text_area("🎯 Instructions(how it works)", "1. Select a functionality. 2. Upload an image. 3. Click on the button to generate.")


# Helper: Text-to-Speech Conversion
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # Speed
    engine.setProperty("volume", 1)  # Volume
    audio_file = "text_to_speech_output.mp3"
    engine.save_to_file(text, audio_file)
    engine.runAndWait()
    st.audio(audio_file, format="audio/mp3")

# Feature 1: Real-Time Scene Understanding
def real_time_scene_understanding(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """You are a real-time scene interpreter for visually impaired users... (trimmed for brevity)"""},  # Instructions for scene understanding
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"},
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.markdown(f"### Scene Description 📸:\n{response_text}")
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Scene understanding failed: {e}")

# Feature 2: Text Extraction
def text_extraction(uploaded_image):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    try:
        extracted_text = pytesseract.image_to_string(uploaded_image)
        st.markdown(f"### Extracted Text 📝:\n{extracted_text}")
        text_to_speech(extracted_text)
    except Exception as e:
        st.error(f"Text extraction failed: {e}")

# Feature 3: Object Detection
def object_detection(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """You are a visual accessibility specialist... (trimmed for brevity)"""},  # Instructions for object detection
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"},
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.markdown(f"### Detected Objects 🕵️‍♂️:\n{response_text}")
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Object detection failed: {e}")

# Feature 4: Personalized Assistance
def personal_assistance(image_base64):
    hmessage = HumanMessage(
        content=[
            {"type": "text", "text": """As an assistive technology specialist... (trimmed for brevity)"""},  # Instructions for personalized assistance
            {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"},
        ]
    )
    try:
        response = chat_model.invoke([hmessage])
        response_text = response.content
        st.markdown(f"### Personalized Assistance 💡:\n{response_text}")
        text_to_speech(response_text)
    except Exception as e:
        st.error(f"Personalized assistance failed: {e}")

# Image Upload and Feature Selection
uploaded_image = st.file_uploader("📤 **Upload an Image**", type=["jpg", "jpeg", "png"])
feature = st.sidebar.radio(
    "### **Select a Feature ⚙️:**",
    ["Real-Time Scene Understanding", "Text-to-Speech Conversion", "Object Detection", "Personalized Assistance"],
    index=0,
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image 📷", use_column_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    if feature == "Real-Time Scene Understanding" and st.button("🔍 **Run Scene Understanding**"):
        with st.spinner("Analyzing... Be Patient!"):
            real_time_scene_understanding(image_base64)

    elif feature == "Text-to-Speech Conversion" and st.button("📝 **Extract & Convert Text**"):
        with st.spinner("Extracting text..."):
            text_extraction(image)

    elif feature == "Object Detection" and st.button("🕵️‍♂️ **Run Object Detection**"):
        with st.spinner("Detecting objects..."):
            object_detection(image_base64)

    elif feature == "Personalized Assistance" and st.button("💡 **Run Personalized Assistance**"):
        with st.spinner("Providing personalized guidance..."):
            personal_assistance(image_base64)
else:
    st.info("🚨 **Please upload an image to proceed.**")


# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align:left;">
        <p>Powered by <strong>Google Gemini API,LangChain,Streamlit </strong> | Syed Huzaifa ❤️</p>
    </footer>
    """,
    unsafe_allow_html=True,
)
