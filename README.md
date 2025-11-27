# 🚀 SmartVisionAI  
### AI-Powered Assistive Vision System for the Visually Impaired  
Empowering users with real-time scene understanding, text extraction, and contextual audio assistance using cutting-edge AI.

---

## ⭐ Overview

**SmartVisionAI** is a multimodal accessibility platform designed to help visually impaired individuals perceive their surroundings. The system interprets visual information, extracts text, detects objects, and converts insights into speech — enabling non-visual interaction with the real world. :contentReference[oaicite:0]{index=0}

---

## 🔥 Problem Statement

Visually impaired users face significant challenges when interacting with their environment:

- Difficulty identifying objects or obstacles  
- Inability to read printed or digital text  
- Limited understanding of surroundings  
- Dependency on external help for visual tasks  

SmartVisionAI tackles these constraints through AI-driven visual perception and voice-based assistance. :contentReference[oaicite:1]{index=1}

---

## 🚀 Key Features

| Feature | Description |
|--------|-------------|
| 📸 **Scene Description** | Generates detailed, context-aware interpretations of uploaded images |
| 📝 **Text Extraction (OCR)** | Reads text content from images using Tesseract |
| 🔊 **Text-to-Speech** | Converts extracted text and scene insights to speech |
| 🎯 **Object Detection** | Identifies objects and obstacles for safe navigation |
| 💬 **Personalized Assistance** | Provides contextual help tailored to the uploaded image :contentReference[oaicite:2]{index=2} |

---

## 🧠 Architecture / System Workflow

```mermaid
flowchart LR
User -->|Upload Image| Streamlit_UI
Streamlit_UI --> AI_Engine[SmartVisionAI Engine]
AI_Engine -->|Scene Description| GoogleGenAI[Google Generative AI]
AI_Engine -->|Object Detection| LangChain
AI_Engine -->|Text Extraction| TesseractOCR
AI_Engine -->|Audio Output| Pyttsx3
Pyttsx3 --> User
