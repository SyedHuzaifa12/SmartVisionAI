# 👁️ SmartVisionAI

An AI-powered assistive vision platform that helps visually impaired individuals understand images — converting visual information into descriptive text, extracted text, object insights, and spoken audio output, built with the guardrails, confidence estimation, and observability a production AI system needs.

---

## 📖 Overview

SmartVisionAI is a Streamlit application that analyzes an uploaded image and, on request, will:

- 🖼 Describe the scene with a navigation-focused summary, environment classification, and prioritized objects
- 🚦 Assess hazards and classify overall safety as **Safe / Caution / Dangerous**, with an evidence-based explanation
- 📝 Extract printed/digital text via OCR and classify what kind of text it is (label, menu, sign, document, etc..)
- 🎯 Detect and prioritize objects relevant to safe navigation, with left/right/front/behind positioning
- 💬 Provide context-specific assistive guidance for a document, label, or item
- 🔊 Speak every result aloud as a concise audio summary

Underneath those four features is a small but real AI engineering pipeline: every Gemini response is schema-validated and retried before it's trusted, every response carries a self-reported confidence that gates whether it's shown as-is, every hazard verdict comes with an evidence-only explanation, and every request is timed and scored for internal QA. None of that is visible as a "feature" in the UI - it's the difference between calling an LLM API and engineering around one.

Visually impaired users struggle with understanding surroundings, reading printed text, identifying objects, interpreting visual cues, and assessing whether a space is safe to move through. SmartVisionAI turns an image into **structured, spoken, contextual knowledge**, letting users interact with visual content more independently.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📸 **Scene Understanding** | Structured, navigation-focused analysis: scene summary, environment type, prioritized objects with position, and a Safe/Caution/Dangerous hazard rating with an evidence-based explanation. |
| 🎯 **Object Detection** | Same structured analysis, weighted toward precisely identifying and positioning individual objects/hazards rather than narrating the scene. |
| 📝 **OCR Text Extraction** | Reads printed/digital text via Tesseract OCR (with a measured confidence score), then classifies it (medicine label, business card, signboard, menu, document, book page, product packaging). |
| 💬 **Personalized Assistance** | Provides context-specific, assistive guidance for a document, label, or item. |
| 🔊 **Audio Summary** | Every feature speaks a concise, purpose-built narration aloud via `gTTS` — not a raw dump of the on-screen text. |
| 🛡️ **Guardrails** *(internal)* | Every structured response is validated, retried once if malformed, and replaced with a safe fallback if it still fails - never surfaced to the user as an error unless truly unrecoverable. |
| 🚫 **Content moderation** *(internal)* | A narrow, explicit check (real nudity, graphic violence, brandished weapons, hate symbols) gates every image once, before any feature runs - fails open on its own technical errors so it never blocks an ordinary photo. |
| 📊 **Confidence gating** *(internal)* | Low-confidence scene/hazard responses degrade gracefully instead of asserting false certainty; hazard verdicts fail toward caution, never toward false reassurance. |
| ⏱️ **Pipeline Metrics** *(dev panel)* | Per-stage latency, model name, and validation status, shown in a collapsible panel below each result. |
| 🧪 **AI Evaluation** *(dev panel)* | Rule-based completeness/safety/navigation/OCR-usefulness/overall scores for every response, shown in a collapsible panel - development signal only, never the primary answer. |
| ⚡ **Response caching** *(internal)* | Identical (feature, image) requests skip Gemini entirely after the first time - across all visitors, not just one session. |
| 🚦 **Rate limiting** *(internal)* | Caps requests per browser session (default 5/hour) to protect the shared free-tier API key from rapid repeated clicks. |

---

## 🏗 AI Pipeline Architecture

```
                        ┌─────────────────────────┐
        User ──────────▶│      Streamlit UI        │  app.py (thin orchestration only)
  (image + feature)     │  upload · sidebar ·       │
                        │  render result            │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │      Service layer        │  src/services/vision_assistant.py
                        │  composes every stage     │  (the actual pipeline)
                        │  below into one            │
                        │  FeatureResult             │
                        └────────────┬─────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
┌───────────────┐          ┌──────────────────┐         ┌──────────────────┐
│  Multimodal    │          │  OCR extraction    │         │  Prompt layer      │
│  Gemini call    │◀────────│  (Tesseract, with  │         │  system prompt +   │
│  (LangChain     │  image  │  measured word-     │         │  user prompt,      │
│  with_structured│  base64 │  level confidence)  │         │  versioned          │
│  _output)       │          └──────────────────┘         │  (src/prompts)     │
└───────┬─────────┘                                         └──────────────────┘
        │ SceneAnalysis / OcrClassification (typed, includes self-reported confidence)
        ▼
┌──────────────────────────┐
│  Guardrails                │  src/guardrails/validation.py
│  validate -> retry once -> │  required fields? hazard enum valid? summary
│  safe fallback              │  meaningful? Dangerous implies objects present?
└────────────┬───────────────┘
             ▼
┌──────────────────────────┐
│  Confidence gate           │  src/guardrails/confidence.py
│  low scene_confidence ->   │  replace with honest "recapture" message
│  low hazard_confidence ->  │  escalate toward Caution, never downgrade
└────────────┬───────────────┘
             │
   ┌─────────┼──────────────────────┐
   ▼                                ▼
┌───────────────────┐   ┌─────────────────────────┐
│  Observability      │   │  AI Evaluation (dev)      │
│  src/observability/  │   │  src/evaluation/           │
│  per-stage latency,  │   │  rule-based scoring, NOT   │
│  model name,          │   │  a second LLM call - no    │
│  validation status    │   │  self-grading bias          │
└───────────────────┘   └─────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│  Text-to-Speech             │  src/speech/tts.py (gTTS)
└────────────┬───────────────┘
             ▼
  Rendered as: Scene Summary → Environment Type → Important Objects →
  Hazard Assessment (+ confidence) → Extracted Text → Suggested Action →
  Audio Summary, with Pipeline Metrics and AI Evaluation in collapsed panels
```

Configuration (API keys, model name, Tesseract path, confidence thresholds) is centralized in `config.py` and loaded from environment variables / a local `.env` file — no secrets live in source code.

---

## 🧩 Structured Output Design

Scene Understanding and Object Detection both return a typed `SceneAnalysis` object (`src/llm/schemas.py`), obtained via LangChain's `with_structured_output` against a Pydantic schema, rather than parsing hazard level / object list out of freeform prose. OCR classification returns a typed `OcrClassification` the same way.

Why this matters: freeform-prose parsing is fragile - a slightly different phrasing ("this looks risky" vs. "Dangerous") silently breaks a regex. A typed schema makes the shape of the answer a contract the model must fill in, not a pattern to be reverse-engineered from text:

```python
class SceneAnalysis(BaseModel):
    scene_summary: str
    environment_type: str
    important_objects: list[str]
    hazard_level: Literal["Safe", "Caution", "Dangerous"]
    hazard_explanation: str
    suggested_action: str
    audio_summary: str
    scene_confidence: float   # 0.0-1.0
    hazard_confidence: float  # 0.0-1.0
```

Confidence fields live *in* the schema, not bolted on afterward - the model must commit to a calibration estimate on every single response, which is what makes gating on it possible at all (see **Confidence Estimation** below).

---

## ✍️ Prompt Engineering Strategy

Prompts are split into three concerns that used to be tangled together in one string:

- **System prompt** — persona and hard constraints shared by every call of that type (never invent facts, never guess identity, be conservative about confidence, state only observable evidence). Rarely changes between features.
- **User prompt** — the per-feature task instruction (navigation-focused narrative vs. object enumeration vs. OCR classification).
- **Output schema** — not prompt text at all; it lives in `src/llm/schemas.py` and is enforced by `with_structured_output`, so field-level instructions (what a hazard_level means, when to lower confidence) live once, next to the field, instead of being repeated in every prompt that produces it.

This project's prompts have gone through four real iterations (documented in full in `src/prompts/vision_prompts.py`), kept here because *why* the current version is better is more useful than the text itself:

| Version | Change | Problem it fixed |
|---|---|---|
| v1 | Single unstructured prompt per feature | Hazard level / object list scraped from prose with string matching - broke on any phrasing variance |
| v2 | Added explicit hazard-first structure | Introduced a new bug: model reasoned "elevated view ⇒ no hazards", a non-sequitur (a cliff edge is exactly what makes elevation hazardous) |
| v3 | Explicitly forbade inferring safety from distance/elevation; differentiated Scene Understanding from Object Detection | Previously the two features produced near-identical prose, defeating the point of having both |
| v4 (current) | Moved to `with_structured_output`; split into system/user prompts; added confidence + evidence-only hazard explanations | Removed the need to repeat formatting instructions per-prompt; made the schema, not prose, the source of truth |

---

## 🛡️ Guardrails

Guardrails apply on both sides of the pipeline - what the user sends in, and what the model sends back.

**Input validation** (`src/utils/image_utils.py:validate_uploaded_image`), applied before an uploaded file touches OCR or Gemini:
- **Size limit** (default 10 MB) - rejects oversized uploads with a clear message rather than the app default's generous, unbounded ceiling.
- **Corruption check** - `Image.open()` is lazy in PIL; it doesn't fully decode until pixel data is accessed. This forces a full `.load()` immediately, so a truncated or non-image file (or one merely renamed to `.jpg`) fails fast with "this file isn't a valid, readable image" instead of crashing deep in the pipeline with a raw traceback.
- **Dimension limit** (default 6000px on the longest side) - blocks pathologically large images that would otherwise burn excessive memory, latency, and API cost for no benefit.

**Content safety - two layers, for different reasons:**

- Gemini's declarative `safety_settings` (`src/llm/vision_client.py`) is still set - all 4 generic categories at `BLOCK_ONLY_HIGH`, the least strict level - but is *not* relied on as the primary gate anymore. In testing, this mechanism was caught twice: first rejecting every request outright with a `400 INVALID_ARGUMENT` (the `HARM_CATEGORY_IMAGE_*` enum values are valid in the Python library but not accepted by the actual API - a real bug, not a guess, confirmed via the error surfaced in `PipelineMetrics.error_detail`), and second, even after removing those, not reliably flagging real explicit content at any threshold tested. A safety mechanism that's an opaque black box - one you can't verify is actually doing anything - isn't a safety mechanism you can rely on.
- The actual gate is an explicit, narrow **content moderation check this app owns end to end** (`src/llm/vision_client.py:check_content_moderation`, schema `ContentModerationResult`): one small Gemini call with a tightly-scoped prompt that flags nudity/partial nudity, bikinis/swimwear/underwear/lingerie, sexually suggestive content, graphic violence/gore, a weapon being brandished to threaten or harm, and explicit hate symbols/speech - while explicitly telling the model what *not* to flag (ordinary, fully/normally clothed photos of people, posters, kitchen knives, vehicles). The threshold for what counts as "suggestive enough to flag" was widened after testing showed the first version let through bikini photos and an intimate underwear scene - a good example of iterating a prompt against real observed failures rather than shipping the first version untested. It runs once per unique image (cached, not once per feature click) and gates all 4 features from one shared call site in `app.py`. Unlike the hazard-detection fallback (which fails toward Caution), this check **fails open**: if the moderation call itself can't be verified after a retry, the image is treated as fine - a technical glitch in a side-check should never block a normal photo. If Gemini's declarative filter still blocks a request for some other reason, the existing generate → retry → fallback path (below) already catches that as a failure and degrades gracefully.

**Output validation**: every structured LLM call goes through one shared middleware (`src/guardrails/validation.py`), not scattered per-feature checks:

```
generate → validate → (if invalid) retry once → (if still invalid) safe fallback
```

Validation checks: required fields are non-empty and meaningful (not just present), `hazard_level` is one of the three allowed values, and — the one deliberately conditional rule — a `"Dangerous"` verdict with zero identified objects is rejected as internally inconsistent, while a `"Safe"` scene with an empty object list is accepted (a real empty field is not the same as a missing one; a blanket "object list must never be empty" rule would misfire on ordinary safe scenes with nothing notable to report).

An exception raised during generation (a transient API error, or output so malformed even structured-output coercion fails) is treated the same as a failed validation - it triggers the same retry-then-fallback path rather than crashing the feature. The retry waits a short backoff (2s) first, in case the cause was transient (e.g. a rate-limit blip). Critically, the *actual* exception (type + message) is captured and surfaced through `PipelineMetrics.error_detail` all the way to the Pipeline Metrics dev panel - without this, a rate-limit error, a Gemini safety block, and a genuine bug all produced the identical generic fallback text, making the real cause undiagnosable from the UI. Every fallback is now a debuggable event, not a dead end.

The fallback itself is intentionally conservative: it defaults `hazard_level` to `"Caution"`, never `"Safe"` - when the system can't verify its own output, failing toward caution is the only acceptable direction for a safety-relevant tool.

---

## 📊 Confidence Estimation

Three different signals feed into "how much should we trust this?", and they are deliberately *not* all the same kind of number:

- **`scene_confidence` / `hazard_confidence`** — Gemini's own self-reported estimate (0.0-1.0), requested explicitly in the schema and steered by the system prompt to be conservative (lower for blurry/dark/cluttered/ambiguous images) rather than defaulting high just because an answer was produced.
- **OCR confidence** — *not* self-reported by an LLM. It's Tesseract's own measured average word-level confidence (`pytesseract.image_to_data`), a real signal from the OCR engine that actually did the extraction, rather than a model guessing how sure it is about text it didn't generate.

Gating (`src/guardrails/confidence.py`):

- `scene_confidence` below threshold → the entire response is replaced with an honest "I'm not confident enough to describe this reliably, please capture another image" message, rather than presenting an unreliable description as fact.
- `hazard_confidence` below threshold → the hazard verdict is **escalated toward Caution, never downgraded**. An uncertain "Safe" becomes "Caution" with a caveat appended; an uncertain "Dangerous" stays "Dangerous" (you cannot become *more* cautious than that). This asymmetry is the core safety property of the whole gate.
- OCR confidence below threshold → the extracted text is still shown (partial OCR is often still useful, unlike a wrong hazard call) but with a visible "may be inaccurate, consider retaking" warning banner.

**Known limitation, stated honestly:** an LLM's self-reported confidence is a heuristic proxy, not a calibrated probability - there is no ground-truth accuracy guarantee behind "0.82". Mitigating that is exactly why hazard-confidence gating fails toward caution instead of trusting the number at face value, and why OCR uses a real measured signal instead of an LLM guess wherever one is available.

**A concrete finding from testing this, not a hypothetical:** the first version of the confidence instruction ("be conservative, don't default high") had no calibration anchor, and in practice the model reported `scene_confidence`/`hazard_confidence` near **0%** even for clear, well-lit, unambiguous photos - a signal that's useless to threshold against isn't a signal at all. Two changes followed: the prompt now gives the model concrete anchor points (a clear photo should score 0.8-1.0; only genuinely blurry/dark/ambiguous images should score low), and - since a prompt tweak alone isn't something to trust without measurement - **gating on these two fields defaults to off** (`SCENE_CONFIDENCE_THRESHOLD`/`HAZARD_CONFIDENCE_THRESHOLD` default to `0.0`, which can mathematically never trigger). The raw values are still computed and shown on every response for transparency; the thresholds are there to raise once the improved calibration is actually verified against real traffic, not before. Shipping a broken gate that rejects good input is worse than shipping an honest, visible number with no gate yet.

---

## 💡 Explainability

Every hazard assessment includes a `hazard_explanation` field constrained (via both the system prompt and the schema's field description) to state **only observable visual evidence**, not a reasoning trace:

> "Caution because a cow and goats occupy the walking path and pottery tools are on the ground."

No chain-of-thought is requested or exposed. The model is explicitly told to skip its reasoning process and state the conclusion plus the visible evidence for it - which is both more useful to a listener (no rambling) and safer (no incentive to fabricate a plausible-sounding thought process for an answer it isn't sure of).

---

## 🧪 AI Evaluation (Development/Demo Only)

Every response is scored on five 0-100 dimensions and shown in a collapsible **"AI Evaluation (dev)"** panel: Completeness, Safety, Navigation usefulness, OCR usefulness (OCR feature only), and Overall quality.

**Design decision worth calling out:** these scores are computed with plain rule-based heuristics (`src/evaluation/response_evaluator.py`), *not* a second Gemini call asking "how good is this response?" That was a deliberate tradeoff, not an oversight:

- A second LLM call grading the first model's own output risks **self-grading bias** - the same model that made a mistake is not a reliable judge of that mistake.
- It would **double the latency and cost** of every request for a panel that's dev-facing, not user-facing.
- Rule-based scores are **deterministic and reproducible** - useful for a demo/debugging tool in a way a second LLM call (with its own variance) is not.

Safety score, for example, is derived directly from `hazard_level` and `hazard_confidence` (a scene rated "Dangerous" with high confidence scores low on safety; the same rating with low confidence scores even lower - confidence discounts the baseline rather than being ignored). Navigation usefulness counts actual position-indicating language (left/right/ahead/behind/near/far) rather than asking a model to self-assess "usefulness."

This panel never gates or alters the primary response above it - it is purely additional signal for whoever is developing or demoing the app.

---

## 🧠 AI System Design Decisions

A few choices worth explaining if asked about them directly:

- **Why not ask the LLM to grade itself for AI Evaluation?** Self-grading bias, latency/cost doubling, and non-reproducibility - see above.
- **Why does OCR confidence come from Tesseract instead of Gemini?** Gemini didn't perform the extraction, so it has no real signal about its accuracy - only Tesseract's own word-level output does. Mixing a measured signal (OCR) with a self-reported one (scene/hazard) where each is actually available is more honest than uniformly asking every component to "rate your own confidence."
- **Why does the confidence gate fail toward Caution instead of hiding the hazard field entirely?** Silently omitting a hazard assessment is worse than an uncertain one for an accessibility tool - the user still needs *some* guidance. Escalating (never downgrading) preserves that while refusing to assert false certainty.
- **Why keep all 4 original buttons instead of merging Scene Understanding and Object Detection?** They now share one schema and one system prompt, differing only in the user-prompt "focus," which is exactly the DRY outcome you'd want - without removing a feature the existing app already promised.
- **Why is Personalized Assistance the one feature *not* touched by guardrails/confidence/evaluation?** It returns free text, not the `SceneAnalysis` schema, and serves a different purpose (interpreting one held item, not hazard/navigation judgment) - forcing it into the same schema would be scope creep, not consistency.
- **Why is rate limiting per-session instead of per-device?** Streamlit Cloud gives no stable per-visitor identity without a backend/database, which is explicitly out of scope here. A per-session limit is honest about what it actually enforces (a new tab resets it) rather than pretending to a guarantee it can't provide.
- **Why doesn't a cache hit skip the rate limit?** They're deliberately independent: the limiter answers "how many times can this session trigger the pipeline," the cache separately answers "how many of those actually call Gemini." Coupling them (exempting cache hits from the count) saves a little API load at the cost of real complexity, for a portfolio-scale demo where that tradeoff isn't worth it.
- **A real incident, not a hypothetical:** an earlier version of the safety settings included `HARM_CATEGORY_IMAGE_*` variants - valid enum members in the `langchain-google-genai` Python library, but rejected by the actual Gemini REST API with a `400 INVALID_ARGUMENT`. Every single request failed, regardless of image content, which looked identical to "guardrails being too strict" from the outside - you cannot tell "the model refused this" apart from "the request itself was malformed" without seeing the underlying error. That ambiguity is exactly why `PipelineMetrics.error_detail` (see Guardrails/Observability above) exists: it turned a multi-round guessing exercise into a one-line diagnosis. The fix was removing the four `IMAGE_*` entries and keeping only the four generic categories, which already cover multimodal (image+text) requests.

---

## 🚦 Caching & Rate Limiting

Two lightweight production-hardening measures, both intentionally simple rather than backed by a database:

- **Response caching** (`src/utils/response_cache.py`) - a bounded (LRU, 128-entry), thread-safe, in-process cache keyed on `(feature, image)`. If any visitor has already run the same feature on the same image, every subsequent request for that exact pair - from any session - returns instantly with **zero additional Gemini calls**. Hit/miss is returned explicitly (not hidden inside a framework), so the Pipeline Metrics panel can honestly say "served from cache" instead of showing stale latency numbers as if they were freshly measured.
- **Rate limiting** (`src/utils/rate_limiter.py`) - a sliding-window limit (default: 5 requests/hour, configurable via `RATE_LIMIT_MAX_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS`) tracked in `st.session_state`. This protects a shared, free-tier API key from rapid repeated clicks within one browser session.

Both are plain, framework-independent Python (the rate limiter takes any dict-like `session_state`, real or a test double) and fully unit-tested without needing Streamlit, a network call, or a database - consistent with the rest of `src/`.

**Honest scope limits, stated directly:** the cache is in-memory only (cleared on every redeploy) and the rate limit is per-browser-session, not per-device or per-IP. Both are correctly described as *lightweight demo-scale safeguards*, not the persistent, distributed versions a real production system with sustained traffic would need (Redis-backed cache, IP- or auth-based limiting). That's an appropriate scope for this project, not a gap to apologize for.

---

## 📂 Folder Structure

```
SmartVisionAI/
│
├── app.py                     # Streamlit entry point (thin orchestration only)
├── config.py                  # Environment-driven settings (API keys, paths, confidence thresholds)
├── requirements.txt           # Python dependencies
├── README.md
├── LICENSE
├── .env.example                # Template for local environment variables
├── .gitignore
│
├── assets/
│   └── vision_logo.webp       # Sidebar logo
│
├── src/
│   ├── llm/                    # Gemini vision-language model integration
│   │   ├── vision_client.py    # Prompt/image -> Gemini call (plain text and structured)
│   │   └── schemas.py          # SceneAnalysis / OcrClassification / ContentModerationResult / ResponseEvaluation
│   ├── ocr/                    # Tesseract OCR text extraction (with measured confidence)
│   │   └── extractor.py
│   ├── speech/                 # Text-to-speech synthesis
│   │   └── tts.py
│   ├── prompts/                 # Versioned prompt templates (system/user split per feature)
│   │   └── vision_prompts.py
│   ├── guardrails/              # Validation middleware + confidence gating
│   │   ├── validation.py        # generate -> validate -> retry -> fallback
│   │   └── confidence.py        # scene/hazard confidence gating (fail toward caution)
│   ├── observability/           # Lightweight pipeline timing/metrics
│   │   └── metrics.py
│   ├── evaluation/               # Dev-only rule-based response quality scoring
│   │   └── response_evaluator.py
│   ├── services/                 # Feature-level orchestration - the actual pipeline
│   │   └── vision_assistant.py   # composes LLM + guardrails + confidence + metrics + eval
│   ├── ui/                      # Streamlit branding/presentation components
│   │   └── branding.py
│   └── utils/                   # Shared helpers (image <-> base64, response cache, rate limiter)
│       └── image_utils.py
│
├── tests/                       # Unit + integration tests (pytest)
│
└── docs/
    ├── screenshots/             # App screenshots (see below)
    └── SmartVisionAI-Presentation.pptx
```

Each `src/` subpackage owns a single responsibility, so a component (say, swapping Tesseract for a cloud OCR API, or the rule-based evaluator for an LLM-judge later) can be changed in one place without touching UI, guardrails, or LLM code.

---

## 🛠 Tech Stack

- **Python** — core application logic
- **Streamlit** — web UI
- **Google Gemini** (`gemini-flash-latest` by default) via **LangChain** — scene understanding, object detection, personalized assistance, OCR classification, all via `with_structured_output` where structured
- **Pydantic** — structured-output schemas, field-level validation constraints (`ge`/`le` on scores and confidences)
- **Tesseract OCR** (`pytesseract`) — text extraction with measured word-level confidence
- **gTTS** — text-to-speech (Google Translate TTS endpoint, no API key required)
- **python-dotenv** — environment-based configuration
- **pytest** — unit and integration testing (Gemini/OCR calls mocked in service-layer tests)

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
  - Windows: [installer](https://github.com/UB-Mannheim/tesseract/wiki), then set `TESSERACT_CMD` in `.env`
  - Linux: `sudo apt install tesseract-ocr`
  - macOS: `brew install tesseract`
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/SyedHuzaifa12/SmartVisionAI.git
cd SmartVisionAI

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then edit .env and set GEMINI_API_KEY (and TESSERACT_CMD on Windows)

# 5. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Running tests

```bash
pytest
```

### Troubleshooting: `404 NOT_FOUND` on the Gemini model

Google periodically retires dated model snapshots for new API keys/projects. If you see an error like
`Error calling model '...' (NOT_FOUND): 404 ... is not found / no longer available`, list the models your
key actually supports and set `GEMINI_MODEL` in `.env` to one of them:

```bash
python -c "import google.generativeai as genai, os; from dotenv import load_dotenv; load_dotenv(); genai.configure(api_key=os.getenv('GEMINI_API_KEY')); [print(m.name) for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]"
```

---

## 📸 Screenshots

> Add screenshots of the running app here.

| Scene Understanding | Object Detection |
|---|---|
| `docs/screenshots/scene-understanding.png` | `docs/screenshots/object-detection.png` |

---

## 🚀 Future Improvements

- 🌐 Multilingual voice output
- 📸 Real-time camera stream support
- 👓 Wearable integration (smart glasses)
- ⚡ Offline/edge deployment
- 🗣️ Speech-to-text input for hands-free feature selection
- 📈 Persisted evaluation history (currently per-request only) to track quality trends over time

---

## 🌟 AI Engineering Highlights

- Designed a **guardrails middleware** (validate → retry once → safe fallback) shared by every structured LLM call, instead of ad-hoc per-feature validation - with a fallback that fails toward caution, never false safety.
- Implemented **confidence-aware gating** on top of self-reported model confidence: low scene confidence triggers an honest "recapture" message; low hazard confidence escalates the verdict toward Caution and never downgrades it.
- Distinguished **measured vs. self-reported confidence** - OCR confidence comes from Tesseract's actual word-level output, not an LLM's guess about text it didn't generate.
- Added **lightweight pipeline observability** (per-stage latency, model name, validation/retry status) without pulling in a tracing stack a single-process app doesn't need.
- Built a **rule-based AI evaluation layer** scoring completeness/safety/navigation-usefulness/OCR-usefulness, deliberately avoiding a second "LLM-as-judge" call to sidestep self-grading bias and doubled latency/cost - and explicitly documented that tradeoff.
- Refactored prompts into **versioned, system/user/schema-separated** templates, with the actual iteration history (why v4 outperforms v1-v3) documented in-repo rather than asserted.
- Constrained hazard explanations to **evidence-only, non-chain-of-thought** text, matching real production-LLM safety practice around not exposing reasoning traces.
- Replaced freeform LLM prose with **typed, schema-validated structured output** (Pydantic + LangChain `with_structured_output`) — reliable to render and unit-test instead of regex-scraped from paragraphs.
- Replaced hardcoded secrets and machine-specific file paths with **environment-based configuration** (`python-dotenv` + a typed `Settings` object, including tunable confidence thresholds).
- Fixed a concurrency bug where all users shared one text-to-speech output file, by generating a unique temp file per request.
- Set up a **pytest** suite spanning schema validation, guardrail retry/fallback logic, confidence-gate edge cases, rule-based evaluation, and mocked service-layer integration - not just happy-path checks.

---

## 📜 License

Released under the [MIT License](LICENSE).

## 🙋 Contact

**Syed Huzaifa** — AI & Data Science Engineer
🔗 [GitHub](https://github.com/SyedHuzaifa12) · [LinkedIn](https://linkedin.com/in/syedhuzaifa34)
