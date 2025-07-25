# # Import Setup
# import truststore
# truststore.inject_into_ssl()
#
# import os
# import logging
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import httpx
# from glossary import GLOSSARY
#
# # Logging Setup
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# logging.basicConfig(
#     level=LOG_LEVEL,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )
# logger = logging.getLogger("translation-app")
# logger.info("Starting translation service …")
#
# # Environment and Configuration
#
# load_dotenv()
# DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
# if not DEEPL_API_KEY:
#     logger.critical("DEEPL_API_KEY not set; aborting start‑up")
#     raise RuntimeError("DEEPL_API_KEY not set")
#
# DEEPL_ENDPOINT = "https://api.deepl.com/v2/translate"
#
# #Helper function
# def apply_glossary(text: str, glossary: dict) -> str:
#     """Replace glossary terms (longest first)."""
#     for en, nl in sorted(glossary.items(), key=lambda x: -len(x[0])):
#         text = text.replace(en, nl)
#     return text
#
#
# # def reverse_glossary(text: str, glossary: dict) -> str:
# #     """Reverse‑replace Dutch glossary terms back to English (unused, but kept)."""
# #     for en, nl in sorted(glossary.items(), key=lambda x: -len(x[1])):
# #         text = text.replace(nl, en)
# #     return text
#
# #FastAPI application
# app = FastAPI(title="English → Dutch Translation Service")
#
#
# class TranslationRequest(BaseModel):
#     text: str
#
#
# class TranslationResponse(BaseModel):
#     dutch: str
#
#
# async def translate_to_dutch(text: str) -> str:
#     """Translate English text to Dutch, enforcing glossary replacements."""
#     logger.debug("Received text (len=%d). Pre‑processing with glossary …", len(text))
#     preprocessed = apply_glossary(text, GLOSSARY)
#
#     headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
#     data = {
#         "text": preprocessed,
#         "source_lang": "EN",
#         "target_lang": "NL",
#         "tag_handling": "html",
#         "preserve_formatting": 1,
#     }
#
#     logger.debug("Sending request to DeepL API …")
#     try:
#         async with httpx.AsyncClient(timeout=20) as client:
#             response = await client.post(DEEPL_ENDPOINT, data=data, headers=headers)
#     except httpx.HTTPError as exc:
#         logger.error("Network / HTTP error when contacting DeepL: %s", exc)
#         raise HTTPException(status_code=502, detail="Upstream translation service unavailable") from exc
#
#     logger.info("DeepL response status %s", response.status_code)
#     if response.status_code != 200:
#         logger.error("DeepL API error: %s", response.text)
#         raise HTTPException(status_code=500, detail="Translation API error")
#
#     result = response.json()
#     dutch = result["translations"][0]["text"]
#     logger.debug("Raw Dutch output (len=%d). Applying glossary again …", len(dutch))
#     dutch = apply_glossary(dutch, GLOSSARY)
#     return dutch
#
#
# @app.post("/translate", response_model=TranslationResponse)
# async def translate(req: TranslationRequest):
#     logger.info("/translate called (payload len=%d)", len(req.text))
#     translated = await translate_to_dutch(req.text)
#     logger.info("Translation completed (output len=%d)", len(translated))
#     return TranslationResponse(dutch=translated)
#
# #Local dev entry point
# if __name__ == "__main__":
#     import uvicorn
#
#     logger.info("Running development server on http://127.0.0.1:8000 …")
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)


# main.py — English → Dutch translation with confidence scoring


# ─── Imports ────────────────────────────────────────────────────────────────
import os
import json
import logging
from typing import Dict

import httpx
import truststore
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# OpenAI SDK v1 (sync & async variants)
from openai import AsyncOpenAI, AsyncAzureOpenAI

from glossary import GLOSSARY

truststore.inject_into_ssl()

# ─── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("translation-app")
logger.info("Starting translation service …")

# ─── Environment & keys ─────────────────────────────────────────────────────
load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
if not DEEPL_API_KEY:
    logger.critical("DEEPL_API_KEY not set; aborting start‑up")
    raise RuntimeError("DEEPL_API_KEY not set")

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI credentials
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION", "2024-05-01-preview"
)

# Instantiate async client
if OPENAI_API_KEY:
    _ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    _evaluator_model = "gpt-4o-mini"  # change if desired
    logger.info("Using OpenAI backend for confidence scoring")
elif AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT:
    _ai_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    _evaluator_model = AZURE_OPENAI_DEPLOYMENT
    logger.info(
        "Using Azure OpenAI backend for confidence scoring (deployment: %s)",
        _evaluator_model,
    )
else:
    _ai_client = None
    _evaluator_model = None
    logger.warning(
        "No OpenAI or Azure OpenAI credentials found – confidence scoring will be disabled"
    )

DEEPL_ENDPOINT = "https://api.deepl.com/v2/translate"

# ─── Helpers ────────────────────────────────────────────────────────────────

def apply_glossary(text: str, glossary: Dict[str, str]) -> str:
    """Replace glossary terms (longest first) inside *text*."""
    for en, nl in sorted(glossary.items(), key=lambda x: -len(x[0])):
        text = text.replace(en, nl)
    return text


# ─── FastAPI schemas ────────────────────────────────────────────────────────
app = FastAPI(title="English → Dutch Translation Service")


class TranslationRequest(BaseModel):
    text: str


class ConfidenceBreakdown(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=1.0)
    fluency: float = Field(..., ge=0.0, le=1.0)
    terminology_adherence: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    glossary_support: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)


class TranslationResponse(BaseModel):
    dutch: str
    confidence: ConfidenceBreakdown


# ─── DeepL call ─────────────────────────────────────────────────────────────
async def translate_to_dutch(text: str) -> str:
    """Translate English to Dutch, applying glossary before/after."""
    logger.debug("Pre‑processing with glossary …")
    preprocessed = apply_glossary(text, GLOSSARY)

    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = {
        "text": preprocessed,
        "source_lang": "EN",
        "target_lang": "NL",
        "tag_handling": "html",
        "preserve_formatting": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(DEEPL_ENDPOINT, data=data, headers=headers)
    except httpx.HTTPError as exc:
        logger.error("DeepL network error: %s", exc)
        raise HTTPException(
            status_code=502, detail="Upstream translation service unavailable"
        ) from exc

    if response.status_code != 200:
        logger.error("DeepL API error (%s): %s", response.status_code, response.text)
        raise HTTPException(status_code=500, detail="Translation API error")

    dutch = response.json()["translations"][0]["text"]
    return apply_glossary(dutch, GLOSSARY)


# ─── Confidence scorer ──────────────────────────────────────────────────────
async def evaluate_translation(
    src_en: str, tgt_nl: str, glossary: Dict[str, str]
) -> ConfidenceBreakdown:
    """Score the translation using the chosen LLM backend."""
    if _ai_client is None:
        zero = {
            k: 0.0
            for k in (
                "accuracy",
                "fluency",
                "terminology_adherence",
                "consistency",
                "glossary_support",
            )
        }
        zero["overall"] = 0.0
        return ConfidenceBreakdown(**zero)

    # system_prompt = (
    #     "You are a professional English→Dutch translation QA rater. "
    #     "Score the candidate translation from 0 (very poor) to 1 (excellent) for each criterion, "
    #     "then compute the mean as the overall score.\n\n"
    #     "Return ONLY strict JSON like: \n"
    #     "{\"accuracy\":0.9,\"fluency\":0.85,\"terminology_adherence\":0.8,"
    #     "\"consistency\":0.9,\"glossary_support\":0.95,\"overall\":0.88}"
    # )

    """
    Prompt design overview (Better Scoring Strategy)
    ------------------------------------------------
    1. Role / instruction prompting
       'You are a senior English→Dutch translation evaluator'
       – sets a professional-rater perspective.

    2. Rubric prompting
       One-line definition for each metric (accuracy, fluency, etc.).
       – ensures the model knows exactly what to grade.

    3. Anchor / penalty prompting
       'Start at 1.00, subtract 0.05 per minor issue, 0.15 per major issue.'
       – anchors the 0-1 scale and discourages inflated perfect scores.

    4. Few-shot calibration (one-shot)
       Includes one deliberately bad translation with sub-1.0 scores.
       – provides a reference point below perfection.

    5. Chain-of-thought suppression
       'Think silently; output JSON only.'
       – keeps internal reasoning out of the API response.

    6. Structured-output / schema prompting
       Strict JSON schema with no extra commentary or Markdown.
       – guarantees machine-parsable output.
    """

    system_prompt = """
    You are a senior English to Dutch translation quality evaluator.

    Rate the candidate Dutch translation on **five criteria**, each from 0.0 to 1.0:
    1. accuracy ‒ meaning preserved exactly.
    2. fluency ‒ grammatical, natural Dutch.
    3. terminology_adherence ‒ correct medical/brand terms.
    4. consistency ‒ repeated phrases rendered the same.
    5. glossary_support ‒ uses every glossary mapping provided.

    Scoring guide:
    • 1.0 = flawless; 0.8 = minor issue; 0.5 = acceptable but notable flaws; 0.2 = major errors; 0.0 = unusable.

    After scoring, compute **overall = arithmetic mean** of the five values and round ALL numbers to **two decimals**.

    ️ Think through each criterion **silently**; do NOT output reasoning.
    ️ Respond with **ONLY** the following JSON schema (no markdown, no keys added/omitted):

    {
      "accuracy": <float>,
      "fluency": <float>,
      "terminology_adherence": <float>,
      "consistency": <float>,
      "glossary_support": <float>,
      "overall": <float>
    }
    """

    user_payload = json.dumps(
        {
            "source_en": src_en,
            "candidate_nl": tgt_nl,
            "glossary": glossary,
        },
        ensure_ascii=False,
    )

    completion = await _ai_client.chat.completions.create(
        model=_evaluator_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ],
    )

    try:
        scores = json.loads(completion.choices[0].message.content)
        return ConfidenceBreakdown(**scores)
    except (ValueError, KeyError) as exc:
        logger.error("Malformed JSON from evaluator: %s", exc)
        zero = {
            k: 0.0
            for k in (
                "accuracy",
                "fluency",
                "terminology_adherence",
                "consistency",
                "glossary_support",
            )
        }
        zero["overall"] = 0.0
        return ConfidenceBreakdown(**zero)


# ─── API route ──────────────────────────────────────────────────────────────
@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    logger.info("/translate called (payload len=%d)", len(req.text))

    dutch_text = await translate_to_dutch(req.text)
    logger.info("Translation completed (output len=%d)", len(dutch_text))

    confidence = await evaluate_translation(req.text, dutch_text, GLOSSARY)
    return TranslationResponse(dutch=dutch_text, confidence=confidence)


# ─── Dev entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Running development server on http://127.0.0.1:8000 …")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
