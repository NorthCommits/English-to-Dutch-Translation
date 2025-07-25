import truststore
truststore.inject_into_ssl()
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

# Load DeepL API key from environment variables
load_dotenv()
DEEPL_API_KEY = os.getenv("DeepL_Api_Key")
if not DEEPL_API_KEY:
    raise RuntimeError("DeepL_Api_Key not found in .env file")

# Initialize FastAPI app
app = FastAPI()

# ----- Pydantic models -----
class TranslationRequest(BaseModel):
    """Schema for incoming Dutch text."""
    text: str

class TranslationResponse(BaseModel):
    """Schema for returned English translation."""
    english: str

# ----- Core translation function -----
async def translate_to_english(text: str) -> str:
    """Translate Dutch text to English using the DeepL API (no glossary)."""
    url = "https://api.deepl.com/v2/translate"  # Endpoint for DeepL Pro accounts
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = {
        "text": text,
        "source_lang": "NL",
        "target_lang": "EN",
        "tag_handling": "html",  # Preserve any HTML if present
        "preserve_formatting": 1,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, data=data, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Translation API error")

        result = response.json()
        # Extract the translated text
        english_translation = result["translations"][0]["text"]

    return english_translation

# ----- API route -----
@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest):
    english_text = await translate_to_english(req.text)
    return TranslationResponse(english=english_text)

# ----- Run with Uvicorn -----
if __name__ == "__main__":
    import uvicorn

    # Use a different port (8001) to avoid clashing with the main EN→NL service
    uvicorn.run("backtranslation:app", host="127.0.0.1", port=8001, reload=True)
