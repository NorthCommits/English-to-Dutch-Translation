# English–Dutch Medical Translation Suite

This repository provides a professional translation and evaluation toolkit for English ↔ Dutch medical and clinical texts. It leverages DeepL for high-quality translation, OpenAI for translation confidence scoring, and includes a domain-specific glossary and semantic similarity tools.

## Features
- **English → Dutch translation API** with medical glossary enforcement and confidence scoring (FastAPI, DeepL, OpenAI/Azure OpenAI)
- **Dutch → English backtranslation API** (FastAPI, DeepL)
- **Medical/clinical glossary** for consistent terminology
- **Semantic similarity scoring** between English sentences (SentenceTransformers)

---

## Project Structure

- `main.py` — Main FastAPI app for English→Dutch translation with confidence scoring and glossary support.
- `backtranslation.py` — FastAPI app for Dutch→English backtranslation (no glossary).
- `glossary.py` — Professional glossary mapping English medical/clinical terms to Dutch.
- `cosineSimilarity.py` — Script to compute semantic similarity between two English sentences using transformer models.

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd translation
```

### 2. Install dependencies
Create a virtual environment and install required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Required packages:**
- fastapi
- uvicorn
- httpx
- python-dotenv
- truststore
- openai
- sentence-transformers

### 3. Environment Variables
Create a `.env` file in the project root with the following keys:
```
DEEPL_API_KEY=your_deepl_api_key
OPENAI_API_KEY=your_openai_api_key  # or Azure OpenAI keys if using Azure
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_azure_deployment_name
```

---

## Usage

### 1. English → Dutch Translation API
Run the main translation service:
```bash
uvicorn main:app --reload
```
- **Endpoint:** `POST /translate`
- **Request body:** `{ "text": "<English text>" }`
- **Response:**
  - `dutch`: Translated Dutch text
  - `confidence`: Object with accuracy, fluency, terminology adherence, consistency, glossary support, and overall score

### 2. Dutch → English Backtranslation API
Run the backtranslation service:
```bash
uvicorn backtranslation:app --port 8001 --reload
```
- **Endpoint:** `POST /translate`
- **Request body:** `{ "text": "<Dutch text>" }`
- **Response:**
  - `english`: Translated English text

### 3. Semantic Similarity
Run the script to compare two English sentences:
```bash
python cosineSimilarity.py
```
- Edit `STATEMENT_A` and `STATEMENT_B` in the script or input interactively.
- Outputs similarity scores using two transformer models.

---

## Glossary
See `glossary.py` for the full list of English–Dutch medical/clinical term mappings used to enforce terminology consistency.


---

## Acknowledgments
- [DeepL API](https://www.deepl.com/pro-api)
- [OpenAI](https://platform.openai.com/)
- [SentenceTransformers](https://www.sbert.net/) 