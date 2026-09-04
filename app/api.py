"""FastAPI app for local inference.
Endpoints:
- POST /transcribe: multipart-form audio file -> returns transcription
- POST /query: JSON {"query": "..."} -> returns RAG-augmented answer

Note: this app uses placeholder model loading. Update `MODEL_DIR` to point to your quantized model.
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()

MODEL_DIR = 'models/quantized-13b'  # set to your model path
INDEX_DIR = 'data/faiss_index'
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Load embedder + index lazily
_embedder = None
_index = None
_mapping = None

class Query(BaseModel):
    query: str


def load_resources():
    global _embedder, _index, _mapping
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        p = Path(INDEX_DIR)
        _index = faiss.read_index(str(p / 'index.faiss'))
        with open(p / 'mapping.json', 'r', encoding='utf8') as f:
            _mapping = json.load(f)

@app.post('/transcribe')
async def transcribe_audio(file: UploadFile = File(...)):
    # Save uploaded audio to a temp file and call the speech pipeline
    tmp = Path('tmp_upload.wav')
    content = await file.read()
    tmp.write_bytes(content)
    from app.speech_pipeline import transcribe_file
    text = transcribe_file(str(tmp))
    return {'text': text}

@app.post('/query')
async def query(q: Query):
    load_resources()
    qtext = q.query
    emb = _embedder.encode([qtext], convert_to_numpy=True)
    D, I = _index.search(emb, k=5)
    sources = []
    context_texts = []
    for idx in I[0]:
        m = _mapping[idx]
        sources.append({'path': m['path'], 'preview': m['text_preview'][:300]})
        context_texts.append(m['text_preview'])

    # Assemble prompt for the LLM (placeholder)
    prompt = f"You are Ultron, a helpful assistant specialized in astrophysics, rocket science, and nuclear physics. Use the following context to answer the question.\n\nCONTEXT:\n{context_texts[:3]}\n\nQUESTION:\n{qtext}"

    # TODO: call the LLM (load quantized model) and return answer.
    # For now we return the retrieved contexts and a placeholder
    return {'answer': 'Hello — LLM not yet loaded. Replace placeholder to call your quantized model.', 'sources': sources}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
