from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

from app.llm_inference import UltronLLM

app = FastAPI()

MODEL_DIR = os.environ.get('ULTRON_LOCAL_MODEL_DIR', 'models/quantized-13b')  # set to your model path
INDEX_DIR = 'data/faiss_index'
EMBED_MODEL = 'all-MiniLM-L6-v2'

# Load embedder + index lazily
_embedder = None
_index = None
_mapping = None

# Initialize LLM helper. If no local model present, UltronLLM will fallback to remote HF inference if configured via env vars.
_llm = UltronLLM(model_dir=MODEL_DIR)

class Query(BaseModel):
    query: str


def load_resources():
    global _embedder, _index, _mapping
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    if _index is None:
        p = Path(INDEX_DIR)
        if not p.exists():
            raise RuntimeError(f'FAISS index not found at {INDEX_DIR}. Build it with scripts/make_rag_index.py')
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
    try:
        tmp.unlink()
    except Exception:
        pass
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
        sources.append({'path': m.get('path'), 'preview': m.get('text_preview', '')[:500]})
        context_texts.append(m.get('text_preview', ''))

    # Assemble prompt for the LLM
    context_join = "\n\n".join(context_texts[:5])
    prompt = (
        "You are Ultron, an expert assistant specialized in astrophysics, rocket science, and nuclear physics. "
        "Use the provided CONTEXT to answer the question succinctly and accurately. If you are unsure, say so. "
        "Do NOT provide operationally harmful instructions.\n\n"
        f"CONTEXT:\n{context_join}\n\nQUESTION:\n{qtext}\n\nANSWER:"
    )

    # Generate answer via LLM helper (local quantized model or HF Inference API fallback)
    try:
        answer = _llm.generate(prompt, max_tokens=512, temperature=0.0)
    except Exception as e:
        answer = f"LLM generation failed: {e}"

    return {'answer': answer, 'sources': sources}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
