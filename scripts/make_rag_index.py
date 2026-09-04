"""Create embeddings and FAISS index from chunked text files.
Usage:
  python scripts/make_rag_index.py --input_dir data/txt_chunks --out_dir data/faiss_index
"""
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json


def main(args):
    model_name = args.embed_model or 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name)

    input_dir = Path(args.input_dir)
    files = list(input_dir.glob('**/*.txt'))
    texts = [f.read_text(encoding='utf8') for f in files]

    print(f"Computing embeddings for {len(texts)} chunks using {model_name}...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / 'index.faiss'))

    mapping = [{'path': str(p), 'text_preview': t[:500]} for p, t in zip(files, texts)]
    with open(out_dir / 'mapping.json', 'w', encoding='utf8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print('Saved FAISS index and mapping to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--embed_model', default=None)
    args = parser.parse_args()
    main(args)
