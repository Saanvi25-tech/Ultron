"""Convert PDFs to plain text and chunk them for RAG indexing.
Usage:
  python scripts/preprocess_pdfs.py --input_dir data/raw_pdfs --out_dir data/txt_chunks --chunk_size 1000
"""
import argparse
import os
import pdfplumber
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def main(args):
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = list(input_dir.glob('**/*.pdf'))
    for pdf in tqdm(pdf_paths, desc='PDFs'):
        text = extract_text_from_pdf(str(pdf))
        if not text.strip():
            continue
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        base = pdf.stem
        for i, c in enumerate(chunks):
            out_path = out_dir / f"{base}_chunk_{i}.txt"
            out_path.write_text(c, encoding='utf8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--overlap', type=int, default=200)
    args = parser.parse_args()
    main(args)
