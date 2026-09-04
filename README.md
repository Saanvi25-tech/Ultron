# Ultron — Personal Astro/Rocket/Nuclear AI

This repository is the starter skeleton for "Ultron": a local-first personal AI that supports speech input (wake word / push-to-talk), speech-to-text, Retrieval-Augmented Generation (RAG) over domain documents (astrophysics, rocket science, nuclear physics), and LoRA/PEFT fine-tuning of an open-source 13B model.

Important notes
- You told me your machine uses a Qualcomm Adreno X1-85 GPU. That GPU is a mobile/SoC GPU and is NOT supported by PyTorch/transformers for model training or inference. Fine-tuning or running a 13B model on that GPU locally is not feasible. The repo includes a cloud-friendly training flow (QLoRA/PEFT) and local inference instructions — you'll likely need a cloud GPU (or a desktop NVIDIA GPU with >=24GB VRAM) to do heavy fine-tuning and large-model inference.
- You asked: clickable citations/links = no. The RAG pipeline will return source metadata (title, short snippet) but will not include clickable links by default.
- Safety: I will not provide operationally harmful or illicit instructions. The assistant will focus on high-level theory, published research references, and safe engineering practices.

Quick start
1. Install dependencies: python -m pip install -r requirements.txt
2. Put PDFs into `data/raw_pdfs/` (see scripts/preprocess_pdfs.py for helpers). The repo includes ingestion & indexing scripts to build the vector store.
3. Build the RAG index locally (FAISS) with:
   python scripts/make_rag_index.py --input_dir data/txt_chunks --out_dir data/faiss_index
4. For fine-tuning: see scripts/train_lora.py. If your GPU can't handle 13B fine-tuning locally, use a cloud instance (I recommend a 48GB+ A100 or equivalent) and the `--use_cloud` flow.
5. Run the demo API: python app/api.py --model_dir path/to/quantized_model --index_dir data/faiss_index

Repository layout
- requirements.txt
- README.md (this file)
- .gitignore
- scripts/
  - preprocess_pdfs.py
  - make_rag_index.py
  - train_lora.py
- app/
  - speech_pipeline.py
  - wake_word.py
  - api.py
- data/ (gitignored) — for raw PDFs, extracted text, and FAISS index

What's next
- I can add a Dockerfile for reproducible local runs if you want.
- I can also run a controlled cloud crawl to fetch public-domain papers (arXiv + NASA TRs) if you confirm.

If anything here should be changed (repo path, file names, or different model choice), tell me and I'll update the files.
