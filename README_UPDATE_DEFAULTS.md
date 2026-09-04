# README updates: default behaviors

- Default behavior: Ultron will use the Hugging Face Inference API for generation if a local quantized 13B model is not available. To enable this, set:
  - HUGGINGFACE_API_KEY environment variable with your token
  - HUGGINGFACE_MODEL environment variable to the HF model id to use (e.g., 'tiiuae/falcon-7b-instruct' or other hosted model)

- VAD-based listening: Frontend can upload audio to /transcribe; a VAD-based helper (app/vad_listener.py) is provided for local capture.

- React frontend scaffold is provided in frontend/. Use your preferred bundler (webpack, Vite) to build and serve the frontend; it's a minimal scaffold to make Ultron look polished.
