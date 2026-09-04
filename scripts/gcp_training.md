# GCP training guide for Ultron (13B PEFT / QLoRA)

This document describes step-by-step how to provision a GCP VM with an A100 GPU, prepare the environment, and run PEFT/LoRA-style fine-tuning (QLoRA-style) for a 13B model using the scripts in this repository.

Prerequisites
- Google Cloud SDK (gcloud) installed and authenticated: gcloud auth login
- Your GCP project has GPU quota for A100 in the desired zone
- Optional: gcloud billing enabled

1) Create the instance (recommended helper)
- From your workstation run:
  bash scripts/gcp_launch.sh
- This creates a VM named `ultron-train-1` with an A100 GPU. You can customize zone and instance name via environment variables:
  PROJECT=my-project ZONE=us-central1-a INSTANCE=ultron-train-1 bash scripts/gcp_launch.sh

2) SSH into the instance
  gcloud compute ssh ultron-train-1 --zone=us-central1-a

3) On the instance: clone your Ultron repo and prepare Python env
  git clone https://github.com/Saanvi25-tech/Ultron.git ulron_repo
  cd ulron_repo
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt

Notes: For training you will also need
  pip install bitsandbytes accelerate transformers safetensors einops

4) Configure accelerate
- Create an accelerate config file (example provided in scripts/accelerate_config.yml)
- Alternatively run interactive:
  accelerate config

5) Prepare your training data
- Use the preprocessing scripts to collect and chunk documents into data/txt_chunks
- Convert your chunks into an instruction tuning dataset (JSONL) with fields: {"instruction":"...", "input":"...", "output":"..."}

6) Example training command (QLoRA-style using accelerate)
  accelerate launch --config_file scripts/accelerate_config.yml \
    scripts/train_lora.py \
    --model_name meta-llama/Llama-2-13b-chat-hf \
    --train_data data/train.jsonl \
    --output_dir models/ultron-lora

Notes on resource sizing and tips
- Use mixed precision (fp16) where possible and bitsandbytes 4-bit quantization to reduce VRAM.
- Keep batch_size small (e.g., per-device batch size 1–2) to avoid OOM.
- Use gradient accumulation steps to increase effective batch size.

7) Save and download adapters
- After training, the PEFT/LoRA adapters are typically small and can be uploaded to HF Hub or downloaded to your laptop for inference.

8) Inference on your laptop
- Use the app/llm_inference.py which supports local quantized models (if you have GPU resources) or the HF Inference API fallback.

If you want, I can prepare a fully automated startup-script that also pulls the repo and runs the full training pipeline — confirm and I will add it (this will make the instance perform a long-running training job automatically).
