"""LLM inference helper for Ultron with a local small-model fallback.
This module supports three modes:
- Local quantized model loading (bitsandbytes / accelerate) when a local 13B quantized model directory is provided via model_dir.
- Remote HF Inference API fallback when HUGGINGFACE_API_KEY and HUGGINGFACE_MODEL are set.
- Local CPU small model fallback (google/flan-t5-small) when no other model is available — useful for quick demos on machines without HF keys or large GPUs.

Usage example:
  from app.llm_inference import UltronLLM
  llm = UltronLLM()  # will try local quantized model, then HF API, then flan-t5-small
  answer = llm.generate(prompt)
"""
import os
from typing import Optional

HF_API_URL = "https://api-inference.huggingface.co/models/"

class UltronLLM:
    def __init__(self, model_dir: Optional[str] = None, hf_model: Optional[str] = None):
        self.model_dir = model_dir
        self.hf_model = hf_model
        self._client = None
        self._local = False
        self._local_seq2seq = False

        # If model_dir exists, attempt to load local model (quantized)
        if model_dir and os.path.isdir(model_dir):
            try:
                # Lazy import heavy libs
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                print('Loading local model from', model_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
                # Attempt to load with device_map auto (requires bitsandbytes)
                self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
                self._local = True
                self._local_seq2seq = False
            except Exception as e:
                print('Local quantized model load failed, will fallback. Error:', e)
                self._local = False

        # If no local model, set HF remote model if provided
        if not self._local:
            self.hf_model = hf_model or os.environ.get('HUGGINGFACE_MODEL')
            self.hf_api_key = os.environ.get('HUGGINGFACE_API_KEY')
            if not self.hf_model and not self.hf_api_key:
                # As a last resort, attempt to load a small local CPU model for demos (flan-t5-small)
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    import torch
                    print('No HF API key or model configured — loading local CPU demo model (flan-t5-small)')
                    demo_model = 'google/flan-t5-small'
                    self.tokenizer = AutoTokenizer.from_pretrained(demo_model)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(demo_model)
                    self._local = True
                    self._local_seq2seq = True
                except Exception as e:
                    print('Local demo model load failed:', e)
                    self._local = False

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        if self._local:
            return self._generate_local(prompt, max_tokens, temperature)
        else:
            return self._generate_remote(prompt, max_tokens, temperature)

    def _generate_local(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Support both causal LM and seq2seq small model
        import torch
        if self._local_seq2seq:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device if hasattr(self.model, 'device') else 'cpu')
            out = self.model.generate(**inputs, max_new_tokens=max_tokens)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            from transformers import GenerationConfig
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
            gen_config = GenerationConfig(temperature=temperature, max_new_tokens=max_tokens)
            with torch.no_grad():
                out = self.model.generate(**inputs, generation_config=gen_config)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _generate_remote(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Use Hugging Face Inference API
        if not (os.environ.get('HUGGINGFACE_API_KEY') and (os.environ.get('HUGGINGFACE_MODEL') or self.hf_model)):
            raise RuntimeError('No remote model configured. Set HUGGINGFACE_API_KEY and HUGGINGFACE_MODEL or provide a local model_dir.')
        import requests
        hf_model = self.hf_model or os.environ.get('HUGGINGFACE_MODEL')
        headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_API_KEY')}", "Content-Type": "application/json"}
        url = HF_API_URL + hf_model
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f'Inference API error: {resp.status_code} {resp.text}')
        data = resp.json()
        if isinstance(data, str):
            return data
        if isinstance(data, list) and len(data) > 0 and 'generated_text' in data[0]:
            return data[0]['generated_text']
        # Common HF inference returns {'generated_text': '...'}
        if isinstance(data, dict) and 'generated_text' in data:
            return data['generated_text']
        return str(data)
