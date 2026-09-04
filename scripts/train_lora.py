"""Template script to train a LoRA/PEFT adapter (QLoRA-style) on a 13B model.
This is a minimal template — you will need a machine with sufficient GPU VRAM or use a cloud instance.
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch


def main(args):
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Load model in 4-bit with bitsandbytes if available
    print('Loading base model (this may use 4-bit + bnb)')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map='auto',
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias='none'
    )
    model = get_peft_model(model, lora_config)

    # Dataset loading / preprocessing placeholder
    # You should prepare an instruction tuning style dataset with `instruction`, `input`, `output` fields
    ds = load_dataset('json', data_files=args.train_data)

    def tokenize_fn(examples):
        # Simple example: concatenate instruction + input and target
        inputs = [ (i.get('instruction','') + '\n' + i.get('input','')) for i in examples['train'] ] if 'train' in examples else []
        return tokenizer(inputs, truncation=True, max_length=2048)

    # ... full training loop omitted for brevity. Use Hugging Face Trainer or accelerate launch with your config.
    print('This script is a template. See README for full training commands and recommended cloud hardware.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='meta-llama/Llama-2-13b-chat-hf')
    parser.add_argument('--train_data', required=False)
    args = parser.parse_args()
    main(args)
