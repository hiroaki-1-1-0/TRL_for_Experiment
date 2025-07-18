#!/usr/bin/env python3
"""
Minimal test for reward model training setup
"""

import os
import warnings

# Set up the same environment as the main script
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_basic_setup():
    print("Testing basic model and tokenizer loading...")
    
    # Test tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Test model loading (without moving to GPU to save memory)
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen3-8B",
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Model config: {model.config}")
    
    # Test tokenization
    test_texts = ["Hello world", "How are you?"]
    tokens = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    print(f"✓ Tokenization works: {tokens['input_ids'].shape}")
    
    print("✓ All basic tests passed!")

if __name__ == "__main__":
    test_basic_setup() 