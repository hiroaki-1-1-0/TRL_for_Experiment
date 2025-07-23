#!/usr/bin/env python3
"""
Single-GPU RLOO Training Script - Stable and Reliable Version
Simplified implementation to avoid distributed training complexities
"""

import os
import sys
import argparse
import logging
import warnings
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Environment setup for stability
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
import gc

from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

def setup_logging() -> logging.Logger:
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def filter_english_samples(dataset: Dataset) -> Dataset:
    """Filter dataset for English language samples only"""
    print("Filtering for English language samples...")
    
    def is_english_sample(example):
        return example.get('language', '').lower() == 'english'
    
    english_dataset = dataset.filter(is_english_sample)
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"After English filtering: {len(english_dataset)} samples")
    print(f"Filtered out: {len(dataset) - len(english_dataset)} non-English samples")
    
    return english_dataset

def prepare_helpsteer_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Prepare HelpSteer3 dataset for RLOO training"""
    
    def extract_and_tokenize_prompt(element):
        """Extract conversation context and tokenize for RLOO training"""
        context = element["context"]
        
        # Build the conversation prompt from context
        if isinstance(context, list) and len(context) > 0:
            # Use the last user message as the prompt for generation
            user_messages = [msg for msg in context if msg.get("role") == "user"]
            if user_messages:
                prompt = user_messages[-1]["content"]
            else:
                prompt = " ".join([msg.get("content", "") for msg in context])
        else:
            prompt = str(context)
        
        # Apply chat template if available
        if tokenizer.chat_template is not None:
            if isinstance(context, list):
                messages = [msg for msg in context if msg.get("role") in ["user", "assistant"]]
                if not messages or messages[-1].get("role") != "user":
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    if user_messages:
                        messages = [user_messages[-1]]
                
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    padding=False,
                    add_generation_prompt=True,
                    return_tensors=None
                )
            else:
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    padding=False,
                    add_generation_prompt=True,
                    return_tensors=None
                )
        else:
            outputs = tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=512,
            )
            input_ids = outputs["input_ids"]
        
        return {"input_ids": input_ids, "lengths": len(input_ids)}
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        extract_and_tokenize_prompt,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    
    # Filter out samples that are too long
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: x["lengths"] <= 512,
        num_proc=4
    )
    
    return tokenized_dataset

def setup_models_and_tokenizer(model_config: ModelConfig, training_args: RLOOConfig, 
                              reward_model_path: Optional[str] = None):
    """Setup policy, reference policy, reward model, and tokenizer"""
    
    policy_model_name = "Qwen/Qwen3-4B"
    
    if reward_model_path:
        reward_model_name = reward_model_path
        print(f"Using trained reward model from: {reward_model_name}")
    else:
        reward_model_name = "experiment/models/qwen3_4b_reward_model"
        print(f"Using default trained reward model path: {reward_model_name}")
    
    print(f"Loading tokenizer from {policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        policy_model_name,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    print(f"Loading reward model from {reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"Loading reference policy from {policy_model_name}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"Loading policy model from {policy_model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    return tokenizer, reward_model, ref_policy, policy

def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Get reward model path
    reward_model_path = getattr(script_args, 'reward_model_path', None)
    if reward_model_path is None:
        reward_model_path = "experiment/models/qwen3_4b_reward_model"
    
    # Remove output directory if it exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    print("="*60)
    print("Single-GPU RLOO Training with HelpSteer3 Dataset")
    print("="*60)
    print(f"Policy Model: Qwen/Qwen3-4B")
    print(f"Reward Model: {reward_model_path}")
    print(f"Dataset: nvidia/HelpSteer3 (English only)")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("="*60)
    
    # Set seeds
    set_seed(42)
    
    # Filter warnings
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)
    
    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    ################
    # Models & Tokenizer
    ################
    tokenizer, reward_model, ref_policy, policy = setup_models_and_tokenizer(
        model_args, training_args, reward_model_path
    )
    
    ################
    # Dataset Loading and Preparation
    ################
    print("Loading HelpSteer3 dataset...")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Filter for English language samples only
    dataset = filter_english_samples(dataset)
    
    # Split into train and eval
    eval_samples = min(1000, len(dataset) // 10)
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets with tokenization
    print("Tokenizing datasets...")
    train_dataset = prepare_helpsteer_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_helpsteer_dataset(eval_dataset, tokenizer)
    
    print(f"Final train dataset size: {len(train_dataset)}")
    print(f"Final eval dataset size: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    print("Initializing RLOO Trainer...")
    trainer = RLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting RLOO training...")
    trainer.train()
    
    ################
    # Save Model
    ################
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        print("Pushing model to hub...")
        trainer.push_to_hub(dataset_name="nvidia/HelpSteer3")
    
    print("Training completed successfully!")
    
    # Generate sample completions to verify training
    print("Generating sample completions...")
    trainer.generate_completions()

if __name__ == "__main__":
    main() 