#!/usr/bin/env python3
"""
Multi-GPU Reward Model Training Script - Simplified Working Version
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import json

# Set environment variables early
os.environ.setdefault('NCCL_SHM_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_P2P_LEVEL', 'LOC')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
# Memory optimization settings
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import set_seed
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_reward_model(model_name: str, use_peft: bool = True, lora_r: int = 64):
    """Create and configure the reward model"""
    
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",  # Explicitly set padding side for consistency
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading reward model from {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    
    # Configure pad token in model
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply PEFT if specified
    if use_peft:
        print("Applying LoRA (PEFT) configuration")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset() -> Dataset:
    """Prepare HelpSteer3 dataset for reward training"""
    
    print("Loading HelpSteer3 dataset...")
    raw_dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Ensure we have a Dataset object (not DatasetDict or IterableDataset)
    if isinstance(raw_dataset, DatasetDict):
        dataset = raw_dataset['train']
    elif hasattr(raw_dataset, '__iter__') and not hasattr(raw_dataset, '__len__'):
        # Handle IterableDataset by converting to Dataset
        print("Converting IterableDataset to Dataset...")
        dataset = Dataset.from_list(list(raw_dataset))
    else:
        dataset = raw_dataset
    
    # Now we should have a proper Dataset object
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Expected Dataset object, got {type(dataset)}")
    
    dataset_size = len(dataset)
    print(f"Original dataset size: {dataset_size}")
    
    # Filter for English language samples
    english_dataset = dataset.filter(lambda x: x.get('language', '').lower() == 'english')
    print(f"After English filtering: {len(english_dataset)} samples")
    
    # Convert to chosen/rejected format
    print("Converting to chosen/rejected format...")
    def convert_to_chosen_rejected(examples):
        chosen_texts = []
        rejected_texts = []
        
        for i in range(len(examples['context'])):
            context = examples['context'][i]
            response1 = examples['response1'][i]
            response2 = examples['response2'][i]
            preference = examples['overall_preference'][i]
            
            # Skip neutral preferences
            if preference == 0:
                continue
            
            # Build conversation from context
            if isinstance(context, list) and len(context) > 0:
                conversation = ""
                for msg in context:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation += f"Human: {content}\n"
                    elif role == "assistant":
                        conversation += f"Assistant: {content}\n"
                
                full_text1 = conversation + f"Assistant: {response1}"
                full_text2 = conversation + f"Assistant: {response2}"
            else:
                prompt = str(context)
                full_text1 = f"Human: {prompt}\nAssistant: {response1}"
                full_text2 = f"Human: {prompt}\nAssistant: {response2}"
            
            # Assign chosen/rejected based on preference
            if preference > 0:  # response1 is preferred
                chosen_texts.append(full_text1)
                rejected_texts.append(full_text2)
            else:  # preference < 0, response2 is preferred
                chosen_texts.append(full_text2)
                rejected_texts.append(full_text1)
        
        return {
            'chosen': chosen_texts,
            'rejected': rejected_texts
        }
    
    # Get column names properly - english_dataset is guaranteed to be a Dataset
    column_names = english_dataset.column_names
    
    # Convert dataset
    reward_dataset = english_dataset.map(
        convert_to_chosen_rejected,
        batched=True,
        remove_columns=column_names,
    )
    
    # Filter out empty results
    reward_dataset = reward_dataset.filter(lambda x: len(x['chosen']) > 0)
    print(f"Converted dataset size: {len(reward_dataset)} preference pairs")
    
    return reward_dataset

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="experiment/models/qwen3_4b_reward_model")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)  # Reduced from 12
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)   # Reduced from 6
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # Increased to maintain effective batch size
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)                # Reduced from 2048
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
    )
    
    rank = accelerator.process_index
    logger = setup_logging(rank)
    
    if rank == 0:
        print("=" * 60)
        print("Multi-GPU Reward Model Training with HelpSteer3 Dataset")
        print("=" * 60)
        print(f"Base Model: {args.model_name}")
        print(f"Dataset: nvidia/HelpSteer3 (English only)")
        print(f"Output Directory: {args.output_dir}")
        print(f"Use PEFT: {args.use_peft}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        effective_batch_size = (args.per_device_train_batch_size * 
                              args.gradient_accumulation_steps * 
                              accelerator.num_processes)
        print(f"Effective Batch Size: {effective_batch_size}")
        print("=" * 60)
    
    # Set seeds
    set_seed(42)
    
    # Filter specific tokenizer warnings to reduce noise
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Enable TF32
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Load model and tokenizer
    model, tokenizer = create_reward_model(args.model_name, args.use_peft, args.lora_r)
    
    # Prepare dataset
    with accelerator.local_main_process_first():
        reward_dataset = prepare_dataset()
    
    # Ensure we have a proper Dataset object
    if not isinstance(reward_dataset, Dataset):
        raise ValueError(f"Expected Dataset object from prepare_dataset, got {type(reward_dataset)}")
    
    # Split dataset
    dataset_size = len(reward_dataset)
    train_size = int(0.9 * dataset_size)
    
    train_indices = list(range(train_size))
    eval_indices = list(range(train_size, dataset_size))
    
    train_dataset = reward_dataset.select(train_indices)
    eval_dataset = reward_dataset.select(eval_indices)
    
    if rank == 0:
        print(f"Dataset split - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Configure training arguments
    training_args = RewardConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        tf32=args.tf32,
        dataloader_num_workers=2,  # Reduced from 4
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=args.run_name or f"qwen3_4b_reward_model_{rank}",
        max_length=args.max_length,
        # Memory optimization settings
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        optim="adamw_torch_fused",  # More memory efficient optimizer
    )
    
    # Create trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train the model
    if rank == 0:
        print("Starting training...")
    
    trainer.train()
    
    # Save the model
    if rank == 0:
        print("Saving final model...")
        trainer.save_model()
        
        # Save training info
        training_info = {
            "model_name": args.model_name,
            "dataset_name": "nvidia/HelpSteer3",
            "training_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "use_peft": args.use_peft,
            "max_length": args.max_length,
        }
        
        with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 