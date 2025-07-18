#!/usr/bin/env python3
"""
Multi-GPU Reward Model Training with Qwen3-8B and HelpSteer3
Optimized for 7x 48GB NVIDIA RTX 6000 Ada GPUs
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json
import numpy as np
from dataclasses import dataclass, field

# Set environment variables early for NCCL configuration
os.environ.setdefault('NCCL_SHM_DISABLE', '1')  # Disable shared memory
os.environ.setdefault('NCCL_IB_DISABLE', '1')   # Disable InfiniBand
os.environ.setdefault('NCCL_P2P_LEVEL', 'LOC')  # Local P2P only
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_SOCKET_NTHREADS', '1')
os.environ.setdefault('NCCL_NSOCKS_PERTHREAD', '1')

# PyTorch and ML imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from transformers.trainer_utils import set_seed

# TRL and related imports
from trl import RewardTrainer, RewardConfig
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType

# Accelerate for distributed training
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed as accelerate_set_seed

# ========================================
# Configuration and Setup
# ========================================

def setup_distributed_environment():
    """Setup distributed training environment with fallback options"""
    
    # Check if we're in a distributed setting
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        print(f"Distributed setup - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        # Additional NCCL optimizations for container environments
        os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
        os.environ.setdefault('NCCL_TIMEOUT', '1800')  # 30 minutes
        
        # Try to use faster networks if available
        if not os.environ.get('NCCL_SOCKET_IFNAME'):
            # Auto-detect best network interface
            import socket
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            print(f"Auto-detected network: {hostname} -> {local_ip}")
            
        # Fallback communication methods
        os.environ.setdefault('NCCL_NET_GDR_LEVEL', '0')  # Disable GPU Direct RDMA
        os.environ.setdefault('NCCL_NET_GDR_READ', '0')
        
    else:
        print("Running in single-GPU mode")

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name: str = field(default="Qwen/Qwen3-8B")
    max_length: int = field(default=2048)
    use_peft: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)

@dataclass
class DataArguments:
    """Arguments for data configuration"""
    dataset_name: str = field(default="nvidia/HelpSteer3")
    max_samples: Optional[int] = field(default=None)
    train_split_ratio: float = field(default=0.9)

# ========================================
# Data Processing
# ========================================

class HelpSteerRewardDataset(Dataset):
    """Custom dataset for HelpSteer3 reward modeling"""
    
    def __init__(self, tokenizer, max_length: int = 2048, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process dataset
        print("Loading HelpSteer3 dataset...")
        dataset = load_dataset("nvidia/HelpSteer3", split="train")
        
        # Filter for English samples only
        print(f"Original dataset size: {len(dataset)}")
        
        # Process the full dataset to find English samples
        english_samples = []
        processed = 0
        
        # Filter for English language samples using the language field
        for idx, item in enumerate(dataset):
            # Check if the sample is marked as English in the language field
            if item.get('language', '').lower() == 'english':
                english_samples.append(item)
                if max_samples and len(english_samples) >= max_samples:
                    break
            
            processed += 1
            if processed % 5000 == 0:
                print(f"Processed {processed} samples, found {len(english_samples)} English samples")
        
        print(f"After English filtering: {len(english_samples)} samples")
        print(f"Filtered out: {processed - len(english_samples)} non-English samples")
        
        # Convert to chosen/rejected format
        print("Converting to chosen/rejected format...")
        self.preference_pairs = self._create_preference_pairs(english_samples)
        print(f"Converted dataset size: {len(self.preference_pairs)} preference pairs")
    
    def _is_english(self, text: str) -> bool:
        """Simple English language detection"""
        if not text or len(text.strip()) < 5:
            return False
        
        # More comprehensive English detection
        english_indicators = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'can', 'could', 'should', 'shall',
            'this', 'that', 'these', 'those', 'what', 'where', 'when', 'why', 'how',
            'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        ]
        
        # Check for English characters and common words
        text_lower = text.lower()
        
        # Basic check: contains mostly ASCII characters
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio < 0.8:  # Less than 80% ASCII characters
            return False
        
        # Count English indicator words
        words = text_lower.split()[:100]  # Check first 100 words
        if len(words) < 3:
            return False
            
        english_count = sum(1 for word in words if any(word == indicator for indicator in english_indicators))
        
        # More lenient threshold: at least 2 English indicators or 15% of words
        return english_count >= 2 or (english_count / len(words)) >= 0.15
    
    def _create_preference_pairs(self, samples: List[Dict]) -> List[Dict]:
        """Convert HelpSteer3 samples to preference pairs"""
        pairs = []
        
        for item in samples:
            # HelpSteer3 has context, response1, response2, and overall_preference
            context = item.get('context', [])
            response1 = item.get('response1', '')
            response2 = item.get('response2', '')
            preference = item.get('overall_preference', 0)
            
            # Skip neutral preferences for clearer training signal
            if preference == 0:
                continue
            
            # Build conversation prompt from context
            if isinstance(context, list) and len(context) > 0:
                # Format as conversation
                conversation = ""
                for msg in context:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        conversation += f"Human: {content}\n"
                    elif role == "assistant":
                        conversation += f"Assistant: {content}\n"
                
                # Create full texts with responses
                full_text1 = conversation + f"Assistant: {response1}"
                full_text2 = conversation + f"Assistant: {response2}"
            else:
                # Fallback for unexpected format
                prompt = str(context)
                full_text1 = f"Human: {prompt}\nAssistant: {response1}"
                full_text2 = f"Human: {prompt}\nAssistant: {response2}"
            
            # Assign chosen/rejected based on preference
            if preference > 0:  # response1 is preferred
                pairs.append({
                    'prompt': conversation if isinstance(context, list) else str(context),
                    'chosen': full_text1,
                    'rejected': full_text2,
                    'chosen_score': preference,
                    'rejected_score': -preference
                })
            else:  # preference < 0, response2 is preferred
                pairs.append({
                    'prompt': conversation if isinstance(context, list) else str(context),
                    'chosen': full_text2,
                    'rejected': full_text1,
                    'chosen_score': -preference,
                    'rejected_score': preference
                })
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.preference_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.preference_pairs[idx]
        
        # Return the chosen and rejected texts directly
        # RewardTrainer will handle tokenization
        return {
            'chosen': item['chosen'],
            'rejected': item['rejected']
        }

# ========================================
# Model Setup
# ========================================

def create_reward_model(model_args: ModelArguments) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Create and configure the reward model"""
    
    print(f"Loading tokenizer from {model_args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading reward model from {model_args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name,
        num_labels=1,  # Single score output for reward modeling
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None  # Let accelerate handle device placement
    )
    
    # Configure pad token in model if needed
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Apply PEFT if specified
    if model_args.use_peft:
        print("Applying LoRA (PEFT) configuration")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

# ========================================
# Training
# ========================================

def main():
    # Setup distributed environment first
    setup_distributed_environment()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-GPU Reward Model Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="nvidia/HelpSteer3")
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="models/qwen3_8b_reward_model")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--remove_unused_columns", type=bool, default=False)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--ddp_timeout", type=int, default=7200)
    parser.add_argument("--dataloader_pin_memory", type=bool, default=False)
    
    args = parser.parse_args()
    
    # Initialize accelerator with timeout handling
    try:
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="bf16" if args.bf16 else "no",
            log_with=args.report_to if args.report_to != "none" else None,
            project_dir=args.output_dir,
            kwargs_handlers=[]
        )
    except Exception as e:
        print(f"Failed to initialize Accelerator: {e}")
        print("Falling back to basic setup...")
        accelerator = None
    
    # Get rank for logging
    rank = accelerator.process_index if accelerator else 0
    logger = setup_logging(rank)
    
    # Only print on main process
    if rank == 0:
        print("=" * 60)
        print("Multi-GPU Reward Model Training with HelpSteer3 Dataset")
        print("=" * 60)
        print(f"Base Model: {args.model_name}")
        print(f"Dataset: {args.dataset_name} (English only)")
        print(f"Output Directory: {args.output_dir}")
        print(f"Use PEFT: {args.use_peft}")
        if accelerator:
            print(f"Number of GPUs: {accelerator.num_processes}")
            effective_batch_size = (args.per_device_train_batch_size * 
                                  args.gradient_accumulation_steps * 
                                  accelerator.num_processes)
            print(f"Effective Batch Size: {effective_batch_size}")
        print("=" * 60)
    
    # Set seeds for reproducibility
    set_seed(42)
    if accelerator:
        accelerate_set_seed(42)
    
    # Enable TF32 for better performance on Ampere GPUs
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Create model arguments
    model_args = ModelArguments(
        model_name=args.model_name,
        max_length=args.max_length,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    # Load model and tokenizer
    model, tokenizer = create_reward_model(model_args)
    
    # Prepare dataset with distributed-aware loading
    if accelerator:
        with accelerator.local_main_process_first():
            print("Loading HelpSteer3 dataset...")
            raw_dataset = load_dataset("nvidia/HelpSteer3", split="train")
            
            # Filter for English language samples
            print(f"Original dataset size: {len(raw_dataset)}")
            english_dataset = raw_dataset.filter(lambda x: x.get('language', '').lower() == 'english')
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
            
            # Convert to chosen/rejected format
            reward_dataset = english_dataset.map(
                convert_to_chosen_rejected,
                batched=True,
                remove_columns=english_dataset.column_names,
                num_proc=1,
            )
            
            # Filter out empty results
            reward_dataset = reward_dataset.filter(lambda x: len(x['chosen']) > 0)
            print(f"Converted dataset size: {len(reward_dataset)} preference pairs")
    else:
        print("Loading HelpSteer3 dataset...")
        raw_dataset = load_dataset("nvidia/HelpSteer3", split="train")
        english_dataset = raw_dataset.filter(lambda x: x.get('language', '').lower() == 'english')
        # ... (same conversion logic)
    
    # Split dataset
    dataset_size = len(reward_dataset)
    train_size = int(0.9 * dataset_size)
    
    # Create train/eval split
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
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        tf32=args.tf32,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        remove_unused_columns=args.remove_unused_columns,
        report_to=args.report_to,
        run_name=args.run_name or f"qwen3_8b_reward_model_{rank}",
        ddp_timeout=args.ddp_timeout,
        max_length=args.max_length,
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
    
    try:
        trainer.train()
        
        # Save the final model
        if rank == 0:
            print("Saving final model...")
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
            
            # Save training info
            training_info = {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "training_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "use_peft": args.use_peft,
                "max_length": args.max_length,
                "final_loss": trainer.state.log_history[-1].get("train_loss", "N/A") if trainer.state.log_history else "N/A"
            }
            
            with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f, indent=2)
            
            print(f"Training completed! Model saved to {args.output_dir}")
            
            # Test the model with a sample prediction
            try:
                print("\nTesting model with sample predictions...")
                
                # Load the saved model for testing
                test_model = AutoModelForSequenceClassification.from_pretrained(
                    args.output_dir,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                
                test_prompts = [
                    "Human: What is the capital of France?\n\nAssistant: The capital of France is Paris.",
                    "Human: What is the capital of France?\n\nAssistant: I don't know.",
                ]
                
                for i, prompt in enumerate(test_prompts):
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(test_model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = test_model(**inputs)
                        reward_score = outputs.logits.cpu().item()
                    
                    print(f"Sample {i+1} reward score: {reward_score:.4f}")
                
                print("Sample predictions completed successfully!")
                
            except Exception as e:
                print(f"Error during sample prediction: {e}")
                print("Model training completed but sample prediction failed")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e

if __name__ == "__main__":
    main() 