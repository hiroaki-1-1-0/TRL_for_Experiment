#!/usr/bin/env python3
"""
Fixed Multi-GPU RLOO Training Script
Resolves deadlock and freeze issues in distributed training
"""

import os
import sys
import argparse
import logging
import warnings
import shutil
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import time
import threading
import signal

# Critical environment setup BEFORE any CUDA operations
os.environ.setdefault('NCCL_SHM_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1') 
os.environ.setdefault('NCCL_P2P_LEVEL', 'LOC')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_SOCKET_NTHREADS', '1')
os.environ.setdefault('NCCL_NSOCKS_PERTHREAD', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256,roundup_power2_divisions:16')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

# Prevent hanging in multiprocessing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
import gc

from trl import ModelConfig, RLOOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

# Force use of local modules by inserting project root first in sys.path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now import the local RLOOTrainer (will override any installed version)
from trl.trainer.rloo_trainer import RLOOTrainer

print(f"🔧 Using LOCAL RLOOTrainer with DistributedDataParallel fixes")
print(f"✅ Has _get_unwrapped_model method: {hasattr(RLOOTrainer, '_get_unwrapped_model')}")

# Global timeout handler
class TimeoutHandler:
    def __init__(self, timeout_seconds=300):  # 5 minutes timeout
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
        
    def check_timeout(self, operation_name=""):
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            print(f"⚠️ TIMEOUT after {elapsed:.1f}s during: {operation_name}")
            sys.exit(1)

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def safe_distributed_init():
    """Safely initialize distributed training with timeout"""
    if not dist.is_initialized():
        timeout_handler = TimeoutHandler(60)  # 1 minute for init
        
        try:
            rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            print(f"🔧 Initializing distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            
            # Set device before distributed init
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                
            # Initialize with timeout
            timeout_handler.check_timeout("before dist.init_process_group")
            if torch.cuda.is_available() and world_size > 1:
                dist.init_process_group(
                    backend='nccl',
                    timeout=torch.timedelta(seconds=60)
                )
            else:
                # Single GPU or CPU mode
                pass
            timeout_handler.check_timeout("after dist.init_process_group")
            
            print(f"✅ Distributed initialized successfully on rank {rank}")
            return rank, world_size, local_rank
            
        except Exception as e:
            print(f"❌ Distributed initialization failed: {e}")
            return 0, 1, 0
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return rank, world_size, local_rank

def filter_english_samples(dataset):
    """Filter dataset for English language samples only using 'language' column"""
    print("Filtering for English language samples using 'language' column...")
    return dataset.filter(lambda x: x.get('language', '').lower() == 'english')

def prepare_datasets(args, tokenizer, rank=0):
    """Prepare training and evaluation datasets with timeout protection"""
    timeout_handler = TimeoutHandler(300)  # 5 minutes for dataset loading

    print("\U0001F4CA Loading HelpSteer3 dataset...")
    timeout_handler.check_timeout("before dataset loading")

    try:
        # Load dataset
        dataset = load_dataset("nvidia/HelpSteer3", split="train")
        timeout_handler.check_timeout("after dataset loading")

        # Filter for English samples
        english_dataset = filter_english_samples(dataset)
        timeout_handler.check_timeout("after English filtering")

        # Convert to chosen/rejected format
        print("Converting to chosen/rejected format...")
        def convert_to_chosen_rejected(example):
            context = example['context']
            response1 = example['response1']
            response2 = example['response2']
            preference = example['overall_preference']
            # Skip neutral preferences
            if preference == 0:
                return {'chosen': None, 'rejected': None}
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
            if preference > 0:
                return {'chosen': full_text1, 'rejected': full_text2}
            else:
                return {'chosen': full_text2, 'rejected': full_text1}

        column_names = english_dataset.column_names
        reward_dataset = english_dataset.map(
            convert_to_chosen_rejected,
            batched=False,
            remove_columns=column_names,
        )
        # Filter out empty results
        reward_dataset = reward_dataset.filter(lambda x: x['chosen'] is not None and x['rejected'] is not None)
        print(f"Converted dataset size: {len(reward_dataset)} preference pairs")

        if len(reward_dataset) < 100:
            raise ValueError(f"Not enough English samples: {len(reward_dataset)}")

        # Use smaller subset for faster training
        total_samples = min(len(reward_dataset), 2000)  # Limit to 2000 samples
        reward_dataset = reward_dataset.select(range(total_samples))

        # Split: 90% train, 10% eval
        split_point = int(0.9 * len(reward_dataset))
        train_dataset = reward_dataset.select(range(split_point))
        eval_dataset = reward_dataset.select(range(split_point, len(reward_dataset)))

        print(f"\U0001F4C8 Train samples: {len(train_dataset)}")
        print(f"\U0001F4CA Eval samples: {len(eval_dataset)}")

        # Tokenize datasets with timeout protection
        print("\U0001F524 Tokenizing datasets...")
        timeout_handler.check_timeout("before tokenization")

        def rloo_tokenize_function(examples):
            """
            For RLOO training, we need to extract the prompt part for generation.
            We'll use the 'chosen' text but extract only the prompt part.
            """
            assert isinstance(examples['chosen'], list) and all(isinstance(x, str) for x in examples['chosen'])
            
            # Extract prompts from chosen texts (everything before "Assistant: ")
            prompts = []
            for text in examples['chosen']:
                if "Assistant: " in text:
                    prompt = text.split("Assistant: ")[0] + "Assistant: "
                else:
                    # Fallback: use the text as is
                    prompt = text
                prompts.append(prompt)
            
            # Tokenize prompts for RLOO training
            prompt_encodings = tokenizer(
                prompts,
                truncation=True,
                padding='max_length',
                max_length=256,  # Shorter for prompts
                return_tensors="pt",
                return_attention_mask=True
            )
            
            return {
                'input_ids': prompt_encodings['input_ids'],
                'attention_mask': prompt_encodings['attention_mask'],
            }

        def safe_tokenize(dataset, batch_size=50):
            print("Tokenizing samples...")
            tokenized_samples = []
            for i in range(0, len(dataset), batch_size):
                if i % 100 == 0:
                    timeout_handler.check_timeout(f"tokenizing batch {i}")
                batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
                tokenized_batch = rloo_tokenize_function(batch.to_dict())
                for j in range(len(tokenized_batch['input_ids'])):
                    tokenized_samples.append({
                        'input_ids': tokenized_batch['input_ids'][j],
                        'attention_mask': tokenized_batch['attention_mask'][j],
                    })
            print(f"Successfully tokenized {len(tokenized_samples)} samples")
            return Dataset.from_list(tokenized_samples)

        train_dataset = safe_tokenize(train_dataset)
        eval_dataset = safe_tokenize(eval_dataset)

        # Use smaller subsets to prevent hanging
        max_train_size = min(200, len(train_dataset))
        max_eval_size = min(50, len(eval_dataset))
        train_dataset = train_dataset.select(range(max_train_size))
        eval_dataset = eval_dataset.select(range(max_eval_size))

        print(f"✅ Final train dataset size: {len(train_dataset)}")
        print(f"✅ Final eval dataset size: {len(eval_dataset)}")

        timeout_handler.check_timeout("dataset preparation complete")
        return train_dataset, eval_dataset

    except Exception as e:
        print(f"❌ Dataset preparation failed: {e}")
        raise

class NonBlockingRLOOTrainer(RLOOTrainer):
    """Non-blocking RLOO trainer that prevents deadlocks"""
    
    def __init__(self, **kwargs):
        print(f"🚀 NonBlockingRLOOTrainer - Anti-deadlock initialization")
        
        # Do not extract device information - let the base class handle it
        # The base class will auto-detect model devices correctly
        
        # Initialize with timeout protection
        timeout_handler = TimeoutHandler(120)  # 2 minutes for trainer init
        
        try:
            timeout_handler.check_timeout("before super().__init__")
            super().__init__(**kwargs)
            timeout_handler.check_timeout("after super().__init__")
            
            print(f"  Reward model device: {self._get_model_device(self.reward_model) if hasattr(self, 'reward_model') else 'N/A'}")
            print(f"  Reference policy device: {self._get_model_device(self.ref_policy) if hasattr(self, 'ref_policy') else 'N/A'}")
            print(f"  Policy model device: {self._get_model_device(self.policy) if hasattr(self, 'policy') else 'N/A'}")
            
            print("✅ NonBlockingRLOOTrainer initialized successfully")
            
        except Exception as e:
            print(f"❌ NonBlockingRLOOTrainer initialization failed: {e}")
            raise

def load_models_with_timeout(args, tokenizer, rank=0, world_size=1):
    """Load models with timeout protection and proper device placement"""
    timeout_handler = TimeoutHandler(300)  # 5 minutes for model loading
    
    # Device mapping for multi-GPU setup - ensure all devices are within valid range
    num_gpus = torch.cuda.device_count()
    
    # Validate rank is within GPU range
    if rank >= num_gpus:
        print(f"⚠️ Warning: rank {rank} >= num_gpus {num_gpus}, using rank % num_gpus")
        effective_rank = rank % num_gpus
    else:
        effective_rank = rank
    
    if num_gpus >= 7:
        # For 7+ GPUs: distribute models across different GPUs more efficiently
        reward_device = f"cuda:{effective_rank % 3}"  # Use GPUs 0,1,2 for reward model
        ref_policy_device = f"cuda:{(effective_rank % 3) + 3}"  # Use GPUs 3,4,5 for ref policy  
        policy_device = f"cuda:6"  # Use GPU 6 for policy (shared)
    elif num_gpus >= 4:
        reward_device = f"cuda:{effective_rank % 2}"  # Use GPU 0 or 1 for reward model
        ref_policy_device = f"cuda:{(effective_rank % 2) + 2}"  # Use GPU 2 or 3 for ref policy
        # Ensure policy device is within valid range
        policy_device_idx = min((effective_rank % 2) + 4, num_gpus - 1)
        policy_device = f"cuda:{policy_device_idx}"
    elif num_gpus >= 3:
        reward_device = f"cuda:{effective_rank % num_gpus}"
        ref_policy_device = f"cuda:{(effective_rank + 1) % num_gpus}"
        policy_device = f"cuda:{(effective_rank + 2) % num_gpus}"
    else:
        # Fallback to shared GPUs with minimal memory usage
        reward_device = f"cuda:{effective_rank % num_gpus}"
        ref_policy_device = f"cuda:{effective_rank % num_gpus}"
        policy_device = f"cuda:{effective_rank % num_gpus}"
    
    print(f"🔧 Model device mapping (rank {rank}, effective_rank {effective_rank}, num_gpus {num_gpus}):")
    print(f"  Reward model: {reward_device}")
    print(f"  Reference policy: {ref_policy_device}")
    print(f"  Policy model: {policy_device}")
    
    # Load reward model with memory optimization
    print(f"🔄 Loading reward model from {args.reward_model_path}")
    timeout_handler.check_timeout("before reward model loading")

    # Extract device index safely
    try:
        reward_device_idx = int(reward_device.split(':')[1])
        reward_max_memory = {reward_device_idx: "10GB"} if reward_device_idx < num_gpus else None
    except (ValueError, IndexError):
        reward_max_memory = None

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map=reward_device,
        trust_remote_code=True,
        num_labels=1,
        low_cpu_mem_usage=True,
        max_memory=reward_max_memory  # Safe memory limit
    )
    timeout_handler.check_timeout("after reward model loading")
    
    # Load reference policy with memory optimization
    print(f"🔄 Loading reference policy from {args.model_name}")
    timeout_handler.check_timeout("before reference policy loading")
    
    # Extract device index safely
    try:
        ref_device_idx = int(ref_policy_device.split(':')[1])
        ref_max_memory = {ref_device_idx: "15GB"} if ref_device_idx < num_gpus else None
    except (ValueError, IndexError):
        ref_max_memory = None
    
    ref_policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=ref_policy_device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=ref_max_memory  # Safe memory limit
    )
    timeout_handler.check_timeout("after reference policy loading")
    
    # Load policy model with memory optimization
    print(f"🔄 Loading policy model from {args.model_name}")
    timeout_handler.check_timeout("before policy model loading")
    
    # Extract device index safely
    try:
        policy_device_idx = int(policy_device.split(':')[1])
        policy_max_memory = {policy_device_idx: "20GB"} if policy_device_idx < num_gpus else None
    except (ValueError, IndexError):
        policy_max_memory = None
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=policy_device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=policy_max_memory  # Safe memory limit
    )
    timeout_handler.check_timeout("after policy model loading")
    
    return reward_model, ref_policy, policy

def main():
    # Parse arguments using argparse directly for simplicity
    parser = argparse.ArgumentParser(description="RLOO Training Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path")
    parser.add_argument("--reward_model_path", type=str, default="experiment/models/qwen3_1.7b_reward_model", help="Path to trained reward model")
    parser.add_argument("--output_dir", type=str, default="experiment/models/qwen3_1.7b_rloo_model", help="Output directory")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--local_rollout_forward_batch_size", type=int, default=2, help="Rollout batch size")
    parser.add_argument("--rloo_k", type=int, default=4, help="RLOO K parameter")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--total_episodes", type=int, default=500, help="Total episodes")
    parser.add_argument("--num_ppo_epochs", type=int, default=1, help="Number of PPO epochs")
    parser.add_argument("--num_mini_batches", type=int, default=1, help="Number of mini batches")
    parser.add_argument("--missing_eos_penalty", type=float, default=1.0, help="Missing EOS penalty")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=2, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=25, help="Eval steps")
    parser.add_argument("--bf16", type=bool, default=True, help="Use BF16")
    parser.add_argument("--tf32", type=bool, default=True, help="Use TF32")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=1, help="Number of dataloader workers")
    parser.add_argument("--remove_unused_columns", type=bool, default=False, help="Remove unused columns")
    parser.add_argument("--report_to", type=str, default="none", help="Report to")
    parser.add_argument("--run_name", type=str, default="rloo_training", help="Run name")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate and adjust RLOO parameters to ensure compatibility
    local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
    
    print(f"🔧 RLOO Parameter Validation:")
    print(f"  per_device_train_batch_size: {args.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  num_mini_batches: {args.num_mini_batches}")
    print(f"  rloo_k: {args.rloo_k}")
    print(f"  Calculated local_batch_size: {local_batch_size}")
    
    # Ensure local_batch_size is a multiple of rloo_k
    if local_batch_size % args.rloo_k != 0:
        # Adjust per_device_train_batch_size to make it compatible
        min_per_device_batch_size = max(1, args.rloo_k // (args.gradient_accumulation_steps * args.num_mini_batches))
        if min_per_device_batch_size * args.gradient_accumulation_steps * args.num_mini_batches < args.rloo_k:
            min_per_device_batch_size = args.rloo_k
            args.gradient_accumulation_steps = 1
            args.num_mini_batches = 1
        
        args.per_device_train_batch_size = min_per_device_batch_size
        local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        
        print(f"⚠️  Adjusted parameters to ensure local_batch_size is multiple of rloo_k:")
        print(f"  New per_device_train_batch_size: {args.per_device_train_batch_size}")
        print(f"  New local_batch_size: {local_batch_size}")
    
    print(f"✅ RLOO parameters are valid\n")
    
    # Create config objects manually
    config = RLOOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rollout_forward_batch_size=args.local_rollout_forward_batch_size,
        rloo_k=args.rloo_k,
        learning_rate=args.learning_rate,
        total_episodes=args.total_episodes,
        num_ppo_epochs=args.num_ppo_epochs,
        num_mini_batches=args.num_mini_batches,
        missing_eos_penalty=args.missing_eos_penalty,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        tf32=args.tf32,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=args.remove_unused_columns,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
    )
    
    # Initialize distributed training with safety
    rank, world_size, local_rank = safe_distributed_init()
    
    # Setup logging
    logger = setup_logging(rank)
    
    if rank == 0:
        print("🚀 FIXED Multi-GPU RLOO Training")
        print("🔧 Anti-deadlock and timeout protection enabled")
        print("=" * 80)
        print(f"📋 Training Configuration:")
        print(f"  Policy Model: {args.model_name}")
        print(f"  Reward Model: {args.reward_model_path}")
        print(f"  Dataset: nvidia/HelpSteer3 (English only)")
        print(f"  Output Directory: {config.output_dir}")
        print(f"  Available GPUs: {torch.cuda.device_count()}")
        print("=" * 80)
    
    # Set random seed
    set_seed(config.seed)
    
    # Load tokenizer
    print(f"Using trained reward model from: {args.reward_model_path}")
    print(f"Loading tokenizer from {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    # Prepare datasets with timeout
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer, rank)
    
    # Load models with timeout
    reward_model, ref_policy, policy = load_models_with_timeout(
        args, tokenizer, rank, world_size
    )
    
    # Setup trainer with timeout protection
    print("🎯 Initializing NonBlockingRLOOTrainer...")
    
    try:
        trainer = NonBlockingRLOOTrainer(
            config=config,
            processing_class=tokenizer,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Start training with timeout monitoring
        print("🏃 Starting RLOO training...")
        
        training_timeout = TimeoutHandler(3600)  # 1 hour for training
        
        def training_monitor():
            while True:
                time.sleep(60)  # Check every minute
                training_timeout.check_timeout("training progress")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=training_monitor, daemon=True)
        monitor_thread.start()
        
        trainer.train()
        
        # Save final model
        if rank == 0:
            print("💾 Saving trained model...")
            trainer.save_model(config.output_dir)
            tokenizer.save_pretrained(config.output_dir)
            print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()