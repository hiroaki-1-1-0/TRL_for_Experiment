#!/usr/bin/env python3
"""
FINAL Multi-GPU RLOO Training Script for HelpSteer3 Dataset
Integrates all technical insights for successful multi-GPU implementation
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

# CRITICAL: Optimized environment variables based on technical insights
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use first 3 GPUs only
os.environ.setdefault('NCCL_SHM_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_P2P_LEVEL', 'LOC')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
# Memory optimization from technical insights
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256,roundup_power2_divisions:16')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')

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

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def filter_english_samples(dataset):
    """Filter dataset for English language samples only - PRESERVED from original"""
    print("Filtering for English language samples...")
    
    english_samples = []
    for i, sample in enumerate(dataset):
        if sample.get('language', '').lower() == 'english':
            english_samples.append(sample)
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"Processed {i} samples, found {len(english_samples)} English samples")
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"After English filtering: {len(english_samples)} samples")
    print(f"Filtered out: {len(dataset) - len(english_samples)} non-English samples")
    
    # Convert back to Dataset
    from datasets import Dataset
    return Dataset.from_list(english_samples)

def prepare_helpsteer_dataset(dataset, tokenizer, max_samples=None):
    """Prepare HelpSteer3 dataset for RLOO training - PRESERVED from original"""
    
    def format_conversation(sample):
        """Format a sample into a conversation prompt"""
        context = sample.get('context', [])
        
        if isinstance(context, list) and len(context) > 0:
            # Multi-turn conversation
            messages = []
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            
            # If last message isn't from user, take the last user message
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                messages = [user_messages[-1]]
        
        prompt = sample.get('prompt', '')
        if prompt:
            messages = [{"role": "user", "content": prompt}]
        else:
            # Fallback
            messages = [{"role": "user", "content": "Please provide a helpful response."}]
        
        return messages

    def tokenize_sample(sample):
        """Tokenize a single sample"""
        try:
            messages = format_conversation(sample)
            
            # Apply chat template with limited length - optimized for memory
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                return_tensors=None,
                max_length=256,  # Reduced for memory efficiency
                truncation=True
            )
            
            return {
                "input_ids": input_ids,
                "lengths": len(input_ids)
            }
        except Exception as e:
            print(f"Error tokenizing sample: {e}")
            # Return a minimal fallback
            return {
                "input_ids": tokenizer.encode("Please help me.", max_length=64, truncation=True),
                "lengths": 10
            }
    
    print("Tokenizing samples...")
    tokenized_samples = []
    
    # Limit samples if specified
    samples_to_process = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    for i, sample in enumerate(samples_to_process):
        tokenized = tokenize_sample(sample)
        if tokenized["lengths"] > 0:
            tokenized_samples.append(tokenized)
        
        if i % 100 == 0:
            print(f"Tokenized {i}/{len(samples_to_process)} samples")
    
    print(f"Successfully tokenized {len(tokenized_samples)} samples")
    
    # Convert to Dataset
    return Dataset.from_list(tokenized_samples)

def setup_models_with_optimal_device_mapping(model_config: ModelConfig, training_args: RLOOConfig, 
                                            reward_model_path: Optional[str] = None):
    """Setup models with optimal device mapping based on technical insights"""
    
    policy_model_name = "Qwen/Qwen3-8B"
    
    if reward_model_path:
        reward_model_name = reward_model_path
        print(f"Using trained reward model from: {reward_model_name}")
    else:
        reward_model_name = "experiment/models/qwen3_8b_reward_model"
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
    
    # TECHNICAL INSIGHT: True Multi-GPU distribution strategy
    print("🔧 Applying technical insights: True Multi-GPU distribution")
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Multi-GPU distribution mapping for maximum memory utilization
    reward_device = "cuda:0"      # GPU 0 for reward model
    ref_policy_device = "cuda:1"  # GPU 1 for reference policy
    policy_device = "cuda:2"      # GPU 2 for policy model (training)
    
    print(f"🎯 Optimal device mapping:")
    print(f"  Reward model: {reward_device} (15GB)")
    print(f"  Reference policy: {ref_policy_device} (15GB)")
    print(f"  Policy model: {policy_device} (15GB + training)")
    
    # Load models with technical insights applied
    print(f"🔄 Loading reward model from {reward_model_name}")
    torch.cuda.empty_cache()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        device_map={"": reward_device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"🔄 Loading reference policy from {policy_model_name}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
        device_map={"": ref_policy_device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"🔄 Loading policy model from {policy_model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.float16,
        device_map={"": policy_device},
        low_cpu_mem_usage=True
    )
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return tokenizer, reward_model, ref_policy, policy, reward_device, ref_policy_device, policy_device

class OptimizedRLOOTrainer(RLOOTrainer):
    """RLOO trainer with all technical insights integrated"""
    
    def __init__(self, reward_device, ref_policy_device, policy_device, **kwargs):
        self.reward_device = reward_device
        self.ref_policy_device = ref_policy_device
        self.policy_device = policy_device
        
        print(f"🚀 OptimizedRLOOTrainer initialized with insights:")
        print(f"  Reward model device: {self.reward_device}")
        print(f"  Reference policy device: {self.ref_policy_device}")
        print(f"  Policy model device: {self.policy_device}")
        
        # Store models and apply technical insights
        self.stored_reward_model = kwargs.get('reward_model')
        self.stored_ref_policy = kwargs.get('ref_policy')
        self.stored_policy = kwargs.get('policy')
        
        # TECHNICAL INSIGHT: Disable device movements during initialization
        if self.stored_reward_model:
            self.stored_reward_model._original_to = self.stored_reward_model.to
            self.stored_reward_model.to = lambda *a, **kw: self.stored_reward_model
        if self.stored_ref_policy:
            self.stored_ref_policy._original_to = self.stored_ref_policy.to
            self.stored_ref_policy.to = lambda *a, **kw: self.stored_ref_policy
        
        # Initialize parent
        super().__init__(**kwargs)
        
        # Restore device movement methods
        if self.stored_reward_model and hasattr(self.stored_reward_model, '_original_to'):
            self.stored_reward_model.to = self.stored_reward_model._original_to
        if self.stored_ref_policy and hasattr(self.stored_ref_policy, '_original_to'):
            self.stored_ref_policy.to = self.stored_ref_policy._original_to
    
    def _move_tensor_to_device(self, tensor, device):
        """Safely move tensor to device"""
        if hasattr(tensor, 'to'):
            return tensor.to(device)
        return tensor
    
    def _move_batch_to_device(self, batch, device):
        """Move batch data to specified device"""
        if isinstance(batch, dict):
            return {k: self._move_tensor_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_tensor_to_device(item, device) for item in batch)
        else:
            return self._move_tensor_to_device(batch, device)
    
    def get_train_dataloader(self):
        """Override dataloader with device management insights"""
        original_dataloader = super().get_train_dataloader()
        
        class OptimizedDataLoader:
            def __init__(self, dataloader, trainer):
                self.dataloader = dataloader
                self.trainer = trainer
            
            def __iter__(self):
                for batch in self.dataloader:
                    # Apply technical insight: move batch to policy device
                    batch = self.trainer._move_batch_to_device(batch, self.trainer.policy_device)
                    yield batch
            
            def __len__(self):
                return len(self.dataloader)
            
            def __getattr__(self, name):
                return getattr(self.dataloader, name)
        
        return OptimizedDataLoader(original_dataloader, self)
    
    def train(self):
        """Training method with comprehensive device management"""
        print("🔥 Starting optimized training with device management...")
        
        # TECHNICAL INSIGHT: Monkey patch forward function for device management
        from trl.trainer import utils
        original_forward = utils.forward
        
        def device_aware_forward(model, input_ids, pad_token_id):
            """Advanced device management for multi-GPU setup"""
            # Determine model's device
            try:
                model_device = next(model.parameters()).device
                print(f"📍 Advanced Forward: model on {model_device}")
            except StopIteration:
                model_device = torch.device(self.policy_device)
            
            # Ensure all inputs are on model's device
            if torch.is_tensor(input_ids):
                input_ids = input_ids.to(model_device)
            elif isinstance(input_ids, dict):
                input_ids = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in input_ids.items()}
            elif isinstance(input_ids, (list, tuple)):
                input_ids = [item.to(model_device) if torch.is_tensor(item) else item for item in input_ids]
            
            print(f"📍 Input moved to {model_device}")
            
            # Execute on correct device with context
            with torch.cuda.device(model_device):
                try:
                    # Call original forward
                    outputs = original_forward(model, input_ids, pad_token_id)
                    print(f"✅ Forward successful on {model_device}")
                    
                    # Important: Move outputs to policy device for consistency
                    if model_device != torch.device(self.policy_device):
                        if hasattr(outputs, 'logits'):
                            outputs.logits = outputs.logits.to(self.policy_device)
                        elif hasattr(outputs, 'to'):
                            outputs = outputs.to(self.policy_device)
                        print(f"📤 Outputs moved to {self.policy_device}")
                    
                    return outputs
                except Exception as e:
                    print(f"❌ Forward failed on {model_device}: {e}")
                    # Enhanced fallback
                    try:
                        if torch.is_tensor(input_ids):
                            attention_mask = torch.ones_like(input_ids).to(model_device)
                        else:
                            attention_mask = None
                        
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        print(f"✅ Fallback successful on {model_device}")
                        return outputs
                    except Exception as e2:
                        print(f"❌ Fallback also failed: {e2}")
                        raise e
        
        # Apply monkey patch
        utils.forward = device_aware_forward
        print("🔧 Applied advanced forward function monkey patch")
        
        try:
            result = super().train()
            print("✅ Training completed successfully!")
            return result
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Restore original function
            utils.forward = original_forward
            print("🔧 Restored original forward function")

def main():
    print("🚀 FINAL Multi-GPU RLOO Training with Technical Insights")
    print("="*70)
    
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    rank = 0
    logger = setup_logging(rank)
    
    # Get reward model path
    reward_model_path = getattr(script_args, 'reward_model_path', None)
    if reward_model_path is None:
        reward_model_path = "experiment/models/qwen3_8b_reward_model"
    
    # Remove output directory if it exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    print("📋 Training Configuration:")
    print(f"  Policy Model: Qwen/Qwen3-8B")
    print(f"  Reward Model: {reward_model_path}")
    print(f"  Dataset: nvidia/HelpSteer3 (English only)")
    print(f"  Output Directory: {training_args.output_dir}")
    print(f"  Available GPUs: {torch.cuda.device_count()}")
    print("="*70)
    
    # Set seeds
    set_seed(42)
    
    # Filter warnings
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)
    
    # TECHNICAL INSIGHT: Comprehensive GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Conservative memory settings
        for i in range(torch.cuda.device_count()):
            if i < 3:  # Only for visible GPUs
                torch.cuda.set_per_process_memory_fraction(0.8, device=i)
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    ################
    # Models & Tokenizer with Technical Insights
    ################
    tokenizer, reward_model, ref_policy, policy, reward_device, ref_policy_device, policy_device = setup_models_with_optimal_device_mapping(
        model_args, training_args, reward_model_path
    )
    
    ################
    # Dataset Loading - PRESERVED from original
    ################
    print("📊 Loading HelpSteer3 dataset...")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Filter for English language samples only
    dataset = filter_english_samples(dataset)
    
    # Split into train and eval
    eval_samples = min(500, len(dataset) // 20)
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    print(f"📈 Train samples: {len(train_dataset)}")
    print(f"📊 Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets with tokenization - PRESERVED from original
    print("🔤 Tokenizing datasets...")
    train_dataset = prepare_helpsteer_dataset(train_dataset, tokenizer, max_samples=200)  # Further reduced for memory
    eval_dataset = prepare_helpsteer_dataset(eval_dataset, tokenizer, max_samples=20)
    
    print(f"✅ Final train dataset size: {len(train_dataset)}")
    print(f"✅ Final eval dataset size: {len(eval_dataset)}")
    
    ################
    # Training with Technical Insights
    ################
    print("🎯 Initializing optimized RLOO trainer...")
    
    trainer = OptimizedRLOOTrainer(
        reward_device=reward_device,
        ref_policy_device=ref_policy_device,
        policy_device=policy_device,
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("🔥 Starting RLOO training with all technical insights...")
    trainer.train()
    
    ################
    # Save Model
    ################
    print(f"💾 Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    print("🎉 Multi-GPU RLOO training completed successfully with technical insights!")

if __name__ == "__main__":
    main() 