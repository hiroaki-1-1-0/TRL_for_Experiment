#!/usr/bin/env python3
"""
CPU Offload RLOO Training Script with HelpSteer3 Dataset
Memory-efficient solution using CPU offload strategy
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
import gc

# Memory optimization environment variables
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.trainer_utils import set_seed
from transformers.hf_argparser import HfArgumentParser
from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def filter_english_samples(dataset):
    """Filter dataset for English language samples only"""
    print("Filtering for English language samples...")
    
    english_samples = []
    for i, sample in enumerate(dataset):
        if sample.get('language', '').lower() == 'english':
            english_samples.append(sample)
        
        if i % 1000 == 0:
            print(f"Processed {i} samples, found {len(english_samples)} English samples")
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"After English filtering: {len(english_samples)} samples")
    
    # Convert back to Dataset
    return Dataset.from_list(english_samples)

def prepare_helpsteer_dataset(dataset, tokenizer, max_samples=None):
    """Prepare HelpSteer3 dataset for RLOO training"""
    
    def format_conversation(sample):
        """Format a sample into a conversation prompt"""
        context = sample.get('context', [])
        
        if isinstance(context, list) and len(context) > 0:
            messages = []
            for msg in context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if user_messages:
                messages = [user_messages[-1]]
        
        prompt = sample.get('prompt', '')
        if prompt:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": "Please provide a helpful response."}]
        
        return messages

    def tokenize_sample(sample):
        """Tokenize a single sample"""
        try:
            messages = format_conversation(sample)
            
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                return_tensors=None,
                max_length=96,  # Very short for memory efficiency
                truncation=True
            )
            
            return {
                "input_ids": input_ids,
                "lengths": len(input_ids)
            }
        except Exception as e:
            print(f"Error tokenizing sample: {e}")
            return {
                "input_ids": tokenizer.encode("Please help me.", max_length=32, truncation=True),
                "lengths": 10
            }
    
    print("Tokenizing samples...")
    tokenized_samples = []
    
    samples_to_process = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    for i, sample in enumerate(samples_to_process):
        tokenized = tokenize_sample(sample)
        if tokenized["lengths"] > 0:
            tokenized_samples.append(tokenized)
        
        if i % 50 == 0:
            print(f"Tokenized {i}/{len(samples_to_process)} samples")
    
    print(f"Successfully tokenized {len(tokenized_samples)} samples")
    return Dataset.from_list(tokenized_samples)

class CPUOffloadRLOOTrainer(RLOOTrainer):
    """RLOO trainer with CPU offload strategy"""
    
    def __init__(self, **kwargs):
        print("🔧 CPUOffloadRLOOTrainer with memory optimization")
        
        # Store models for device management
        self.stored_reward_model = kwargs.get('reward_model')
        self.stored_ref_policy = kwargs.get('ref_policy')
        self.stored_policy = kwargs.get('policy')
        
        # Prevent automatic device movements during init
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
            return tensor.to(device, non_blocking=True)
        return tensor
    
    def get_train_dataloader(self):
        """Override dataloader for device management"""
        original_dataloader = super().get_train_dataloader()
        
        class CPUOffloadDataLoader:
            def __init__(self, dataloader, trainer):
                self.dataloader = dataloader
                self.trainer = trainer
            
            def __iter__(self):
                for batch in self.dataloader:
                    # Move batch to GPU (policy device)
                    if isinstance(batch, dict):
                        batch = {k: self.trainer._move_tensor_to_device(v, "cuda:0") for k, v in batch.items()}
                    yield batch
            
            def __len__(self):
                return len(self.dataloader)
            
            def __getattr__(self, name):
                return getattr(self.dataloader, name)
        
        return CPUOffloadDataLoader(original_dataloader, self)
    
    def train(self):
        """Training with CPU offload device management"""
        print("🔥 Starting CPU offload training...")
        
        # Override TRL forward function
        from trl.trainer import utils
        original_forward = utils.forward
        
        def cpu_offload_forward(model, input_ids, pad_token_id):
            """Forward function with CPU offload handling"""
            # Determine model device
            try:
                model_device = next(model.parameters()).device
                print(f"📍 CPU Offload Forward: model on {model_device}")
            except StopIteration:
                model_device = torch.device("cuda:0")
            
            # Move inputs to model device
            if torch.is_tensor(input_ids):
                input_ids = input_ids.to(model_device, non_blocking=True)
            
            # Execute forward pass
            try:
                with torch.cuda.device(model_device) if model_device.type == 'cuda' else torch.device('cpu'):
                    if torch.is_tensor(input_ids):
                        attention_mask = (input_ids != pad_token_id).long().to(model_device)
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                            return_dict=True
                        )
                    else:
                        outputs = original_forward(model, input_ids, pad_token_id)
                
                print(f"✅ CPU Offload Forward successful on {model_device}")
                
                # Move outputs to GPU if model was on CPU
                if model_device.type == 'cpu' and hasattr(outputs, 'logits'):
                    outputs.logits = outputs.logits.to("cuda:0", non_blocking=True)
                    print(f"📤 Outputs moved from CPU to CUDA")
                
                return outputs
                
            except Exception as e:
                print(f"❌ CPU Offload Forward failed: {e}")
                if torch.is_tensor(input_ids):
                    attention_mask = torch.ones_like(input_ids).to(model_device)
                    return model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    raise e
        
        # Apply monkey patch
        utils.forward = cpu_offload_forward
        print("🔧 Applied CPU offload forward function")
        
        try:
            result = super().train()
            print("✅ CPU offload training completed successfully!")
            return result
        except Exception as e:
            print(f"❌ CPU offload training error: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            utils.forward = original_forward
            print("🔧 Restored original forward function")

def main():
    print("🚀 CPU Offload RLOO Training with HelpSteer3")
    print("="*60)
    
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    logger = setup_logging()
    
    # Get reward model path
    reward_model_path = getattr(script_args, 'reward_model_path', None)
    if reward_model_path is None:
        reward_model_path = "experiment/models/qwen3_8b_reward_model"
    
    # Remove output directory if it exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    print("📋 CPU Offload Training Configuration:")
    print(f"  Policy Model: Qwen/Qwen3-8B")
    print(f"  Reward Model: {reward_model_path}")
    print(f"  Dataset: nvidia/HelpSteer3 (English only)")
    print(f"  Output Directory: {training_args.output_dir}")
    print(f"  Device Strategy: CPU Offload")
    print("="*60)
    
    # Set seeds
    set_seed(42)
    
    # Filter warnings
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)
    
    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
    
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Device mapping - CPU offload strategy
    reward_device = "cpu"         # Offload to CPU
    ref_policy_device = "cpu"     # Offload to CPU
    policy_device = "cuda:0"      # Keep on GPU for training
    
    print(f"🎯 CPU Offload device mapping:")
    print(f"  Reward model: {reward_device}")
    print(f"  Reference policy: {ref_policy_device}")
    print(f"  Policy model: {policy_device}")
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    # Load models with CPU offload
    print(f"🔄 Loading reward model to {reward_device}...")
    torch.cuda.empty_cache()
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map={"": reward_device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"🔄 Loading reference policy to {ref_policy_device}...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map={"": ref_policy_device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"🔄 Loading policy model to {policy_device}...")
    policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map={"": policy_device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    ################
    # Dataset Loading - HelpSteer3
    ################
    print("📊 Loading HelpSteer3 dataset...")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Filter for English language samples only
    dataset = filter_english_samples(dataset)
    
    # Split into train and eval
    eval_samples = min(50, len(dataset) // 100)
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    print(f"📈 Train samples: {len(train_dataset)}")
    print(f"📊 Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets with tokenization
    print("🔤 Tokenizing datasets...")
    train_dataset = prepare_helpsteer_dataset(train_dataset, tokenizer, max_samples=30)
    eval_dataset = prepare_helpsteer_dataset(eval_dataset, tokenizer, max_samples=5)
    
    print(f"✅ Final train dataset size: {len(train_dataset)}")
    print(f"✅ Final eval dataset size: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    print("🎯 Initializing CPU offload RLOO trainer...")
    
    trainer = CPUOffloadRLOOTrainer(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("🔥 Starting CPU offload RLOO training...")
    trainer.train()
    
    ################
    # Save Model
    ################
    print(f"💾 Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    print("🎉 CPU offload RLOO training completed successfully!")

if __name__ == "__main__":
    main() 