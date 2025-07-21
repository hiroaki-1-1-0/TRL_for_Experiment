#!/usr/bin/env python3
"""
Multi-GPU Distributed RLOO Training Script for HelpSteer3 Dataset
Uses multiple GPUs with proper device mapping for memory distribution
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

# Set environment variables early
os.environ.setdefault('NCCL_SHM_DISABLE', '1')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_P2P_LEVEL', 'LOC')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
# Conservative memory allocation to avoid PyTorch allocator bugs
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128,roundup_power2_divisions:16')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
# Additional memory optimizations
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')

from accelerate import Accelerator, PartialState
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
    """Setup logging for distributed training"""
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
    """Prepare HelpSteer3 dataset for RLOO training"""
    
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
            
            # Apply chat template with limited length
            input_ids = tokenizer.apply_chat_template(
                messages,
                padding=False,
                add_generation_prompt=True,
                return_tensors=None,
                max_length=512,  # Shorter prompts for memory efficiency
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

def setup_models_and_tokenizer_distributed(model_config: ModelConfig, training_args: RLOOConfig, 
                                          reward_model_path: Optional[str] = None, accelerator=None):
    """Setup models with distributed device mapping"""
    
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
    
    # Calculate device mapping for 4 GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Distribute models across available GPUs
    if num_gpus >= 3:
        # Distribute 3 models across available GPUs
        reward_device = f"cuda:0"
        ref_policy_device = f"cuda:1" 
        policy_device = f"cuda:2"
    elif num_gpus >= 2:
        reward_device = f"cuda:0"
        ref_policy_device = f"cuda:1"
        policy_device = f"cuda:0"  # Share with reward model
    else:
        reward_device = "cuda:0"
        ref_policy_device = "cuda:0"
        policy_device = "cuda:0"
    
    print(f"Device mapping:")
    print(f"  Reward model: {reward_device}")
    print(f"  Reference policy: {ref_policy_device}")
    print(f"  Policy model: {policy_device}")
    
    # Load models with memory optimization and error handling
    print(f"Loading reward model from {reward_model_name}")
    torch.cuda.empty_cache()  # Clear before each model load
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map={"": reward_device},  # More explicit device mapping
        low_cpu_mem_usage=True,
        max_memory={reward_device: "15GB"}  # Conservative memory limit
    )
    
    print(f"Loading reference policy from {policy_model_name}")
    torch.cuda.empty_cache()  # Clear before each model load
    ref_policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map={"": ref_policy_device},  # More explicit device mapping
        low_cpu_mem_usage=True,
        max_memory={ref_policy_device: "15GB"}  # Conservative memory limit
    )
    
    print(f"Loading policy model from {policy_model_name}")
    torch.cuda.empty_cache()  # Clear before each model load
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map={"": policy_device},  # More explicit device mapping
        low_cpu_mem_usage=True,
        max_memory={policy_device: "15GB"}  # Conservative memory limit
    )
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    return tokenizer, reward_model, ref_policy, policy, reward_device, ref_policy_device, policy_device

def main():
    # Parse arguments first
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Skip Accelerator for single-process multi-GPU setup  
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    #     mixed_precision="bf16",
    # )
    accelerator = None
    
    rank = 0  # Single process, main rank
    logger = setup_logging(rank)
    
    # Get reward model path
    reward_model_path = getattr(script_args, 'reward_model_path', None)
    if reward_model_path is None:
        reward_model_path = "experiment/models/qwen3_8b_reward_model"
    
    # Remove output directory if it exists
    if rank == 0:
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    # Wait for all processes
    # accelerator.wait_for_everyone()  # Skip for single process
    
    if rank == 0:
        print("="*60)
        print("Multi-GPU RLOO Training with HelpSteer3 Dataset")
        print("="*60)
        print(f"Policy Model: Qwen/Qwen3-8B")
        print(f"Reward Model: {reward_model_path} (Qwen/Qwen3-8B trained)")
        print(f"Dataset: nvidia/HelpSteer3 (English only)")
        print(f"Output Directory: {training_args.output_dir}")
        print(f"Number of processes: {torch.cuda.device_count()}")
        print("="*60)
    
    # Set seeds
    set_seed(42)
    
    # Filter warnings
    warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*method.*", category=UserWarning)
    
    # Comprehensive GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Force garbage collection
        import gc
        gc.collect()
        
        # Set conservative memory fraction
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_per_process_memory_fraction(0.85, device=i)
    
    # Enable optimizations with conservative settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    ################
    # Models & Tokenizer
    ################
    tokenizer, reward_model, ref_policy, policy, reward_device, ref_policy_device, policy_device = setup_models_and_tokenizer_distributed(
        model_args, training_args, reward_model_path, accelerator
    )
    
    ################
    # Dataset Loading and Preparation
    ################
    # with accelerator.local_main_process_first():  # Skip for single process
    if True:  # Always execute in single process
        if rank == 0:
            print("Loading HelpSteer3 dataset...")
        dataset = load_dataset("nvidia/HelpSteer3", split="train")
        
        # Filter for English language samples only
        dataset = filter_english_samples(dataset)
        
        # Split into train and eval
        eval_samples = min(500, len(dataset) // 20)  # 5% for eval, max 500 samples
        train_dataset = dataset.select(range(len(dataset) - eval_samples))
        eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
        
        if rank == 0:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Eval samples: {len(eval_dataset)}")
        
        # Prepare datasets with tokenization - limit for memory
        if rank == 0:
            print("Tokenizing datasets...")
        train_dataset = prepare_helpsteer_dataset(train_dataset, tokenizer, max_samples=2000)
        eval_dataset = prepare_helpsteer_dataset(eval_dataset, tokenizer, max_samples=200)
        
        if rank == 0:
            print(f"Final train dataset size: {len(train_dataset)}")
            print(f"Final eval dataset size: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    if rank == 0:
        print("Initializing RLOO Trainer...")
    
    # Create a comprehensive custom trainer for multi-GPU device management
    class MultiGPURLOOTrainer(RLOOTrainer):
        def __init__(self, reward_device, ref_policy_device, policy_device, *args, **kwargs):
            # Store device mapping for consistent use
            self.reward_device = reward_device
            self.ref_policy_device = ref_policy_device
            self.policy_device = policy_device
            
            print(f"Initializing MultiGPURLOOTrainer with device mapping:")
            print(f"  Reward model: {self.reward_device}")
            print(f"  Reference policy: {self.ref_policy_device}")
            print(f"  Policy model: {self.policy_device}")
            print(f"  CUDA_VISIBLE_DEVICES effect: GPUs 0,1,2,3 are visible as cuda:0,1,2,3")
            
            # Store models before parent initialization
            self.stored_reward_model = kwargs.get('reward_model')
            self.stored_ref_policy = kwargs.get('ref_policy')
            self.stored_policy = kwargs.get('policy')
            
            # Temporarily replace device movement methods
            if self.stored_reward_model:
                self.stored_reward_model._original_to = self.stored_reward_model.to
                self.stored_reward_model.to = lambda *a, **kw: self.stored_reward_model
            if self.stored_ref_policy:
                self.stored_ref_policy._original_to = self.stored_ref_policy.to
                self.stored_ref_policy.to = lambda *a, **kw: self.stored_ref_policy
            
            # Initialize parent with device movement disabled
            super().__init__(*args, **kwargs)
            
            # Restore original device movement methods
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
        
        def training_step(self, model, inputs):
            """Override training step to handle device placement"""
            # Move inputs to policy device
            inputs = self._move_batch_to_device(inputs, self.policy_device)
            return super().training_step(model, inputs)
        
        def get_train_dataloader(self):
            """Override dataloader to handle device placement"""
            original_dataloader = super().get_train_dataloader()
            
            class MultiGPUDataLoader:
                def __init__(self, dataloader, trainer):
                    self.dataloader = dataloader
                    self.trainer = trainer
                
                def __iter__(self):
                    for batch in self.dataloader:
                        # Move all batch data to policy device
                        batch = self.trainer._move_batch_to_device(batch, self.trainer.policy_device)
                        yield batch
                
                def __len__(self):
                    return len(self.dataloader)
                
                def __getattr__(self, name):
                    return getattr(self.dataloader, name)
            
            return MultiGPUDataLoader(original_dataloader, self)
        
        def _forward_with_device_management(self, model, inputs, target_device):
            """Forward pass with explicit device management"""
            # Move inputs to model's device
            inputs_on_device = self._move_batch_to_device(inputs, target_device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs_on_device)
            
            # Move outputs back to policy device if needed
            if target_device != self.policy_device:
                if hasattr(outputs, 'logits'):
                    outputs.logits = outputs.logits.to(self.policy_device)
                elif hasattr(outputs, 'to'):
                    outputs = outputs.to(self.policy_device)
            
            return outputs
        
        def train(self):
            """Override train method to handle device placement throughout training"""
            # Import required modules
            from trl.trainer import utils
            import torch.nn.functional as F
            
            # Store original functions
            original_forward = utils.forward
            
            def device_aware_forward(model, input_ids, pad_token_id):
                """Custom forward function with comprehensive device management"""
                print(f"Forward called - Model device: {next(model.parameters()).device}, Input device: {input_ids.device if hasattr(input_ids, 'device') else 'N/A'}")
                
                # Determine which device the model is on
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device("cuda:0")  # fallback
                
                # Handle different input types and move to model device
                if torch.is_tensor(input_ids):
                    input_ids = input_ids.to(model_device)
                elif isinstance(input_ids, dict):
                    input_ids = {k: v.to(model_device) if torch.is_tensor(v) else v for k, v in input_ids.items()}
                elif isinstance(input_ids, (list, tuple)):
                    input_ids = [item.to(model_device) if torch.is_tensor(item) else item for item in input_ids]
                
                print(f"After device move - Input device: {input_ids.device if hasattr(input_ids, 'device') else 'N/A'}")
                
                # Call original forward on the correct device
                with torch.cuda.device(model_device):
                    try:
                        outputs = original_forward(model, input_ids, pad_token_id)
                    except Exception as e:
                        print(f"Forward error: {e}")
                        # Fallback: call model directly
                        attention_mask = torch.ones_like(input_ids)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                print(f"Forward completed on device: {model_device}")
                return outputs
            
            # Replace the forward function
            utils.forward = device_aware_forward
            print("Monkey patching forward function for device management")
            
            try:
                # Call parent train method
                print("Starting parent train method...")
                result = super().train()
                return result
            except Exception as e:
                print(f"Training error: {e}")
                raise
            finally:
                # Restore original forward function
                utils.forward = original_forward
                print("Restored original forward function")
        
        def get_eval_dataloader(self, eval_dataset=None):
            """Override eval dataloader to handle device placement"""
            original_dataloader = super().get_eval_dataloader(eval_dataset)
            
            class MultiGPUDataLoader:
                def __init__(self, dataloader, trainer):
                    self.dataloader = dataloader
                    self.trainer = trainer
                
                def __iter__(self):
                    for batch in self.dataloader:
                        # Move all batch data to policy device
                        batch = self.trainer._move_batch_to_device(batch, self.trainer.policy_device)
                        yield batch
                
                def __len__(self):
                    return len(self.dataloader)
                
                def __getattr__(self, name):
                    return getattr(self.dataloader, name)
            
            return MultiGPUDataLoader(original_dataloader, self)
    
    trainer = MultiGPURLOOTrainer(
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
    
    if rank == 0:
        print("Starting RLOO training...")
    
    trainer.train()
    
    ################
    # Save Model
    ################
    if rank == 0:
        print(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        print("Training completed successfully!")

if __name__ == "__main__":
    main() 