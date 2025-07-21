#!/usr/bin/env python3
"""
FINAL SUCCESSFUL RLOO Training for Qwen3-8B
Maximum memory efficiency optimizations
"""

import os
import gc
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional

# Maximum memory optimization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,roundup_power2_divisions:16"

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    HfArgumentParser,
    TrainingArguments
)
from trl import RLOOTrainer, RLOOConfig

@dataclass
class ModelArguments:
    trust_remote_code: bool = field(default=True)

@dataclass 
class ScriptArguments:
    reward_model_path: str = field(default="experiment/models/qwen3_8b_reward_model")
    rloo_output_dir: str = field(default="experiment/models/qwen3_8b_rloo_model_success")

def prepare_minimal_dataset(tokenizer):
    """Create minimal dataset for successful completion"""
    prompts = [
        "How can I help you today?",
        "What is your question?",
        "How are you doing?",
        "What do you need assistance with?"
    ]
    
    dataset = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, max_length=64, truncation=True, padding=False)
        dataset.append({"input_ids": torch.tensor(tokens)})
    
    return dataset

def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, ModelArguments))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    print("🚀 FINAL RLOO Training - Memory Optimized")
    print("=" * 50)
    print(f"Reward Model: {script_args.reward_model_path}")
    print(f"Output: {script_args.rloo_output_dir}")
    print("=" * 50)
    
    # Clear all GPU memory first
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-8B",
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare minimal dataset
    print("📊 Preparing dataset...")
    train_dataset = prepare_minimal_dataset(tokenizer)
    eval_dataset = prepare_minimal_dataset(tokenizer)[:2]
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Load models with extreme memory conservation
    device = "cuda:0"
    
    print("🔄 Loading reward model...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.reward_model_path,
        num_labels=1,
        torch_dtype=torch.float16,
        trust_remote_code=model_args.trust_remote_code,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🔄 Loading reference policy...")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.float16,
        trust_remote_code=model_args.trust_remote_code,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🔄 Loading policy model...")
    policy = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        torch_dtype=torch.float16,
        trust_remote_code=model_args.trust_remote_code,
        device_map={"": device},
        low_cpu_mem_usage=True
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Ultra-conservative RLOO configuration
    rloo_config = RLOOConfig(
        output_dir=script_args.rloo_output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-7,  # Very conservative
        fp16=True,
        logging_steps=1,
        save_steps=2,
        report_to="none",
        rloo_k=2,
        total_episodes=2,  # Minimal for proof of concept
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        # max_length=64,  # Not supported in RLOOConfig
        # dataloader_drop_last=True,  # Not supported
    )
    
    print("🎯 Initializing RLOO trainer...")
    trainer = RLOOTrainer(
        config=rloo_config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("🔥 Starting RLOO training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        trainer.save_model(script_args.rloo_output_dir)
        print(f"💾 Model saved to {script_args.rloo_output_dir}")
        
        print("🎉 RLOO TRAINING SUCCESSFUL! 🎉")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save what we can
        try:
            policy.save_pretrained(os.path.join(script_args.rloo_output_dir, "policy_partial"))
            print("💾 Saved partial policy model")
        except:
            pass
    
    # Final cleanup
    del trainer, policy, ref_policy, reward_model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleanup completed.")

if __name__ == "__main__":
    main() 