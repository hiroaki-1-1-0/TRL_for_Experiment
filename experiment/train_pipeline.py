#!/usr/bin/env python3
"""
Complete Training Pipeline for RLOO with Custom Reward Model

This script runs the complete training pipeline:
1. Train Qwen/Qwen3-8B as a reward model using RewardTrainer
2. Use the trained reward model for RLOO training

Usage:
python train_pipeline.py [--reward_model_only] [--rloo_only] [--reward_model_path PATH]
"""

import subprocess
import argparse
import os
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def train_reward_model(args):
    """Train the reward model"""
    cmd = [
        "python", "reward_model_training.py",
        "--output_dir", args.reward_model_dir,
        "--per_device_train_batch_size", str(args.reward_batch_size),
        "--gradient_accumulation_steps", str(args.reward_grad_accum),
        "--learning_rate", str(args.reward_lr),
        "--num_train_epochs", str(args.reward_epochs),
        "--max_length", str(args.max_length),
        "--warmup_steps", str(args.warmup_steps),
        "--eval_size", str(args.eval_size),
        "--dataset_num_proc", str(args.num_proc),
    ]
    
    if args.use_peft:
        cmd.extend([
            "--use_peft",
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout)
        ])
    
    if args.push_to_hub:
        cmd.append("--push_to_hub")
    
    return run_command(cmd, "Reward Model Training")


def train_rloo(args):
    """Train the RLOO model"""
    cmd = [
        "python", "rloo_helpsteer_training.py",
        "--output_dir", args.rloo_output_dir,
        "--per_device_train_batch_size", str(args.rloo_batch_size),
        "--gradient_accumulation_steps", str(args.rloo_grad_accum),
        "--learning_rate", str(args.rloo_lr),
        "--total_episodes", str(args.total_episodes),
        "--num_ppo_epochs", str(args.num_ppo_epochs),
        "--num_mini_batches", str(args.num_mini_batches),
        "--rloo_k", str(args.rloo_k),
        "--local_rollout_forward_batch_size", str(args.rollout_batch_size),
        "--missing_eos_penalty", str(args.eos_penalty),
        "--reward_model_path", args.reward_model_dir,
    ]
    
    if args.push_to_hub:
        cmd.append("--push_to_hub")
    
    return run_command(cmd, "RLOO Training")


def main():
    parser = argparse.ArgumentParser(description="Complete RLOO training pipeline")
    
    # Pipeline control
    parser.add_argument("--reward_model_only", action="store_true",
                       help="Only train the reward model")
    parser.add_argument("--rloo_only", action="store_true", 
                       help="Only run RLOO training (requires existing reward model)")
    
    # Directory settings
    parser.add_argument("--reward_model_dir", type=str, default="models/qwen3_8b_reward_model",
                       help="Directory for reward model")
    parser.add_argument("--rloo_output_dir", type=str, default="models/rloo_helpsteer3",
                       help="Directory for RLOO output")
    
    # Reward model training settings
    parser.add_argument("--reward_batch_size", type=int, default=4,
                       help="Batch size for reward model training")
    parser.add_argument("--reward_grad_accum", type=int, default=4,
                       help="Gradient accumulation for reward model")
    parser.add_argument("--reward_lr", type=float, default=1e-5,
                       help="Learning rate for reward model")
    parser.add_argument("--reward_epochs", type=int, default=3,
                       help="Number of epochs for reward model")
    
    # RLOO training settings
    parser.add_argument("--rloo_batch_size", type=int, default=2,
                       help="Batch size for RLOO training")
    parser.add_argument("--rloo_grad_accum", type=int, default=8,
                       help="Gradient accumulation for RLOO")
    parser.add_argument("--rloo_lr", type=float, default=1e-6,
                       help="Learning rate for RLOO")
    parser.add_argument("--total_episodes", type=int, default=50000,
                       help="Total episodes for RLOO")
    parser.add_argument("--num_ppo_epochs", type=int, default=1,
                       help="PPO epochs")
    parser.add_argument("--num_mini_batches", type=int, default=1,
                       help="Number of mini batches")
    parser.add_argument("--rloo_k", type=int, default=4,
                       help="RLOO k parameter")
    parser.add_argument("--rollout_batch_size", type=int, default=2,
                       help="Rollout batch size")
    parser.add_argument("--eos_penalty", type=float, default=1.0,
                       help="EOS penalty")
    
    # General settings
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--eval_size", type=int, default=1000,
                       help="Evaluation set size")
    parser.add_argument("--num_proc", type=int, default=4,
                       help="Number of processes for data processing")
    
    # PEFT settings
    parser.add_argument("--use_peft", action="store_true", default=True,
                       help="Use PEFT (LoRA) for training")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Hub settings
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push models to Hugging Face Hub")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.reward_model_only and args.rloo_only:
        print("Error: Cannot specify both --reward_model_only and --rloo_only")
        sys.exit(1)
    
    if args.rloo_only and not os.path.exists(args.reward_model_dir):
        print(f"Error: Reward model directory {args.reward_model_dir} does not exist")
        print("Please train the reward model first or specify correct path")
        sys.exit(1)
    
    # Create output directories
    Path(args.reward_model_dir).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rloo_output_dir).parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting RLOO Training Pipeline")
    print(f"Reward Model Directory: {args.reward_model_dir}")
    print(f"RLOO Output Directory: {args.rloo_output_dir}")
    print(f"Use PEFT: {args.use_peft}")
    
    # Run training steps
    success = True
    
    if not args.rloo_only:
        success &= train_reward_model(args)
        
        if not success:
            print("\n❌ Reward model training failed. Stopping pipeline.")
            sys.exit(1)
    
    if not args.reward_model_only:
        success &= train_rloo(args)
    
    if success:
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"Reward Model: {args.reward_model_dir}")
        if not args.reward_model_only:
            print(f"RLOO Model: {args.rloo_output_dir}")
    else:
        print(f"\n❌ Training pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 