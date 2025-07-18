# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RLOO Training Script using HelpSteer3 Dataset

This script performs RLOO (REINFORCE Leave-One-Out) training using:
- Dataset: nvidia/HelpSteer3 (preference subset, train split, English language only)
- Policy Model: Qwen/Qwen3-8B (Now available on Hugging Face)
- Reward Model: Qwen/Qwen3-8B trained as reward model using RewardTrainer

The script filters HelpSteer3 to use only samples with language='english' for training.
The reward model should be trained first using the reward_model_training.py script.

Usage:
python rloo_helpsteer_training.py \
    --output_dir models/rloo_helpsteer3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --total_episodes 50000 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --local_rollout_forward_batch_size 2 \
    --missing_eos_penalty 1.0 \
    --reward_model_path models/qwen3_8b_reward_model
"""

import shutil
import argparse
from typing import Dict, Any, Union, Optional

from accelerate import PartialState
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.hf_argparser import HfArgumentParser

from trl import ModelConfig, RLOOConfig, RLOOTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def filter_english_samples(dataset: Union[Dataset, Any]) -> Dataset:
    """
    Filter the HelpSteer3 dataset to include only samples with language='english'.
    HelpSteer3 has multiple domains and languages, we want only English language samples.
    """
    
    def is_english_sample(example):
        # Filter for samples where language field is 'english'
        return example.get('language', '').lower() == 'english'
    
    print(f"Original dataset size: {len(dataset)}") # 38,459
    
    # Filter to only English language samples
    english_dataset = dataset.filter(is_english_sample)
    
    print(f"After English filtering: {len(english_dataset)} samples") # 22,380
    print(f"Filtered out: {len(dataset) - len(english_dataset)} non-English samples") # 16,079
    
    return english_dataset


def prepare_helpsteer_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """
    Prepare HelpSteer3 dataset for RLOO training.
    
    HelpSteer3 format:
    - context: List of conversation turns (messages)
    - response1: First response option
    - response2: Second response option  
    - overall_preference: Overall preference (-2 to 2)
    - individual_preference: Detailed preference annotations
    - domain: Task domain (e.g., code, general)
    - language: Sample language
    """
    
    def extract_and_tokenize_prompt(element):
        """Extract conversation context and tokenize for RLOO training"""
        # Extract the conversation context
        context = element["context"]
        
        # Build the conversation prompt from context
        if isinstance(context, list) and len(context) > 0:
            # Use the last user message as the prompt for generation
            user_messages = [msg for msg in context if msg.get("role") == "user"]
            if user_messages:
                # Take the last user message as the main prompt
                prompt = user_messages[-1]["content"]
            else:
                # Fallback: join all messages
                prompt = " ".join([msg.get("content", "") for msg in context])
        else:
            # Fallback for unexpected format
            prompt = str(context)
        
        # Apply chat template if available
        if tokenizer.chat_template is not None:
            # Format the full context as a conversation
            if isinstance(context, list):
                # Use the conversation context as-is, but only include user messages for generation
                messages = [msg for msg in context if msg.get("role") in ["user", "assistant"]]
                # Add generation prompt for the assistant
                if not messages or messages[-1].get("role") != "user":
                    # If last message isn't from user, take the last user message
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
                # Fallback to simple user message
                messages = [{"role": "user", "content": prompt}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    padding=False,
                    add_generation_prompt=True,
                    return_tensors=None
                )
        else:
            # Fallback to direct tokenization
            outputs = tokenizer(
                prompt,
                padding=False,
                truncation=True,
                max_length=512,  # Limit input length
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


def setup_models_and_tokenizer(model_config: ModelConfig, training_args: RLOOConfig, reward_model_path: Optional[str] = None):
    """Setup policy, reference policy, reward model, and tokenizer"""
    
    # Using Qwen3-8B which is now available on Hugging Face
    policy_model_name = "Qwen/Qwen3-8B"
    
    # Use trained Qwen3-8B reward model if path provided, otherwise use base model
    if reward_model_path:
        reward_model_name = reward_model_path
        print(f"Using trained reward model from: {reward_model_name}")
    else:
        reward_model_name = "models/qwen3_8b_reward_model"  # Default path for our trained model
        print(f"Using default trained reward model path: {reward_model_name}")
    
    print(f"Loading tokenizer from {policy_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        policy_model_name,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    print(f"Loading reward model from {reward_model_name}")
    # Load as sequence classification model since it was trained for reward modeling
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype="auto"
    )
    
    print(f"Loading reference policy from {policy_model_name}")
    ref_policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype="auto"
    )
    
    print(f"Loading policy model from {policy_model_name}")
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype="auto"
    )
    
    return tokenizer, reward_model, ref_policy, policy


def main():
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))  # type: ignore
    parser.add_argument("--reward_model_path", type=str, default=None,
                       help="Path to trained reward model (default: models/qwen3_8b_reward_model)")
    script_args, training_args, model_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # Parse the additional argument
    import argparse
    additional_parser = argparse.ArgumentParser()
    additional_parser.add_argument("--reward_model_path", type=str, default=None)
    additional_args = additional_parser.parse_args(remaining_args)
    reward_model_path = additional_args.reward_model_path
    
    # Remove output directory if it exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    print("="*60)
    print("RLOO Training with HelpSteer3 Dataset")
    print("="*60)
    print(f"Policy Model: Qwen/Qwen3-8B")
    print(f"Reward Model: {reward_model_path or 'models/qwen3_8b_reward_model'} (Qwen/Qwen3-8B trained)")
    print(f"Dataset: nvidia/HelpSteer3 (English only)")
    print(f"Output Directory: {training_args.output_dir}")
    print("="*60)
    
    ################
    # Models & Tokenizer
    ################
    tokenizer, reward_model, ref_policy, policy = setup_models_and_tokenizer(
        model_args, training_args, reward_model_path
    )
    
    ################
    # Dataset Loading and Preparation
    ################
    print("Loading HelpSteer3 dataset (preference format, train split)...")
    print("Note: HelpSteer3 is already in preference format with response1/response2 pairs")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Filter for English language samples only
    dataset = filter_english_samples(dataset)
    
    # Split into train and eval
    eval_samples = min(1000, len(dataset) // 10)  # 10% for eval, max 1000 samples
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # Prepare datasets with tokenization
    with PartialState().local_main_process_first():
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