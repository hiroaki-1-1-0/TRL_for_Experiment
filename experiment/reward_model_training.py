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
Reward Model Training Script using HelpSteer3 Dataset

This script trains a reward model using:
- Dataset: nvidia/HelpSteer3 (preference subset, train split, English language only)
- Base Model: Qwen/Qwen3-8B 
- Trainer: RewardTrainer from HuggingFace TRL

The script converts HelpSteer3 preference data into the format expected by RewardTrainer
(chosen/rejected pairs) and trains a reward model that can be used for RLOO training.

Usage:
python reward_model_training.py \
    --output_dir models/qwen3_8b_reward_model \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --max_length 1024 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32
"""

import shutil
import argparse
from typing import Dict, Any, Union, List
import torch

from accelerate import PartialState
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.hf_argparser import HfArgumentParser
from peft import LoraConfig, TaskType

from trl import ModelConfig, RewardConfig, RewardTrainer, ScriptArguments
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


def filter_english_samples(dataset: Union[Dataset, Any]) -> Dataset:
    """
    Filter the HelpSteer3 dataset to include only samples with language='english'.
    """
    def is_english_sample(example):
        return example.get('language', '').lower() == 'english'
    
    print(f"Original dataset size: {len(dataset)}")
    english_dataset = dataset.filter(is_english_sample)
    print(f"After English filtering: {len(english_dataset)} samples")
    print(f"Filtered out: {len(dataset) - len(english_dataset)} non-English samples")
    
    return english_dataset


def prepare_helpsteer_for_reward_training(dataset: Dataset, tokenizer) -> Dataset:
    """
    Prepare HelpSteer3 dataset for RewardTrainer.
    
    RewardTrainer expects a dataset with 'chosen' and 'rejected' columns.
    HelpSteer3 has response1, response2, and overall_preference.
    
    overall_preference ranges from -2 to 2:
    - Positive values mean response1 is preferred
    - Negative values mean response2 is preferred
    - 0 means neutral (we'll skip these for clearer training signal)
    """
    
    def convert_to_chosen_rejected(examples):
        chosen_texts = []
        rejected_texts = []
        
        for i in range(len(examples['context'])):
            context = examples['context'][i]
            response1 = examples['response1'][i]
            response2 = examples['response2'][i]
            preference = examples['overall_preference'][i]
            
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
                        conversation += f"User: {content}\n"
                    elif role == "assistant":
                        conversation += f"Assistant: {content}\n"
                
                # Add the responses
                full_text1 = conversation + f"Assistant: {response1}"
                full_text2 = conversation + f"Assistant: {response2}"
            else:
                # Fallback for unexpected format
                prompt = str(context)
                full_text1 = f"User: {prompt}\nAssistant: {response1}"
                full_text2 = f"User: {prompt}\nAssistant: {response2}"
            
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
    reward_dataset = dataset.map(
        convert_to_chosen_rejected,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    
    # Filter out empty results (from neutral preferences)
    reward_dataset = reward_dataset.filter(lambda x: len(x['chosen']) > 0)
    
    print(f"Converted dataset size: {len(reward_dataset)} preference pairs")
    
    return reward_dataset


def setup_reward_model_and_tokenizer(model_config: ModelConfig, training_args: RewardConfig):
    """Setup reward model and tokenizer for training"""
    
    model_name = "Qwen/Qwen3-8B"
    
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",  # RewardTrainer typically uses right padding
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template if not present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    print(f"Loading reward model from {model_name}")
    # Load as sequence classification model for reward modeling
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Reward models output a single score
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype="auto"
    )
    
    return tokenizer, model


def setup_peft_config(args):
    """Setup PEFT configuration for LoRA training"""
    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification for reward modeling
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["score"],  # Save the classification head
        )
        return peft_config
    return None


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Qwen/Qwen3-8B as reward model")
    
    # Model and training arguments
    parser.add_argument("--output_dir", type=str, default="models/qwen3_8b_reward_model",
                        help="Output directory for trained model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device during evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of updates steps to accumulate gradients")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Run evaluation every X steps")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    
    # PEFT arguments
    parser.add_argument("--use_peft", action="store_true",
                        help="Use PEFT (LoRA) for efficient training")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")
    
    # Data arguments
    parser.add_argument("--dataset_num_proc", type=int, default=4,
                        help="Number of processes for dataset processing")
    parser.add_argument("--eval_size", type=int, default=1000,
                        help="Size of evaluation set")
    
    args = parser.parse_args()
    
    # Create model and training configs
    model_config = ModelConfig(trust_remote_code=True)
    
    training_args = RewardConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # Change to "wandb" if you want to use wandb
        remove_unused_columns=False,
        push_to_hub=args.push_to_hub,
        bf16=True,  # Use bfloat16 for better performance
        dataloader_num_workers=4,
        dataset_num_proc=args.dataset_num_proc,
    )
    
    # Remove output directory if it exists
    if training_args.output_dir:
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    
    print("="*60)
    print("Reward Model Training with HelpSteer3 Dataset")
    print("="*60)
    print(f"Base Model: Qwen/Qwen3-8B")
    print(f"Dataset: nvidia/HelpSteer3 (English only)")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Use PEFT: {args.use_peft}")
    print("="*60)
    
    ################
    # Model & Tokenizer Setup
    ################
    tokenizer, model = setup_reward_model_and_tokenizer(model_config, training_args)
    
    ################
    # PEFT Configuration
    ################
    peft_config = setup_peft_config(args)
    
    ################
    # Dataset Loading and Preparation
    ################
    print("Loading HelpSteer3 dataset...")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    # Filter for English language samples only
    dataset = filter_english_samples(dataset)
    
    # Prepare dataset for reward training
    with PartialState().local_main_process_first():
        print("Converting to chosen/rejected format...")
        reward_dataset = prepare_helpsteer_for_reward_training(dataset, tokenizer)
    
    # Split into train and eval
    eval_samples = min(args.eval_size, len(reward_dataset) // 10)
    train_dataset = reward_dataset.select(range(len(reward_dataset) - eval_samples))
    eval_dataset = reward_dataset.select(range(len(reward_dataset) - eval_samples, len(reward_dataset)))
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    ################
    # Training
    ################
    print("Initializing RewardTrainer...")
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config if peft_config is not None else None,
    )
    
    print("Starting reward model training...")
    trainer.train()
    
    ################
    # Save Model
    ################
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        print("Pushing model to hub...")
        trainer.push_to_hub(dataset_name="nvidia/HelpSteer3")
    
    print("Reward model training completed successfully!")
    
    # Show some sample predictions to verify the model
    print("\nGenerating sample predictions...")
    if hasattr(trainer, 'model') and trainer.model is not None:
        trainer.model.eval()
        sample_data = eval_dataset.select(range(min(3, len(eval_dataset))))
        
        for i, example in enumerate(sample_data):
            print(f"\nSample {i+1}:")
            chosen_text = example['chosen']
            rejected_text = example['rejected']
            print(f"Chosen: {chosen_text[:200]}...")
            print(f"Rejected: {rejected_text[:200]}...")
            
            # Tokenize and get predictions
            chosen_inputs = tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=args.max_length)
            rejected_inputs = tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=args.max_length)
            
            device = next(trainer.model.parameters()).device
            chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
            rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
            
            with torch.no_grad():
                chosen_score = trainer.model(**chosen_inputs).logits.item()
                rejected_score = trainer.model(**rejected_inputs).logits.item()
                
                print(f"Chosen score: {chosen_score:.4f}")
                print(f"Rejected score: {rejected_score:.4f}")
                print(f"Preference correct: {chosen_score > rejected_score}")
    else:
        print("Model not available for sample predictions.")


if __name__ == "__main__":
    main() 