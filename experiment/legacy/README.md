# RLOO Training with Custom Qwen/Qwen3-8B Reward Model

This directory contains scripts for training a reward model using HuggingFace TRL's RewardTrainer and then using it for RLOO (REINFORCE Leave-One-Out) training.

## Overview

The training process consists of two main steps:

1. **Reward Model Training**: Train Qwen/Qwen3-8B as a reward model using the HelpSteer3 dataset
2. **RLOO Training**: Use the trained reward model for RLOO training of the policy model

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install transformers datasets trl peft accelerate torch
```

## Step 1: Train the Reward Model

First, train Qwen/Qwen3-8B as a reward model using the `reward_model_training.py` script:

### Basic Usage

```bash
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
```

### Key Arguments

- `--output_dir`: Directory to save the trained reward model
- `--use_peft`: Enable LoRA training for memory efficiency
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--max_length`: Maximum sequence length (default: 1024)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-5)

### Dataset Processing

The script automatically:
- Downloads the nvidia/HelpSteer3 dataset
- Filters for English-language samples only
- Converts preference data to chosen/rejected pairs
- Skips neutral preferences for clearer training signal

### Expected Output

After training, you'll have a reward model saved in the specified output directory that can score text completions.

## Step 2: RLOO Training with Custom Reward Model

Use the trained reward model for RLOO training with the `rloo_helpsteer_training.py` script:

### Basic Usage

```bash
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
```

### Key Arguments

- `--reward_model_path`: Path to your trained reward model (default: models/qwen3_8b_reward_model)
- `--output_dir`: Directory to save the RLOO-trained policy model
- `--total_episodes`: Total number of training episodes
- `--rloo_k`: Number of samples for RLOO

## Complete Training Pipeline

### Option 1: Use the Pipeline Script (Recommended)

The easiest way to run the complete pipeline is using the `train_pipeline.py` script:

```bash
# Run the complete pipeline with default settings
python train_pipeline.py

# Run with custom settings
python train_pipeline.py \
    --reward_batch_size 4 \
    --reward_epochs 3 \
    --rloo_batch_size 2 \
    --total_episodes 50000 \
    --use_peft

# Train only the reward model
python train_pipeline.py --reward_model_only

# Run only RLOO training (if reward model already exists)
python train_pipeline.py --rloo_only --reward_model_dir models/qwen3_8b_reward_model
```

### Option 2: Manual Step-by-Step

Here's the complete command sequence to train both models manually:

```bash
# Step 1: Train the reward model
python reward_model_training.py \
    --output_dir models/qwen3_8b_reward_model \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32

# Step 2: Use the reward model for RLOO training
python rloo_helpsteer_training.py \
    --output_dir models/rloo_helpsteer3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --total_episodes 50000 \
    --rloo_k 4 \
    --reward_model_path models/qwen3_8b_reward_model
```

## Key Improvements

### Advantages of Using Qwen/Qwen3-8B as Reward Model

1. **Consistency**: Same base architecture for both policy and reward models
2. **Efficiency**: Can use PEFT/LoRA for memory-efficient training
3. **Customization**: Trained specifically on your preference data
4. **Flexibility**: Easy to modify and retrain as needed

### Dataset Processing Improvements

1. **Automatic Filtering**: Only English samples are used
2. **Preference Conversion**: HelpSteer3 preferences converted to chosen/rejected pairs
3. **Quality Control**: Neutral preferences skipped for clearer training signal

## Memory Requirements

### Reward Model Training
- With PEFT/LoRA: ~16-24 GB GPU memory
- Without PEFT: ~40-60 GB GPU memory

### RLOO Training
- Depends on batch size and sequence length
- Recommended: 24-40 GB GPU memory

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Increase batch size or use multiple GPUs
3. **Poor Reward Signal**: Check reward model predictions on sample data

### Verification

After reward model training, the script shows sample predictions to verify the model is working correctly. The reward model should assign higher scores to chosen responses than rejected responses.

## File Structure

```
experiment/
├── reward_model_training.py     # Reward model training script
├── rloo_helpsteer_training.py   # RLOO training script (updated)
├── train_pipeline.py            # Complete pipeline script
├── README.md                    # This file
└── models/                      # Directory for saved models
    ├── qwen3_8b_reward_model/   # Trained reward model
    └── rloo_helpsteer3/         # RLOO-trained policy model
```

## Next Steps

After training, you can:

1. **Evaluate the Models**: Test on held-out preference data
2. **Fine-tune Further**: Continue training with different hyperparameters
3. **Deploy**: Use the trained models for inference
4. **Iterate**: Retrain with different preference data or architectures

## References

- [HuggingFace TRL RewardTrainer Documentation](https://huggingface.co/docs/trl/main/en/reward_trainer)
- [nvidia/HelpSteer3 Dataset](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [Qwen/Qwen3-8B Model](https://huggingface.co/Qwen/Qwen3-8B) 