# RLOO Training with HelpSteer3 Dataset

This project implements RLOO (REINFORCE Leave-One-Out) training using the TRL library with the following configuration:

- **Dataset**: nvidia/HelpSteer3 (preference subset, train split, English language only)
- **Policy Model**: Qwen/Qwen3-8B (Now available on Hugging Face)
- **Reward Model**: nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual
- **Training Method**: RLOO (REINFORCE Leave-One-Out)

## Overview

RLOO is a policy optimization technique that generates K completions for each prompt and uses the mean scores from the other K-1 completions as a baseline to calculate the advantage. This approach is more memory-efficient than PPO and has been shown to achieve competitive results in RLHF.

## Files

- `rloo_helpsteer_training.py` - Main training script
- `run_rloo_training.sh` - Shell script for easy execution with different configurations
- `README_RLOO_Training.md` - This documentation file

## Requirements

Make sure you have the following installed:

```bash
pip install torch transformers datasets accelerate deepspeed tensorboard
pip install trl  # Latest version with RLOO support
```

## Dataset Information

**HelpSteer3** is a comprehensive preference dataset from NVIDIA with:
- 40.5k total preference samples in the training set (nearly double HelpSteer2)
- Multilingual support (English, Chinese, Korean + 11 more languages)
- Diverse task domains (General, STEM, Code, Multilingual scenarios)  
- Higher quality annotations with 3-5 annotators per sample
- Preference format with response1/response2 pairs and detailed reasoning
- Permissive CC-BY-4.0 license

**English Language Filtering**: This implementation specifically filters the dataset to use only samples with `language='english'`, ensuring high-quality English language training data.

## Model Information

### Policy Model: Qwen/Qwen3-8B
- Large multilingual language model
- Optimized for instruction following
- Good performance-to-size ratio

### Reward Model: nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual
- Large generative reward model from NVIDIA
- Multilingual support with strong English performance
- Designed for human preference evaluation

## Training Configuration

The training uses the following optimized parameters:

- **Batch Size**: 2 per device
- **Gradient Accumulation**: 8 steps
- **Learning Rate**: 1e-6
- **Total Episodes**: 50,000
- **RLOO K**: 4 (number of completions per prompt)
- **Mixed Precision**: bf16
- **Gradient Checkpointing**: Enabled for memory efficiency

## Usage

### Quick Start

For single GPU training:
```bash
./run_rloo_training.sh
# or
./run_rloo_training.sh single
```

### Multi-GPU Training

For multiple GPUs with Accelerate:
```bash
./run_rloo_training.sh multi
```

### Large Scale Training with DeepSpeed

For DeepSpeed ZeRO-3 (recommended for limited GPU memory):
```bash
./run_rloo_training.sh deepspeed
```

### Manual Execution

You can also run the training script directly:

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
    --bf16 \
    --gradient_checkpointing
```

## Training Process

1. **Dataset Loading**: Downloads HelpSteer3 and filters for English language samples only
2. **Model Setup**: Loads policy model, reference policy, and reward model
3. **Tokenization**: Prepares prompts for training with proper chat formatting
4. **RLOO Training**: Runs the RLOO algorithm with the specified parameters
5. **Model Saving**: Saves the trained model to the output directory

## Monitoring Training

The training logs are saved with TensorBoard. To monitor progress:

```bash
tensorboard --logdir models/rloo_helpsteer3/runs
```

Key metrics to watch:
- `objective/rlhf_reward`: Should increase over time
- `objective/kl`: KL divergence with reference model
- `policy/entropy_avg`: Policy entropy
- `val/ratio`: Policy ratio (should stay around 1.0)

## Memory Requirements

### Approximate GPU Memory Usage:
- **Single GPU**: ~40-50GB (requires A100 or similar)
- **Multi-GPU**: ~20-25GB per GPU (can use multiple A40s/V100s)
- **DeepSpeed ZeRO-3**: ~15-20GB per GPU (most memory efficient)

### Optimizations Included:
- Gradient checkpointing
- Mixed precision (bf16)
- Efficient tokenization
- Optional CPU offloading with DeepSpeed

## Customization

### Adjusting Training Parameters

Edit the variables in `run_rloo_training.sh`:

```bash
BATCH_SIZE=2        # Increase if you have more GPU memory
GRAD_ACCUM=8        # Adjust for effective batch size
LEARNING_RATE=1e-6  # Lower for more stable training
TOTAL_EPISODES=50000 # Increase for longer training
RLOO_K=4           # Number of completions per prompt
```

### Using Different Models

Modify the model names in `rloo_helpsteer_training.py`:

```python
policy_model_name = "your-policy-model"
reward_model_name = "your-reward-model"
```

## Expected Results

After training, you should see:
- Improved response quality on human preference metrics
- Better alignment with human preferences
- Generated samples that are more helpful, correct, and coherent

## Troubleshooting

### Common Issues:

1. **Out of Memory**: 
   - Reduce batch size
   - Use DeepSpeed ZeRO-3
   - Enable gradient checkpointing

2. **Slow Training**:
   - Increase batch size if memory allows
   - Use multiple GPUs
   - Check GPU utilization

3. **Model Loading Errors**:
   - Ensure you have access to the models
   - Check internet connection
   - Verify model names are correct

### Error Messages:

- **"CUDA out of memory"**: Reduce batch size or use DeepSpeed
- **"Model not found"**: Check model names and HuggingFace access
- **"Dataset not found"**: Check internet connection and dataset name

## Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [RLOO Paper](https://huggingface.co/papers/2402.14740)
- [HelpSteer3 Dataset](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-8B)
- [Nemotron Reward Model](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual)

## Citation

If you use this code, please cite:

```bibtex
@misc{helpsteer3_2024,
  title={HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages},
  author={Wang, Zhilin and Zeng, Jiaqi and Delalleau, Olivier and others},
  year={2024},
  journal={arXiv preprint arXiv:2505.11475}
}
``` 