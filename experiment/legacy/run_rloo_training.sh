#!/bin/bash

# RLOO Training Script for HelpSteer3 Dataset
# This script runs RLOO training using HelpSteer3 dataset with optimal parameters

set -e  # Exit on any error

echo "=========================================="
echo "RLOO Training with HelpSteer3 Dataset"
echo "=========================================="

# Check if we're in the correct directory
if [ ! -f "rloo_helpsteer_training.py" ]; then
    echo "Error: rloo_helpsteer_training.py not found in current directory"
    exit 1
fi

# Set default values
OUTPUT_DIR="experiment/models/rloo_helpsteer3"
BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=1e-6
TOTAL_EPISODES=50000
RLOO_K=4
EPOCHS=1
MINI_BATCHES=1

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "Warning: No GPU detected. Training will be very slow on CPU."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training Parameters:"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Total Episodes: $TOTAL_EPISODES"
echo "  RLOO K: $RLOO_K"
echo "  PPO Epochs: $EPOCHS"
echo "  Mini Batches: $MINI_BATCHES"
echo "=========================================="

# For single GPU training
if [ "$1" == "single" ] || [ "$#" -eq 0 ]; then
    echo "Starting single GPU training..."
    python rloo_helpsteer_training.py \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --total_episodes "$TOTAL_EPISODES" \
        --num_ppo_epochs "$EPOCHS" \
        --num_mini_batches "$MINI_BATCHES" \
        --rloo_k "$RLOO_K" \
        --local_rollout_forward_batch_size "$BATCH_SIZE" \
        --missing_eos_penalty 1.0 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 1000 \
        --eval_steps 500 \
        --bf16=true \
        --gradient_checkpointing \
        --dataloader_num_workers 4 \
        --remove_unused_columns false \
        --report_to "tensorboard" \
        --run_name "rloo_helpsteer3_qwen_nemotron" \
        --trust_remote_code true

# For multi-GPU training with accelerate
elif [ "$1" == "multi" ]; then
    echo "Starting multi-GPU training with accelerate..."
    
    # Check if accelerate config exists
    if [ ! -f "accelerate_config.yaml" ]; then
        echo "Creating default accelerate config..."
        cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: auto
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    fi
    
    accelerate launch --config_file accelerate_config.yaml \
        rloo_helpsteer_training.py \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --total_episodes "$TOTAL_EPISODES" \
        --num_ppo_epochs "$EPOCHS" \
        --num_mini_batches "$MINI_BATCHES" \
        --rloo_k "$RLOO_K" \
        --local_rollout_forward_batch_size "$BATCH_SIZE" \
        --missing_eos_penalty 1.0 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 1000 \
        --eval_steps 500 \
        --bf16 true \
        --gradient_checkpointing \
        --dataloader_num_workers 4 \
        --remove_unused_columns false \
        --report_to "tensorboard" \
        --run_name "rloo_helpsteer3_qwen_nemotron" \
        --trust_remote_code true

# For DeepSpeed training (Zero3)
elif [ "$1" == "deepspeed" ]; then
    echo "Starting DeepSpeed ZeRO-3 training..."
    
    # Check if DeepSpeed config exists
    if [ ! -f "deepspeed_zero3.json" ]; then
        echo "Creating DeepSpeed ZeRO-3 config..."
        cat > deepspeed_zero3.json << EOF
{
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": "auto",
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
EOF
    fi
    
    deepspeed --include localhost:0,1,2,3 \
        rloo_helpsteer_training.py \
        --output_dir "$OUTPUT_DIR" \
        --per_device_train_batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        --learning_rate "$LEARNING_RATE" \
        --total_episodes "$TOTAL_EPISODES" \
        --num_ppo_epochs "$EPOCHS" \
        --num_mini_batches "$MINI_BATCHES" \
        --rloo_k "$RLOO_K" \
        --local_rollout_forward_batch_size "$BATCH_SIZE" \
        --missing_eos_penalty 1.0 \
        --warmup_steps 100 \
        --logging_steps 10 \
        --save_steps 1000 \
        --eval_steps 500 \
        --bf16 true \
        --gradient_checkpointing \
        --dataloader_num_workers 4 \
        --remove_unused_columns false \
        --report_to "tensorboard" \
        --run_name "rloo_helpsteer3_qwen_nemotron" \
        --trust_remote_code true \
        --deepspeed deepspeed_zero3.json

else
    echo "Usage: $0 [single|multi|deepspeed]"
    echo "  single    - Single GPU training (default)"
    echo "  multi     - Multi-GPU training with accelerate"
    echo "  deepspeed - DeepSpeed ZeRO-3 training"
    exit 1
fi

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "View logs with: tensorboard --logdir $OUTPUT_DIR/runs"
echo "==========================================" 