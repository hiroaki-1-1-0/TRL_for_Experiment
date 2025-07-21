#!/bin/bash

# Stable Single-GPU Training Launch Script for Qwen3-4B Reward Model + RLOO
# Simplified approach to avoid distributed training complexities

# ========================================
# Configuration
# ========================================
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

# Memory and Performance Optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# GPU Memory and Performance
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

REWARD_MODEL_DIR="experiment/models/qwen3_4b_reward_model"
RLOO_MODEL_DIR="experiment/models/qwen3_4b_rloo_model"
LOG_DIR="./logs"

# ========================================
# Functions
# ========================================
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining processes
    pkill -f "reward_model_training_multi_gpu_fixed.py" 2>/dev/null || true
    pkill -f "rloo_helpsteer_training_single_gpu.py" 2>/dev/null || true
    pkill -f "tensorboard" 2>/dev/null || true
    
    echo "Cleanup completed."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

show_gpu_status() {
    echo "GPU Status (Single GPU):"
    nvidia-smi -i 0
}

start_tensorboard() {
    echo "Starting TensorBoard..."
    # Check if tensorboard is available
    if command -v tensorboard &> /dev/null; then
        nohup tensorboard --logdir=$LOG_DIR --host=0.0.0.0 --port=6006 > /dev/null 2>&1 &
        echo "TensorBoard started at http://localhost:6006"
    else
        echo "TensorBoard not available, skipping..."
    fi
}

train_reward_model() {
    echo "Starting Reward Model Training (Single GPU)..."
    
    # Optimized batch size for single GPU
    local per_device_batch_size=6   # Optimized for 48GB memory
    local gradient_accumulation_steps=4  # Maintain reasonable effective batch size
    local effective_batch_size=$((per_device_batch_size * gradient_accumulation_steps))
    echo "Effective batch size: $effective_batch_size (1 GPU × $per_device_batch_size × $gradient_accumulation_steps)"
    
    python experiment/reward_model_training_multi_gpu_fixed.py \
        --model_name "Qwen/Qwen3-4B" \
        --output_dir "$REWARD_MODEL_DIR" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $per_device_batch_size \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --eval_steps 200 \
        --save_steps 400 \
        --logging_steps 10 \
        --learning_rate 1e-5 \
        --weight_decay 0.01 \
        --warmup_ratio 0.1 \
        --max_length 1024 \
        --use_peft \
        --lora_r 64 \
        --bf16 \
        --tf32 \
        --report_to "none" \
        --run_name "qwen3_4b_reward_model_single_gpu_$(date +%Y%m%d_%H%M%S)"
}

train_rloo() {
    echo "Starting Single-GPU RLOO Training..."
    
    # Single-GPU RLOO training parameters
    local per_device_batch_size=2    # Conservative for stability
    local gradient_accumulation_steps=8   # Maintain effective batch size
    local rollout_batch_size=2      # Conservative for memory
    local rloo_k=2                  # Conservative for stability
    
    echo "Single-GPU RLOO Parameters:"
    echo "  Batch size per device: $per_device_batch_size"
    echo "  Gradient accumulation: $gradient_accumulation_steps"
    echo "  Rollout batch size: $rollout_batch_size"
    echo "  RLOO K: $rloo_k"
    
    python experiment/rloo_helpsteer_training_single_gpu.py \
        --reward_model_path "$REWARD_MODEL_DIR" \
        --output_dir "$RLOO_MODEL_DIR" \
        --per_device_train_batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --local_rollout_forward_batch_size $rollout_batch_size \
        --rloo_k $rloo_k \
        --learning_rate 1e-6 \
        --total_episodes 1000 \
        --num_ppo_epochs 1 \
        --num_mini_batches 1 \
        --missing_eos_penalty 1.0 \
        --warmup_steps 10 \
        --logging_steps 5 \
        --save_steps 100 \
        --eval_steps 50 \
        --bf16 \
        --tf32 \
        --gradient_checkpointing \
        --dataloader_num_workers 1 \
        --remove_unused_columns false \
        --report_to "none" \
        --run_name "qwen3_4b_rloo_single_gpu_$(date +%Y%m%d_%H%M%S)" \
        --trust_remote_code
}

# ========================================
# Main Execution
# ========================================
echo "========================================"
echo "Single-GPU Training Setup (Stable Mode)"
echo "========================================"
echo "GPU: $CUDA_VISIBLE_DEVICES (Single GPU)"
echo "Memory per GPU: 48GB"
echo "Model: Qwen3-4B (simplified for stability)"
echo "TensorBoard logs: $LOG_DIR"
echo "Note: Using single GPU to avoid distributed training issues"
echo "========================================"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$REWARD_MODEL_DIR")"
mkdir -p "$(dirname "$RLOO_MODEL_DIR")"

# Show system status
show_gpu_status
echo "========================================"

# Parse command line arguments
case "${1:-both}" in
    "reward")
        start_tensorboard
        train_reward_model
        ;;
    "rloo")
        if [ ! -d "$REWARD_MODEL_DIR" ]; then
            echo "Error: Reward model not found at $REWARD_MODEL_DIR"
            echo "Please train the reward model first with: $0 reward"
            exit 1
        fi
        start_tensorboard
        train_rloo
        ;;
    "both")
        echo "Running complete training pipeline (Single GPU)..."
        start_tensorboard
        
        # Train reward model first
        train_reward_model
        reward_exit_code=$?
        
        if [ $reward_exit_code -eq 0 ]; then
            echo "Reward model training completed successfully!"
            echo "Starting RLOO training..."
            train_rloo
        else
            echo "Reward model training failed with exit code: $reward_exit_code"
            exit $reward_exit_code
        fi
        ;;
    *)
        echo "Usage: $0 [reward|rloo|both]"
        echo "  reward - Train only the reward model"
        echo "  rloo   - Train only the RLOO model (requires existing reward model)"
        echo "  both   - Train both models sequentially (default)"
        echo ""
        echo "This is the stable script using single GPU"
        echo "to avoid distributed training complexities."
        exit 1
        ;;
esac

echo "========================================"
echo "Training completed!"
echo "========================================" 