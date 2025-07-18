#!/bin/bash

# Fallback Multi-GPU Training Launch Script for Qwen3-8B Reward Model + RLOO
# Uses 4 GPUs instead of 7 to work around shared memory limitations

# ========================================
# Configuration - Reduced GPU count
# ========================================
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use only 4 GPUs
export OMP_NUM_THREADS=1

# NCCL Configuration - Optimized for container environments
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_SHM_DISABLE=1  # Disable shared memory to avoid /dev/shm issues
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1

# Additional NCCL optimizations
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0

# Memory and Performance Optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# PyTorch Distributed Settings
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # Different port to avoid conflicts

# GPU Memory and Performance
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

NUM_GPUS=4  # Reduced from 7 to 4
REWARD_MODEL_DIR="models/qwen3_8b_reward_model"
RLOO_MODEL_DIR="models/qwen3_8b_rloo_model"
LOG_DIR="./logs"

# ========================================
# Functions
# ========================================
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining processes
    pkill -f "reward_model_training_multi_gpu_fixed.py" 2>/dev/null || true
    pkill -f "rloo_helpsteer_training.py" 2>/dev/null || true
    pkill -f "tensorboard" 2>/dev/null || true
    
    # Clean up NCCL shared memory segments if any exist
    rm -f /dev/shm/nccl-* 2>/dev/null || true
    
    echo "Cleanup completed."
}

# Set up signal handlers
trap cleanup EXIT INT TERM

clear_shm() {
    echo "Clearing shared memory..."
    # Clean up any existing NCCL shared memory
    rm -f /dev/shm/nccl-* 2>/dev/null || true
    rm -f /dev/shm/sem.* 2>/dev/null || true
    echo "Shared memory cleared."
}

show_gpu_status() {
    echo "GPU Status (Using GPUs 0-3 only):"
    nvidia-smi -i 0,1,2,3
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
    echo "Starting Reward Model Training (4 GPUs)..."
    clear_shm  # Clear shared memory before training
    
    # Adjust batch size for memory constraints - optimized for 1024 sequence length
    local per_device_batch_size=4   # Optimized for 48GB memory with 1024 max_length
    local gradient_accumulation_steps=4  # Adjusted to maintain reasonable effective batch size
    local effective_batch_size=$((NUM_GPUS * per_device_batch_size * gradient_accumulation_steps))
    echo "Effective batch size: $effective_batch_size (4 GPUs × $per_device_batch_size × $gradient_accumulation_steps)"
    
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        --max_restarts=0 \
        experiment/reward_model_training_multi_gpu_fixed.py \
        --model_name "Qwen/Qwen3-8B" \
        --output_dir "$REWARD_MODEL_DIR" \
        --num_train_epochs 1 \
        --per_device_train_batch_size $per_device_batch_size \
        --per_device_eval_batch_size 2 \
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
        --run_name "qwen3_8b_reward_model_4gpu_$(date +%Y%m%d_%H%M%S)"
}

train_rloo() {
    echo "Starting RLOO Training..."
    clear_shm  # Clear shared memory before training
    
    python experiment/rloo_helpsteer_training.py \
        --reward_model_path "$REWARD_MODEL_DIR" \
        --output_dir "$RLOO_MODEL_DIR" \
        --bf16 \
        --tf32 \
        --report_to "tensorboard" \
        --run_name "qwen3_8b_rloo_4gpu_$(date +%Y%m%d_%H%M%S)"
}

# ========================================
# Main Execution
# ========================================
echo "========================================"
echo "Multi-GPU Training Setup (Fallback Mode)"
echo "========================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES (4 GPUs only)"
echo "Number of GPUs: $NUM_GPUS"
echo "Memory per GPU: 48GB"
echo "TensorBoard logs: $LOG_DIR"
echo "Note: Using 4 GPUs instead of 7 to avoid shared memory issues"
echo "========================================"

# Initial cleanup
clear_shm

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
        echo "Running complete training pipeline (4 GPUs)..."
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
        echo "This is the fallback script using 4 GPUs instead of 7"
        echo "to work around shared memory limitations."
        exit 1
        ;;
esac

echo "========================================"
echo "Training completed!"
echo "========================================" 