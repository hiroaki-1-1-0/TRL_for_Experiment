#!/bin/bash

# Fallback Multi-GPU Training Launch Script for Qwen3-1.7B Reward Model + RLOO
# Uses adaptive GPU allocation: 4 GPUs for Reward Model, 7 GPUs for RLOO

# ========================================
# Configuration - Different GPU settings for different models
# ========================================
# GPU configuration will be set dynamically based on training type
export OMP_NUM_THREADS=1

# NCCL Configuration - Optimized for container environments with shared memory issues
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_SHM_DISABLE=1  # Disable shared memory to avoid /dev/shm issues
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1

# Additional NCCL optimizations - Force network-based communication
export NCCL_TREE_THRESHOLD=0
export NCCL_ALGO=Tree
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=2
export NCCL_MIN_NCHANNELS=2

# Memory and Performance Optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,roundup_power2_divisions:16
export TOKENIZERS_PARALLELISM=false
export CUDA_MODULE_LOADING=LAZY
export TORCH_CUDNN_V8_API_ENABLED=1

# PyTorch Distributed Settings
export MASTER_ADDR=localhost
export MASTER_PORT=29501  # Different port to avoid conflicts

# GPU Memory and Performance
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPU configurations for different training types
REWARD_MODEL_GPUS="0,1,2,3"        # 4 GPUs for Reward Modeling
RLOO_GPUS="0,1,2,3,4,5,6"         # 7 GPUs for RLOO
REWARD_MODEL_NUM_GPUS=4
RLOO_NUM_GPUS=7
REWARD_MODEL_DIR="experiment/models/qwen3_1.7b_reward_model"
RLOO_MODEL_DIR="experiment/models/qwen3_1.7b_rloo_model"
LOG_DIR="./logs"

# ========================================
# Functions
# ========================================
cleanup() {
    echo "Cleaning up..."
    # Kill any remaining processes
    pkill -f "reward_model_training_multi_gpu.py" 2>/dev/null || true
    pkill -f "rloo_helpsteer_training_multi_gpu.py" 2>/dev/null || true
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
    
    # Check /dev/shm space
    echo "Checking /dev/shm space:"
    df -h /dev/shm
    
    # Free up additional space if needed
    find /dev/shm -type f -name "*" -mtime +1 -delete 2>/dev/null || true
    
    echo "Shared memory cleared."
}

show_gpu_status() {
    local gpus=$1
    echo "GPU Status (Using GPUs $gpus):"
    nvidia-smi -i $gpus
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
    echo "Starting Reward Model Training (4 GPUs: $REWARD_MODEL_GPUS)..."
    
    # Set GPU configuration for reward model training
    export CUDA_VISIBLE_DEVICES=$REWARD_MODEL_GPUS
    
    clear_shm  # Clear shared memory before training
    
    # Adjust batch size for memory constraints - reduced for shared memory issues
    local per_device_batch_size=8   # Reduced from 16 to minimize memory pressure
    local gradient_accumulation_steps=4  # Increased to maintain effective batch size
    local effective_batch_size=$((REWARD_MODEL_NUM_GPUS * per_device_batch_size * gradient_accumulation_steps))
    echo "Effective batch size: $effective_batch_size ($REWARD_MODEL_NUM_GPUS GPUs × $per_device_batch_size × $gradient_accumulation_steps)"
    
    torchrun \
        --standalone \
        --nproc_per_node=$REWARD_MODEL_NUM_GPUS \
        --max_restarts=0 \
        experiment/reward_model_training_multi_gpu.py \
        --model_name "Qwen/Qwen3-1.7B" \
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
        --run_name "qwen3_1.7b_reward_model_4gpu_$(date +%Y%m%d_%H%M%S)"
}

train_rloo() {
    echo "Starting Multi-GPU Distributed RLOO Training (7 GPUs: $RLOO_GPUS)..."
    
    # Set GPU configuration for RLOO training
    export CUDA_VISIBLE_DEVICES=$RLOO_GPUS
    
    clear_shm  # Clear shared memory before training
    
    # Multi-GPU distributed RLOO training parameters - optimized for 1.7B model using 7 GPUs
    local per_device_batch_size=4    # Increased for 1.7B model (much smaller memory footprint)
    local gradient_accumulation_steps=2   # Reduced since we can use larger batch size
    local rollout_batch_size=4      # Increased for consistency with batch size
    local rloo_k=7                  # Use 7 to match number of GPUs
    
    echo "Multi-GPU RLOO Parameters:"
    echo "  Batch size per device: $per_device_batch_size"
    echo "  Gradient accumulation: $gradient_accumulation_steps"
    echo "  Rollout batch size: $rollout_batch_size"
    echo "  RLOO K: $rloo_k"
    echo "  Using distributed model loading across all 7 GPUs"
    
    torchrun \
        --standalone \
        --nproc_per_node=$RLOO_NUM_GPUS \
        --max_restarts=0 \
        experiment/rloo_helpsteer_training_multi_gpu.py \
        --reward_model_path "$REWARD_MODEL_DIR" \
        --output_dir "$RLOO_MODEL_DIR" \
        --per_device_train_batch_size $per_device_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --local_rollout_forward_batch_size $rollout_batch_size \
        --rloo_k $rloo_k \
        --learning_rate 1e-6 \
        --total_episodes 2000 \
        --num_ppo_epochs 1 \
        --num_mini_batches 1 \
        --missing_eos_penalty 1.0 \
        --warmup_steps 20 \
        --logging_steps 5 \
        --save_steps 200 \
        --eval_steps 100 \
        --bf16 True \
        --tf32 True \
        --gradient_checkpointing \
        --dataloader_num_workers 1 \
        --remove_unused_columns false \
        --report_to "none" \
        --run_name "qwen3_1.7b_rloo_multi_gpu_$(date +%Y%m%d_%H%M%S)" \
        --trust_remote_code
}

# ========================================
# Main Execution
# ========================================
echo "========================================"
echo "Multi-GPU Training Setup (Adaptive GPU Mode)"
echo "========================================"
echo "Reward Model GPUs: $REWARD_MODEL_GPUS ($REWARD_MODEL_NUM_GPUS GPUs)"
echo "RLOO GPUs: $RLOO_GPUS ($RLOO_NUM_GPUS GPUs)"
echo "Memory per GPU: 48GB"
echo "Model: Qwen3-1.7B"
echo "TensorBoard logs: $LOG_DIR"
echo "Note: Using different GPU configurations for optimal performance"
echo "========================================"

# Initial cleanup
clear_shm

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$REWARD_MODEL_DIR")"
mkdir -p "$(dirname "$RLOO_MODEL_DIR")"

# Show system status
show_gpu_status $REWARD_MODEL_GPUS
echo "Available for Reward Model training ^"
echo ""
show_gpu_status $RLOO_GPUS  
echo "Available for RLOO training ^"
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
        echo "Running complete training pipeline (Adaptive GPU allocation)..."
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
        echo "  reward - Train only the reward model (uses GPUs $REWARD_MODEL_GPUS)"
        echo "  rloo   - Train only the RLOO model (uses GPUs $RLOO_GPUS, requires existing reward model)"
        echo "  both   - Train both models sequentially (default)"
        echo ""
        echo "This script uses adaptive GPU allocation:"
        echo "  - Reward Model: $REWARD_MODEL_NUM_GPUS GPUs ($REWARD_MODEL_GPUS)"
        echo "  - RLOO Training: $RLOO_NUM_GPUS GPUs ($RLOO_GPUS)"
        echo "Using Qwen3-1.7B model for optimal performance and memory efficiency."
        exit 1
        ;;
esac

echo "========================================"
echo "Training completed!"
echo "========================================" 