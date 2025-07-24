#!/bin/bash

# Multi-GPU RLOO vs Baseline Model Comparison Script
# This script uses multiple GPUs for faster evaluation

echo "Multi-GPU RLOO vs Baseline Model Comparison"
echo "============================================"

# Check if model directory exists
MODEL_DIR="experiment/models/qwen3_1.7b_rloo_model"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Error: Model directory '$MODEL_DIR' not found!"
    echo "Please run rloo_helpsteer_training_multi_gpu.py first to train the model."
    exit 1
fi

echo "✅ Model directory found: $MODEL_DIR"

# Check available GPUs
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --list-gpus
    echo ""
else
    echo "⚠️  nvidia-smi not found. Cannot check GPU status."
fi

# Set CUDA visible devices to use 7 GPUs and configure memory settings
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
export CUDA_LAUNCH_BLOCKING=1  # Better error reporting
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Better distributed debugging
echo "🔧 Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "🔧 CUDA debugging enabled for better error reporting"
echo ""

# Function to run evaluation with multi-GPU and better error handling
run_multi_gpu_eval() {
    local description="$1"
    local extra_args="$2"
    
    echo "Running: $description"
    echo "Command: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python experiment/eval.py $extra_args"
    echo ""
    
    # Run with error checking
    if python experiment/eval.py $extra_args; then
        echo "✅ Evaluation completed successfully"
    else
        echo "❌ Evaluation failed with exit code $?"
        echo "This might be due to DDP model loading issues."
        echo "The improved eval.py should handle DDP models better now."
    fi
    echo ""
    echo "----------------------------------------"
    echo ""
}

# Quick test with 20 examples (higher batch size for multi-GPU)
echo "Choose evaluation mode:"
echo "1. Manual comparison only (no judge evaluation)"
echo "2. With lightweight local judge (microsoft/DialoGPT-medium - free)"
echo "3. With good quality local judge (Qwen/Qwen3-1.7B - free)"
echo "4. With high quality local judge (Qwen/Qwen3-30B-A3B - may require API payment)"
echo "5. With OpenAI GPT judge (requires API key)"
echo ""
read -p "Enter choice (1-5): " judge_choice

# Set judge arguments based on choice
case $judge_choice in
    1)
        judge_args="--skip_judge"
        judge_desc="manual comparison only"
        ;;
    2)
        judge_args="--judge_model microsoft/DialoGPT-medium"
        judge_desc="lightweight local judge (DialoGPT-medium)"
        ;;
    3)
        judge_args="--judge_model Qwen/Qwen3-1.7B"
        judge_desc="good quality local judge (Qwen3-1.7B)"
        ;;
    4)
        judge_args="--judge_model Qwen/Qwen3-30B-A3B"
        judge_desc="high quality judge (Qwen3-30B-A3B)"
        ;;
    5)
        judge_args="--judge_model gpt-3.5-turbo"
        judge_desc="OpenAI GPT judge"
        ;;
    *)
        echo "Invalid choice, using lightweight local judge"
        judge_args="--judge_model microsoft/DialoGPT-medium"
        judge_desc="lightweight local judge (DialoGPT-medium)"
        ;;
esac

echo "Selected: $judge_desc"
echo ""

# Quick test with 20 examples (higher batch size for multi-GPU)
run_multi_gpu_eval "Quick multi-GPU test (20 examples, $judge_desc)" "--num_examples 20 --batch_size 8 $judge_args"

# Medium evaluation with 100 examples
run_multi_gpu_eval "Medium multi-GPU comparison (100 examples, $judge_desc)" "--num_examples 100 --batch_size 12 $judge_args"

# Standard evaluation with 500 examples
run_multi_gpu_eval "Standard multi-GPU comparison (500 examples, $judge_desc)" "--num_examples 500 --batch_size 16 $judge_args"

# Full evaluation with all examples
echo "Do you want to run full evaluation with ALL examples? This may take a long time."
echo "Press Enter to continue or Ctrl+C to cancel..."
read -r

run_multi_gpu_eval "Full multi-GPU comparison (all examples, $judge_desc)" "--batch_size 20 $judge_args"

echo "Multi-GPU evaluation completed!"
echo ""
echo "Summary:"
echo "- RLOO Model: experiment/models/qwen3_1.7b_rloo_model"
echo "- Baseline Model: Qwen/Qwen3-1.7B"
echo "- GPUs used: $CUDA_VISIBLE_DEVICES"
echo "- Multi-GPU evaluation provides faster processing"
echo "- Compare the win rates to see the improvement from RLOO training."
echo "- A win rate > 50% means RLOO model outperforms the baseline."
echo ""
echo "💡 Judge Model Recommendations:"
echo "   - Use microsoft/DialoGPT-medium for quick, free local evaluation"
echo "   - Use Qwen/Qwen3-1.7B for better quality, still free local evaluation"
echo "   - Avoid large models like Qwen3-30B-A3B to prevent API charges"
