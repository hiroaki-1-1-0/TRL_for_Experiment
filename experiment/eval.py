#!/usr/bin/env python3
"""
Improved RLOO Model Evaluation Script

This script compares RLOO-trained models against baseline models with enhanced:
- Better model loading for distributed training checkpoints
- Improved tokenizer padding settings for decoder-only models
- Robust error handling and fallback options
- Enhanced multi-GPU support
- Optional judge evaluation with fallbacks
"""

from dataclasses import dataclass, field
from typing import Optional
import sys
import gc
import torch
import os
import warnings
import glob
import tempfile
import shutil
from collections import OrderedDict

from datasets import load_dataset
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import safetensors.torch
from safetensors import safe_open
from safetensors.torch import save_file

# Suppress some common warnings
warnings.filterwarnings("ignore", message="The following generation flags are not valid")
warnings.filterwarnings("ignore", message="A decoder-only architecture is being used")

try:
    from trl import HfPairwiseJudge, OpenAIPairwiseJudge
    TRL_AVAILABLE = True
except ImportError:
    print("TRL not available. Judge evaluation will be skipped.")
    TRL_AVAILABLE = False


class LocalPairwiseJudge:
    """
    Local implementation of pairwise judge that loads and runs the model locally
    instead of using HuggingFace Inference API
    """
    def __init__(self, model_name: str, device_map="auto", torch_dtype=torch.bfloat16):
        self.model_name = model_name
        print(f"Loading local judge model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model locally
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            max_memory={i: "15GB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None
        )
        self.model.eval()
        print(f"✅ Local judge model loaded successfully")
    
    def judge(self, prompts, completions):
        """
        Judge pairwise completions locally
        Returns list of indices (0 for first completion, 1 for second completion)
        """
        results = []
        
        for i, (prompt, completion_pair) in enumerate(zip(prompts, completions)):
            if i % 10 == 0:
                print(f"Judging example {i+1}/{len(prompts)}...")
            
            completion_a, completion_b = completion_pair
            
            # Create judge prompt
            judge_prompt = self._create_judge_prompt(prompt, completion_a, completion_b)
            
            # Tokenize
            inputs = self.tokenizer(
                judge_prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response to get winner (0 or 1)
            winner = self._parse_judge_response(response)
            results.append(winner)
            
            # Clean up GPU memory periodically
            if i % 20 == 0:
                torch.cuda.empty_cache()
        
        return results
    
    def _create_judge_prompt(self, prompt, completion_a, completion_b):
        """Create a judge prompt for pairwise comparison"""
        return f"""Please evaluate which response is better for the given prompt.

Prompt: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Which response is better? Please consider helpfulness, accuracy, relevance, and overall quality.
Answer with only "A" or "B" followed by a brief explanation.

Answer:"""
    
    def _parse_judge_response(self, response):
        """Parse judge response to determine winner (0 for A, 1 for B)"""
        response = response.strip().upper()
        
        # Look for explicit A or B choice
        if response.startswith('A') or 'RESPONSE A' in response:
            return 0
        elif response.startswith('B') or 'RESPONSE B' in response:
            return 1
        
        # Fallback: look for A or B anywhere in the response
        if 'A' in response and 'B' not in response:
            return 0
        elif 'B' in response and 'A' not in response:
            return 1
        
        # Random choice if unclear (shouldn't happen often with good prompts)
        import random
        return random.choice([0, 1])


def setup_multi_gpu():
    """Setup multi-GPU environment"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPUs available")
        
        # Print GPU info
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return num_gpus
    else:
        print("No CUDA GPUs available")
        return 0


def load_ddp_model_correctly(model_path: str, torch_dtype=torch.bfloat16, device_map="auto", num_gpus=1):
    """
    Load a model that was saved with DistributedDataParallel (DDP) format.
    This handles the 'model.module.*' prefix issue by creating corrected weight files.
    Based on the implementation from output_check.py
    """
    import logging
    import tempfile
    import shutil
    import glob
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    print(f"Loading model: {model_path}")
    
    try:
        # Configure for multi-GPU
        if num_gpus > 1:
            device_map = "auto"
            max_memory = {i: "20GB" for i in range(num_gpus)}
        else:
            max_memory = None
        
        # Check if this is a DDP checkpoint by looking at safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            print("No safetensors files found, using standard loading")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                max_memory=max_memory
            )
            print(f"Model loaded successfully using device_map: {device_map}")
            return model
        
        # Check if this is actually a DDP checkpoint by inspecting keys
        is_ddp_checkpoint = False
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if any(key.startswith("model.module.") or key.startswith("module.") for key in keys):
                    is_ddp_checkpoint = True
                    break
        
        if not is_ddp_checkpoint:
            print("ℹ️ No DDP format detected, using standard loading")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                max_memory=max_memory
            )
            print(f"Model loaded successfully using device_map: {device_map}")
            return model
        
        print("🔧 DDP format detected, creating corrected model checkpoint...")
        
        # Create a temporary directory for corrected weights
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy all non-safetensors files to temp directory
            for file_name in os.listdir(model_path):
                if not file_name.endswith('.safetensors'):
                    src_path = os.path.join(model_path, file_name)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, temp_dir)
            
            # Process and save corrected weights
            print(f"Processing {len(safetensor_files)} safetensor files...")
            for i, safetensor_file in enumerate(safetensor_files):
                print(f"  Processing {os.path.basename(safetensor_file)}...")
                
                corrected_state_dict = {}
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        # Remove DDP prefix - handle both "model.module." and "module."
                        clean_key = key
                        if key.startswith("model.module."):
                            clean_key = key[13:]  # Remove "model.module."
                        elif key.startswith("module."):
                            clean_key = key[7:]   # Remove "module."
                        
                        corrected_state_dict[clean_key] = tensor
                
                # Save corrected weights with proper naming
                if len(safetensor_files) == 1:
                    output_file = os.path.join(temp_dir, "model.safetensors")
                else:
                    output_file = os.path.join(temp_dir, f"model-{i+1:05d}-of-{len(safetensor_files):05d}.safetensors")
                
                save_file(corrected_state_dict, output_file)
                print(f"    ✅ Saved corrected weights to {os.path.basename(output_file)}")
            
            # Now load the model from the corrected directory
            print("🔄 Loading model with corrected weights...")
            model = AutoModelForCausalLM.from_pretrained(
                temp_dir,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                max_memory=max_memory
            )
            
            print("✅ DDP model loaded successfully with corrected weights")
            return model
                
    except Exception as e:
        print(f"DDP loading failed: {e}")
        print("Falling back to standard loading...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            max_memory=max_memory
        )
        return model


def load_distributed_model(model_path, torch_dtype=torch.bfloat16, device_map="auto", num_gpus=1):
    """
    Load model saved with distributed training (DistributedDataParallel)
    with robust handling of different checkpoint formats
    """
    return load_ddp_model_correctly(model_path, torch_dtype, device_map, num_gpus)


def setup_tokenizer(tokenizer):
    """Setup tokenizer with proper padding for decoder-only models"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_completions(model, tokenizer, prompts, batch_size, model_name="Model"):
    """Generate completions with robust error handling"""
    print(f"Generating {model_name} completions...")
    completions = []
    
    model.eval()
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            # Tokenize
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode
            for j, output in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_tokens = output[input_length:]
                completion = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                completions.append(completion)
            
            # Progress
            if (i + batch_size) % 20 == 0 or (i + batch_size) >= len(prompts):
                completed = min(i + batch_size, len(prompts))
                print(f"Generated {completed}/{len(prompts)} {model_name} completions")
                
        except Exception as e:
            print(f"Error generating batch {i//batch_size}: {e}")
            # Add empty completions for failed batch
            for _ in range(len(batch_prompts)):
                completions.append("")
    
    return completions


def manual_comparison(prompts, rloo_completions, baseline_completions, num_examples=5):
    """Display manual comparison of outputs"""
    print("=" * 80)
    print("MANUAL COMPARISON SAMPLES")
    print("=" * 80)
    
    num_to_show = min(num_examples, len(prompts))
    
    for i in range(num_to_show):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {prompts[i][:200]}...")
        print(f"\nRLOO Output: {rloo_completions[i][:300]}...")
        print(f"\nBaseline Output: {baseline_completions[i][:300]}...")
        print("-" * 80)


@dataclass
class ScriptArguments:
    """Arguments for the evaluation script"""
    
    model_name_or_path: str = field(
        default="experiment/models/qwen3_1.7b_rloo_model",
        metadata={"help": "Path to the RLOO model to evaluate"}
    )
    baseline_model: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "Path to the baseline model"}
    )
    judge_model: str = field(
        default="microsoft/DialoGPT-medium",  # Lightweight local judge, or use Qwen/Qwen3-1.7B for better quality
        metadata={"help": "Judge model for evaluation. Options: 'microsoft/DialoGPT-medium' (lightweight), 'Qwen/Qwen3-1.7B' (good quality), 'Qwen/Qwen3-30B-A3B' (high quality but large), 'gpt-3.5-turbo' (OpenAI API)"}
    )
    num_examples: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of examples to evaluate (None for all)"}
    )
    batch_size: int = field(
        default=4, 
        metadata={"help": "Batch size for generation"}
    )
    use_multi_gpu: bool = field(
        default=True, 
        metadata={"help": "Whether to use multiple GPUs"}
    )
    skip_judge: bool = field(
        default=False,
        metadata={"help": "Skip judge evaluation and only show manual comparison"}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setup environment
    num_gpus = setup_multi_gpu() if script_args.use_multi_gpu else 1
    if not script_args.use_multi_gpu:
        num_gpus = 1
    
    effective_batch_size = script_args.batch_size * max(1, num_gpus)
    
    print("=" * 80)
    print("IMPROVED RLOO MODEL EVALUATION")
    print("=" * 80)
    print(f"RLOO Model: {script_args.model_name_or_path}")
    print(f"Baseline Model: {script_args.baseline_model}")
    print(f"Judge Model: {script_args.judge_model}")
    print(f"Examples: {script_args.num_examples or 'All'}")
    print(f"Batch Size: {script_args.batch_size} (effective: {effective_batch_size})")
    print(f"GPUs: {num_gpus}")
    print(f"Skip Judge: {script_args.skip_judge}")
    print("-" * 80)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("trl-lib/tldr", split="validation")
    if script_args.num_examples is not None:
        dataset = dataset.select(range(script_args.num_examples))
    print(f"Dataset loaded: {len(dataset)} examples")
    
    prompts = dataset["prompt"]
    
    # Generate RLOO completions
    try:
        print(f"\nLoading RLOO model: {script_args.model_name_or_path}")
        rloo_tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
        rloo_tokenizer = setup_tokenizer(rloo_tokenizer)
        
        rloo_model = load_distributed_model(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            num_gpus=num_gpus
        )
        
        rloo_completions = generate_completions(
            rloo_model, rloo_tokenizer, prompts, effective_batch_size, "RLOO"
        )
        
        # Clean up
        del rloo_model
        del rloo_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error with RLOO model: {e}")
        sys.exit(1)
    
    # Generate baseline completions
    try:
        print(f"\nLoading baseline model: {script_args.baseline_model}")
        baseline_tokenizer = AutoTokenizer.from_pretrained(script_args.baseline_model, trust_remote_code=True)
        baseline_tokenizer = setup_tokenizer(baseline_tokenizer)
        
        baseline_model = AutoModelForCausalLM.from_pretrained(
            script_args.baseline_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if num_gpus > 1 else "cuda",
            trust_remote_code=True,
            max_memory={i: "20GB" for i in range(num_gpus)} if num_gpus > 1 else None
        )
        
        baseline_completions = generate_completions(
            baseline_model, baseline_tokenizer, prompts, effective_batch_size, "Baseline"
        )
        
        # Clean up
        del baseline_model
        del baseline_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error with baseline model: {e}")
        sys.exit(1)
    
    # Show manual comparison
    manual_comparison(prompts, rloo_completions, baseline_completions)
    
    # Judge evaluation (optional)
    if not script_args.skip_judge and TRL_AVAILABLE:
        try:
            print(f"\nInitializing judge: {script_args.judge_model}")
            
            if "gpt" in script_args.judge_model:
                judge = OpenAIPairwiseJudge(script_args.judge_model)
                print("Using OpenAI judge (API-based)")
            else:
                # Try local judge first to avoid API costs
                try:
                    print("Attempting to use local judge model to avoid API costs...")
                    judge = LocalPairwiseJudge(
                        script_args.judge_model,
                        device_map="auto",
                        torch_dtype=torch.bfloat16
                    )
                    print("✅ Using local judge model")
                except Exception as local_e:
                    print(f"❌ Local judge loading failed: {local_e}")
                    print("Falling back to HuggingFace API judge...")
                    judge = HfPairwiseJudge(script_args.judge_model)
                    print("⚠️  Using HuggingFace API judge (may incur costs)")
            
            print("Running pairwise evaluation...")
            completions = [[baseline_comp, rloo_comp] for baseline_comp, rloo_comp in 
                          zip(baseline_completions, rloo_completions)]
            
            best_idxs = judge.judge(prompts, completions)
            rloo_win_rate = best_idxs.count(1) / len(best_idxs)
            baseline_win_rate = best_idxs.count(0) / len(best_idxs)
            
            print("=" * 80)
            print("JUDGE EVALUATION RESULTS")
            print("=" * 80)
            print(f"🏆 RLOO Win Rate: {rloo_win_rate * 100:.2f}%")
            print(f"📊 Baseline Win Rate: {baseline_win_rate * 100:.2f}%")
            
            if rloo_win_rate > baseline_win_rate:
                improvement = rloo_win_rate - baseline_win_rate
                print(f"✅ RLOO outperforms baseline by {improvement * 100:.2f} points!")
            elif baseline_win_rate > rloo_win_rate:
                degradation = baseline_win_rate - rloo_win_rate
                print(f"❌ RLOO underperforms baseline by {degradation * 100:.2f} points")
            else:
                print(f"🤝 RLOO and baseline perform equally")
            
            # Clean up judge model
            if hasattr(judge, 'model'):
                del judge.model
                del judge.tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"\nJudge evaluation failed: {e}")
            print("You can still manually compare the outputs shown above.")
            print("\nTo fix judge evaluation:")
            print("1. For OpenAI: export OPENAI_API_KEY=your_key")
            print("2. For free local evaluation, try:")
            print("   --judge_model microsoft/DialoGPT-medium (lightweight)")
            print("   --judge_model Qwen/Qwen3-1.7B (good quality)")
            print("3. Or use --skip_judge to skip evaluation")
            print("4. Note: Large models like Qwen3-30B-A3B may require API payment")
    
    else:
        print("\nJudge evaluation skipped. Use the manual comparison above.")
    
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
