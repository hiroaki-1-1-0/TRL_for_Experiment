#!/usr/bin/env python3
"""
RLOO Training Output Verification Script
Checks the output quality of models/qwen3_1.7b_rloo_model_chinese_01 trained with RLOO
"""

import os
import sys
import argparse
import torch
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import safetensors.torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_ddp_model_correctly(model_path: str, model_class=AutoModelForCausalLM):
    """
    Load a model that was saved with DistributedDataParallel (DDP) format.
    This handles the 'model.module.*' prefix issue by creating corrected weight files.
    """
    import logging
    import tempfile
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if this is a DDP checkpoint by looking at safetensors files
        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not safetensor_files:
            logger.info("No safetensors files found, using standard loading")
            return model_class.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Check if this is actually a DDP checkpoint by inspecting keys
        is_ddp_checkpoint = False
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if any(key.startswith("model.module.") or key.startswith("module.") for key in keys):
                    is_ddp_checkpoint = True
                    break
        
        if not is_ddp_checkpoint:
            logger.info("ℹ️ No DDP format detected, using standard loading")
            return model_class.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        logger.info("🔧 DDP format detected, creating corrected model checkpoint...")
        
        # Create a temporary directory for corrected weights
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy all non-safetensors files to temp directory
            for file_name in os.listdir(model_path):
                if not file_name.endswith('.safetensors'):
                    src_path = os.path.join(model_path, file_name)
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, temp_dir)
            
            # Process and save corrected weights
            logger.info(f"Processing {len(safetensor_files)} safetensor files...")
            for i, safetensor_file in enumerate(safetensor_files):
                logger.info(f"  Processing {os.path.basename(safetensor_file)}...")
                
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
                logger.info(f"    ✅ Saved corrected weights to {os.path.basename(output_file)}")
            
            # Now load the model from the corrected directory
            logger.info("🔄 Loading model with corrected weights...")
            model = model_class.from_pretrained(
                temp_dir,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            logger.info("✅ DDP model loaded successfully with corrected weights")
            return model
                
    except Exception as e:
        logger.error(f"DDP loading failed: {e}")
        logger.info("Falling back to standard loading...")
        return model_class.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
    except Exception as e:
        logger.error(f"Failed to load model with DDP correction: {e}")
        # Fallback to normal loading
        return model_class.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class RLOOOutputChecker:
    """
    Class to verify RLOO model outputs and compare with baseline
    """
    
    def __init__(self, 
                 model_path: str = "experiment/models/qwen3_1.7b_rloo_model_chinese_01",
                 base_model_name: str = "Qwen/Qwen3-1.7B",
                 reward_model_path: str = "experiment/models/qwen3_1.7b_reward_model"):
        """
        Initialize the output checker
        
        Args:
            model_path: Path to the RLOO-trained model
            base_model_name: Name/path of the base model for comparison
            reward_model_path: Path to the reward model for scoring
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.reward_model_path = reward_model_path
        self.logger = setup_logging()
        
        # Test prompts in both English and Chinese
        self.test_prompts = {
            "english": [
                "How can I improve my programming skills?",
                "What is the best way to learn machine learning?",
                "Explain the concept of neural networks in simple terms.",
                "How do I write clean and maintainable code?",
                "What are the key principles of software engineering?"
            ],
            "chinese": [
                "如何提高编程技能？",
                "学习机器学习的最佳方法是什么？",
                "用简单的术语解释神经网络的概念。",
                "如何编写干净且可维护的代码？",
                "软件工程的关键原则是什么？"
            ]
        }
        
    def load_models(self):
        """Load the RLOO model, base model, and reward model"""
        self.logger.info("Loading models...")
        
        # Convert relative paths to absolute paths if needed
        if not os.path.isabs(self.model_path):
            abs_model_path = os.path.abspath(self.model_path)
        else:
            abs_model_path = self.model_path
        
        # Check multiple possible locations for RLOO model
        # Include checkpoint directories
        base_paths = [
            abs_model_path,
            self.model_path,
            os.path.join(os.getcwd(), self.model_path),
            os.path.join("/work/hiroaki/dev/TRL_for_Experiment", self.model_path),
            # Additional paths based on the actual model location found
            os.path.join("/TRL_for_Experiment", self.model_path),
            os.path.join(os.getcwd(), "experiment", "models", "qwen3_1.7b_rloo_model_chinese_01"),
            "/TRL_for_Experiment/experiment/models/qwen3_1.7b_rloo_model_chinese_01"
        ]
        
        # Add checkpoint paths
        possible_paths = base_paths.copy()
        for base_path in base_paths:
            if os.path.exists(base_path):
                # Look for checkpoint directories
                checkpoints = []
                try:
                    for item in os.listdir(base_path):
                        if item.startswith('checkpoint-') and os.path.isdir(os.path.join(base_path, item)):
                            checkpoint_num = int(item.split('-')[1])
                            checkpoints.append((checkpoint_num, os.path.join(base_path, item)))
                    # Sort by checkpoint number and use the latest
                    if checkpoints:
                        checkpoints.sort(key=lambda x: x[0], reverse=True)
                        latest_checkpoint = checkpoints[0][1]
                        possible_paths.insert(0, latest_checkpoint)  # Try latest checkpoint first
                        self.logger.info(f"Found latest checkpoint: {latest_checkpoint}")
                except Exception as e:
                    self.logger.warning(f"Error scanning for checkpoints in {base_path}: {e}")
        
        rloo_model_found = False
        actual_model_path = None
        
        self.logger.info(f"Searching for RLOO model in the following locations:")
        for path in possible_paths:
            self.logger.info(f"  - {path}")
            if os.path.exists(path):
                # Check if it's a valid model directory
                required_files = ['config.json']
                model_files = ['model.safetensors', 'pytorch_model.bin']
                
                has_config = any(os.path.exists(os.path.join(path, f)) for f in required_files)
                has_model = any(os.path.exists(os.path.join(path, f)) for f in model_files)
                
                if has_config and has_model:
                    actual_model_path = path
                    rloo_model_found = True
                    self.logger.info(f"  ✅ Found valid model at: {path}")
                    break
                else:
                    self.logger.info(f"  ❌ Not a valid model directory: {path}")
            else:
                self.logger.info(f"  ❌ Not found at: {path}")
        
        if not rloo_model_found:
            self.logger.warning(f"RLOO model not found in any of the searched locations")
            self.logger.info("Note: Model might be stored on a different server as mentioned by user")
            self.rloo_model = None
            self.rloo_tokenizer = None
        else:
            try:
                self.logger.info(f"Loading RLOO model from {actual_model_path}")
                self.rloo_tokenizer = AutoTokenizer.from_pretrained(actual_model_path)
                
                # Use custom DDP loader for RLOO model
                self.logger.info("Using DDP-aware model loader...")
                self.rloo_model = load_ddp_model_correctly(actual_model_path, AutoModelForCausalLM)
                
                self.logger.info("✅ RLOO model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load RLOO model: {e}")
                self.rloo_model = None
                self.rloo_tokenizer = None
        
        # Load base model for comparison
        try:
            self.logger.info(f"Loading base model: {self.base_model_name}")
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.logger.info("✅ Base model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            self.base_model = None
            self.base_tokenizer = None
            
        # Load reward model for scoring
        reward_possible_paths = [
            self.reward_model_path,
            os.path.abspath(self.reward_model_path),
            os.path.join(os.getcwd(), self.reward_model_path),
            os.path.join("/work/hiroaki/dev/TRL_for_Experiment", self.reward_model_path),
            # Additional paths based on the actual model location found
            os.path.join("/TRL_for_Experiment", self.reward_model_path),
            os.path.join(os.getcwd(), "experiment", "models", "qwen3_1.7b_reward_model"),
            "/TRL_for_Experiment/experiment/models/qwen3_1.7b_reward_model"
        ]
        
        reward_model_found = False
        actual_reward_path = None
        
        self.logger.info(f"Searching for reward model in the following locations:")
        for path in reward_possible_paths:
            self.logger.info(f"  - {path}")
            if os.path.exists(path):
                actual_reward_path = path
                reward_model_found = True
                self.logger.info(f"  ✅ Found at: {path}")
                break
            else:
                self.logger.info(f"  ❌ Not found at: {path}")
        
        if not reward_model_found:
            self.logger.warning(f"Reward model not found in any of the searched locations")
            self.reward_model = None
            self.reward_tokenizer = None
        else:
            try:
                self.logger.info(f"Loading reward model from {actual_reward_path}")
                self.reward_tokenizer = AutoTokenizer.from_pretrained(actual_reward_path)
                
                # Try to load reward model with proper configuration
                try:
                    self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                        actual_reward_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        num_labels=1  # Specify single output for reward model
                    )
                except Exception as config_error:
                    self.logger.warning(f"Failed to load reward model with config: {config_error}")
                    # Try loading as base model and add classification head
                    self.logger.info("Attempting to load as base model...")
                    base_reward_model = AutoModelForCausalLM.from_pretrained(actual_reward_path)
                    # For now, skip reward scoring if we can't load the reward model properly
                    self.reward_model = None
                    self.logger.warning("Using base model comparison only (no reward scoring)")
                
                if self.reward_model is not None:
                    self.logger.info("✅ Reward model loaded successfully")
                else:
                    self.logger.warning("⚠️ Reward model not available - using response comparison only")
                    
            except Exception as e:
                self.logger.error(f"Failed to load reward model: {e}")
                self.reward_model = None
                self.reward_tokenizer = None
    
    def generate_response(self, model, tokenizer, prompt: str, max_length: int = 512) -> str:
        """Generate response from a model"""
        if model is None or tokenizer is None:
            return "[Model not available]"
            
        try:
            # Prepare input
            if tokenizer.chat_template is None:
                # Simple format if no chat template
                input_text = f"User: {prompt}\nAssistant: "
            else:
                # Use chat template if available
                messages = [{"role": "user", "content": prompt}]
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"[Error: {str(e)}]"
    
    def calculate_reward_score(self, prompt: str, response: str) -> float:
        """Calculate reward score for a prompt-response pair"""
        if self.reward_model is None or self.reward_tokenizer is None:
            return 0.0
            
        try:
            # Format input for reward model
            input_text = f"User: {prompt}\nAssistant: {response}"
            inputs = self.reward_tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}
            
            # Get reward score
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                score = outputs.logits.squeeze().item()
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating reward score: {e}")
            return 0.0
    
    def calculate_response_quality_metrics(self, prompt: str, response: str) -> Dict[str, float]:
        """Calculate alternative quality metrics when reward model is not available"""
        metrics = {}
        
        # Length metrics
        metrics['response_length'] = len(response.split())
        metrics['response_char_length'] = len(response)
        
        # Content quality heuristics
        # Check if response actually addresses the prompt
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        # Keyword overlap
        prompt_words = set(prompt_lower.split())
        response_words = set(response_lower.split())
        metrics['keyword_overlap'] = len(prompt_words.intersection(response_words)) / len(prompt_words) if prompt_words else 0
        
        # Response completeness (not ending abruptly)
        if response.endswith(('.', '!', '?', '。', '！', '？')):
            metrics['completeness_score'] = 1.0
        else:
            metrics['completeness_score'] = 0.5
        
        # Avoid repetitive responses
        words = response.split()
        if len(words) > 0:
            unique_words = len(set(words))
            metrics['diversity_score'] = unique_words / len(words)
        else:
            metrics['diversity_score'] = 0.0
        
        # Check for error indicators
        error_indicators = ['[Model not available]', '[Error:', 'error', 'failed', 'cannot']
        metrics['error_score'] = 1.0 if not any(indicator in response_lower for indicator in error_indicators) else 0.0
        
        # Calculate composite quality score
        metrics['composite_quality'] = (
            min(metrics['response_length'] / 50, 1.0) * 0.2 +  # Reasonable length
            metrics['keyword_overlap'] * 0.2 +                  # Relevance
            metrics['completeness_score'] * 0.2 +               # Completeness
            metrics['diversity_score'] * 0.2 +                  # Diversity
            metrics['error_score'] * 0.2                        # No errors
        )
        
        return metrics
    
    def compare_responses(self, prompt: str, language: str) -> Dict[str, Any]:
        """Compare RLOO model response with base model response"""
        self.logger.info(f"Testing prompt ({language}): {prompt[:50]}...")
        
        # Generate responses
        rloo_response = self.generate_response(self.rloo_model, self.rloo_tokenizer, prompt)
        base_response = self.generate_response(self.base_model, self.base_tokenizer, prompt)
        
        # Calculate reward scores (if available)
        rloo_score = self.calculate_reward_score(prompt, rloo_response)
        base_score = self.calculate_reward_score(prompt, base_response)
        
        # Calculate alternative quality metrics
        rloo_quality = self.calculate_response_quality_metrics(prompt, rloo_response)
        base_quality = self.calculate_response_quality_metrics(prompt, base_response)
        
        result = {
            "prompt": prompt,
            "language": language,
            "rloo_response": rloo_response,
            "base_response": base_response,
            "rloo_score": rloo_score,
            "base_score": base_score,
            "improvement": rloo_score - base_score,
            "rloo_length": len(rloo_response.split()),
            "base_length": len(base_response.split()),
            "rloo_quality_metrics": rloo_quality,
            "base_quality_metrics": base_quality,
            "quality_improvement": rloo_quality['composite_quality'] - base_quality['composite_quality']
        }
        
        return result
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive output verification"""
        self.logger.info("🔍 Starting comprehensive RLOO output verification...")
        
        # Load models
        self.load_models()
        
        results = {
            "model_info": {
                "rloo_model_path": self.model_path,
                "base_model_name": self.base_model_name,
                "reward_model_path": self.reward_model_path,
                "rloo_model_available": self.rloo_model is not None,
                "base_model_available": self.base_model is not None,
                "reward_model_available": self.reward_model is not None
            },
            "test_results": [],
            "summary": {}
        }
        
        all_improvements = []
        all_quality_improvements = []
        language_results = {"english": [], "chinese": []}
        language_quality_results = {"english": [], "chinese": []}
        
        # Test all prompts
        for language, prompts in self.test_prompts.items():
            self.logger.info(f"\n📝 Testing {language.upper()} prompts...")
            
            for prompt in prompts:
                result = self.compare_responses(prompt, language)
                results["test_results"].append(result)
                all_improvements.append(result["improvement"])
                all_quality_improvements.append(result["quality_improvement"])
                language_results[language].append(result["improvement"])
                language_quality_results[language].append(result["quality_improvement"])
                
                # Print immediate results
                print(f"\n--- {language.upper()} Test ---")
                print(f"Prompt: {prompt}")
                print(f"RLOO Response: {result['rloo_response'][:100]}{'...' if len(result['rloo_response']) > 100 else ''}")
                print(f"Base Response: {result['base_response'][:100]}{'...' if len(result['base_response']) > 100 else ''}")
                print(f"RLOO Score: {result['rloo_score']:.3f}")
                print(f"Base Score: {result['base_score']:.3f}")
                print(f"Reward Improvement: {result['improvement']:.3f}")
                print(f"RLOO Quality: {result['rloo_quality_metrics']['composite_quality']:.3f}")
                print(f"Base Quality: {result['base_quality_metrics']['composite_quality']:.3f}")
                print(f"Quality Improvement: {result['quality_improvement']:.3f}")
        
        # Calculate summary statistics
        if all_improvements:
            results["summary"] = {
                "total_tests": len(all_improvements),
                "average_improvement": sum(all_improvements) / len(all_improvements),
                "average_quality_improvement": sum(all_quality_improvements) / len(all_quality_improvements),
                "positive_improvements": sum(1 for x in all_improvements if x > 0),
                "positive_quality_improvements": sum(1 for x in all_quality_improvements if x > 0),
                "negative_improvements": sum(1 for x in all_improvements if x < 0),
                "english_avg_improvement": sum(language_results["english"]) / len(language_results["english"]) if language_results["english"] else 0,
                "chinese_avg_improvement": sum(language_results["chinese"]) / len(language_results["chinese"]) if language_results["chinese"] else 0,
                "english_avg_quality_improvement": sum(language_quality_results["english"]) / len(language_quality_results["english"]) if language_quality_results["english"] else 0,
                "chinese_avg_quality_improvement": sum(language_quality_results["chinese"]) / len(language_quality_results["chinese"]) if language_quality_results["chinese"] else 0,
                "max_improvement": max(all_improvements),
                "min_improvement": min(all_improvements),
                "max_quality_improvement": max(all_quality_improvements),
                "min_quality_improvement": min(all_quality_improvements)
            }
        
        return results
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a comprehensive summary report"""
        print("\n" + "="*80)
        print("🎯 RLOO TRAINING VERIFICATION REPORT")
        print("="*80)
        
        # Model availability
        model_info = results["model_info"]
        print(f"📁 RLOO Model Path: {model_info['rloo_model_path']}")
        print(f"✅ RLOO Model Available: {model_info['rloo_model_available']}")
        print(f"✅ Base Model Available: {model_info['base_model_available']}")
        print(f"✅ Reward Model Available: {model_info['reward_model_available']}")
        
        if not results["summary"]:
            print("\n❌ No test results available")
            return
        
        summary = results["summary"]
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Average Reward Improvement: {summary['average_improvement']:.3f}")
        print(f"   Average Quality Improvement: {summary['average_quality_improvement']:.3f}")
        print(f"   Positive Reward Improvements: {summary['positive_improvements']}/{summary['total_tests']}")
        print(f"   Positive Quality Improvements: {summary['positive_quality_improvements']}/{summary['total_tests']}")
        print(f"   Reward Success Rate: {summary['positive_improvements']/summary['total_tests']*100:.1f}%")
        print(f"   Quality Success Rate: {summary['positive_quality_improvements']/summary['total_tests']*100:.1f}%")
        
        print(f"\n🌍 LANGUAGE-SPECIFIC RESULTS:")
        print(f"   English Avg Reward Improvement: {summary['english_avg_improvement']:.3f}")
        print(f"   English Avg Quality Improvement: {summary['english_avg_quality_improvement']:.3f}")
        print(f"   Chinese Avg Reward Improvement: {summary['chinese_avg_improvement']:.3f}")
        print(f"   Chinese Avg Quality Improvement: {summary['chinese_avg_quality_improvement']:.3f}")
        
        print(f"\n📈 IMPROVEMENT RANGE:")
        print(f"   Best Reward Improvement: {summary['max_improvement']:.3f}")
        print(f"   Worst Reward Performance: {summary['min_improvement']:.3f}")
        print(f"   Best Quality Improvement: {summary['max_quality_improvement']:.3f}")
        print(f"   Worst Quality Performance: {summary['min_quality_improvement']:.3f}")
        
        # Overall assessment - use quality metrics as primary if reward model unavailable
        primary_metric = summary['average_quality_improvement'] if summary['average_improvement'] == 0 else summary['average_improvement']
        primary_success_rate = summary['positive_quality_improvements'] / summary['total_tests'] if summary['average_improvement'] == 0 else summary['positive_improvements'] / summary['total_tests']
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if primary_metric > 0.1:
            print("   🟢 EXCELLENT: RLOO training shows significant improvement")
        elif primary_metric > 0.0:
            print("   🟡 GOOD: RLOO training shows modest improvement")
        elif primary_metric > -0.1:
            print("   🟠 NEUTRAL: RLOO training shows minimal change")
        else:
            print("   🔴 CONCERNING: RLOO training may have degraded performance")
        
        if primary_success_rate > 0.7:
            print("   🟢 CONSISTENT: Most test cases show improvement")
        elif primary_success_rate > 0.5:
            print("   🟡 MIXED: Balanced results across test cases")
        else:
            print("   🔴 INCONSISTENT: Many test cases show degradation")
    
    def diagnose_model_weights(self, model, model_name: str):
        """Diagnose if model weights are properly loaded"""
        if model is None:
            self.logger.info(f"❌ {model_name}: Model is None")
            return
            
        state_dict = model.state_dict()
        
        # Sample a few layers and check their statistics
        sample_keys = list(state_dict.keys())[:5]  # First 5 layers
        
        self.logger.info(f"🔍 Diagnosing {model_name} weights:")
        
        for key in sample_keys:
            param = state_dict[key]
            mean_val = param.mean().item()
            std_val = param.std().item()
            min_val = param.min().item()
            max_val = param.max().item()
            
            # Check if weights look reasonable (not randomly initialized)
            if std_val > 1.0 or abs(mean_val) > 1.0:
                status = "⚠️  SUSPICIOUS"
            elif std_val < 0.001:
                status = "⚠️  TOO_UNIFORM"
            else:
                status = "✅ NORMAL"
                
            self.logger.info(f"  {key[:50]:50} | Mean: {mean_val:7.4f} | Std: {std_val:7.4f} | Range: [{min_val:7.4f}, {max_val:7.4f}] | {status}")
    
    def run_model_diagnostics(self):
        """Run comprehensive model diagnostics"""
        self.logger.info("🔬 Running model diagnostics...")
        self.load_models()
        
        self.diagnose_model_weights(self.rloo_model, "RLOO Model")
        self.diagnose_model_weights(self.base_model, "Base Model")
        
        # Test a simple generation
        if self.rloo_model and self.rloo_tokenizer:
            self.logger.info("🧪 Testing RLOO model generation...")
            test_response = self.generate_response(self.rloo_model, self.rloo_tokenizer, "Hello")
            self.logger.info(f"RLOO test response: {test_response[:100]}")
            
        if self.base_model and self.base_tokenizer:
            self.logger.info("🧪 Testing base model generation...")
            test_response = self.generate_response(self.base_model, self.base_tokenizer, "Hello")
            self.logger.info(f"Base test response: {test_response[:100]}")
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str = "rloo_verification_results.json"):
        """Save detailed results to JSON file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"📄 Detailed results saved to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

def main():
    """Main function to run RLOO output verification"""
    parser = argparse.ArgumentParser(description="Verify RLOO training output quality")
    parser.add_argument("--model_path", type=str, 
                       default="experiment/models/qwen3_1.7b_rloo_model_chinese_01",
                       help="Path to RLOO-trained model")
    parser.add_argument("--base_model", type=str,
                       default="Qwen/Qwen3-1.7B",
                       help="Base model for comparison")
    parser.add_argument("--reward_model_path", type=str,
                       default="experiment/models/qwen3_1.7b_reward_model",
                       help="Path to reward model")
    parser.add_argument("--output_file", type=str,
                       default="rloo_verification_results.json",
                       help="Output file for detailed results")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run only a subset of tests for quick verification")
    parser.add_argument("--list_models", action="store_true",
                       help="List available models in experiment directory")
    parser.add_argument("--use_checkpoint", type=str,
                       help="Use specific checkpoint (e.g., 'checkpoint-14')")
    parser.add_argument("--use_latest_checkpoint", action="store_true",
                       help="Automatically use the latest checkpoint")
    parser.add_argument("--diagnose_model", action="store_true",
                       help="Run model diagnostic to check if weights are properly loaded")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("🔍 Searching for available models...")
        search_dirs = [
            "experiment/models",
            "models",
            "experiment",
            "/work/hiroaki/dev/TRL_for_Experiment/experiment/models",
            "/work/hiroaki/dev/TRL_for_Experiment/models"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"\n📁 Contents of {search_dir}:")
                for item in os.listdir(search_dir):
                    item_path = os.path.join(search_dir, item)
                    if os.path.isdir(item_path):
                        print(f"  📂 {item}/")
                        # Check if it looks like a model directory
                        model_files = ["config.json", "pytorch_model.bin", "model.safetensors"]
                        if any(os.path.exists(os.path.join(item_path, f)) for f in model_files):
                            print(f"    ✅ Looks like a model directory")
                    else:
                        print(f"  📄 {item}")
            else:
                print(f"❌ Directory not found: {search_dir}")
        return
    
    # Handle checkpoint specification
    if args.use_checkpoint or args.use_latest_checkpoint:
        base_model_path = args.model_path
        if args.use_checkpoint:
            checkpoint_path = os.path.join(base_model_path, args.use_checkpoint)
            if os.path.exists(checkpoint_path):
                args.model_path = checkpoint_path
                print(f"🎯 Using specified checkpoint: {checkpoint_path}")
            else:
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return
        elif args.use_latest_checkpoint:
            # Find latest checkpoint
            if os.path.exists(base_model_path):
                checkpoints = []
                try:
                    for item in os.listdir(base_model_path):
                        if item.startswith('checkpoint-') and os.path.isdir(os.path.join(base_model_path, item)):
                            checkpoint_num = int(item.split('-')[1])
                            checkpoints.append((checkpoint_num, os.path.join(base_model_path, item)))
                    if checkpoints:
                        checkpoints.sort(key=lambda x: x[0], reverse=True)
                        latest_checkpoint = checkpoints[0][1]
                        args.model_path = latest_checkpoint
                        print(f"🎯 Using latest checkpoint: {latest_checkpoint}")
                    else:
                        print(f"❌ No checkpoints found in {base_model_path}")
                        return
                except Exception as e:
                    print(f"❌ Error scanning for checkpoints: {e}")
                    return
            else:
                print(f"❌ Base model path not found: {base_model_path}")
                return
    
    # Run diagnostics if requested
    if args.diagnose_model:
        checker = RLOOOutputChecker(
            model_path=args.model_path,
            base_model_name=args.base_model,
            reward_model_path=args.reward_model_path
        )
        checker.run_model_diagnostics()
        return
    
    # Initialize checker
    checker = RLOOOutputChecker(
        model_path=args.model_path,
        base_model_name=args.base_model,
        reward_model_path=args.reward_model_path
    )
    
    # Reduce test prompts for quick test
    if args.quick_test:
        checker.test_prompts = {
            "english": checker.test_prompts["english"][:2],
            "chinese": checker.test_prompts["chinese"][:2]
        }
        print("🚀 Running quick test with reduced prompt set...")
    
    # Run verification
    start_time = time.time()
    results = checker.run_comprehensive_check()
    end_time = time.time()
    
    # Print summary
    checker.print_summary_report(results)
    
    # Save detailed results
    checker.save_detailed_results(results, args.output_file)
    
    print(f"\n⏱️  Total verification time: {end_time - start_time:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
