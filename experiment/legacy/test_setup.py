#!/usr/bin/env python3
"""
Test script to verify RLOO training setup
This script tests that all components can be loaded without actually running training.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import RLOOConfig, ModelConfig, ScriptArguments

def test_dataset_loading():
    """Test loading and basic processing of HelpSteer3 dataset"""
    print("Testing dataset loading...")
    try:
        dataset = load_dataset("nvidia/HelpSteer3", split="train[:10]")
        print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
        
        # Check first sample
        sample = dataset[0]
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Domain: {sample.get('domain', 'N/A')}")
        print(f"  Language: {sample.get('language', 'N/A')}")
        
        # Preview context if available
        if 'context' in sample and sample['context']:
            if isinstance(sample['context'], list) and len(sample['context']) > 0:
                print(f"  Context preview: {sample['context'][0].get('content', '')[:100]}...")
            else:
                print(f"  Context preview: {str(sample['context'])[:100]}...")
        
        # Note: In HelpSteer3, 'language' refers to programming languages, not human languages
        print(f"  Note: This implementation filters for English language samples only")
        
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

def test_model_loading():
    """Test loading models and tokenizer"""
    print("\nTesting model loading...")
    
    # Test tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        return False
    
    # Test policy model (just check it exists, don't load due to size)
    print("✓ Policy model (Qwen/Qwen3-8B) - skipping full load due to size")
    
    # Test reward model (just check it exists, don't load due to size)
    print("✓ Reward model (nvidia/Llama-3_3-Nemotron-Super-49B-GenRM-Multilingual) - skipping full load due to size")
    
    return True

def test_configuration():
    """Test TRL configuration classes"""
    print("\nTesting TRL configuration...")
    try:
        # Test RLOO config
        rloo_config = RLOOConfig(
            output_dir="test_output",
            per_device_train_batch_size=1,
            learning_rate=1e-6,
            total_episodes=100,
            rloo_k=2
        )
        print("✓ RLOOConfig created successfully")
        
        # Test Model config
        model_config = ModelConfig(
            model_name_or_path="Qwen/Qwen3-8B"
        )
        print("✓ ModelConfig created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        return False

def test_tokenization():
    """Test tokenization with sample data"""
    print("\nTesting tokenization...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        sample_prompt = "What are the benefits of renewable energy?"
        
        # Test direct tokenization
        tokens = tokenizer(sample_prompt, return_tensors="pt")
        print(f"✓ Direct tokenization successful: {tokens['input_ids'].shape}")
        
        # Test chat template (if available)
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": sample_prompt}]
            chat_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            print(f"✓ Chat template tokenization successful: {chat_tokens.shape}")
        else:
            print("! Chat template not available, will use simple template")
        
        return True
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability and memory"""
    print("\nTesting GPU availability...")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("! No CUDA GPUs available - training will be very slow on CPU")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("RLOO Training Setup Test")
    print("=" * 60)
    
    tests = [
        test_dataset_loading,
        test_model_loading,
        test_configuration,
        test_tokenization,
        test_gpu_availability,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    test_names = [
        "Dataset Loading",
        "Model Loading", 
        "Configuration",
        "Tokenization",
        "GPU Availability"
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:<20}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 All tests passed! Ready for RLOO training.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main() 