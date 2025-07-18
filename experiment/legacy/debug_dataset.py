#!/usr/bin/env python3
"""
Debug script to check HelpSteer3 dataset content
"""

from datasets import load_dataset

def check_dataset():
    print("Loading HelpSteer3 dataset...")
    dataset = load_dataset("nvidia/HelpSteer3", split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    # Check first 10 samples
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        print(f"\n--- Sample {i} ---")
        print(f"Keys: {list(sample.keys())}")
        
        if 'prompt' in sample:
            prompt = sample['prompt'][:200] + "..." if len(sample['prompt']) > 200 else sample['prompt']
            print(f"Prompt: {prompt}")
        
        if 'response' in sample:
            response = sample['response'][:200] + "..." if len(sample['response']) > 200 else sample['response']
            print(f"Response: {response}")
            
        if 'helpfulness' in sample:
            print(f"Helpfulness: {sample['helpfulness']}")

if __name__ == "__main__":
    check_dataset() 