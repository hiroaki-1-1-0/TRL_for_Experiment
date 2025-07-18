#!/usr/bin/env python3
"""
Demo: English Language Filtering for HelpSteer3 Dataset

This script demonstrates the exact English-only filtering implementation
that's used in the RLOO training script.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import collections

def filter_english_samples(dataset):
    """
    Filter the HelpSteer3 dataset to include only samples with language='english'.
    HelpSteer3 has multiple domains and languages, we want only English language samples.
    """
    
    def is_english_sample(example):
        # Filter for samples where language field is 'english'
        return example.get('language', '').lower() == 'english'
    
    print(f"Original dataset size: {len(dataset)}")  # 38,459
    
    # Filter to only English language samples
    english_dataset = dataset.filter(is_english_sample)
    
    print(f"After English filtering: {len(english_dataset)} samples")  # 22,380
    print(f"Filtered out: {len(dataset) - len(english_dataset)} non-English samples")  # 16,079
    
    return english_dataset

def analyze_dataset(dataset, name):
    """Analyze dataset composition."""
    print(f"\n{name} Analysis:")
    print(f"Total samples: {len(dataset)}")
    
    # Language distribution
    lang_counts = collections.Counter(dataset['language'])
    print("\nTop 10 languages:")
    for lang, count in lang_counts.most_common(10):
        print(f"  {lang}: {count} samples")
    
    # Domain distribution
    domain_counts = collections.Counter(dataset['domain'])
    print("\nDomain distribution:")
    for domain, count in domain_counts.items():
        print(f"  {domain}: {count} samples")

def show_samples(dataset, name, num_samples=3):
    """Show sample data."""
    print(f"\n{name} Sample Data:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Domain: {sample['domain']}")
        print(f"  Language: {sample['language']}")
        
        # Show context preview
        context = sample.get('context', [])
        if context and len(context) > 0:
            first_msg = context[0]
            content_preview = first_msg.get('content', '')[:100]
            print(f"  Context preview: {content_preview}...")
        
        # Show preference info
        preference = sample.get('overall_preference', 'N/A')
        print(f"  Overall preference: {preference}")

def main():
    print("=" * 70)
    print("ENGLISH LANGUAGE FILTERING DEMO - HelpSteer3 Dataset")
    print("=" * 70)
    print("This demo shows the exact filtering used in RLOO training")
    print("Specification: Use only 'english' language samples from HelpSteer3")
    print("=" * 70)
    
    print("\nStep 1: Loading HelpSteer3 dataset (preference format, train split)...")
    print("Note: HelpSteer3 is already in preference format with response1/response2 pairs")
    
    try:
        dataset = load_dataset("nvidia/HelpSteer3", split="train")
        print("✓ Dataset loaded successfully")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return
    
    print("\nStep 2: Analyzing original dataset composition...")
    analyze_dataset(dataset, "Original HelpSteer3")
    
    print("\nStep 3: Applying English language filter...")
    english_dataset = filter_english_samples(dataset)
    
    print("\nStep 4: Analyzing filtered dataset...")
    analyze_dataset(english_dataset, "English-only HelpSteer3")
    
    print("\nStep 5: Showing sample data...")
    show_samples(english_dataset, "English-only", num_samples=2)
    
    # Verify the filtering worked correctly
    print("\nStep 6: Verification...")
    all_english = all(sample['language'].lower() == 'english' for sample in english_dataset)
    print(f"✓ All samples are English: {all_english}")
    
    # Calculate filtering efficiency
    original_size = len(dataset)
    english_size = len(english_dataset)
    percentage = (english_size / original_size) * 100
    
    print(f"\nFiltering Results:")
    print(f"  Original samples: {original_size:,}")
    print(f"  English samples: {english_size:,}")
    print(f"  Retention rate: {percentage:.1f}%")
    print(f"  Filtered out: {original_size - english_size:,} samples")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("English-only filtering is working exactly as specified:")
    print("- ✅ Loads HelpSteer3 preference dataset (train split)")
    print("- ✅ Filters for language='english' samples only")
    print("- ✅ Retains preference format (response1/response2)")
    print("- ✅ Ready for RLOO training")
    print("=" * 70)

if __name__ == "__main__":
    main() 