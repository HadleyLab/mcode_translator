#!/usr/bin/env python3
"""
Test script to verify that all prompts load correctly from the file-based prompt library.
"""

import sys
import os
# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from utils.prompt_loader import PromptLoader

def test_all_prompts():
    """Test loading all prompts from the file-based library."""
    print("Testing prompt loading from file-based library...")
    print("=" * 60)
    
    loader = PromptLoader()
    
    # Test NLP extraction prompts
    print("\nNLP Extraction Prompts:")
    print("-" * 30)
    
    nlp_prompts = [
        "generic_extraction",
        "comprehensive_extraction",
        "minimal_extraction",
        "structured_extraction",
        "basic_extraction",
        "minimal_extraction_optimization"
    ]
    
    for prompt_name in nlp_prompts:
        try:
            prompt = loader.get_prompt(prompt_name)
            print(f"✓ {prompt_name}: Loaded successfully ({len(prompt)} chars)")
        except Exception as e:
            print(f"✗ {prompt_name}: Failed to load - {e}")
    
    # Test mCODE mapping prompts
    print("\nMcode Mapping Prompts:")
    print("-" * 30)
    
    Mcode_prompts = [
        "generic_mapping",
        "standard_mapping",
        "detailed_mapping",
        "error_robust_mapping",
        "comprehensive_mapping",
        "simple_mapping"
    ]
    
    for prompt_name in Mcode_prompts:
        try:
            prompt = loader.get_prompt(prompt_name)
            print(f"✓ {prompt_name}: Loaded successfully ({len(prompt)} chars)")
        except Exception as e:
            print(f"✗ {prompt_name}: Failed to load - {e}")
    
    # Test prompt configuration
    print("\nPrompt Configuration:")
    print("-" * 30)
    
    try:
        config = loader.list_available_prompts()
        print(f"✓ Config loaded: {len(config)} prompts configured")
        for prompt_name, prompt_info in config.items():
            print(f"  - {prompt_name}: {prompt_info.get('category', 'unknown')}")
    except Exception as e:
        print(f"✗ Config failed to load: {e}")
    
    print("\n" + "=" * 60)
    print("Prompt loading test completed!")

if __name__ == "__main__":
    test_all_prompts()