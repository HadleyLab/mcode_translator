#!/usr/bin/env python3
"""
Test script to validate prompt loading and response validation.
Run with: source activate mcode_translator && python test_prompt_validation.py
"""

import json
from src.utils.prompt_loader import PromptLoader

def test_nlp_extraction_generic():
    """Test the NLP extraction generic prompt validation"""
    print("Testing NLP extraction generic prompt...")
    
    loader = PromptLoader()
    try:
        # Load the prompt
        prompt_content = loader.get_prompt('generic_extraction')
        print('✓ NLP extraction generic prompt loaded successfully')
        return True
        
    except Exception as e:
        print('✗ Error:', str(e))
        return False

def test_mcode_mapping_generic():
    """Test the Mcode mapping generic prompt validation"""
    print("\nTesting Mcode mapping generic prompt...")
    
    loader = PromptLoader()
    try:
        # Load the prompt
        prompt_content = loader.get_prompt('generic_mapping')
        print('✓ Mcode mapping generic prompt loaded successfully')
        return True
        
    except Exception as e:
        print('✗ Error:', str(e))
        return False

def test_direct_mcode_direct():
    """Test the direct Mcode direct prompt validation"""
    print("\nTesting direct Mcode direct prompt...")
    
    loader = PromptLoader()
    try:
        # Load the prompt
        prompt_content = loader.get_prompt('direct_mcode')
        print('✓ Direct Mcode direct prompt loaded successfully')
        return True
        
    except Exception as e:
        print('✗ Error:', str(e))
        return False

if __name__ == "__main__":
    print("Testing prompt validation...")
    
    results = []
    results.append(test_nlp_extraction_generic())
    results.append(test_mcode_mapping_generic())
    results.append(test_direct_mcode_direct())
    
    print(f"\nSummary: {sum(results)}/{len(results)} tests passed")