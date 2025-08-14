#!/usr/bin/env python3
"""
Test script for enhanced code extraction functionality
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from code_extraction import CodeExtractionModule

def main():
    """Main function to test enhanced code extraction"""
    print("Testing Enhanced Code Extraction...")
    
    # Create an instance of the code extraction module
    code_extractor = CodeExtractionModule()
    
    # Test entities with enhanced term mappings
    entities = [
        {'text': 'breast cancer', 'confidence': 0.9},
        {'text': 'chemotherapy', 'confidence': 0.8},
        {'text': 'paclitaxel', 'confidence': 0.7},
        {'text': 'mri', 'confidence': 0.85}
    ]
    
    # Extract codes from entities
    mapped_codes = code_extractor.extract_codes_from_entities(entities)
    
    print(f"Extracted {len(mapped_codes)} codes:")
    for mc in mapped_codes:
        print(f"  - {mc['entity']['text']}: {mc['codes']}")
    
    # Test confidence calculation with enhanced features
    print("\nTesting enhanced confidence calculation:")
    test_codes = [
        {'code': 'C50.911', 'validated': True, 'direct_reference': True, 'text': 'confirmed diagnosis'},
        {'code': '12345', 'validated': False, 'ambiguous': True},
        {'code': 'ABC', 'deprecated': True}
    ]
    
    for code_info in test_codes:
        confidence = code_extractor.calculate_code_confidence(code_info)
        print(f"  - Code {code_info['code']}: confidence {confidence:.2f}")

if __name__ == "__main__":
    main()