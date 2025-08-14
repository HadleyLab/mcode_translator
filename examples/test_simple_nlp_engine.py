#!/usr/bin/env python3
"""
Test script for the Simple NLP Engine
"""

import sys
import os
import json

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simple_nlp_engine import SimpleNLPEngine


def test_simple_nlp_engine():
    """Test the Simple NLP Engine with sample criteria"""
    print("Testing Simple NLP Engine...")
    
    # Initialize the NLP engine
    try:
        nlp_engine = SimpleNLPEngine()
        print("Simple NLP Engine initialized successfully")
    except Exception as e:
        print(f"Error initializing Simple NLP Engine: {str(e)}")
        return None
    
    # Sample criteria text
    sample_text = """
    Inclusion Criteria:
    - Male or female patients aged 18 years or older
    - Histologically confirmed diagnosis of breast cancer
    - Must have received prior chemotherapy treatment
    - Currently receiving radiation therapy
    - Laboratory values within normal limits
    
    Exclusion Criteria:
    - Pregnant or nursing women
    - History of other malignancies within the past 5 years
    - Allergy to contrast agents
    - Unable to undergo MRI scanning
    """
    
    print("\nProcessing sample criteria text...")
    
    try:
        # Process the sample text
        result = nlp_engine.process_criteria(sample_text)
        
        print("Processing complete. Results:")
        print(f"- Text length: {result['metadata']['text_length']} characters")
        print(f"- Conditions found: {result['metadata']['condition_count']}")
        print(f"- Procedures found: {result['metadata']['procedure_count']}")
        print(f"- Age criteria: {len(result['demographics']['age'])}")
        print(f"- Gender criteria: {len(result['demographics']['gender'])}")
        
        # Show some extracted information
        print("\nSample extracted conditions:")
        for i, condition in enumerate(result['conditions'][:3]):  # Show first 3 conditions
            print(f"  {i+1}. {condition['text']} (Condition: {condition['condition']})")
        
        # Show demographic information
        if result['demographics']['age']:
            print("\nAge criteria found:")
            for age in result['demographics']['age']:
                print(f"  - {age['text']}")
        
        if result['demographics']['gender']:
            print("\nGender criteria found:")
            for gender in result['demographics']['gender']:
                print(f"  - {gender['text']} ({gender['gender']})")
        
        return result
    except Exception as e:
        print(f"Error processing criteria: {str(e)}")
        return None


def main():
    """Main test function"""
    print("Running Simple NLP Engine Tests")
    print("=" * 50)
    
    # Test NLP engine functionality
    result = test_simple_nlp_engine()
    
    print("\nTests completed.")
    
    if result:
        # Save results to file for review
        output_file = "simple_nlp_test_results.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nDetailed results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")


if __name__ == '__main__':
    main()