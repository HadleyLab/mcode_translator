#!/usr/bin/env python3
"""
Test script to validate the improved direct Mcode prompt with breast cancer data.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.mcode_pipeline import McodePipeline
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Define paths
EXAMPLES_DIR = Path("examples/breast_cancer_data")
TEST_CASES_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.trial.json"
GOLD_STANDARD_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.gold.json"

def load_test_data() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load test case and gold standard data"""
    logger.info("Loading test data...")
    
    # Load test cases
    with open(TEST_CASES_FILE, "r") as f:
        test_cases_data = json.load(f)
    
    # Load gold standard
    with open(GOLD_STANDARD_FILE, "r") as f:
        gold_standard_data = json.load(f)
    
    # Extract the breast cancer test case
    test_case = test_cases_data["test_cases"]["breast_cancer_her2_positive"]
    gold_standard = gold_standard_data["gold_standard"]["breast_cancer_her2_positive"]
    
    logger.info("Test data loaded successfully")
    return test_case, gold_standard

def test_improved_direct_mcode_prompt(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test the improved direct Mcode prompt"""
    try:
        logger.info("Testing improved direct Mcode prompt...")
        
        # Initialize pipeline with the improved prompt
        pipeline = McodePipeline(prompt_name="direct_mcode_improved")
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        logger.info("Pipeline execution completed successfully")
        return {
            'success': True,
            'result': result,
            'error': None
        }
    except Exception as e:
        logger.error(f"Pipeline failed with improved prompt: {e}")
        return {
            'success': False,
            'result': None,
            'error': str(e)
        }

def main():
    """Main function to test the improved prompt"""
    logger.info("Starting test of improved direct Mcode prompt...")
    
    try:
        # Load test data
        test_case, gold_standard = load_test_data()
        
        # Test the improved prompt
        result = test_improved_direct_mcode_prompt(test_case)
        
        if result['success']:
            logger.info("✅ Improved direct Mcode prompt test PASSED")
            print("✅ Improved direct Mcode prompt test PASSED")
            
            # Show some details about the result
            mappings = result['result'].mcode_mappings
            logger.info(f"Generated {len(mappings)} Mcode mappings")
            print(f"Generated {len(mappings)} Mcode mappings")
            
            # Show first few mappings as examples
            for i, mapping in enumerate(mappings[:3]):
                print(f"  {i+1}. {mapping.get('Mcode_element', 'Unknown')} - {mapping.get('value', 'No value')}")
            
            # Save result to file for inspection
            output_file = Path("direct_pipeline_output_breast_cancer_her2_positive_improved.json")
            output_data = {
                'mcode_mappings': mappings,
                'metadata': result['result'].metadata,
                'validation_results': result['result'].validation_results
            }
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            logger.info(f"Output saved to {output_file}")
            print(f"Output saved to {output_file}")
            
            return True
        else:
            logger.error("❌ Improved direct Mcode prompt test FAILED")
            print("❌ Improved direct Mcode prompt test FAILED")
            print(f"Error: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Error running test: {e}")
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)