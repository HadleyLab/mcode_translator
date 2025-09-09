#!/usr/bin/env python3
"""
Test script to verify that all fixed direct mCODE prompts work correctly.
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
from src.utils.prompt_loader import PromptLoader

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Define paths
EXAMPLES_DIR = Path("examples/breast_cancer_data")
TEST_CASES_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.trial.json"

def load_test_data() -> Dict[str, Any]:
    """Load test case data"""
    logger.info("Loading test data...")
    
    # Load test cases
    with open(TEST_CASES_FILE, "r") as f:
        test_cases_data = json.load(f)
    
    # Extract the breast cancer test case
    test_case = test_cases_data["test_cases"]["breast_cancer_her2_positive"]
    
    logger.info("Test data loaded successfully")
    return test_case

def get_all_direct_mcode_prompts() -> list:
    """Get all direct mCODE prompt names"""
    loader = PromptLoader()
    all_prompts = loader.list_available_prompts()
    
    # Filter for direct mCODE prompts
    direct_prompts = []
    for prompt_name, prompt_config in all_prompts.items():
        prompt_type = prompt_config.get('prompt_type', '')
        if prompt_type == 'DIRECT_MCODE':
            direct_prompts.append(prompt_name)
    
    logger.info(f"Found {len(direct_prompts)} direct mCODE prompts: {direct_prompts}")
    return direct_prompts

def test_prompt(prompt_name: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test a specific prompt"""
    try:
        logger.info(f"Testing prompt: {prompt_name}")
        
        # Initialize pipeline with the prompt
        pipeline = McodePipeline(prompt_name=prompt_name)
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        # Count mapped elements
        mapped_count = len(result.mcode_mappings)
        
        logger.info(f"✅ Prompt '{prompt_name}' succeeded with {mapped_count} mappings")
        return {
            'success': True,
            'prompt_name': prompt_name,
            'mapped_count': mapped_count,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"❌ Prompt '{prompt_name}' failed: {e}")
        return {
            'success': False,
            'prompt_name': prompt_name,
            'mapped_count': 0,
            'error': str(e)
        }

def main():
    """Main function to test all direct mCODE prompts"""
    logger.info("Starting test of all direct mCODE prompts...")
    
    try:
        # Load test data
        test_case = load_test_data()
        
        # Get all direct mCODE prompts
        prompt_names = get_all_direct_mcode_prompts()
        
        # Test each prompt
        results = []
        for prompt_name in prompt_names:
            result = test_prompt(prompt_name, test_case)
            results.append(result)
        
        # Summarize results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total prompts tested: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
        
        if failed > 0:
            logger.info("\nFailed prompts:")
            for result in results:
                if not result['success']:
                    logger.info(f"  - {result['prompt_name']}: {result['error']}")
        
        logger.info("\nSuccessful prompts:")
        for result in results:
            if result['success']:
                logger.info(f"  - {result['prompt_name']}: {result['mapped_count']} mappings")
        
        return successful == len(results)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)