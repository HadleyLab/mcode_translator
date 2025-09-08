#!/usr/bin/env python3
"""
Simple test script to verify that all prompts work correctly across all pipeline types.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.mcode_pipeline import McodePipeline
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
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

def get_all_prompts_by_type() -> Dict[str, List[str]]:
    """Get all prompts grouped by type"""
    loader = PromptLoader()
    all_prompts = loader.list_available_prompts()
    
    # Group prompts by type
    prompt_groups = {
        'NLP_EXTRACTION': [],
        'MCODE_MAPPING': [],
        'DIRECT_MCODE': []
    }
    
    for prompt_name, prompt_config in all_prompts.items():
        prompt_type = prompt_config.get('prompt_type', 'UNKNOWN')
        if prompt_type in prompt_groups:
            prompt_groups[prompt_type].append(prompt_name)
        else:
            prompt_groups['DIRECT_MCODE'].append(prompt_name)  # Default to DIRECT_MCODE for unknown types
    
    logger.info(f"Found prompts: {prompt_groups}")
    return prompt_groups

def test_nlp_pipeline(extraction_prompt: str, mapping_prompt: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test NLP to mCODE pipeline with specific prompts"""
    try:
        logger.info(f"Testing NLP pipeline: extraction='{extraction_prompt}', mapping='{mapping_prompt}'")
        
        # Initialize pipeline
        pipeline = NlpMcodePipeline(
            extraction_prompt_name=extraction_prompt,
            mapping_prompt_name=mapping_prompt
        )
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        # Count mapped elements
        mapped_count = len(result.mcode_mappings)
        
        logger.info(f"✅ NLP pipeline '{extraction_prompt}' + '{mapping_prompt}' succeeded with {mapped_count} mappings")
        return {
            'success': True,
            'pipeline_type': 'NLP to mCODE',
            'extraction_prompt': extraction_prompt,
            'mapping_prompt': mapping_prompt,
            'mapped_count': mapped_count,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"❌ NLP pipeline '{extraction_prompt}' + '{mapping_prompt}' failed: {e}")
        return {
            'success': False,
            'pipeline_type': 'NLP to mCODE',
            'extraction_prompt': extraction_prompt,
            'mapping_prompt': mapping_prompt,
            'mapped_count': 0,
            'error': str(e)
        }

def test_direct_pipeline(prompt: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Test Direct to mCODE pipeline with specific prompt"""
    try:
        logger.info(f"Testing Direct pipeline: prompt='{prompt}'")
        
        # Initialize pipeline
        pipeline = McodePipeline(prompt_name=prompt)
        
        # Process test case
        result = pipeline.process_clinical_trial(test_case)
        
        # Count mapped elements
        mapped_count = len(result.mcode_mappings)
        
        logger.info(f"✅ Direct pipeline '{prompt}' succeeded with {mapped_count} mappings")
        return {
            'success': True,
            'pipeline_type': 'Direct to mCODE',
            'direct_prompt': prompt,
            'mapped_count': mapped_count,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"❌ Direct pipeline '{prompt}' failed: {e}")
        return {
            'success': False,
            'pipeline_type': 'Direct to mCODE',
            'direct_prompt': prompt,
            'mapped_count': 0,
            'error': str(e)
        }

def main():
    """Main function to test all prompts"""
    logger.info("Starting test of all prompts...")
    
    try:
        # Load test data
        test_case = load_test_data()
        
        # Get all prompts grouped by type
        prompt_groups = get_all_prompts_by_type()
        
        # Test all prompt combinations
        results = []
        
        # Test NLP to mCODE pipeline combinations
        logger.info("Testing NLP to mCODE pipeline combinations...")
        for extraction_prompt in prompt_groups['NLP_EXTRACTION']:
            for mapping_prompt in prompt_groups['MCODE_MAPPING']:
                result = test_nlp_pipeline(extraction_prompt, mapping_prompt, test_case)
                results.append(result)
        
        # Test Direct to mCODE pipeline
        logger.info("Testing Direct to mCODE pipeline...")
        for direct_prompt in prompt_groups['DIRECT_MCODE']:
            result = test_direct_pipeline(direct_prompt, test_case)
            results.append(result)
        
        # Summarize results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total tests: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success rate: {successful/len(results)*100:.1f}%")
        
        if failed > 0:
            logger.info("\nFailed tests:")
            for result in results:
                if not result['success']:
                    if result['pipeline_type'] == 'NLP to mCODE':
                        logger.info(f"  - NLP Pipeline: extraction='{result['extraction_prompt']}', mapping='{result['mapping_prompt']}'")
                    else:
                        logger.info(f"  - Direct Pipeline: prompt='{result['direct_prompt']}'")
                    logger.info(f"    Error: {result['error']}")
        
        logger.info("\nSuccessful tests:")
        for result in results:
            if result['success']:
                if result['pipeline_type'] == 'NLP to mCODE':
                    logger.info(f"  - NLP Pipeline: extraction='{result['extraction_prompt']}', mapping='{result['mapping_prompt']}' -> {result['mapped_count']} mappings")
                else:
                    logger.info(f"  - Direct Pipeline: prompt='{result['direct_prompt']}' -> {result['mapped_count']} mappings")
        
        return successful == len(results)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)