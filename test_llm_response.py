#!/usr/bin/env python3
"""
Test script to understand LLM response parsing issues.
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
from src.pipeline.llm_base import LlmBase

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

def test_llm_call():
    """Test direct LLM call to see the raw response"""
    logger.info("Testing direct LLM call...")
    
    # Load test data
    test_case = load_test_data()
    
    # Initialize pipeline with the improved prompt
    pipeline = McodePipeline(prompt_name="direct_mcode_improved")
    
    # Get a section to test with
    from src.pipeline.document_ingestor import DocumentIngestor
    document_ingestor = DocumentIngestor()
    document_sections = document_ingestor.ingest_clinical_trial_document(test_case)
    
    # Use the first section for testing
    section = document_sections[0]
    logger.info(f"Testing with section: {section.name}")
    logger.info(f"Section content: {section.content}")
    
    # Get the prompt template and format it
    prompt_template = pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE
    prompt = prompt_template.format(clinical_text=section.content)
    
    # Make a direct LLM call
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Call the LLM directly
        response_text, metrics = pipeline.llm_mapper._call_llm_api(messages, {"test": "debug"})
        logger.info(f"LLM response text:\n{response_text}")
        logger.info(f"Response metrics: {metrics}")
        
        # Try to parse the response
        try:
            parsed = pipeline.llm_mapper._parse_and_validate_json_response(response_text)
            logger.info(f"Parsed response: {parsed}")
        except Exception as parse_error:
            logger.error(f"Failed to parse response: {parse_error}")
            # Let's see what the response actually looks like
            logger.info(f"Raw response (first 500 chars): {response_text[:500]}")
            
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return False
    
    return True

def main():
    """Main function to test LLM response"""
    logger.info("Starting LLM response test...")
    
    try:
        # Test LLM call
        success = test_llm_call()
        
        if success:
            logger.info("LLM response test completed successfully")
            return True
        else:
            logger.error("LLM response test failed")
            return False
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)