#!/usr/bin/env python3
"""
Debug script to understand what's happening with the direct Mcode prompt.
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

def test_prompt_formatting():
    """Test the prompt formatting to see what's being sent to the LLM"""
    logger.info("Testing prompt formatting...")
    
    # Load test data
    test_case = load_test_data()
    
    # Initialize pipeline with the improved prompt
    pipeline = McodePipeline(prompt_name="direct_mcode_improved")
    
    # Get the prompt template
    prompt_template = pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE
    logger.info(f"Prompt template:\n{prompt_template}")
    
    # Test with a small section of the data
    from src.pipeline.document_ingestor import DocumentIngestor
    document_ingestor = DocumentIngestor()
    document_sections = document_ingestor.ingest_clinical_trial_document(test_case)
    
    # Use the first section for testing
    section = document_sections[0]
    logger.info(f"Testing with section: {section.name}")
    logger.info(f"Section content: {section.content[:200]}...")
    
    # Format the prompt
    prompt = prompt_template.format(clinical_text=section.content)
    logger.info(f"Formatted prompt (first 500 chars):\n{prompt[:500]}...")
    
    return prompt

def main():
    """Main function to debug the prompt"""
    logger.info("Starting debug of direct Mcode prompt...")
    
    try:
        # Test prompt formatting
        prompt = test_prompt_formatting()
        
        logger.info("Debug completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in debug: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)