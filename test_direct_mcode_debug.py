#!/usr/bin/env python3
"""
Debug script to understand what's happening with the direct mCODE prompt.
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

def test_direct_pipeline():
    """Test the direct pipeline with debugging"""
    logger.info("Testing direct pipeline with debugging...")
    
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
    
    try:
        # Try to process just this section directly
        section_context = {
            'name': section.name,
            'type': section.source_type,
            'position': section.position
        }
        
        # Call the mapper directly with just the clinical text
        mapping_result = pipeline.llm_mapper.map_to_mcode(
            entities=[],  # No entities for direct mapping
            trial_context=test_case,
            source_references=[],
            clinical_text=section.content
        )
        
        logger.info("Direct mapping successful!")
        logger.info(f"Mapped elements: {len(mapping_result['mapped_elements'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Direct mapping failed: {e}")
        logger.error(f"Error type: {type(e)}")
        
        # Let's try to understand what's happening by looking at the prompt
        prompt_template = pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE
        prompt = prompt_template.format(clinical_text=section.content)
        
        logger.info("Formatted prompt:")
        logger.info(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        
        return False

def main():
    """Main function to debug the direct pipeline"""
    logger.info("Starting direct pipeline debug...")
    
    try:
        # Test direct pipeline
        success = test_direct_pipeline()
        
        if success:
            logger.info("Direct pipeline debug completed successfully")
            return True
        else:
            logger.error("Direct pipeline debug failed")
            return False
        
    except Exception as e:
        logger.error(f"Error in debug: {e}")
        logger.exception(e)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)