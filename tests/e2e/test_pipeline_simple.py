#!/usr/bin/env python3
"""
Simple test to verify the pipeline works outside of the test environment
"""

import json
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.utils.logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

def test_pipeline_simple():
    """Test the pipeline with simple clinical trial data"""
    logger.info("🧪 Testing pipeline with simple clinical trial data")
    
    # Sample clinical trial data
    sample_trial_data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Study of Novel Therapy in HER2-Positive Breast Cancer"
            },
            "conditionsModule": {
                "conditions": ["HER2-Positive Breast Cancer", "Metastatic Breast Cancer"]
            },
            "eligibilityModule": {
                "eligibilityCriteria": """
                    INCLUSION CRITERIA:
                    - Histologically confirmed HER2-positive metastatic breast cancer
                    - Measurable disease per RECIST 1.1
                    - ECOG performance status 0-1
                    - Adequate organ function
                    
                    EXCLUSION CRITERIA:
                    - Prior treatment with Trastuzumab Deruxtecan
                    - Active brain metastases
                    - Pregnancy or breastfeeding
                """
            }
        }
    }
    
    try:
        # Initialize pipeline
        pipeline = NlpMcodePipeline()
        logger.info("✅ Pipeline initialized successfully")
        
        # Process the trial data
        result = pipeline.process_clinical_trial(sample_trial_data)
        logger.info("✅ Pipeline processing completed")
        
        # Check if there was an error
        if result.error:
            logger.error(f"❌ Pipeline error: {result.error}")
            return False
        
        # Display results
        logger.info(f"📊 Extracted entities: {len(result.extracted_entities)}")
        logger.info(f"📊 Mapped mCODE elements: {len(result.mcode_mappings)}")
        logger.info(f"📊 Validation valid: {result.validation_results['valid']}")
        logger.info(f"📊 Compliance score: {result.validation_results['compliance_score']}")
        
        # Show some sample entities
        if result.extracted_entities:
            logger.info("\n🔍 Sample extracted entities:")
            for i, entity in enumerate(result.extracted_entities[:3]):
                logger.info(f"  {i+1}. {entity.get('text', 'No text')} ({entity.get('type', 'Unknown type')})")
        
        # Show some sample mappings
        if result.mcode_mappings:
            logger.info("\n🔍 Sample mCODE mappings:")
            for i, mapping in enumerate(result.mcode_mappings[:3]):
                logger.info(f"  {i+1}. {mapping.get('resourceType', 'Unknown')}: {mapping.get('element_name', 'No name')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline_simple()
    if success:
        logger.info("\n✅ Pipeline test completed successfully!")
    else:
        logger.error("\n❌ Pipeline test failed!")
        sys.exit(1)