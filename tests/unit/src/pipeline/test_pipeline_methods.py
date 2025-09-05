#!/usr/bin/env python3
"""
Test script to demonstrate calling methods of StrictDynamicExtractionPipeline
"""

import sys
import os
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)

def test_clinical_text_processing():
    """Test processing individual clinical text"""
    logger.info("Testing clinical text processing...")
    
    # Initialize pipeline
    pipeline = StrictDynamicExtractionPipeline()
    
    # Sample clinical text
    clinical_text = """
    Patient is a 45-year-old female with metastatic breast cancer, HER2 positive status.
    She has received prior treatment with trastuzumab and pertuzumab. 
    ECOG performance status is 1. 
    Recent CT scan shows multiple liver metastases measuring up to 3.5 cm.
    """
    
    try:
        result = pipeline.process_clinical_text(clinical_text)
        logger.info(f"✅ Clinical text processing successful!")
        logger.info(f"Extracted {len(result.extracted_entities)} entities")
        logger.info(f"Mapped {len(result.mcode_mappings)} Mcode elements")
        
        # Show some sample results
        if result.extracted_entities:
            logger.info("Sample extracted entities:")
            for entity in result.extracted_entities[:3]:
                logger.info(f"  - {entity.get('entity_type', 'N/A')}: {entity.get('text', 'N/A')}")
        
        if result.mcode_mappings:
            logger.info("Sample Mcode mappings:")
            for mapping in result.mcode_mappings[:3]:
                logger.info(f"  - {mapping.get('resourceType', 'N/A')}: {mapping.get('element_name', 'N/A')}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ Clinical text processing failed: {str(e)}")
        return False

def test_clinical_trial_processing():
    """Test processing complete clinical trial data"""
    logger.info("Testing clinical trial processing...")
    
    # Initialize pipeline
    pipeline = StrictDynamicExtractionPipeline()
    
    # Sample clinical trial data
    trial_data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Phase II Study of Novel Therapy in HER2+ Breast Cancer"
            },
            "designModule": {
                "designInfo": "Single-arm, open-label phase II study"
            },
            "eligibilityModule": {
                "eligibilityCriteria": """
                    INCLUSION CRITERIA:
                    - Histologically confirmed HER2-positive breast cancer
                    - Measurable disease per RECIST 1.1
                    - Age ≥ 18 years
                    - ECOG performance status 0-1
                    
                    EXCLUSION CRITERIA:
                    - Prior treatment with investigational agent
                    - Uncontrolled brain metastases
                    - Pregnancy or breastfeeding
                """
            },
            "conditionsModule": {
                "conditions": ["Breast Cancer", "HER2-positive Breast Cancer"]
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "interventionType": "Drug",
                        "interventionName": "Novel Anti-HER2 Therapy"
                    }
                ]
            }
        }
    }
    
    try:
        result = pipeline.process_clinical_trial(trial_data)
        logger.info(f"✅ Clinical trial processing successful!")
        logger.info(f"Extracted {len(result.extracted_entities)} entities")
        logger.info(f"Mapped {len(result.mcode_mappings)} Mcode elements")
        logger.info(f"Compliance score: {result.validation_results.get('compliance_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Clinical trial processing failed: {str(e)}")
        return False

def test_eligibility_criteria_processing():
    """Test processing eligibility criteria text"""
    logger.info("Testing eligibility criteria processing...")
    
    # Initialize pipeline
    pipeline = StrictDynamicExtractionPipeline()
    
    # Sample eligibility criteria
    criteria_text = """
    INCLUSION CRITERIA:
    - Histologically or cytologically confirmed diagnosis of cancer
    - Must have at least one measurable lesion per RECIST 1.1
    - Age 18 years or older
    - ECOG performance status of 0 or 1
    - Adequate organ function
    
    EXCLUSION CRITERIA:
    - Prior malignancy within 5 years except adequately treated basal cell carcinoma
    - Uncontrolled intercurrent illness including active infection
    - Pregnancy or lactation
    - Known hypersensitivity to study drug components
    """
    
    try:
        result = pipeline.process_eligibility_criteria(criteria_text)
        logger.info(f"✅ Eligibility criteria processing successful!")
        logger.info(f"Extracted {len(result.extracted_entities)} entities")
        logger.info(f"Mapped {len(result.mcode_mappings)} Mcode elements")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Eligibility criteria processing failed: {str(e)}")
        return False

def test_custom_prompts():
    """Test pipeline with custom prompts"""
    logger.info("Testing pipeline with custom prompts...")
    
    try:
        # Initialize with custom prompts
        pipeline = StrictDynamicExtractionPipeline(
            extraction_prompt_name="nlp_extraction/comprehensive_extraction",
            mapping_prompt_name="Mcode_mapping/comprehensive_mapping"
        )
        
        clinical_text = "Patient with metastatic breast cancer, HER2 positive, ECOG 1"
        result = pipeline.process_clinical_text(clinical_text)
        
        logger.info(f"✅ Custom prompts processing successful!")
        logger.info(f"Extracted {len(result.extracted_entities)} entities")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Custom prompts processing failed: {str(e)}")
        return False

def main():
    """Run all pipeline method tests"""
    logger.info("🚀 Starting StrictDynamicExtractionPipeline method tests")
    logger.info("=" * 60)
    
    results = {}
    
    # Test individual methods
    results['clinical_text'] = test_clinical_text_processing()
    logger.info("-" * 40)
    
    results['clinical_trial'] = test_clinical_trial_processing()
    logger.info("-" * 40)
    
    results['eligibility_criteria'] = test_eligibility_criteria_processing()
    logger.info("-" * 40)
    
    results['custom_prompts'] = test_custom_prompts()
    logger.info("=" * 60)
    
    # Summary
    logger.info("📊 Test Results Summary:")
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.warning("⚠️  Some tests failed. Check logs for details.")

if __name__ == "__main__":
    main()