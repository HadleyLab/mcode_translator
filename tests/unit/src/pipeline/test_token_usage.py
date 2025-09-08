#!/usr/bin/env python3
"""
Test script to verify token usage tracking across different LLM providers
"""

import sys
import os
import json
import time
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.utils.logging_config import get_logger
from src.utils.token_tracker import global_token_tracker

# Set up logging
logger = get_logger(__name__)

def create_test_data():
    """Create test clinical trial data"""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Test Clinical Trial for Token Usage"
            },
            "conditionsModule": {
                "conditions": ["Breast Cancer", "HER2-Positive"]
            },
            "eligibilityModule": {
                "eligibilityCriteria": """
                    INCLUSION CRITERIA:
                    - Histologically confirmed HER2-positive breast cancer
                    - Age ≥ 18 years
                    - ECOG performance status 0-1
                    
                    EXCLUSION CRITERIA:
                    - Prior treatment with Trastuzumab Deruxtecan
                    - Pregnancy or breastfeeding
                """
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "Drug",
                        "name": "Trastuzumab Deruxtecan",
                        "description": "Novel antibody-drug conjugate targeting HER2"
                    }
                ]
            }
        }
    }

def test_token_usage():
    """Test token usage tracking across different pipeline operations"""
    logger.info("🧪 Starting token usage tracking test")
    
    # Reset token tracker
    global_token_tracker.reset()
    
    # Create test data
    test_data = create_test_data()
    
    # Initialize pipeline
    pipeline = NlpMcodePipeline()
    
    # Process the clinical trial
    logger.info("🚀 Processing clinical trial...")
    start_time = time.time()
    result = pipeline.process_clinical_trial(test_data)
    end_time = time.time()
    
    # Get token usage information from the result (this includes aggregate usage)
    extraction_token_usage = result.metadata.get('token_usage', {})
    aggregate_token_usage_dict = result.metadata.get('aggregate_token_usage', {})
    
    # Create TokenUsage object from the result data
    from src.utils.token_tracker import TokenUsage
    aggregate_token_usage = TokenUsage(
        prompt_tokens=aggregate_token_usage_dict.get('prompt_tokens', 0),
        completion_tokens=aggregate_token_usage_dict.get('completion_tokens', 0),
        total_tokens=aggregate_token_usage_dict.get('total_tokens', 0),
        model_name=aggregate_token_usage_dict.get('model_name', ''),
        provider_name=aggregate_token_usage_dict.get('provider_name', '')
    )
    
    # Log results
    logger.info(f"✅ Processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"📊 Extraction token usage: {extraction_token_usage}")
    logger.info(f"📊 Aggregate token usage: {aggregate_token_usage_dict}")
    logger.info(f"📊 Total tokens used: {aggregate_token_usage_dict.get('total_tokens', 0)}")
    logger.info(f"📊 Prompt tokens: {aggregate_token_usage_dict.get('prompt_tokens', 0)}")
    logger.info(f"📊 Completion tokens: {aggregate_token_usage_dict.get('completion_tokens', 0)}")
    
    # Save results
    results_dir = Path("test_token_usage_results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        "run_id": str(int(time.time())),
        "duration_seconds": end_time - start_time,
        "extraction_token_usage": extraction_token_usage,
        "aggregate_token_usage": aggregate_token_usage_dict,
        "entities_extracted": len(result.extracted_entities),
        "mcode_mappings": len(result.mcode_mappings)
    }
    
    results_file = results_dir / f"benchmark_{results['run_id']}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to: {results_file}")
    
    return results

def main():
    """Run token usage tests"""
    logger.info("=" * 60)
    logger.info("Token Usage Tracking Test")
    logger.info("=" * 60)
    
    try:
        results = test_token_usage()
        
        logger.info("\n📋 Test Results Summary:")
        logger.info(f"  Duration: {results['duration_seconds']:.2f} seconds")
        logger.info(f"  Entities extracted: {results['entities_extracted']}")
        logger.info(f"  mCODE mappings: {results['mcode_mappings']}")
        logger.info(f"  Total tokens: {results['aggregate_token_usage'].get('total_tokens', 0)}")
        logger.info(f"  Prompt tokens: {results['aggregate_token_usage'].get('prompt_tokens', 0)}")
        logger.info(f"  Completion tokens: {results['aggregate_token_usage'].get('completion_tokens', 0)}")
        
        logger.info("=" * 60)
        logger.info("✅ Token usage tracking test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Token usage tracking test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()