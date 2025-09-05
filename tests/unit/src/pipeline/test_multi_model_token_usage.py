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

from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.utils.logging_config import get_logger
from src.utils.config import Config

# Set up logging
logger = get_logger(__name__)

def create_test_data():
    """Create test clinical trial data"""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Multi-Model Token Usage Test"
            },
            "conditionsModule": {
                "conditions": ["Breast Cancer"]
            },
            "eligibilityModule": {
                "eligibilityCriteria": """
                    INCLUSION CRITERIA:
                    - Histologically confirmed breast cancer
                    - Age ≥ 18 years
                    
                    EXCLUSION CRITERIA:
                    - Pregnancy
                """
            }
        }
    }

def test_model_token_usage(model_name):
    """Test token usage tracking for a specific model"""
    logger.info(f"🧪 Testing token usage tracking for model: {model_name}")
    
    # Create test data
    test_data = create_test_data()
    
    # Initialize pipeline with specific model
    pipeline = StrictDynamicExtractionPipeline()
    
    # Override the model in the pipeline components
    pipeline.nlp_extractor.model_name = model_name
    pipeline.llm_mapper.model_name = model_name
    
    # Process the clinical trial
    logger.info(f"🚀 Processing clinical trial with {model_name}...")
    start_time = time.time()
    result = pipeline.process_clinical_trial(test_data)
    end_time = time.time()
    
    # Get token usage information
    extraction_token_usage = result.metadata.get('token_usage', {})
    aggregate_token_usage = result.metadata.get('aggregate_token_usage', {})
    
    # Log results
    logger.info(f"✅ Processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"📊 Extraction token usage: {extraction_token_usage}")
    logger.info(f"📊 Aggregate token usage: {aggregate_token_usage}")
    logger.info(f"📊 Total tokens used: {aggregate_token_usage.get('total_tokens', 0)}")
    logger.info(f"📊 Prompt tokens: {aggregate_token_usage.get('prompt_tokens', 0)}")
    logger.info(f"📊 Completion tokens: {aggregate_token_usage.get('completion_tokens', 0)}")
    
    return {
        "model_name": model_name,
        "duration_seconds": end_time - start_time,
        "extraction_token_usage": extraction_token_usage,
        "aggregate_token_usage": aggregate_token_usage,
        "entities_extracted": len(result.extracted_entities),
        "mcode_mappings": len(result.mcode_mappings)
    }

def main():
    """Run token usage tests across different models"""
    logger.info("=" * 60)
    logger.info("Multi-Model Token Usage Tracking Test")
    logger.info("=" * 60)
    
    # Get available models from configuration
    config = Config()
    llm_providers = config.get_llm_providers()
    model_names = [provider.get('model') for provider in llm_providers if provider.get('model')]
    
    logger.info(f"📋 Testing token usage tracking for {len(model_names)} models: {model_names}")
    
    results = []
    
    try:
        for model_name in model_names:
            try:
                model_result = test_model_token_usage(model_name)
                results.append(model_result)
                
                logger.info(f"\n📊 Results for {model_name}:")
                logger.info(f"  Duration: {model_result['duration_seconds']:.2f} seconds")
                logger.info(f"  Entities extracted: {model_result['entities_extracted']}")
                logger.info(f"  Mcode mappings: {model_result['mcode_mappings']}")
                logger.info(f"  Total tokens: {model_result['aggregate_token_usage'].get('total_tokens', 0)}")
                logger.info(f"  Prompt tokens: {model_result['aggregate_token_usage'].get('prompt_tokens', 0)}")
                logger.info(f"  Completion tokens: {model_result['aggregate_token_usage'].get('completion_tokens', 0)}")
                
            except Exception as e:
                logger.error(f"❌ Failed to test model {model_name}: {str(e)}")
                results.append({
                    "model_name": model_name,
                    "error": str(e)
                })
        
        # Save results
        results_dir = Path("test_token_usage_results")
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / f"multi_model_benchmark_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        logger.info("\n📋 Overall Test Results Summary:")
        for result in results:
            if 'error' in result:
                logger.info(f"  {result['model_name']}: ❌ Error - {result['error']}")
            else:
                logger.info(f"  {result['model_name']}: ✅ {result['aggregate_token_usage'].get('total_tokens', 0)} tokens")
        
        logger.info("=" * 60)
        logger.info("✅ Multi-model token usage tracking test completed!")
        
    except Exception as e:
        logger.error(f"❌ Multi-model token usage tracking test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()