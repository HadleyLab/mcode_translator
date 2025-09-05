#!/usr/bin/env python3
import sys
import os
import json
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.optimization.prompt_optimization_framework import (
    PromptOptimizationFramework, PromptVariant, PromptType, APIConfig
)
from src.utils.logging_config import get_logger
from src.utils.config import Config

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# Also configure the framework logger to show debug messages
framework_logger = get_logger(PromptOptimizationFramework.__name__)
framework_logger.setLevel(logging.DEBUG)

def main():
    """Run a single benchmark to validate token usage"""
    logger.info("🚀 Starting token usage validation test")
    
    # Initialize framework
    framework = PromptOptimizationFramework(results_dir="./test_token_usage_results")

    # Add a single API config
    config = Config()
    api_config = APIConfig(
        name="default_api",
        model=config.get_model_name()
    )
    framework.add_api_config(api_config)

    # Add a single prompt variant
    variant = PromptVariant(
        name="minimal_extraction",
        prompt_type=PromptType.NLP_EXTRACTION,
        prompt_key="minimal_extraction",
    )
    framework.add_prompt_variant(variant)

    # Inline test data and gold standard
    trial_data = {
        "test_cases": {
            "token_test": {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT00000000",
                        "briefTitle": "Token Test"
                    },
                    "eligibilityModule": {
                        "eligibilityCriteria": "HER2-Positive Breast Cancer"
                    }
                }
            }
        }
    }

    gold_standard = {
        "gold_standard": {
            "token_test": {
                "expected_extraction": {
                    "entities": [
                        {
                            "text": "HER2-Positive Breast Cancer",
                            "type": "condition"
                        }
                    ]
                },
                "expected_mcode_mappings": {
                    "mapped_elements": [
                        {
                            "Mcode_element": "CancerCondition",
                            "value": "HER2-Positive Breast Cancer"
                        }
                    ]
                }
            }
        }
    }
    
    test_case = trial_data['test_cases']['token_test']
    framework.add_test_case("token_test", test_case)
    
    expected_data = gold_standard['gold_standard']['token_test']
    expected_entities = expected_data['expected_extraction']['entities']
    expected_mappings = expected_data['expected_mcode_mappings']['mapped_elements']

    # Define pipeline callback
    def pipeline_callback(test_data, prompt_content, prompt_variant_id):
        pipeline = StrictDynamicExtractionPipeline()
        variant = framework.prompt_variants.get(prompt_variant_id)
        if variant.prompt_type == PromptType.NLP_EXTRACTION:
            pipeline.nlp_extractor.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
        elif variant.prompt_type == PromptType.MCODE_MAPPING:
            pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
        return pipeline.process_clinical_trial(test_data)

    # Run the benchmark
    result = framework.run_benchmark(
        prompt_variant_id=variant.id,
        api_config_name="default_api",
        test_case_id="token_test",
        pipeline_callback=pipeline_callback,
        expected_entities=expected_entities,
        expected_mappings=expected_mappings
    )

    logger.info(f"📊 Token Usage: {result.token_usage}")
    
    if result.token_usage > 0:
        logger.info("✅ Token usage validation successful!")
    else:
        logger.error("❌ Token usage validation failed!")

if __name__ == "__main__":
    main()