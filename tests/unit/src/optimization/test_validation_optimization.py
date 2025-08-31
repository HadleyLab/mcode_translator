#!/usr/bin/env python3
"""
Test script for validation and optimization methods with paired gold standard outputs
"""

import sys
import os
import json
import pandas as pd
import time
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.pipeline.strict_dynamic_extraction_pipeline import StrictDynamicExtractionPipeline
from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework, PromptVariant, PromptType, APIConfig
)
from src.utils.logging_config import get_logger
from src.utils.config import Config

# Set up logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# Also configure the framework logger to show debug messages
from src.optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework
framework_logger = get_logger(StrictPromptOptimizationFramework.__name__)
framework_logger.setLevel(logging.DEBUG)

def create_paired_gold_standard():
    """Create paired gold standard outputs for validation testing"""
    
    # Sample clinical trial data
    trial_data = {
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
                    - Age ≥ 18 years
                    - ECOG performance status 0-1
                    - Adequate organ function
                    
                    EXCLUSION CRITERIA:
                    - Prior treatment with Trastuzumab Deruxtecan
                    - Active brain metastases
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
    
    # Expected gold standard output
    gold_standard = {
        "gold_standard": {
            "breast_cancer_her2_positive": {
                "expected_extraction": {
                    "entities": [
                        {
                            "text": "HER2-Positive Breast Cancer",
                            "type": "condition",
                            "attributes": {"status": "positive"},
                            "confidence": 0.95,
                            "source_context": {"section": "conditionsModule"}
                        },
                        {
                            "text": "Metastatic Breast Cancer", 
                            "type": "condition",
                            "attributes": {"status": "metastatic"},
                            "confidence": 0.95,
                            "source_context": {"section": "conditionsModule"}
                        },
                        {
                            "text": "HER2-positive metastatic breast cancer",
                            "type": "condition", 
                            "attributes": {"status": "positive", "metastatic": True},
                            "confidence": 0.9,
                            "source_context": {"section": "eligibilityModule"}
                        },
                        {
                            "text": "Measurable disease",
                            "type": "condition",
                            "attributes": {"status": "present"},
                            "confidence": 0.85,
                            "source_context": {"section": "eligibilityModule"}
                        },
                        {
                            "text": "ECOG performance status 0-1",
                            "type": "demographic",
                            "attributes": {"value": "0-1", "scale": "ECOG"},
                            "confidence": 0.9,
                            "source_context": {"section": "eligibilityModule"}
                        },
                        {
                            "text": "Trastuzumab Deruxtecan",
                            "type": "medication",
                            "attributes": {"type": "antibody-drug conjugate"},
                            "confidence": 0.95,
                            "source_context": {"section": "armsInterventionsModule"}
                        }
                    ],
                    "metadata": {
                        "extraction_method": "llm_based",
                        "text_length": 800,
                        "entity_count": 6
                    }
                },
                "expected_mcode_mappings": {
                    "mapped_elements": [
                        {
                            "mcode_element": "CancerCondition",
                            "value": "HER2-Positive Breast Cancer",
                            "confidence": 0.95,
                            "mapping_rationale": "Primary cancer diagnosis with HER2 biomarker status"
                        },
                        {
                            "mcode_element": "CancerCondition", 
                            "value": "Metastatic Breast Cancer",
                            "confidence": 0.95,
                            "mapping_rationale": "Metastatic cancer condition"
                        },
                        {
                            "mcode_element": "CancerCondition",
                            "value": "HER2-positive metastatic breast cancer",
                            "confidence": 0.9,
                            "mapping_rationale": "Specific cancer diagnosis with biomarker and metastatic status"
                        },
                        {
                            "mcode_element": "CancerDiseaseStatus",
                            "value": "Measurable disease present",
                            "confidence": 0.85,
                            "mapping_rationale": "Disease status indicating measurable disease"
                        },
                        {
                            "mcode_element": "ECOGPerformanceStatus",
                            "value": "0-1",
                            "confidence": 0.9,
                            "mapping_rationale": "ECOG performance status score range"
                        },
                        {
                            "mcode_element": "CancerRelatedMedication",
                            "value": "Trastuzumab Deruxtecan",
                            "confidence": 0.95,
                            "mapping_rationale": "HER2-targeted antibody-drug conjugate medication"
                        }
                    ],
                    "metadata": {
                        "mapping_method": "llm_based",
                        "total_entities": 6,
                        "mapped_count": 6,
                        "unmapped_count": 0
                    }
                }
            }
        }
    }
    
    # Save files
    trial_file = "test_validation_data/trial_data.json"
    gold_file = "test_validation_data/gold_standard.json"
    
    os.makedirs("test_validation_data", exist_ok=True)
    
    with open(trial_file, 'w') as f:
        json.dump({"test_cases": {"breast_cancer_her2_positive": trial_data}}, f, indent=2)
    
    with open(gold_file, 'w') as f:
        json.dump(gold_standard, f, indent=2)
    
    logger.info(f"Created paired gold standard files:")
    logger.info(f"  - Trial data: {trial_file}")
    logger.info(f"  - Gold standard: {gold_file}")
    
    return trial_file, gold_file

def load_breast_cancer_data():
    """Load breast cancer HER2 positive test data from examples directory"""
    trial_file = "examples/breast_cancer_data/breast_cancer_her2_positive.trial.json"
    gold_file = "examples/breast_cancer_data/breast_cancer_her2_positive.gold.json"
    
    # Load test data
    with open(trial_file, 'r') as f:
        trial_data = json.load(f)
    
    with open(gold_file, 'r') as f:
        gold_standard = json.load(f)
    
    logger.info(f"Loaded breast cancer data:")
    logger.info(f"  - Trial data: {trial_file}")
    logger.info(f"  - Gold standard: {gold_file}")
    
    return trial_data, gold_standard

def test_validation_with_gold_standard():
    """Test validation methods using gold standard outputs"""
    logger.info("Testing validation with gold standard...")
    
    # Load breast cancer data
    trial_data, gold_standard = load_breast_cancer_data()
    
    # Initialize pipeline
    pipeline = StrictDynamicExtractionPipeline()
    
    # Process the trial data
    test_case = trial_data['test_cases']['breast_cancer_her2_positive']
    result = pipeline.process_clinical_trial(test_case)
    
    # Save processed results
    os.makedirs("test_results", exist_ok=True)
    
    # Save extraction results
    extraction_file = "test_results/extraction_results.json"
    with open(extraction_file, 'w') as f:
        json.dump({
            "extracted_entities": result.extracted_entities,
            "metadata": {
                "extraction_method": "llm_based",
                "entity_count": len(result.extracted_entities),
                "processing_time": getattr(result, 'processing_time', None)
            }
        }, f, indent=2)
    
    # Save mapping results
    mapping_file = "test_results/mapping_results.json"
    with open(mapping_file, 'w') as f:
        json.dump({
            "mcode_mappings": result.mcode_mappings,
            "metadata": {
                "mapping_method": "llm_based",
                "mapped_count": len(result.mcode_mappings),
                "processing_time": getattr(result, 'processing_time', None)
            }
        }, f, indent=2)
    
    logger.info(f"Saved extraction results to: {extraction_file}")
    logger.info(f"Saved mapping results to: {mapping_file}")
    
    # Get expected results from gold standard
    expected_data = gold_standard['gold_standard']['breast_cancer_her2_positive']
    expected_entities = expected_data['expected_extraction']['entities']
    expected_mappings = expected_data['expected_mcode_mappings']['mapped_elements']
    
    # Perform validation
    logger.info("Performing validation against gold standard...")
    
    # Extraction validation
    extracted_count = len(result.extracted_entities)
    expected_count = len(expected_entities)
    extraction_completeness = extracted_count / expected_count if expected_count > 0 else 0
    
    logger.info(f"Extraction validation:")
    logger.info(f"  - Extracted: {extracted_count} entities")
    logger.info(f"  - Expected: {expected_count} entities")
    logger.info(f"  - Completeness: {extraction_completeness:.2%}")
    
    # Mapping validation
    mapped_count = len(result.mcode_mappings)
    expected_mapped_count = len(expected_mappings)
    mapping_completeness = mapped_count / expected_mapped_count if expected_mapped_count > 0 else 0
    
    logger.info(f"Mapping validation:")
    logger.info(f"  - Mapped: {mapped_count} mCODE elements")
    logger.info(f"  - Expected: {expected_mapped_count} mCODE elements")
    logger.info(f"  - Completeness: {mapping_completeness:.2%}")
    
    # Compliance validation
    compliance_score = result.validation_results.get('compliance_score', 0)
    logger.info(f"Compliance score: {compliance_score:.2%}")
    
    # Calculate precision metrics (simplified)
    extracted_texts = [e.get('text', '') for e in result.extracted_entities]
    expected_texts = [e.get('text', '') for e in expected_entities]
    
    true_positives = len(set(extracted_texts) & set(expected_texts))
    false_positives = len(set(extracted_texts) - set(expected_texts))
    false_negatives = len(set(expected_texts) - set(extracted_texts))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"Precision metrics:")
    logger.info(f"  - Precision: {precision:.2%}")
    logger.info(f"  - Recall: {recall:.2%}")
    logger.info(f"  - F1 Score: {f1_score:.2%}")
    
    # Benchmarking metrics
    processing_time = getattr(result, 'processing_time', None)
    token_usage = getattr(result, 'token_usage', {})
    
    benchmark_metrics = {
        'extraction_completeness': extraction_completeness,
        'mapping_completeness': mapping_completeness,
        'compliance_score': compliance_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'processing_time': processing_time,
        'token_usage': token_usage,
        'extracted_count': extracted_count,
        'mapped_count': mapped_count
    }
    
    # Save benchmark results
    benchmark_file = "test_results/benchmark_results.json"
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_metrics, f, indent=2)
    
    logger.info(f"Saved benchmark results to: {benchmark_file}")
    
    return benchmark_metrics

def test_optimization_framework():
    """Test optimization framework with prompt library integration"""
    logger.info("Testing optimization framework with prompt library...")

    # Initialize framework
    framework = StrictPromptOptimizationFramework(results_dir="./test_optimization_results")

    # API configs are now automatically loaded from configuration
    # The framework automatically adds all models from the configuration
    config = Config()
    
    # Log information about the loaded API configs
    llm_providers = config.get_llm_providers()
    logger.info(f"Loaded {len(llm_providers)} LLM providers from configuration:")
    for provider in llm_providers:
        logger.info(f"  - {provider.get('name')}: {provider.get('model')}")

    # Load prompt library configuration - STRICT MODE
    prompt_config_path = Path("prompts/prompts_config.json")
    if not prompt_config_path.exists():
        raise FileNotFoundError(f"STRICT: Prompt library configuration not found at {prompt_config_path}")

    # Load all prompts from the library
    from src.utils.prompt_loader import PromptLoader
    prompt_loader = PromptLoader(str(prompt_config_path))
    all_prompts = prompt_loader.list_available_prompts()

    # Add prompt variants from the library
    for prompt_name, prompt_config in all_prompts.items():
        # Convert string prompt type to PromptType enum
        prompt_type_str = prompt_config.get('prompt_type', '')
        if prompt_type_str == 'NLP_EXTRACTION':
            prompt_type = PromptType.NLP_EXTRACTION
        elif prompt_type_str == 'MCODE_MAPPING':
            prompt_type = PromptType.MCODE_MAPPING
        else:
            raise ValueError(f"STRICT: Unknown prompt type '{prompt_type_str}' for prompt '{prompt_name}'")
        
        variant = PromptVariant(
            name=prompt_name,
            prompt_type=prompt_type,
            prompt_key=prompt_name,  # Use prompt name as key
            description=prompt_config.get('description', ''),
            version=prompt_config.get('version', '1.0'),
            tags=prompt_config.get('tags', []),
            parameters={}
        )
        
        # STRICT: Fail immediately if prompt variant validation fails
        framework.add_prompt_variant(variant)
        logger.info(f"Added prompt variant: {prompt_name} ({prompt_type_str})")

    # Load test case and gold standard data
    trial_data, gold_standard = load_breast_cancer_data()
    test_case = trial_data['test_cases']['breast_cancer_her2_positive']
    framework.add_test_case("breast_cancer_test", test_case)
    
    # Extract expected entities and mappings from gold standard
    expected_data = gold_standard['gold_standard']['breast_cancer_her2_positive']
    expected_entities = expected_data['expected_extraction']['entities']
    expected_mappings = expected_data['expected_mcode_mappings']['mapped_elements']

    # Define pipeline callback with prompt library integration
    def pipeline_callback(test_data, prompt_content, prompt_variant_id):
        # Get the prompt variant to determine prompt type
        variant = framework.prompt_variants.get(prompt_variant_id)
        if not variant:
            raise ValueError(f"Prompt variant {prompt_variant_id} not found")
        
        # Create pipeline instance
        pipeline = StrictDynamicExtractionPipeline()
        
        # Set the prompt content directly on the NLP engine based on prompt type
        if variant.prompt_type == PromptType.NLP_EXTRACTION:
            # For extraction prompts, set the extraction prompt template
            pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
        elif variant.prompt_type == PromptType.MCODE_MAPPING:
            # For mapping prompts, set the mapping prompt template
            # Note: This would require similar changes to StrictMcodeMapper
            pipeline.llm_mapper.MCODE_MAPPING_PROMPT_TEMPLATE = prompt_content
        else:
            # Default to extraction prompt if type is unknown
            pipeline.nlp_engine.ENTITY_EXTRACTION_PROMPT_TEMPLATE = prompt_content
        
        # Process the test data with the configured pipeline
        return pipeline.process_clinical_trial(test_data)
    
    # Run comprehensive benchmarking for all prompt variants and API configs
    logger.info(f"📊 Starting optimization with {len(framework.prompt_variants)} prompt variants and {len(framework.api_configs)} API configs")
    logger.info(f"   📋 Test Case: breast_cancer_test")
    
    # Use the framework's run_all_combinations method to automatically run benchmarks
    # across all combinations of prompt variants and API configurations
    framework.run_all_combinations(
        test_case_ids=["breast_cancer_test"],
        pipeline_callback=pipeline_callback,
        expected_entities=expected_entities,
        expected_mappings=expected_mappings
    )
    
    # Get the results from the framework
    results = framework.benchmark_results
    
    # Generate comprehensive optimization report
    if results:
        df = framework.get_results_dataframe()
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        logger.info(f"\n📊 Optimization Results Summary:")
        logger.info(f"Total prompt variants tested: {len(results)}")
        logger.info(f"✅ Successful benchmarks: {len(successful_results)}")
        logger.info(f"❌ Failed benchmarks: {len(failed_results)}")
        
        if successful_results:
            # Sort by F1 score (descending)
            successful_results.sort(key=lambda x: x.f1_score, reverse=True)
            
            # Calculate average metrics
            avg_f1 = sum(r.f1_score for r in successful_results) / len(successful_results)
            avg_precision = sum(r.precision for r in successful_results) / len(successful_results)
            avg_recall = sum(r.recall for r in successful_results) / len(successful_results)
            avg_duration = sum(r.duration_ms for r in successful_results) / len(successful_results)
            
            logger.info(f"\n📈 Average Metrics (successful runs):")
            logger.info(f"   F1 Score: {avg_f1:.3f}")
            logger.info(f"   Precision: {avg_precision:.3f}")
            logger.info(f"   Recall: {avg_recall:.3f}")
            logger.info(f"   Duration: {avg_duration:.0f}ms")
            
            logger.info(f"\n🏆 Top Performing Prompts:")
            for i, result in enumerate(successful_results[:5]):  # Show top 5
                variant = framework.prompt_variants.get(result.prompt_variant_id, {})
                model_config = framework.api_configs.get(result.api_config_name, {})
                logger.info(f"{i+1}. Model: {getattr(model_config, 'model', 'Unknown')}, Prompt: {getattr(variant, 'name', 'Unknown')}:")
                logger.info(f"   F1 Score: {result.f1_score:.3f}")
                logger.info(f"   Precision: {result.precision:.3f}")
                logger.info(f"   Recall: {result.recall:.3f}")
                logger.info(f"   Duration: {result.duration_ms:.0f}ms")
                logger.info(f"   Type: {getattr(variant, 'prompt_type', 'Unknown')}")
        
        if failed_results:
            logger.info(f"\n⚠️ Failed Benchmarks:")
            for result in failed_results[:3]:  # Show first 3 failures
                variant = framework.prompt_variants.get(result.prompt_variant_id, {})
                logger.info(f"❌ {getattr(variant, 'name', 'Unknown')}: {result.error_message}")
            if len(failed_results) > 3:
                logger.info(f"   ... and {len(failed_results) - 3} more failures")
        
        # Save detailed results
        os.makedirs("./test_optimization_results", exist_ok=True)
        
        # Save CSV report
        report_file = "./test_optimization_results/optimization_report.csv"
        df.to_csv(report_file, index=False)
        logger.info(f"📋 Optimization report saved to: {report_file}")
        
        # Save JSON results
        json_file = "./test_optimization_results/optimization_results.json"
        results_data = [r.to_dict() for r in results]
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"💾 Detailed results saved to: {json_file}")
        
        # Generate performance comparison
        comparison_file = "./test_optimization_results/performance_comparison.md"
        with open(comparison_file, 'w') as f:
            f.write("# Prompt Performance Comparison\n\n")
            f.write(f"**Total benchmarks:** {len(results)}\n")
            f.write(f"**Successful benchmarks:** {len(successful_results)}\n")
            f.write(f"**Failed benchmarks:** {len(failed_results)}\n\n")
            
            if successful_results:
                f.write("## Average Metrics (Successful Runs)\n\n")
                f.write(f"- **Average F1 Score:** {avg_f1:.3f}\n")
                f.write(f"- **Average Precision:** {avg_precision:.3f}\n")
                f.write(f"- **Average Recall:** {avg_recall:.3f}\n")
                f.write(f"- **Average Duration:** {avg_duration:.0f}ms\n\n")
            
            f.write("## Top Performing Prompts\n\n")
            f.write("| Rank | Prompt Name | Type | F1 Score | Precision | Recall | Duration |\n")
            f.write("|------|-------------|------|----------|-----------|--------|----------|\n")
            
            for i, result in enumerate(successful_results[:10]):
                variant = framework.prompt_variants.get(result.prompt_variant_id, {})
                f.write(f"| {i+1} | {getattr(variant, 'name', 'Unknown')} | {getattr(variant, 'prompt_type', 'Unknown')} | {result.f1_score:.3f} | {result.precision:.3f} | {result.recall:.3f} | {result.duration_ms:.0f}ms |\n")
            
            if failed_results:
                f.write("\n## Failed Benchmarks\n\n")
                f.write("| Prompt Name | Error Message |\n")
                f.write("|-------------|---------------|\n")
                for result in failed_results[:5]:
                    variant = framework.prompt_variants.get(result.prompt_variant_id, {})
                    f.write(f"| {getattr(variant, 'name', 'Unknown')} | {result.error_message[:100]}... |\n")
        
        logger.info(f"📝 Performance comparison saved to: {comparison_file}")
        
        return df
    else:
        raise RuntimeError("STRICT: No benchmark results available - optimization framework failed completely")

def main():
    """Run validation and optimization tests"""
    logger.info("🚀 Starting validation and optimization tests")
    logger.info("=" * 60)
    
    # Test 1: Validation with gold standard
    logger.info("1. Testing validation with gold standard...")
    validation_results = test_validation_with_gold_standard()
    logger.info("-" * 40)
    
    # Test 2: Optimization framework
    logger.info("2. Testing optimization framework...")
    optimization_results = test_optimization_framework()
    logger.info("-" * 40)
    
    # Summary
    logger.info("📊 Test Results Summary:")
    logger.info("Validation metrics:")
    for metric, value in validation_results.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.2%}")
        else:
            logger.info(f"  {metric}: {value}")
    
    if optimization_results is not None:
        logger.info(f"Optimization tests completed: {len(optimization_results)} variants tested")
    else:
        raise RuntimeError("STRICT: Optimization tests failed completely - no results available")
    
    logger.info("=" * 60)
    logger.info("✅ Validation and optimization tests completed!")

if __name__ == "__main__":
    main()