#!/usr/bin/env python3
"""
Comprehensive prompt evaluation script
Tests each prompt by running it through the pipeline and evaluating the results
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.prompt_loader import PromptLoader
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.utils.logging_config import setup_logging

# Setup logging with explicit WARNING level to reduce verbosity
setup_logging(logging.WARNING)
logger = logging.getLogger(__name__)

# Sample clinical text for testing
SAMPLE_CLINICAL_TEXT = """
Patient is a 45-year-old female with HER2-positive metastatic breast cancer.
ECOG performance status is 1.
Tumor is ER-negative, PR-negative, and HER2-positive by IHC.
Patient has no prior history of cardiac disease.
Liver metastases present with largest lesion measuring 2.5 cm.
"""

# Sample gold standard for validation - matched to SAMPLE_CLINICAL_TEXT
SAMPLE_GOLD_STANDARD = {
    "gold_standard": {
        "breast_cancer_her2_positive": {
            "expected_extraction": {
                "entities": [
                    {
                        "text": "45-year-old",
                        "type": "demographic",
                        "attributes": {
                            "age": "45",
                            "unit": "years"
                        },
                        "confidence": 0.95
                    },
                    {
                        "text": "female",
                        "type": "demographic",
                        "attributes": {
                            "gender": "female"
                        },
                        "confidence": 0.95
                    },
                    {
                        "text": "HER2-positive metastatic breast cancer",
                        "type": "condition",
                        "attributes": {
                            "status": "positive",
                            "metastatic": True,
                            "cancer_type": "breast"
                        },
                        "confidence": 0.95
                    },
                    {
                        "text": "ECOG performance status is 1",
                        "type": "assessment",
                        "attributes": {
                            "score": "1",
                            "scale": "ECOG"
                        },
                        "confidence": 0.9
                    },
                    {
                        "text": "ER-negative",
                        "type": "biomarker",
                        "attributes": {
                            "status": "negative",
                            "receptor": "ER"
                        },
                        "confidence": 0.9
                    },
                    {
                        "text": "PR-negative",
                        "type": "biomarker",
                        "attributes": {
                            "status": "negative",
                            "receptor": "PR"
                        },
                        "confidence": 0.9
                    },
                    {
                        "text": "HER2-positive",
                        "type": "biomarker",
                        "attributes": {
                            "status": "positive",
                            "receptor": "HER2"
                        },
                        "confidence": 0.95
                    },
                    {
                        "text": "no prior history of cardiac disease",
                        "type": "exclusion",
                        "attributes": {
                            "status": "absent",
                            "condition": "cardiac disease"
                        },
                        "confidence": 0.85
                    },
                    {
                        "text": "Liver metastases",
                        "type": "condition",
                        "attributes": {
                            "site": "liver",
                            "status": "present",
                            "metastatic": True
                        },
                        "confidence": 0.9
                    },
                    {
                        "text": "largest lesion measuring 2.5 cm",
                        "type": "measurement",
                        "attributes": {
                            "size": "2.5",
                            "unit": "cm",
                            "type": "lesion"
                        },
                        "confidence": 0.85
                    }
                ],
                "relationships": [],
                "metadata": {
                    "extraction_method": "manual_curation",
                    "text_length": len(SAMPLE_CLINICAL_TEXT.strip()),
                    "entity_count": 10
                }
            },
            "expected_mcode_mappings": {
                "mapped_elements": [
                    {
                        "source_entity_index": 0,
                        "Mcode_element": "Patient",
                        "value": "45-year-old",
                        "confidence": 0.9,
                        "mapping_rationale": "Patient age demographic"
                    },
                    {
                        "source_entity_index": 1,
                        "Mcode_element": "Patient",
                        "value": "female",
                        "confidence": 0.95,
                        "mapping_rationale": "Patient gender demographic"
                    },
                    {
                        "source_entity_index": 2,
                        "Mcode_element": "CancerCondition",
                        "value": "HER2-positive metastatic breast cancer",
                        "confidence": 0.95,
                        "mapping_rationale": "Primary cancer diagnosis with biomarker status"
                    },
                    {
                        "source_entity_index": 3,
                        "Mcode_element": "ECOGPerformanceStatus",
                        "value": "1",
                        "confidence": 0.9,
                        "mapping_rationale": "ECOG performance status score"
                    },
                    {
                        "source_entity_index": 4,
                        "Mcode_element": "Observation",
                        "value": "ER-negative",
                        "confidence": 0.9,
                        "mapping_rationale": "Estrogen receptor biomarker status"
                    },
                    {
                        "source_entity_index": 5,
                        "Mcode_element": "Observation",
                        "value": "PR-negative",
                        "confidence": 0.9,
                        "mapping_rationale": "Progesterone receptor biomarker status"
                    },
                    {
                        "source_entity_index": 6,
                        "Mcode_element": "Observation",
                        "value": "HER2-positive",
                        "confidence": 0.95,
                        "mapping_rationale": "HER2 biomarker status"
                    },
                    {
                        "source_entity_index": 8,
                        "Mcode_element": "CancerCondition",
                        "value": "Liver metastases",
                        "confidence": 0.9,
                        "mapping_rationale": "Metastatic cancer site"
                    },
                    {
                        "source_entity_index": 9,
                        "Mcode_element": "Observation",
                        "value": "2.5 cm lesion",
                        "confidence": 0.85,
                        "mapping_rationale": "Tumor measurement observation"
                    }
                ],
                "unmapped_entities": [
                    {
                        "entity_index": 7,
                        "reason": "Exclusion criteria - absence of cardiac disease is not a mCODE oncology element for current condition"
                    }
                ],
                "metadata": {
                    "mapping_method": "manual_curation",
                    "total_entities": 10,
                    "mapped_count": 9,
                    "unmapped_count": 1
                }
            }
        }
    }
}

def test_prompt_with_pipeline(prompt_name: str, prompt_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single prompt by running it through the pipeline
    
    Args:
        prompt_name: Name of the prompt to test
        prompt_config: Prompt configuration
        
    Returns:
        Dictionary with test results
    """
    results = {
        'prompt_name': prompt_name,
        'prompt_type': prompt_config.get('prompt_type', 'unknown'),
        'valid': False,
        'error': None,
        'extraction_results': None,
        'mapping_results': None,
        'entities_count': 0,
        'mappings_count': 0,
        'gold_standard_metrics': None
    }
    
    try:
        logger.info(f"Testing: {prompt_name} ({results['prompt_type']})")
        
        # Initialize pipeline with this prompt
        if results['prompt_type'] == 'NLP_EXTRACTION':
            pipeline = NlpMcodePipeline(extraction_prompt_name=prompt_name)
        elif results['prompt_type'] == 'MCODE_MAPPING':
            pipeline = NlpMcodePipeline(mapping_prompt_name=prompt_name)
        else:
            results['error'] = f"Unknown prompt type: {results['prompt_type']}"
            return results
        
        # Process sample clinical text
        result = pipeline.process_clinical_text(SAMPLE_CLINICAL_TEXT)
        
        # Record results
        results['valid'] = True
        results['extraction_results'] = {
            'entities_count': len(result.extracted_entities),
            'sample_entities': [{'text': e.get('text', ''), 'type': e.get('type', '')} 
                              for e in result.extracted_entities[:3]]
        }
        results['mapping_results'] = {
            'mappings_count': len(result.mcode_mappings),
            'sample_mappings': [{'element_name': m.get('element_name', ''), 
                               'resourceType': m.get('resourceType', '')}
                              for m in result.mcode_mappings[:3]]
        }
        results['entities_count'] = len(result.extracted_entities)
        results['mappings_count'] = len(result.mcode_mappings)
        
        # Calculate gold standard metrics
        results['gold_standard_metrics'] = calculate_gold_standard_metrics(result)
        
        if results['gold_standard_metrics']:
            logger.info(f"  ✅ {results['entities_count']} entities, {results['mappings_count']} mappings, "
                       f"F1={results['gold_standard_metrics']['f1_score']:.3f}")
        else:
            logger.info(f"  ✅ {results['entities_count']} entities, {results['mappings_count']} mappings")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"  ❌ Failed: {e}")
    
    return results

def evaluate_all_prompts():
    """Evaluate all available prompts through the pipeline"""
    
    logger.info("Starting prompt evaluation")
    logger.info("=" * 40)
    
    # Load all prompts
    loader = PromptLoader()
    all_prompts = loader.list_available_prompts()
    
    results = []
    
    # Test each prompt
    for prompt_name, prompt_config in all_prompts.items():
        result = test_prompt_with_pipeline(prompt_name, prompt_config)
        results.append(result)
    
    # Generate summary report
    generate_summary_report(results)
    
    return results

def generate_summary_report(results: List[Dict[str, Any]]):
    """Generate a comprehensive summary report"""
    
    logger.info("\n" + "=" * 40)
    logger.info("SUMMARY REPORT")
    logger.info("=" * 40)
    
    # Count statistics
    total_prompts = len(results)
    successful_prompts = sum(1 for r in results if r['valid'])
    failed_prompts = total_prompts - successful_prompts
    
    extraction_prompts = [r for r in results if r['prompt_type'] == 'NLP_EXTRACTION']
    mapping_prompts = [r for r in results if r['prompt_type'] == 'MCODE_MAPPING']
    
    logger.info(f"Total: {total_prompts}, Success: {successful_prompts}, Failed: {failed_prompts}")
    logger.info(f"Extraction: {len(extraction_prompts)}, Mapping: {len(mapping_prompts)}")
    
    # Gold standard metrics summary
    successful_with_metrics = [r for r in results if r['valid'] and r['gold_standard_metrics']]
    if successful_with_metrics:
        avg_f1 = sum(r['gold_standard_metrics']['f1_score'] for r in successful_with_metrics) / len(successful_with_metrics)
        avg_precision = sum(r['gold_standard_metrics']['precision'] for r in successful_with_metrics) / len(successful_with_metrics)
        avg_recall = sum(r['gold_standard_metrics']['recall'] for r in successful_with_metrics) / len(successful_with_metrics)
        logger.info(f"Avg F1: {avg_f1:.3f}, Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}")
    
    # Only show detailed results for failed prompts or if verbose mode
    failed_results = [r for r in results if not r['valid']]
    if failed_results:
        logger.info("\nFAILED PROMPTS:")
        logger.info("-" * 20)
        for result in failed_results:
            logger.info(f"❌ {result['prompt_name']}: {result['error']}")
    
    # Recommendations - simplified
    working_extraction = [r for r in extraction_prompts if r['valid']]
    working_mapping = [r for r in mapping_prompts if r['valid']]
    
    if working_extraction:
        best_extraction = max(working_extraction, key=lambda x: x.get('gold_standard_metrics', {}).get('f1_score', 0) if x.get('gold_standard_metrics') else 0)
        logger.info(f"\nBest extraction: {best_extraction['prompt_name']} (F1={best_extraction['gold_standard_metrics']['f1_score']:.3f})")
    
    if working_mapping:
        best_mapping = max(working_mapping, key=lambda x: x.get('gold_standard_metrics', {}).get('mapping_f1', 0) if x.get('gold_standard_metrics') else 0)
        logger.info(f"Best mapping: {best_mapping['prompt_name']} (F1={best_mapping['gold_standard_metrics']['mapping_f1']:.3f})")
    
    if not working_extraction:
        logger.info("\n❌ No extraction prompts working")
    
    if not working_mapping:
        logger.info("❌ No mapping prompts working")

def calculate_gold_standard_metrics(pipeline_result) -> Dict[str, float]:
    """Calculate gold standard validation metrics against sample gold standard"""
    try:
        # Use inline gold standard data
        gold_standard = SAMPLE_GOLD_STANDARD['gold_standard']['breast_cancer_her2_positive']
        expected_entities = gold_standard['expected_extraction']['entities']
        expected_mappings = gold_standard['expected_mcode_mappings']['mapped_elements']
        
        # Get actual results from pipeline
        actual_entities = pipeline_result.extracted_entities if hasattr(pipeline_result, 'extracted_entities') else []
        actual_mappings = pipeline_result.mcode_mappings if hasattr(pipeline_result, 'mcode_mappings') else []
        
        # Calculate extraction metrics
        extracted_texts = {e.get('text', '').lower().strip() for e in actual_entities if e.get('text')}
        expected_texts = {e.get('text', '').lower().strip() for e in expected_entities if e.get('text')}
        
        true_positives_ext = len(extracted_texts & expected_texts)
        false_positives_ext = len(extracted_texts - expected_texts)
        false_negatives_ext = len(expected_texts - extracted_texts)
        
        precision_ext = true_positives_ext / (true_positives_ext + false_positives_ext) if (true_positives_ext + false_positives_ext) > 0 else 0
        recall_ext = true_positives_ext / (true_positives_ext + false_negatives_ext) if (true_positives_ext + false_negatives_ext) > 0 else 0
        f1_ext = 2 * (precision_ext * recall_ext) / (precision_ext + recall_ext) if (precision_ext + recall_ext) > 0 else 0
        
        # Calculate mapping metrics
        actual_mcode_elements = {m.get('element_name', '').lower().strip() for m in actual_mappings if m.get('element_name')}
        expected_mcode_elements = {m.get('Mcode_element', '').lower().strip() for m in expected_mappings if m.get('Mcode_element')}
        
        true_positives_map = len(actual_mcode_elements & expected_mcode_elements)
        false_positives_map = len(actual_mcode_elements - expected_mcode_elements)
        false_negatives_map = len(expected_mcode_elements - actual_mcode_elements)
        
        precision_map = true_positives_map / (true_positives_map + false_positives_map) if (true_positives_map + false_positives_map) > 0 else 0
        recall_map = true_positives_map / (true_positives_map + false_negatives_map) if (true_positives_map + false_negatives_map) > 0 else 0
        f1_map = 2 * (precision_map * recall_map) / (precision_map + recall_map) if (precision_map + recall_map) > 0 else 0
        
        # Calculate overall weighted F1 score
        total_expected = len(expected_entities) + len(expected_mappings)
        if total_expected > 0:
            extraction_weight = len(expected_entities) / total_expected
            mapping_weight = len(expected_mappings) / total_expected
            overall_f1 = (f1_ext * extraction_weight) + (f1_map * mapping_weight)
        else:
            overall_f1 = 0
        
        return {
            'precision': precision_ext,
            'recall': recall_ext,
            'f1_score': f1_ext,
            'mapping_precision': precision_map,
            'mapping_recall': recall_map,
            'mapping_f1': f1_map,
            'overall_f1': overall_f1,
            'true_positives_ext': true_positives_ext,
            'false_positives_ext': false_positives_ext,
            'false_negatives_ext': false_negatives_ext,
            'true_positives_map': true_positives_map,
            'false_positives_map': false_positives_map,
            'false_negatives_map': false_negatives_map
        }
        
    except Exception as e:
        logger.warning(f"Gold standard validation failed: {e}")
        return None

def test_prompt_validation_only():
    """Test prompt validation without running full pipeline"""
    
    logger.info("Testing prompt validation")
    logger.info("=" * 30)
    
    loader = PromptLoader()
    all_prompts = loader.list_available_prompts()
    
    for prompt_name, prompt_config in all_prompts.items():
        prompt_type = prompt_config.get('prompt_type', 'unknown')
        
        try:
            # Try to load the prompt (this triggers validation)
            prompt_content = loader.get_prompt(prompt_name)
            logger.info(f"✅ {prompt_name}: PASSED ({len(prompt_content)} chars)")
            
        except Exception as e:
            logger.info(f"❌ {prompt_name}: FAILED - {e}")

if __name__ == "__main__":
    logger.info("Prompt Evaluation - Choose mode:")
    logger.info("1. Full pipeline evaluation")
    logger.info("2. Validation only")
    
    # Check command line argument for mode selection
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        logger.info("\nRunning validation...")
        test_prompt_validation_only()
    else:
        # Run full pipeline evaluation
        logger.info("\nRunning pipeline evaluation...")
        evaluate_all_prompts()
        
        # Skip additional validation to reduce output