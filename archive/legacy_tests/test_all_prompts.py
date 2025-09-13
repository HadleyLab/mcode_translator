#!/usr/bin/env python3
"""
Test script to run all prompts across all pipelines using breast cancer data
and generate a report of failures and performance metrics.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.pipeline.mcode_pipeline import McodePipeline
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.utils.logging_config import setup_logging
from src.utils.prompt_loader import PromptLoader

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Define paths
EXAMPLES_DIR = Path("examples/breast_cancer_data")
TEST_CASES_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.trial.json"
GOLD_STANDARD_FILE = EXAMPLES_DIR / "breast_cancer_her2_positive.gold.json"

def load_test_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load test case and gold standard data"""
    logger.info("Loading test data...")
    
    # Load test cases
    with open(TEST_CASES_FILE, "r") as f:
        test_cases_data = json.load(f)
    
    # Load gold standard
    with open(GOLD_STANDARD_FILE, "r") as f:
        gold_standard_data = json.load(f)
    
    # Extract the breast cancer test case
    test_case = test_cases_data["test_cases"]["breast_cancer_her2_positive"]
    gold_standard = gold_standard_data["gold_standard"]["breast_cancer_her2_positive"]
    
    logger.info("Test data loaded successfully")
    return test_case, gold_standard

def get_all_prompts() -> Dict[str, List[str]]:
    """Get all available prompts grouped by type"""
    loader = PromptLoader()
    all_prompts = loader.list_available_prompts()
    
    # Group prompts by type
    prompt_groups = {
        'NLP_EXTRACTION': [],
        'MCODE_MAPPING': [],
        'DIRECT_MCODE': []
    }
    
    for prompt_name, prompt_config in all_prompts.items():
        prompt_type = prompt_config.get('prompt_type', 'UNKNOWN')
        if prompt_type in prompt_groups:
            prompt_groups[prompt_type].append(prompt_name)
        else:
            prompt_groups['DIRECT_MCODE'].append(prompt_name)  # Default to DIRECT_MCODE for unknown types
    
    logger.info(f"Found prompts: {prompt_groups}")
    return prompt_groups

def run_nlp_to_mcode_pipeline(extraction_prompt: str, mapping_prompt: str, 
                             test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run the NLP to mCODE pipeline with specific prompts"""
    try:
        logger.info(f"Running NLP to mCODE pipeline with extraction='{extraction_prompt}', mapping='{mapping_prompt}'")
        
        # Initialize pipeline
        pipeline = NlpMcodePipeline(
            extraction_prompt_name=extraction_prompt,
            mapping_prompt_name=mapping_prompt
        )
        
        # Process test case
        start_time = time.time()
        result = pipeline.process_clinical_trial(test_case)
        end_time = time.time()
        
        return {
            'success': True,
            'result': result,
            'execution_time': end_time - start_time,
            'error': None
        }
    except Exception as e:
        logger.error(f"Pipeline failed with extraction='{extraction_prompt}', mapping='{mapping_prompt}': {e}")
        return {
            'success': False,
            'result': None,
            'execution_time': 0,
            'error': str(e)
        }

def run_direct_mcode_pipeline(prompt: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Direct to mCODE pipeline with specific prompt"""
    try:
        logger.info(f"Running Direct to mCODE pipeline with prompt='{prompt}'")
        
        # Initialize pipeline
        pipeline = McodePipeline(prompt_name=prompt)
        
        # Process test case
        start_time = time.time()
        result = pipeline.process_clinical_trial(test_case)
        end_time = time.time()
        
        return {
            'success': True,
            'result': result,
            'execution_time': end_time - start_time,
            'error': None
        }
    except Exception as e:
        logger.error(f"Direct pipeline failed with prompt='{prompt}': {e}")
        return {
            'success': False,
            'result': None,
            'execution_time': 0,
            'error': str(e)
        }

def calculate_metrics(result: Any, gold_standard: Dict[str, Any]) -> Dict[str, float]:
    """Calculate precision, recall, and F1-score against gold standard"""
    try:
        # Extract mappings from result
        if hasattr(result, 'mcode_mappings'):
            actual_mappings = result.mcode_mappings
        elif isinstance(result, dict) and 'mcode_mappings' in result:
            actual_mappings = result['mcode_mappings']
        else:
            actual_mappings = []
        
        # Extract expected mappings from gold standard
        expected_mappings = gold_standard['expected_mcode_mappings']['mapped_elements']
        
        # Create simplified representations for comparison
        def simplify_mapping(mapping):
            """Create a simplified representation of a mapping for comparison"""
            # Focus on key identifying elements
            key_parts = []
            
            # Include resource type if available
            if mapping.get('resourceType'):
                key_parts.append(f"resourceType:{mapping['resourceType']}")
            
            # Include element name if available
            if mapping.get('element_name'):
                key_parts.append(f"element_name:{mapping['element_name']}")
            
            # Include mCODE element if available (from gold standard)
            if mapping.get('mcode_element'):
                key_parts.append(f"mcode_element:{mapping['mcode_element']}")
            
            # Include code if available
            if mapping.get('code'):
                code = mapping['code']
                if isinstance(code, dict):
                    if code.get('coding'):
                        for coding in code['coding']:
                            if coding.get('code'):
                                key_parts.append(f"code:{coding['code']}")
                            if coding.get('system'):
                                key_parts.append(f"system:{coding['system']}")
                elif isinstance(code, str):
                    key_parts.append(f"code:{code}")
            
            # Sort and join to create a consistent representation
            key_parts.sort()
            return tuple(key_parts)
        
        # Convert to sets of simplified representations
        actual_set = {simplify_mapping(m) for m in actual_mappings}
        expected_set = {simplify_mapping(m) for m in expected_mappings}
        
        # Calculate metrics
        true_positives = len(actual_set.intersection(expected_set))
        false_positives = len(actual_set - expected_set)
        false_negatives = len(expected_set - actual_set)
        
        # Calculate precision, recall, F1-score
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_actual': len(actual_mappings),
            'total_expected': len(expected_mappings)
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_actual': 0,
            'total_expected': 0
        }

def test_all_prompt_combinations(test_case: Dict[str, Any], gold_standard: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Test all prompt combinations across all pipelines"""
    prompt_groups = get_all_prompts()
    results = []
    
    # Test NLP to mCODE pipeline combinations
    logger.info("Testing NLP to mCODE pipeline combinations...")
    for extraction_prompt in prompt_groups['NLP_EXTRACTION']:
        for mapping_prompt in prompt_groups['MCODE_MAPPING']:
            # Run pipeline
            pipeline_result = run_nlp_to_mcode_pipeline(extraction_prompt, mapping_prompt, test_case)
            
            # Calculate metrics if successful
            metrics = {}
            if pipeline_result['success']:
                metrics = calculate_metrics(pipeline_result['result'], gold_standard)
            
            # Store result
            results.append({
                'pipeline_type': 'NLP to mCODE',
                'extraction_prompt': extraction_prompt,
                'mapping_prompt': mapping_prompt,
                'direct_prompt': None,
                'success': pipeline_result['success'],
                'execution_time': pipeline_result['execution_time'],
                'error': pipeline_result['error'],
                'metrics': metrics
            })
    
    # Test Direct to mCODE pipeline
    logger.info("Testing Direct to mCODE pipeline...")
    for direct_prompt in prompt_groups['DIRECT_MCODE']:
        # Run pipeline
        pipeline_result = run_direct_mcode_pipeline(direct_prompt, test_case)
        
        # Calculate metrics if successful
        metrics = {}
        if pipeline_result['success']:
            metrics = calculate_metrics(pipeline_result['result'], gold_standard)
        
        # Store result
        results.append({
            'pipeline_type': 'Direct to mCODE',
            'extraction_prompt': None,
            'mapping_prompt': None,
            'direct_prompt': direct_prompt,
            'success': pipeline_result['success'],
            'execution_time': pipeline_result['execution_time'],
            'error': pipeline_result['error'],
            'metrics': metrics
        })
    
    return results

def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive report of test results"""
    report = []
    report.append("# Prompt Pipeline Test Report")
    report.append("")
    report.append("## Summary")
    report.append("")
    
    # Calculate summary statistics
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - successful_tests
    
    report.append(f"Total tests: {total_tests}")
    report.append(f"Successful tests: {successful_tests}")
    report.append(f"Failed tests: {failed_tests}")
    report.append(f"Success rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "Success rate: 0%")
    report.append("")
    
    # Group results by pipeline type
    nlp_results = [r for r in results if r['pipeline_type'] == 'NLP to mCODE']
    direct_results = [r for r in results if r['pipeline_type'] == 'Direct to mCODE']
    
    report.append(f"NLP to mCODE tests: {len(nlp_results)}")
    report.append(f"Direct to mCODE tests: {len(direct_results)}")
    report.append("")
    
    # Show failed tests
    failed_results = [r for r in results if not r['success']]
    if failed_results:
        report.append("## Failed Tests")
        report.append("")
        for result in failed_results:
            if result['pipeline_type'] == 'NLP to mCODE':
                report.append(f"- **NLP Pipeline**: Extraction='{result['extraction_prompt']}', Mapping='{result['mapping_prompt']}'")
            else:
                report.append(f"- **Direct Pipeline**: Prompt='{result['direct_prompt']}'")
            report.append(f"  - Error: {result['error']}")
            report.append("")
    
    # Show performance metrics for successful tests
    successful_results = [r for r in results if r['success']]
    if successful_results:
        report.append("## Successful Tests Performance")
        report.append("")
        
        # Sort by F1 score
        successful_results.sort(key=lambda x: x['metrics'].get('f1_score', 0), reverse=True)
        
        report.append("| Pipeline Type | Prompt(s) | F1 Score | Precision | Recall | Execution Time (s) |")
        report.append("|---------------|-----------|----------|-----------|--------|-------------------|")
        
        for result in successful_results:
            metrics = result['metrics']
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            exec_time = result['execution_time']
            
            if result['pipeline_type'] == 'NLP to mCODE':
                prompt_info = f"Ext:{result['extraction_prompt']} / Map:{result['mapping_prompt']}"
            else:
                prompt_info = f"Direct:{result['direct_prompt']}"
            
            report.append(f"| {result['pipeline_type']} | {prompt_info} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {exec_time:.2f} |")
    
    # Show best performers
    if successful_results:
        report.append("")
        report.append("## Best Performers (by F1 Score)")
        report.append("")
        
        # Get top 5 performers
        top_performers = successful_results[:5]
        for i, result in enumerate(top_performers, 1):
            metrics = result['metrics']
            f1 = metrics.get('f1_score', 0)
            
            if result['pipeline_type'] == 'NLP to mCODE':
                report.append(f"{i}. **NLP Pipeline**: Extraction='{result['extraction_prompt']}', Mapping='{result['mapping_prompt']}' (F1: {f1:.3f})")
            else:
                report.append(f"{i}. **Direct Pipeline**: Prompt='{result['direct_prompt']}' (F1: {f1:.3f})")
    
    return "\n".join(report)

def main():
    """Main function to run all tests and generate report"""
    logger.info("Starting prompt pipeline testing...")
    
    try:
        # Load test data
        test_case, gold_standard = load_test_data()
        
        # Test all prompt combinations
        results = test_all_prompt_combinations(test_case, gold_standard)
        
        # Generate report
        report = generate_report(results)
        
        # Save report to file
        report_file = Path("prompt_pipeline_test_report.md")
        with open(report_file, "w") as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")
        print(report)
        
        # Also save detailed results as JSON
        results_file = Path("prompt_pipeline_test_results.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)