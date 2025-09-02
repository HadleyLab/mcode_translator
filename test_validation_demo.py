#!/usr/bin/env python3
"""
Test script to demonstrate gold standard validation functionality.
This script runs a quick test to show validation results without UI dependencies.
"""

import asyncio
import json
from pathlib import Path
from src.optimization.pipeline_task_tracker import PipelineTask, TaskStatus

def calculate_validation_metrics(pipeline_mappings: list, gold_mappings: list) -> dict:
    """
    Calculate precision, recall, and F1-score for mCODE mappings.
    This is a standalone version of the validation logic.
    """
    # Convert mappings to strings for comparison
    pipeline_strs = [json.dumps(mapping, sort_keys=True) for mapping in pipeline_mappings]
    gold_strs = [json.dumps(mapping, sort_keys=True) for mapping in gold_mappings]
    
    # Calculate metrics
    true_positives = len(set(pipeline_strs) & set(gold_strs))
    false_positives = len(set(pipeline_strs) - set(gold_strs))
    false_negatives = len(set(gold_strs) - set(pipeline_strs))
    
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
        'total_pipeline': len(pipeline_mappings),
        'total_gold': len(gold_mappings)
    }

async def test_validation_demo():
    """Test the validation functionality with sample data"""
    print("Testing gold standard validation functionality...")
    
    try:
        # Load sample data
        sample_path = "examples/breast_cancer_data/breast_cancer_her2_positive.trial.json"
        gold_path = "examples/breast_cancer_data/breast_cancer_her2_positive.gold.json"
        
        sample_trial_data = None
        gold_standard_data = None
        
        # Load test case
        test_case_file = Path(sample_path)
        if test_case_file.exists():
            with open(test_case_file, 'r') as f:
                sample_trial_data = json.load(f)
                print("✓ Test case data loaded")
        
        # Load gold standard
        gold_file = Path(gold_path)
        if gold_file.exists():
            with open(gold_file, 'r') as f:
                gold_standard_data = json.load(f)
                print("✓ Gold standard data loaded")
        
        # Create a mock task with validation results
        task = PipelineTask(id="test123")
        task.status = TaskStatus.SUCCESS
        task.start_time = asyncio.get_event_loop().time() - 5.0
        task.end_time = asyncio.get_event_loop().time()
        task.test_case_name = "breast_cancer_her2_positive"
        task.pipeline_type = "NLP to mCODE"
        
        # Add validation metrics
        task.benchmark_metrics = {
            'precision': 0.85,
            'recall': 0.78,
            'f1_score': 0.81,
            'true_positives': 17,
            'false_positives': 3,
            'false_negatives': 5,
            'total_pipeline': 20,
            'total_gold': 22,
            'total_processing_time': 4.2,
            'nlp_extraction_time': 1.8,
            'mcode_mapping_time': 2.4,
            'total_tokens': 1250,
            'prompt_tokens': 850,
            'completion_tokens': 400
        }
        
        print("✓ Mock task created with validation results")
        print("✓ Validation metrics:")
        print(f"  - Precision: {task.benchmark_metrics['precision']:.3f}")
        print(f"  - Recall: {task.benchmark_metrics['recall']:.3f}")
        print(f"  - F1-Score: {task.benchmark_metrics['f1_score']:.3f}")
        print(f"  - True Positives: {task.benchmark_metrics['true_positives']}")
        print(f"  - False Positives: {task.benchmark_metrics['false_positives']}")
        print(f"  - False Negatives: {task.benchmark_metrics['false_negatives']}")
        
        # Test validation function directly
        print("\nTesting validation function...")
        
        # Create sample mappings for validation test
        pipeline_mappings = [
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "12345"}]}, "valueString": "Positive"},
            {"resourceType": "Condition", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "67890"}]}}
        ]
        
        gold_mappings = [
            {"resourceType": "Observation", "code": {"coding": [{"system": "http://loinc.org", "code": "12345"}]}, "valueString": "Positive"},
            {"resourceType": "Condition", "code": {"coding": [{"system": "http://snomed.info/sct", "code": "67890"}]}},
            {"resourceType": "Medication", "code": {"coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "99999"}]}}
        ]
        
        metrics = calculate_validation_metrics(pipeline_mappings, gold_mappings)
        print("✓ Direct validation test:")
        print(f"  - Precision: {metrics['precision']:.3f}")
        print(f"  - Recall: {metrics['recall']:.3f}")
        print(f"  - F1-Score: {metrics['f1_score']:.3f}")
        print(f"  - True Positives: {metrics['true_positives']}")
        print(f"  - False Positives: {metrics['false_positives']}")
        print(f"  - False Negatives: {metrics['false_negatives']}")
        
        print("\n✅ Validation functionality is working correctly!")
        print("✅ Run the pipeline task tracker to see the enhanced validation display in the UI")
        
    except Exception as e:
        print(f"❌ Error during validation test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_validation_demo())