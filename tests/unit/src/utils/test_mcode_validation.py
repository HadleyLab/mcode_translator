#!/usr/bin/env python3
"""
Test script to verify mCODE-based validation is working correctly
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework

def test_mcode_validation():
    """Test the mCODE-based validation with existing benchmark results"""
    
    # Initialize the framework
    framework = StrictPromptOptimizationFramework(results_dir="./breast_cancer_optimization_results_fixed")
    
    # Load the gold standard test case
    gold_standard_path = "examples/breast_cancer_data/breast_cancer_her2_positive.gold.json"
    with open(gold_standard_path, 'r') as f:
        gold_standard = json.load(f)
    
    # Add the test case to the framework
    test_case_data = gold_standard['gold_standard']['breast_cancer_her2_positive']
    framework.add_test_case("breast_cancer_her2_positive", test_case_data)
    
    # Load a specific benchmark result file
    benchmark_file = "breast_cancer_optimization_results_fixed/benchmark_3edebd02-e5df-472f-9332-750a7735d98b.json"
    
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    print(f"Loaded benchmark result: {benchmark_data.get('run_id', 'N/A')}")
    
    # Test the mCODE conversion function
    expected_entities = test_case_data['expected_extraction']['entities']
    expected_mappings = test_case_data['expected_mcode_mappings']['mapped_elements']
    
    print(f"Expected entities: {len(expected_entities)}")
    print(f"Expected mCODE mappings: {len(expected_mappings)}")
    
    # Test converting entities to mCODE format
    converted_mcode = framework._convert_entities_to_mcode(expected_entities)
    print(f"Converted entities to mCODE: {len(converted_mcode)} elements")
    
    for i, mapping in enumerate(converted_mcode[:3]):
        print(f"  {i+1}. {mapping.get('element_name', 'N/A')} -> {mapping.get('resourceType', 'N/A')}")
    
    # Test with the benchmark result data
    extracted_entities = benchmark_data.get('extracted_entities', [])
    mcode_mappings = benchmark_data.get('mcode_mappings', [])
    
    print(f"\nBenchmark extracted entities: {len(extracted_entities)}")
    print(f"Benchmark mCODE mappings: {len(mcode_mappings)}")
    
    # Convert benchmark entities to mCODE format
    benchmark_mcode = framework._convert_entities_to_mcode(extracted_entities)
    print(f"Benchmark entities converted to mCODE: {len(benchmark_mcode)} elements")
    
    # Calculate metrics manually
    true_positives_ext = len(set(m.get('element_name', '') for m in benchmark_mcode) &
                           set(m.get('element_name', '') for m in converted_mcode))
    false_positives_ext = len(set(m.get('element_name', '') for m in benchmark_mcode) -
                            set(m.get('element_name', '') for m in converted_mcode))
    false_negatives_ext = len(set(m.get('element_name', '') for m in converted_mcode) -
                            set(m.get('element_name', '') for m in benchmark_mcode))
    
    precision = true_positives_ext / (true_positives_ext + false_positives_ext) if (true_positives_ext + false_positives_ext) > 0 else 0
    recall = true_positives_ext / (true_positives_ext + false_negatives_ext) if (true_positives_ext + false_negatives_ext) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    
    return precision > 0 or recall > 0 or f1_score > 0

if __name__ == "__main__":
    success = test_mcode_validation()
    if success:
        print("\n✅ mCODE-based validation test PASSED!")
    else:
        print("\n❌ mCODE-based validation test FAILED!")