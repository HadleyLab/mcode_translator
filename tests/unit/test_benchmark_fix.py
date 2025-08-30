#!/usr/bin/env python3
"""
Test script to verify the benchmark metric calculation fix
"""

import json
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optimization.strict_prompt_optimization_framework import (
    StrictPromptOptimizationFramework, BenchmarkResult
)

def test_benchmark_fix():
    """Test that the benchmark metric calculation fix works"""
    print("🧪 Testing benchmark metric calculation fix...")
    
    # Load one of the existing benchmark results
    benchmark_dir = Path("test_optimization_results")
    benchmark_files = list(benchmark_dir.glob("benchmark_*.json"))
    
    if not benchmark_files:
        print("❌ No benchmark files found to test")
        return False
    
    # Load the first benchmark file
    benchmark_file = benchmark_files[0]
    print(f"📊 Testing with benchmark file: {benchmark_file.name}")
    
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    # Create a BenchmarkResult object and manually calculate metrics
    result = BenchmarkResult()
    result.run_id = benchmark_data.get('run_id', '')
    result.prompt_variant_id = benchmark_data.get('prompt_variant_id', '')
    result.api_config_name = benchmark_data.get('api_config_name', '')
    result.test_case_id = benchmark_data.get('test_case_id', '')
    result.success = benchmark_data.get('success', False)
    result.extracted_entities = benchmark_data.get('extracted_entities', [])
    result.mcode_mappings = benchmark_data.get('mcode_mappings', [])
    result.validation_results = benchmark_data.get('validation_results', {})
    
    # Load expected data from gold standard
    gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
    if gold_file.exists():
        with open(gold_file, 'r') as f:
            gold_data = json.load(f)
        
        expected_data = gold_data['gold_standard']['breast_cancer_her2_positive']
        expected_entities = expected_data['expected_extraction']['entities']
        expected_mappings = expected_data['expected_mcode_mappings']['mapped_elements']
        
        # Create framework instance for metric calculation
        framework = StrictPromptOptimizationFramework()
        
        # Calculate metrics using the fixed method
        result.calculate_metrics(expected_entities, expected_mappings, framework)
        
        print(f"✅ Metrics calculated successfully:")
        print(f"   - Precision: {result.precision:.3f}")
        print(f"   - Recall: {result.recall:.3f}")
        print(f"   - F1 Score: {result.f1_score:.3f}")
        print(f"   - Extraction Completeness: {result.extraction_completeness:.3f}")
        print(f"   - Mapping Accuracy: {result.mapping_accuracy:.3f}")
        
        # Verify metrics are not zero
        if result.precision > 0 or result.recall > 0 or result.f1_score > 0:
            print("✅ SUCCESS: Metrics are no longer zero!")
            return True
        else:
            print("❌ FAILED: Metrics are still zero")
            return False
    else:
        print("❌ Gold standard file not found")
        return False

if __name__ == "__main__":
    success = test_benchmark_fix()
    if success:
        print("\n🎉 Benchmark metric calculation fix verified!")
        sys.exit(0)
    else:
        print("\n💥 Benchmark metric calculation fix failed!")
        sys.exit(1)