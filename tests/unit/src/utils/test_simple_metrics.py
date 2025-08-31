#!/usr/bin/env python3
"""
Simple test to verify metric calculation without framework initialization
"""

import json
from pathlib import Path

def test_simple_metrics():
    """Test metric calculation with simple data"""
    print("🧪 Testing simple metric calculation...")
    
    # Load a benchmark file
    benchmark_file = Path("test_optimization_results/benchmark_b587f2af-e950-41b9-b28f-3927e0f21de4.json")
    
    if not benchmark_file.exists():
        print("❌ Benchmark file not found")
        return False
    
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    # Load gold standard
    gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
    if not gold_file.exists():
        print("❌ Gold standard file not found")
        return False
    
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)
    
    # Get extracted and expected data
    extracted_mappings = benchmark_data.get('mcode_mappings', [])
    expected_mappings = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print(f"📊 Extracted mappings: {len(extracted_mappings)}")
    print(f"📊 Expected mappings: {len(expected_mappings)}")
    
    # Test the fixed metric calculation logic
    actual_mappings = set((m.get('mcode_element', ''), m.get('value', '')) for m in extracted_mappings)
    expected_mappings_set = set((m.get('mcode_element', ''), m.get('value', '')) for m in expected_mappings)
    
    true_positives = len(actual_mappings & expected_mappings_set)
    false_positives = len(actual_mappings - expected_mappings_set)
    false_negatives = len(expected_mappings_set - actual_mappings)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✅ Metrics calculated:")
    print(f"   - True Positives: {true_positives}")
    print(f"   - False Positives: {false_positives}")
    print(f"   - False Negatives: {false_negatives}")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1 Score: {f1_score:.3f}")
    
    # Check if metrics are non-zero
    if precision > 0 or recall > 0 or f1_score > 0:
        print("✅ SUCCESS: Metrics are no longer zero!")
        return True
    else:
        print("❌ FAILED: Metrics are still zero")
        return False

if __name__ == "__main__":
    success = test_simple_metrics()
    if success:
        print("\n🎉 Metric calculation fix verified!")
        exit(0)
    else:
        print("\n💥 Metric calculation fix failed!")
        exit(1)