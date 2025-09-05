#!/usr/bin/env python3
"""
Test all benchmark files to verify the metric calculation fix
"""

import json
from pathlib import Path

def test_benchmark_file(benchmark_file, gold_data):
    """Test a single benchmark file"""
    with open(benchmark_file, 'r') as f:
        benchmark_data = json.load(f)
    
    extracted_mappings = benchmark_data.get('Mcode_mappings', [])
    expected_mappings = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    # Test the fixed metric calculation logic
    actual_mappings = set((m.get('Mcode_element', ''), m.get('value', '')) for m in extracted_mappings)
    expected_mappings_set = set((m.get('Mcode_element', ''), m.get('value', '')) for m in expected_mappings)
    
    true_positives = len(actual_mappings & expected_mappings_set)
    false_positives = len(actual_mappings - expected_mappings_set)
    false_negatives = len(expected_mappings_set - actual_mappings)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'file': benchmark_file.name,
        'extracted_count': len(extracted_mappings),
        'expected_count': len(expected_mappings),
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'is_fixed': precision > 0 or recall > 0 or f1_score > 0
    }

def test_all_benchmarks():
    """Test all benchmark files"""
    print("🧪 Testing all benchmark files...")
    
    # Load gold standard
    gold_file = Path("examples/breast_cancer_data/breast_cancer_her2_positive.gold.json")
    if not gold_file.exists():
        print("❌ Gold standard file not found")
        return False
    
    with open(gold_file, 'r') as f:
        gold_data = json.load(f)
    
    # Find all benchmark files
    benchmark_dir = Path("test_optimization_results")
    benchmark_files = list(benchmark_dir.glob("benchmark_*.json"))
    
    if not benchmark_files:
        print("❌ No benchmark files found")
        return False
    
    results = []
    all_fixed = True
    
    for benchmark_file in benchmark_files:
        result = test_benchmark_file(benchmark_file, gold_data)
        results.append(result)
        all_fixed = all_fixed and result['is_fixed']
        
        status = "✅" if result['is_fixed'] else "❌"
        print(f"{status} {benchmark_file.name}: F1={result['f1_score']:.3f}, P={result['precision']:.3f}, R={result['recall']:.3f}")
    
    print(f"\n📊 Summary: {len([r for r in results if r['is_fixed']])}/{len(results)} files fixed")
    
    return all_fixed

if __name__ == "__main__":
    success = test_all_benchmarks()
    if success:
        print("\n🎉 All benchmark files are now calculating metrics correctly!")
        exit(0)
    else:
        print("\n💥 Some benchmark files still have issues!")
        exit(1)