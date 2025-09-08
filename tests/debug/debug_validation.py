#!/usr/bin/env python3
"""
Debug script to understand why validation metrics are zero.
"""

import json
from pathlib import Path

def debug_validation():
    """Debug the validation logic step by step"""
    
    # Load gold standard
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the mCODE mappings from the gold standard
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print("Gold Standard mCODE Elements:")
    for i, entity in enumerate(gold_standard):
        print(f"  {i+1}. {entity.get('Mcode_element', 'Unknown')}: {entity.get('value', 'No value')}")
    
    # Sample extracted entities from benchmark result
    sample_extracted = [
        {
            "resourceType": "Condition",
            "element_name": "CancerCondition",
            "value": "HER2-positive breast cancer",
            "mapping_confidence": 1.0
        },
        {
            "resourceType": "Condition",
            "element_name": "CancerCondition",
            "value": "Metastatic breast cancer",
            "mapping_confidence": 1.0
        },
        {
            "resourceType": "Observation",
            "element_name": "GenomicVariant",
            "value": "HER2-positive",
            "mapping_confidence": 1.0
        },
        {
            "resourceType": "Observation",
            "element_name": "ECOGPerformanceStatus",
            "value": "0-1",
            "mapping_confidence": 1.0
        }
    ]
    
    print("\nSample Extracted mCODE Elements:")
    for i, entity in enumerate(sample_extracted):
        print(f"  {i+1}. {entity.get('element_name', 'Unknown')}: {entity.get('value', 'No value')}")
    
    # Now let's manually calculate the metrics
    gold_elements = set(m.get('Mcode_element', '') for m in gold_standard)
    extracted_elements = set(m.get('element_name', '') for m in sample_extracted)
    
    print(f"\nGold Elements: {gold_elements}")
    print(f"Extracted Elements: {extracted_elements}")
    
    # Calculate metrics manually
    true_positives_ext = len(extracted_elements & gold_elements)
    false_positives_ext = len(extracted_elements - gold_elements)
    false_negatives_ext = len(gold_elements - extracted_elements)
    
    print(f"\nManual Calculation:")
    print(f"True Positives: {true_positives_ext}")
    print(f"False Positives: {false_positives_ext}")
    print(f"False Negatives: {false_negatives_ext}")
    
    precision = true_positives_ext / (true_positives_ext + false_positives_ext) if (true_positives_ext + false_positives_ext) > 0 else 0
    recall = true_positives_ext / (true_positives_ext + false_negatives_ext) if (true_positives_ext + false_negatives_ext) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    
    # Check if there are any common elements
    common_elements = extracted_elements & gold_elements
    print(f"\nCommon Elements: {common_elements}")
    
    # Check for specific element mismatches
    print("\nElement Comparison:")
    for gold_elem in gold_elements:
        for extracted_elem in extracted_elements:
            if gold_elem == extracted_elem:
                print(f"  ✅ MATCH: {gold_elem} == {extracted_elem}")
            else:
                print(f"  ❌ MISMATCH: {gold_elem} != {extracted_elem}")

if __name__ == "__main__":
    debug_validation()