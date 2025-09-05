#!/usr/bin/env python3
"""
Test script to verify the validation fix for Mcode field comparison.
This script tests the BenchmarkResult.calculate_metrics() method with the fixed logic.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.prompt_optimization_framework import BenchmarkResult

def test_validation_fix():
    """Test the validation fix with sample data"""
    
    # Load gold standard
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the Mcode mappings from the gold standard
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print(f"Gold standard contains {len(gold_standard)} expected Mcode mappings")
    
    # Create a sample benchmark result with extracted entities
    sample_result = {
        "entities_extracted": 22,
        "entities_mapped": 19,
        "Mcode_mappings": [
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
    }
    
    # Create benchmark result instance
    benchmark = BenchmarkResult(
        run_id="test_run",
        prompt_variant_id="test_prompt",
        api_config_name="test_api",
        test_case_id="breast_cancer_test"
    )
    
    # Set the result data
    benchmark.entities_extracted = sample_result["entities_extracted"]
    benchmark.entities_mapped = sample_result["entities_mapped"]
    benchmark.mcode_mappings = sample_result["Mcode_mappings"]
    
    # Create a mock framework instance for the conversion method
    class MockFramework:
        def __init__(self):
            from src.pipeline.mcode_mapper import McodeMapper
            self.mcode_mapper = McodeMapper()
        
        def _convert_entities_to_mcode(self, entities):
            """Mock conversion method - return simplified Mcode format"""
            if not entities:
                return []
            
            # For testing, just create simple Mcode elements based on the sample data
            Mcode_elements = []
            for entity in entities:
                if isinstance(entity, dict) and 'element_name' in entity:
                    Mcode_elements.append({
                        'element_name': entity['element_name'],
                        'value': entity.get('value', '')
                    })
            return Mcode_elements
    
    mock_framework = MockFramework()
    
    # Calculate metrics with the fixed validation logic
    benchmark.calculate_metrics(expected_entities=[], expected_mappings=gold_standard, framework=mock_framework)
    
    print("\nValidation Metrics:")
    print(f"Precision: {benchmark.precision:.3f}")
    print(f"Recall: {benchmark.recall:.3f}")
    print(f"F1 Score: {benchmark.f1_score:.3f}")
    print(f"Extraction Completeness: {benchmark.extraction_completeness:.3f}")
    print(f"Mapping Accuracy: {benchmark.mapping_accuracy:.3f}")
    
    # Check if metrics are non-zero
    if benchmark.precision > 0 or benchmark.recall > 0 or benchmark.f1_score > 0:
        print("\n✅ SUCCESS: Validation metrics are now non-zero!")
        print("The validation fix is working correctly.")
        return True
    else:
        print("\n❌ FAILURE: Validation metrics are still zero.")
        print("The validation fix may not be working.")
        return False

def analyze_gold_standard():
    """Analyze the gold standard structure"""
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the Mcode mappings
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print("Gold Standard Analysis:")
    print(f"Total Mcode mappings: {len(gold_standard)}")
    
    # Count by Mcode element type
    Mcode_elements = {}
    
    for entity in gold_standard:
        Mcode_element = entity.get('Mcode_element', 'Unknown')
        Mcode_elements[Mcode_element] = Mcode_elements.get(Mcode_element, 0) + 1
    
    print("\nMcode Elements:")
    for element, count in Mcode_elements.items():
        print(f"  {element}: {count}")
    
    # Show first few entities
    print("\nSample Mcode mappings:")
    for i, entity in enumerate(gold_standard[:5]):
        print(f"  {i+1}. {entity.get('Mcode_element', 'Unknown')}: {entity.get('value', 'No value')}")

if __name__ == "__main__":
    print("Testing Validation Fix for Mcode Field Comparison")
    print("=" * 50)
    
    # First analyze the gold standard
    analyze_gold_standard()
    
    print("\n" + "=" * 50)
    
    # Test the validation fix
    success = test_validation_fix()
    
    if success:
        print("\n🎉 Validation fix test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Validation fix test FAILED!")
        sys.exit(1)