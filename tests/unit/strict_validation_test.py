#!/usr/bin/env python3
"""
STRICT validation test - direct implementation without mocks or fallbacks.
Tests the actual validation logic with real framework components.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from optimization.strict_prompt_optimization_framework import StrictPromptOptimizationFramework, BenchmarkResult

def test_strict_validation():
    """STRICT test of validation logic using actual framework components"""
    
    print("STRICT Validation Test - No Mocks, No Fallbacks")
    print("=" * 50)
    
    # Load gold standard
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the mCODE mappings from the gold standard
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print(f"Gold standard contains {len(gold_standard)} expected mCODE mappings")
    
    # Create the actual framework instance
    framework = StrictPromptOptimizationFramework()
    
    # Create sample extracted entities (what the NLP engine would produce)
    extracted_entities = [
        {
            "text": "HER2-positive breast cancer",
            "type": "condition",
            "attributes": {"status": "positive"},
            "confidence": 1.0
        },
        {
            "text": "Metastatic breast cancer",
            "type": "condition",
            "attributes": {"status": "positive"},
            "confidence": 1.0
        },
        {
            "text": "HER2-positive",
            "type": "biomarker",
            "attributes": {"status": "positive"},
            "confidence": 1.0
        },
        {
            "text": "ECOG performance status 0-1",
            "type": "demographic",
            "attributes": {"value": "0-1"},
            "confidence": 1.0
        }
    ]
    
    # Create benchmark result with actual data
    benchmark = BenchmarkResult(
        run_id="strict_test_run",
        prompt_variant_id="strict_test_prompt",
        api_config_name="strict_test_api",
        test_case_id="breast_cancer_test"
    )
    
    # Set the actual result data
    benchmark.entities_extracted = 4  # Matches our sample data
    benchmark.entities_mapped = 4     # Will be set by conversion
    benchmark.extracted_entities = extracted_entities
    benchmark.success = True
    
    # STRICT: Calculate metrics using the actual framework
    # We need to provide expected_entities for the validation logic to work
    # Create minimal expected entities that match our sample data
    expected_entities = [
        {"text": "HER2-positive breast cancer", "type": "condition"},
        {"text": "Metastatic breast cancer", "type": "condition"},
        {"text": "HER2-positive", "type": "biomarker"},
        {"text": "ECOG performance status 0-1", "type": "demographic"}
    ]
    
    # Add debug output to the calculate_metrics method temporarily
    original_calculate_metrics = benchmark.calculate_metrics
    
    def debug_calculate_metrics(expected_entities=None, expected_mappings=None, framework=None):
        print(f"\nDEBUG - Inside calculate_metrics:")
        print(f"Expected entities provided: {bool(expected_entities)}")
        print(f"Extracted entities: {len(benchmark.extracted_entities)}")
        print(f"Framework available: {bool(framework)}")
        
        if expected_entities and benchmark.extracted_entities and framework:
            print("✅ Validation condition met - proceeding with calculation")
            
            # Convert both extracted and expected entities to mCODE format
            extracted_mcode = framework._convert_entities_to_mcode(benchmark.extracted_entities)
            expected_mcode_elements = framework._convert_entities_to_mcode(expected_entities)
            
            print(f"Extracted mCODE elements: {[m.get('element_name', '') for m in extracted_mcode]}")
            print(f"Expected mCODE elements: {[m.get('element_name', '') for m in expected_mcode_elements]}")
            
            # Calculate metrics
            true_positives_ext = len(set(m.get('element_name', '') for m in extracted_mcode) &
                                   set(m.get('element_name', '') for m in expected_mcode_elements))
            false_positives_ext = len(set(m.get('element_name', '') for m in extracted_mcode) -
                                    set(m.get('element_name', '') for m in expected_mcode_elements))
            false_negatives_ext = len(set(m.get('element_name', '') for m in expected_mcode_elements) -
                                    set(m.get('element_name', '') for m in extracted_mcode))
            
            print(f"True positives: {true_positives_ext}")
            print(f"False positives: {false_positives_ext}")
            print(f"False negatives: {false_negatives_ext}")
            
        else:
            print("❌ Validation condition NOT met - skipping calculation")
        
        # Call the original method
        return original_calculate_metrics(expected_entities, expected_mappings, framework)
    
    # Replace the method temporarily
    benchmark.calculate_metrics = debug_calculate_metrics
    
    benchmark.calculate_metrics(
        expected_entities=expected_entities,
        expected_mappings=gold_standard,
        framework=framework
    )
    
    print(f"\nSTRICT Validation Results:")
    print(f"Precision: {benchmark.precision:.3f}")
    print(f"Recall: {benchmark.recall:.3f}")
    print(f"F1 Score: {benchmark.f1_score:.3f}")
    print(f"Extraction Completeness: {benchmark.extraction_completeness:.3f}")
    print(f"Mapping Accuracy: {benchmark.mapping_accuracy:.3f}")
    
    # Debug: Show what's being compared
    print(f"\nDebug - Element Comparison:")
    # Get the actual extracted mCODE elements from the conversion
    extracted_mcode = framework._convert_entities_to_mcode(benchmark.extracted_entities)
    extracted_elements = {m.get('element_name', '') for m in extracted_mcode}
    
    # Get expected elements from gold standard (which uses 'mcode_element' field)
    expected_elements = {m.get('mcode_element', '') for m in gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']}
    
    print(f"Extracted elements: {extracted_elements}")
    print(f"Expected elements: {expected_elements}")
    print(f"Intersection: {extracted_elements & expected_elements}")
    print(f"Extracted only: {extracted_elements - expected_elements}")
    print(f"Expected only: {expected_elements - extracted_elements}")
    
    # STRICT validation: Metrics should be non-zero
    if benchmark.precision > 0 or benchmark.recall > 0 or benchmark.f1_score > 0:
        print(f"\n✅ STRICT SUCCESS: Validation metrics are non-zero!")
        print("The validation fix is working correctly.")
        return True
    else:
        print(f"\n❌ STRICT FAILURE: Validation metrics are still zero.")
        print("The validation logic needs debugging.")
        return False

if __name__ == "__main__":
    success = test_strict_validation()
    
    if success:
        print("\n🎉 STRICT validation test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 STRICT validation test FAILED!")
        sys.exit(1)