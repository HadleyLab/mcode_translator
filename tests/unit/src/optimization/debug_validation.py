#!/usr/bin/env python3
"""
Debug script to test the validation logic
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.optimization.prompt_optimization_framework import BenchmarkResult

def test_validation_logic():
    """Test the validation logic with simple data"""
    
    # Create a simple benchmark result
    benchmark = BenchmarkResult(
        run_id="test_run",
        prompt_variant_id="test_prompt",
        api_config_name="test_api",
        test_case_id="test_case"
    )
    
    # Set the result data
    benchmark.entities_extracted = 4
    benchmark.entities_mapped = 4
    benchmark.mcode_mappings = [
        {
            "Mcode_element": "CancerCondition",
            "value": "HER2-Positive Breast Cancer"
        },
        {
            "Mcode_element": "CancerCondition",
            "value": "Metastatic Breast Cancer"
        },
        {
            "Mcode_element": "ECOGPerformanceStatus",
            "value": "0-1"
        },
        {
            "Mcode_element": "CancerDiseaseStatus",
            "value": "Measurable disease present"
        }
    ]
    
    # Create expected mappings
    expected_mappings = [
        {
            "Mcode_element": "CancerCondition",
            "value": "HER2-Positive Breast Cancer"
        },
        {
            "Mcode_element": "CancerCondition",
            "value": "Metastatic Breast Cancer"
        },
        {
            "Mcode_element": "ECOGPerformanceStatus",
            "value": "0-1"
        },
        {
            "Mcode_element": "CancerDiseaseStatus",
            "value": "Measurable disease present"
        },
        {
            "Mcode_element": "CancerRelatedMedication",
            "value": "Trastuzumab Deruxtecan"
        }
    ]
    
    # Mark as successful
    benchmark.success = True
    
    # Create a mock framework with a logger
    class MockFramework:
        def __init__(self):
            pass
            
        class logger:
            @staticmethod
            def debug(message):
                print(f"DEBUG: {message}")
                
            @staticmethod
            def warning(message):
                print(f"WARNING: {message}")
                
            @staticmethod
            def info(message):
                print(f"INFO: {message}")
    
    mock_framework = MockFramework()
    
    # Calculate metrics
    print("Before calculating metrics:")
    print(f"  mcode_mappings: {len(benchmark.mcode_mappings)}")
    print(f"  Expected mappings: {len(expected_mappings)}")
    
    benchmark.calculate_metrics(expected_entities=[], expected_mappings=expected_mappings, framework=mock_framework)
    
    print("\nValidation Metrics:")
    print(f"Precision: {benchmark.precision:.3f}")
    print(f"Recall: {benchmark.recall:.3f}")
    print(f"F1 Score: {benchmark.f1_score:.3f}")
    print(f"Mapping Accuracy: {benchmark.mapping_accuracy:.3f}")
    
    # Check if mapping accuracy is non-zero
    if benchmark.mapping_accuracy > 0:
        print("\n✅ SUCCESS: Mapping accuracy is non-zero!")
        return True
    else:
        print("\n❌ FAILURE: Mapping accuracy is still zero.")
        return False

if __name__ == "__main__":
    print("Testing Validation Logic")
    print("=" * 30)
    
    success = test_validation_logic()
    
    if success:
        print("\n🎉 Validation logic test PASSED!")
        sys.exit(0)
    else:
        print("\n💥 Validation logic test FAILED!")
        sys.exit(1)