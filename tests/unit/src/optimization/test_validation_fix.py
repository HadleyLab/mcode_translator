#!/usr/bin/env python3
"""
Test script to verify the validation fix for mCODE field comparison.
This script tests the BenchmarkResult.calculate_metrics() method with the fixed logic.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.optimization.prompt_optimization_framework import BenchmarkResult

def test_validation_fix():
    """Test the validation fix with sample data"""
    
    # Load gold standard
    gold_standard_path = Path('examples/breast_cancer_data/breast_cancer_her2_positive.gold.json')
    with open(gold_standard_path, 'r') as f:
        gold_data = json.load(f)
    
    # Extract the expected entities and mCODE mappings from the gold standard
    expected_entities = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_extraction']['entities']
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print(f"Gold standard contains {len(gold_standard)} expected mCODE mappings")
    
    # Create sample extracted entities that would be produced by the NLP extraction
    sample_extracted_entities = [
        {
            "text": "HER2-Positive Breast Cancer",
            "type": "condition",
            "attributes": {
                "status": "positive"
            },
            "confidence": 0.95,
            "source_context": {
                "section": "conditionsModule",
                "position": "0-25",
                "context_sentence": "HER2-Positive Breast Cancer"
            }
        },
        {
            "text": "Metastatic Breast Cancer",
            "type": "condition",
            "attributes": {
                "status": "metastatic"
            },
            "confidence": 0.95,
            "source_context": {
                "section": "conditionsModule",
                "position": "26-49",
                "context_sentence": "Metastatic Breast Cancer"
            }
        },
        {
            "text": "HER2-positive metastatic breast cancer",
            "type": "condition",
            "attributes": {
                "status": "positive",
                "metastatic": True
            },
            "confidence": 0.9,
            "source_context": {
                "section": "eligibilityModule",
                "position": "25-60",
                "context_sentence": "Histologically confirmed HER2-positive metastatic breast cancer"
            }
        },
        {
            "text": "Measurable disease",
            "type": "condition",
            "attributes": {
                "status": "present"
            },
            "confidence": 0.85,
            "source_context": {
                "section": "eligibilityModule",
                "position": "61-79",
                "context_sentence": "Measurable disease per RECIST 1.1"
            }
        },
        {
            "text": "ECOG performance status 0-1",
            "type": "demographic",
            "attributes": {
                "value": "0-1",
                "scale": "ECOG"
            },
            "confidence": 0.9,
            "source_context": {
                "section": "eligibilityModule",
                "position": "80-105",
                "context_sentence": "ECOG performance status 0-1"
            }
        }
    ]
    
    # Create sample mCODE mappings that match the gold standard structure exactly
    sample_mcode_mappings = [
        {
            "source_entity_index": 0,
            "Mcode_element": "CancerCondition",
            "value": "HER2-Positive Breast Cancer",
            "confidence": 0.95,
            "mapping_rationale": "Primary cancer diagnosis with HER2 biomarker status"
        },
        {
            "source_entity_index": 1,
            "Mcode_element": "CancerCondition",
            "value": "Metastatic Breast Cancer",
            "confidence": 0.95,
            "mapping_rationale": "Metastatic cancer condition"
        },
        {
            "source_entity_index": 2,
            "Mcode_element": "CancerCondition",
            "value": "HER2-positive metastatic breast cancer",
            "confidence": 0.9,
            "mapping_rationale": "Specific cancer diagnosis with biomarker and metastatic status"
        },
        {
            "source_entity_index": 3,
            "Mcode_element": "CancerDiseaseStatus",
            "value": "Measurable disease present",
            "confidence": 0.85,
            "mapping_rationale": "Disease status indicating measurable disease"
        },
        {
            "source_entity_index": 4,
            "Mcode_element": "ECOGPerformanceStatus",
            "value": "0-1",
            "confidence": 0.9,
            "mapping_rationale": "ECOG performance status score range"
        },
        {
            "source_entity_index": 5,
            "Mcode_element": "Observation",
            "value": "Adequate organ function",
            "confidence": 0.8,
            "mapping_rationale": "General observation about organ function status"
        },
        {
            "source_entity_index": 10,
            "Mcode_element": "CancerRelatedMedication",
            "value": "Trastuzumab Deruxtecan",
            "confidence": 0.95,
            "mapping_rationale": "HER2-targeted antibody-drug conjugate medication"
        },
        {
            "source_entity_index": 11,
            "Mcode_element": "CancerRelatedMedication",
            "value": "Trastuzumab",
            "confidence": 0.9,
            "mapping_rationale": "Standard HER2-targeted therapy medication"
        },
        {
            "source_entity_index": 12,
            "Mcode_element": "CancerRelatedMedication",
            "value": "Taxane chemotherapy",
            "confidence": 0.85,
            "mapping_rationale": "Chemotherapy backbone medication"
        },
        {
            "source_entity_index": 13,
            "Mcode_element": "CancerRelatedMedication",
            "value": "HER2-targeted therapy",
            "confidence": 0.9,
            "mapping_rationale": "General category of HER2-targeted treatments"
        },
        {
            "source_entity_index": 14,
            "Mcode_element": "CancerCondition",
            "value": "HER2-positive metastatic breast cancer",
            "confidence": 0.95,
            "mapping_rationale": "Cancer condition from description section"
        },
        {
            "source_entity_index": 15,
            "Mcode_element": "CancerRelatedMedication",
            "value": "Trastuzumab-based therapy",
            "confidence": 0.9,
            "mapping_rationale": "Prior treatment regimen based on trastuzumab"
        }
    ]
    
    # Create benchmark result instance
    benchmark = BenchmarkResult(
        run_id="test_run",
        prompt_variant_id="test_prompt",
        api_config_name="test_api",
        test_case_id="breast_cancer_test"
    )
    
    # Set success to True to allow metric calculation to proceed
    benchmark.success = True
    
    # Set the result data
    benchmark.extracted_entities = sample_extracted_entities
    benchmark.entities_extracted = len(sample_extracted_entities)
    benchmark.mcode_mappings = sample_mcode_mappings
    benchmark.entities_mapped = len(sample_mcode_mappings)
    
    # Create a mock framework instance for the conversion method
    class MockFramework:
        def __init__(self):
            from src.pipeline.mcode_mapper import McodeMapper
            from src.utils.logging_config import get_logger
            self.mcode_mapper = McodeMapper()
            self.logger = get_logger(__name__)
            # Enable debug logging for the framework logger
            self.logger.setLevel(logging.DEBUG)
        
        def _convert_entities_to_mcode(self, entities):
            """Mock conversion method - return simplified mCODE format"""
            if not entities:
                return []
            
            # For testing, just create simple mCODE elements based on the sample data
            Mcode_elements = []
            for entity in entities:
                if isinstance(entity, dict) and 'element_name' in entity:
                    Mcode_elements.append({
                        'element_name': entity['element_name'],
                        'value': entity.get('value', '')
                    })
            return Mcode_elements
    
    # Enable debug logging for testing
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    
    mock_framework = MockFramework()
    
    # Calculate metrics with the fixed validation logic
    print(f"\nCalling calculate_metrics with:")
    print(f"  expected_entities: {len(expected_entities) if expected_entities else 0}")
    print(f"  extracted_entities: {len(sample_extracted_entities) if sample_extracted_entities else 0}")
    print(f"  expected_mappings: {len(gold_standard) if gold_standard else 0}")
    print(f"  mcode_mappings: {len(sample_mcode_mappings) if sample_mcode_mappings else 0}")
    
    benchmark.calculate_metrics(expected_entities=expected_entities, expected_mappings=gold_standard, framework=mock_framework)
    
    print("\nValidation Metrics:")
    print(f"Precision: {benchmark.precision:.3f}")
    print(f"Recall: {benchmark.recall:.3f}")
    print(f"F1 Score: {benchmark.f1_score:.3f}")
    print(f"Extraction Completeness: {benchmark.extraction_completeness:.3f}")
    print(f"Mapping Accuracy: {benchmark.mapping_accuracy:.3f}")
    
    # Check if metrics are non-zero
    if benchmark.precision > 0 or benchmark.recall > 0 or benchmark.f1_score > 0 or benchmark.mapping_accuracy > 0:
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
    
    # Extract the mCODE mappings
    gold_standard = gold_data['gold_standard']['breast_cancer_her2_positive']['expected_mcode_mappings']['mapped_elements']
    
    print("Gold Standard Analysis:")
    print(f"Total mCODE mappings: {len(gold_standard)}")
    
    # Count by mCODE element type
    Mcode_elements = {}
    
    for entity in gold_standard:
        Mcode_element = entity.get('Mcode_element', 'Unknown')
        Mcode_elements[Mcode_element] = Mcode_elements.get(Mcode_element, 0) + 1
    
    print("\nMcode Elements:")
    for element, count in Mcode_elements.items():
        print(f"  {element}: {count}")
    
    # Show first few entities
    print("\nSample mCODE mappings:")
    for i, entity in enumerate(gold_standard[:5]):
        print(f"  {i+1}. {entity.get('Mcode_element', 'Unknown')}: {entity.get('value', 'No value')}")

if __name__ == "__main__":
    print("Testing Validation Fix for mCODE Field Comparison")
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