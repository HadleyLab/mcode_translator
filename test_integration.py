#!/usr/bin/env python3
"""
Integration test for the mCODE Translator project
Tests the complete workflow from criteria parsing to mCODE generation
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nlp_engine import NLPEngine
from code_extraction import CodeExtractionModule
from mcode_mapping_engine import MCODEMappingEngine
from structured_data_generator import StructuredDataGenerator
from output_formatter import OutputFormatter

def test_complete_workflow():
    """Test the complete workflow from criteria parsing to mCODE generation"""
    print("Testing Complete mCODE Translator Workflow")
    print("=" * 50)
    
    # Sample clinical trial eligibility criteria
    sample_criteria = """
    INCLUSION CRITERIA:
    - Male or female patients aged 18 years or older
    - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
    - Must have received prior chemotherapy treatment (CPT: 12345)
    - Currently receiving radiation therapy
    - Laboratory values within normal limits (LOINC: 12345-6)
    - Currently taking medication (RxNorm: 123456)
    
    EXCLUSION CRITERIA:
    - Pregnant or nursing women
    - History of other malignancies within the past 5 years (ICD-10-CM: C18.9)
    - Allergy to contrast agents (SNOMED CT: 77386006)
    - Unable to undergo MRI scanning
    """
    
    try:
        # Step 1: Process criteria with NLP engine
        print("\n1. Processing criteria with NLP Engine...")
        nlp_engine = NLPEngine()
        nlp_result = nlp_engine.process_criteria(sample_criteria)
        print(f"   Extracted {len(nlp_result['entities'])} entities")
        print(f"   Extracted {len(nlp_result['conditions'])} conditions")
        print(f"   Extracted {len(nlp_result['procedures'])} procedures")
        print(f"   Extracted {len(nlp_result['demographics']['age'])} age criteria")
        print(f"   Extracted {len(nlp_result['demographics']['gender'])} gender criteria")
        
        # Step 2: Extract codes
        print("\n2. Extracting codes with Code Extraction Module...")
        code_extractor = CodeExtractionModule()
        code_result = code_extractor.process_criteria_for_codes(sample_criteria, nlp_result['entities'])
        print(f"   Extracted {code_result['metadata']['total_codes']} codes")
        print(f"   Found codes in systems: {', '.join(code_result['metadata']['systems_found'])}")
        
        # Step 3: Map to mCODE
        print("\n3. Mapping to mCODE with Mapping Engine...")
        mapper = MCODEMappingEngine()
        mapping_result = mapper.process_nlp_output(nlp_result)
        print(f"   Mapped {mapping_result['metadata']['total_mapped_elements']} elements")
        print(f"   mCODE validation: {'Passed' if mapping_result['validation']['valid'] else 'Failed'}")
        
        # Step 4: Generate structured data
        print("\n4. Generating structured mCODE data...")
        generator = StructuredDataGenerator()
        structured_result = generator.generate_mcode_resources(
            mapping_result['mapped_elements'],
            nlp_result.get('demographics', {})
        )
        print(f"   Generated {len(structured_result['resources'])} FHIR resources")
        print(f"   Bundle validation: {'Passed' if structured_result['validation']['valid'] else 'Failed'}")
        
        # Display quality metrics
        quality_metrics = structured_result['validation']['quality_metrics']
        print(f"\n5. Quality Metrics:")
        print(f"   Completeness: {quality_metrics['completeness']:.2f}")
        print(f"   Accuracy: {quality_metrics['accuracy']:.2f}")
        print(f"   Consistency: {quality_metrics['consistency']:.2f}")
        print(f"   Resource Coverage: {quality_metrics['resource_coverage']:.2f}")
        
        # Display resource type summary
        print(f"\n6. Resource Type Summary:")
        for resource_type, count in structured_result['validation']['resource_type_summary'].items():
            print(f"   {resource_type}: {count}")
        
        # Step 5: Format output
        print(f"\n7. Formatting output...")
        formatter = OutputFormatter()
        
        # Format as JSON
        json_output = formatter.to_json(structured_result['bundle'])
        print(f"   JSON output length: {len(json_output)} characters")
        
        # Format validation report
        validation_report = formatter.format_validation_report(structured_result['validation'])
        print(f"   Validation report length: {len(validation_report)} characters")
        
        # Format resource summary
        resource_summary = formatter.format_resource_summary(structured_result['resources'])
        print(f"   Resource summary length: {len(resource_summary)} characters")
        
        print("\n" + "=" * 50)
        print("COMPLETE WORKFLOW TEST: PASSED")
        return True
        
    except Exception as e:
        print(f"\nError in workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 50)
        print("COMPLETE WORKFLOW TEST: FAILED")
        return False

def main():
    """Main function"""
    success = test_complete_workflow()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()