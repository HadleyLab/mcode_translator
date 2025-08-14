#!/usr/bin/env python3
"""
Test script for enhanced structured data generator functionality
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from structured_data_generator import StructuredDataGenerator

def main():
    """Main function to test enhanced structured data generator"""
    print("Testing Enhanced Structured Data Generator...")
    
    # Create an instance of the generator
    generator = StructuredDataGenerator()
    
    # Test generating AllergyIntolerance resource
    print("\n1. Testing AllergyIntolerance resource generation:")
    allergy_data = {
        "mcode_element": "AllergyIntolerance",
        "primary_code": {"system": "SNOMEDCT", "code": "77386006"},
        "mapped_codes": {"ICD10CM": "Z88.0"}
    }
    
    allergy_resource = generator.generate_allergy_intolerance_resource(allergy_data)
    print(f"   Generated AllergyIntolerance resource with ID: {allergy_resource['id']}")
    
    # Validate the resource
    validation_result = generator.validate_resource(allergy_resource)
    print(f"   Validation result: {'Passed' if validation_result['valid'] else 'Failed'}")
    
    # Test generating Specimen resource
    print("\n2. Testing Specimen resource generation:")
    specimen_data = {
        "mcode_element": "Specimen",
        "primary_code": {"system": "SNOMEDCT", "code": "119376003"},
        "mapped_codes": {}
    }
    
    specimen_resource = generator.generate_specimen_resource(specimen_data)
    print(f"   Generated Specimen resource with ID: {specimen_resource['id']}")
    
    # Validate the resource
    validation_result = generator.validate_resource(specimen_resource)
    print(f"   Validation result: {'Passed' if validation_result['valid'] else 'Failed'}")
    
    # Test generating DiagnosticReport resource
    print("\n3. Testing DiagnosticReport resource generation:")
    report_data = {
        "mcode_element": "DiagnosticReport",
        "primary_code": {"system": "LOINC", "code": "24357-6"},
        "mapped_codes": {}
    }
    
    report_resource = generator.generate_diagnostic_report_resource(report_data)
    print(f"   Generated DiagnosticReport resource with ID: {report_resource['id']}")
    
    # Validate the resource
    validation_result = generator.validate_resource(report_resource)
    print(f"   Validation result: {'Passed' if validation_result['valid'] else 'Failed'}")
    
    # Test generating FamilyMemberHistory resource
    print("\n4. Testing FamilyMemberHistory resource generation:")
    family_history_data = {
        "mcode_element": "FamilyMemberHistory",
        "primary_code": {"system": "SNOMEDCT", "code": "410534003"},
        "mapped_codes": {}
    }
    
    family_history_resource = generator.generate_family_member_history_resource(family_history_data)
    print(f"   Generated FamilyMemberHistory resource with ID: {family_history_resource['id']}")
    
    # Validate the resource
    validation_result = generator.validate_resource(family_history_resource)
    print(f"   Validation result: {'Passed' if validation_result['valid'] else 'Failed'}")
    
    # Test generating mCODE resources with all new resource types
    print("\n5. Testing complete mCODE resource generation with all resource types:")
    mapped_elements = [
        {
            "mcode_element": "Condition",
            "primary_code": {"system": "ICD10CM", "code": "C50.911"},
            "mapped_codes": {"SNOMEDCT": "254837009"}
        },
        {
            "mcode_element": "MedicationStatement",
            "primary_code": {"system": "RxNorm", "code": "123456"},
            "mapped_codes": {}
        },
        {
            "mcode_element": "AllergyIntolerance",
            "primary_code": {"system": "SNOMEDCT", "code": "77386006"},
            "mapped_codes": {"ICD10CM": "Z88.0"}
        },
        {
            "mcode_element": "Specimen",
            "primary_code": {"system": "SNOMEDCT", "code": "119376003"},
            "mapped_codes": {}
        },
        {
            "mcode_element": "DiagnosticReport",
            "primary_code": {"system": "LOINC", "code": "24357-6"},
            "mapped_codes": {}
        },
        {
            "mcode_element": "FamilyMemberHistory",
            "primary_code": {"system": "SNOMEDCT", "code": "410534003"},
            "mapped_codes": {}
        }
    ]
    
    demographics = {
        "gender": "female",
        "age": "55",
        "ethnicity": "hispanic-or-latino"
    }
    
    result = generator.generate_mcode_resources(mapped_elements, demographics)
    
    print(f"   Generated {len(result['resources'])} resources:")
    for i, resource in enumerate(result['resources']):
        print(f"     {i+1}. {resource['resourceType']} (ID: {resource['id']})")
    
    print(f"   Bundle validation: {'Passed' if result['validation']['valid'] else 'Failed'}")
    print(f"   Compliance score: {result['validation']['compliance_score']:.2f}")
    
    if result['validation']['errors']:
        print("   Validation errors:")
        for error in result['validation']['errors']:
            print(f"     - {error}")

if __name__ == "__main__":
    main()