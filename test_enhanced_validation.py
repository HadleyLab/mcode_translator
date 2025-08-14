#!/usr/bin/env python3
"""
Test script for enhanced validation and quality metrics functionality
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from structured_data_generator import StructuredDataGenerator

def main():
    """Main function to test enhanced validation and quality metrics"""
    print("Testing Enhanced Validation and Quality Metrics...")
    
    # Create an instance of the generator
    generator = StructuredDataGenerator()
    
    # Test validating a complete bundle with quality metrics
    print("\n1. Testing bundle validation with quality metrics:")
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
    
    print(f"\n   Bundle validation: {'Passed' if result['validation']['valid'] else 'Failed'}")
    print(f"   Compliance score: {result['validation']['compliance_score']:.2f}")
    
    # Display quality metrics
    quality_metrics = result['validation']['quality_metrics']
    print(f"\n   Quality Metrics:")
    print(f"     Completeness: {quality_metrics['completeness']:.2f}")
    print(f"     Accuracy: {quality_metrics['accuracy']:.2f}")
    print(f"     Consistency: {quality_metrics['consistency']:.2f}")
    print(f"     Resource Coverage: {quality_metrics['resource_coverage']:.2f}")
    
    # Display resource type summary
    print(f"\n   Resource Type Summary:")
    for resource_type, count in result['validation']['resource_type_summary'].items():
        print(f"     {resource_type}: {count}")
    
    if result['validation']['errors']:
        print("\n   Validation errors:")
        for error in result['validation']['errors']:
            print(f"     - {error}")
    
    if result['validation']['warnings']:
        print("\n   Validation warnings:")
        for warning in result['validation']['warnings']:
            print(f"     - {warning}")
    
    # Test validating an invalid resource
    print("\n2. Testing validation of invalid resource:")
    invalid_resource = {
        "resourceType": "Condition"
        # Missing required 'code' element
    }
    
    validation_result = generator.validate_resource(invalid_resource)
    print(f"   Resource validation: {'Passed' if validation_result['valid'] else 'Failed'}")
    
    # Display quality metrics for invalid resource
    quality_metrics = validation_result['quality_metrics']
    print(f"   Quality Metrics:")
    print(f"     Completeness: {quality_metrics['completeness']:.2f}")
    print(f"     Accuracy: {quality_metrics['accuracy']:.2f}")
    print(f"     Consistency: {quality_metrics['consistency']:.2f}")
    
    if validation_result['errors']:
        print("   Validation errors:")
        for error in validation_result['errors']:
            print(f"     - {error}")

if __name__ == "__main__":
    main()