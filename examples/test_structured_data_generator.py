#!/usr/bin/env python3
"""
Example script demonstrating the StructuredDataGenerator
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from structured_data_generator import StructuredDataGenerator

def main():
    """Main function to demonstrate the StructuredDataGenerator"""
    print("Testing StructuredDataGenerator...")
    
    # Create an instance of the generator
    generator = StructuredDataGenerator()
    
    # Sample mapped elements
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
        }
    ]
    
    # Sample demographics
    demographics = {
        "gender": "female",
        "age": "55",
        "ethnicity": "hispanic-or-latino"
    }
    
    # Generate mCODE resources
    result = generator.generate_mcode_resources(mapped_elements, demographics)
    
    print(f"Generated {len(result['resources'])} FHIR resources:")
    for i, resource in enumerate(result['resources']):
        print(f"  {i+1}. {resource['resourceType']} (ID: {resource['id']})")
    
    print(f"\nBundle validation: {'Passed' if result['validation']['valid'] else 'Failed'}")
    print(f"Compliance score: {result['validation']['compliance_score']:.2f}")
    
    if result['validation']['errors']:
        print("Validation errors:")
        for error in result['validation']['errors']:
            print(f"  - {error}")
    
    # Convert to JSON
    json_output = generator.to_json(result['bundle'])
    print(f"\nJSON output length: {len(json_output)} characters")
    
    # Convert to XML
    xml_output = generator.to_xml(result['bundle'])
    print(f"XML output length: {len(xml_output)} characters")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()