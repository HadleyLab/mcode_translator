# Structured mCODE Representation Generator Design

## Overview
This document outlines the design for the Structured mCODE Representation Generator, which creates standardized mCODE FHIR resources from extracted clinical trial criteria elements.

## Core Functionality

### 1. FHIR Resource Generation
- Create Patient resources for demographic data
- Generate Condition resources for medical conditions
- Produce Procedure resources for treatments and interventions
- Create MedicationStatement resources for drug therapies
- Generate Observation resources for lab values and vital signs

### 2. mCODE Profile Compliance
- Ensure generated resources comply with mCODE Implementation Guide
- Apply required extensions and modifiers
- Validate against mCODE FHIR profiles
- Handle cardinality constraints

### 3. Resource Relationship Management
- Establish proper references between resources
- Maintain logical connections between related elements
- Handle composition and bundle creation
- Manage resource identifiers and cross-references

## Implementation Approach

### Data Input Format
```json
{
  "patient_characteristics": {
    "demographics": {
      "age": {"min": 18, "max": 75, "unit": "years"},
      "gender": ["female"]
    },
    "medical_history": [
      {
        "condition": "breast cancer",
        "codes": {
          "ICD10CM": "C50.911",
          "SNOMEDCT": "254837009"
        }
      }
    ]
  },
  "extracted_codes": [
    {
      "code": "C50.911",
      "system": "ICD-10-CM",
      "confidence": 0.95
    }
  ]
}
```

### FHIR Resource Generation

#### Patient Resource
```python
def generate_patient_resource(demographics):
    patient = {
        "resourceType": "Patient",
        "gender": demographics.get("gender", "unknown"),
        "birthDate": calculate_birthdate(demographics.get("age"))
    }
    
    # Add mCODE extensions
    if "ethnicity" in demographics:
        patient["extension"] = [{
            "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity",
            "valueCodeableConcept": {
                "coding": [{
                    "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity",
                    "code": demographics["ethnicity"]
                }]
            }
        }]
    
    return patient
```

#### Condition Resource
```python
def generate_condition_resource(condition_data):
    condition = {
        "resourceType": "Condition",
        "code": {
            "coding": []
        }
    }
    
    # Add all available codes
    for system, code in condition_data.get("codes", {}).items():
        system_uris = {
            "ICD10CM": "http://hl7.org/fhir/sid/icd-10-cm",
            "SNOMEDCT": "http://snomed.info/sct"
        }
        
        if system in system_uris:
            condition["code"]["coding"].append({
                "system": system_uris[system],
                "code": code
            })
    
    return condition
```

## mCODE-Specific Components

### Cancer Condition Profile
```python
def generate_cancer_condition(condition_data):
    cancer_condition = generate_condition_resource(condition_data)
    
    # Apply mCODE cancer condition profile
    cancer_condition["meta"] = {
        "profile": [
            "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition"
        ]
    }
    
    # Add body site if available
    if "body_site" in condition_data:
        cancer_condition["bodySite"] = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": condition_data["body_site"]["code"],
                "display": condition_data["body_site"]["display"]
            }]
        }
    
    return cancer_condition
```

### Treatment Plan Profile
```python
def generate_treatment_plan(treatment_data):
    treatment_plan = {
        "resourceType": "CarePlan",
        "meta": {
            "profile": [
                "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-cancer-treatment-plan"
            ]
        },
        "activity": []
    }
    
    # Add treatment activities
    for treatment in treatment_data:
        treatment_plan["activity"].append({
            "detail": {
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": treatment["code"],
                        "display": treatment["display"]
                    }]
                }
            }
        })
    
    return treatment_plan
```

## Resource Bundling

### Bundle Creation
```python
def create_mcode_bundle(resources):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": []
    }
    
    for resource in resources:
        bundle["entry"].append({
            "resource": resource
        })
    
    return bundle
```

## Validation and Quality Assurance

### Profile Validation
- Check resource structure against FHIR profiles
- Validate required elements and cardinality
- Ensure proper use of value sets
- Verify extension usage compliance

### Data Completeness
- Identify missing required elements
- Flag potentially inconsistent data
- Suggest improvements for data quality
- Provide validation reports

## Output Formats

### FHIR JSON
```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "gender": "female",
        "birthDate": "1950-01-01"
      }
    },
    {
      "resource": {
        "resourceType": "Condition",
        "meta": {
          "profile": [
            "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-primary-cancer-condition"
          ]
        },
        "code": {
          "coding": [
            {
              "system": "http://hl7.org/fhir/sid/icd-10-cm",
              "code": "C50.911",
              "display": "Malignant neoplasm of upper-outer quadrant of right female breast"
            }
          ]
        }
      }
    }
  ]
}
```

### XML Format
- Convert JSON FHIR resources to XML
- Maintain FHIR specification compliance
- Support standard FHIR XML formatting

## Integration Points

### Input Sources
- Receives data from patient characteristic identification module
- Gets coded elements from mCODE code extraction module
- Uses mappings from mCODE mapping engine

### Output Destinations
- Feeds structured data to output formatter
- Provides validation results to quality assurance
- Supplies data for testing and evaluation

## Error Handling

### Data Quality Issues
- Handle missing or incomplete data gracefully
- Provide default values where appropriate
- Flag critical missing information
- Generate error reports for review

### Validation Failures
- Identify profile compliance issues
- Suggest corrections for common problems
- Log validation errors for troubleshooting
- Continue processing when possible