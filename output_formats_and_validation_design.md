# Output Formats and Data Validation Design

## Overview
This document outlines the design for output formats and data validation components of the mCODE translator. It defines how processed clinical trial criteria will be formatted and validated for quality assurance.

## Output Formats

### 1. FHIR JSON Format
The primary output format following HL7 FHIR standards and mCODE Implementation Guide.

#### Structure
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

### 2. XML Format
Alternative FHIR XML representation for systems requiring XML.

#### Structure
```xml
<Bundle xmlns="http://hl7.org/fhir">
  <type value="collection"/>
  <entry>
    <resource>
      <Patient>
        <gender value="female"/>
        <birthDate value="1950-01-01"/>
      </Patient>
    </resource>
  </entry>
</Bundle>
```

### 3. CSV/Tabular Format
Simplified format for easy data analysis and import into spreadsheets.

#### Structure
```csv
NCTId,PatientGender,MinAge,MaxAge,ConditionCode,ConditionSystem,ConditionDisplay,TreatmentType
NCT00000001,female,18,75,C50.911,ICD-10-CM,Breast Cancer,Chemotherapy
```

### 4. Summary Report Format
Human-readable summary of extracted information.

#### Structure
```markdown
# Clinical Trial mCODE Analysis: NCT00000001

## Patient Eligibility Criteria
- Gender: Female only
- Age: 18-75 years
- Ethnicity: All ethnicities accepted

## Medical Conditions
- Primary condition: Breast Cancer (C50.911, ICD-10-CM)

## Treatments
- Chemotherapy
- Radiation therapy

## mCODE Compliance
- Patient demographics: ✓ Complete
- Cancer condition: ✓ Complete
- Treatment plan: ✓ Complete
```

## Data Validation

### Validation Levels

#### 1. Structural Validation
- FHIR resource structure compliance
- Required field presence
- Data type correctness
- Cardinality constraints

#### 2. Terminology Validation
- Code system validity
- Code existence in reference terminologies
- Value set membership
- Cross-system code consistency

#### 3. Clinical Validation
- Logical consistency of clinical data
- Age/gender appropriateness
- Temporal relationship validity
- Clinical plausibility

### Validation Rules

#### FHIR Profile Validation
```python
def validate_fhir_profile(resource, profile_url):
    # Check if resource conforms to specified profile
    if "meta" not in resource:
        resource["meta"] = {}
    
    if "profile" not in resource["meta"]:
        resource["meta"]["profile"] = []
    
    return profile_url in resource["meta"]["profile"]
```

#### Code Validation
```python
def validate_code(code, system):
    # Validate code against terminology server
    valid_systems = [
        "http://hl7.org/fhir/sid/icd-10-cm",
        "http://snomed.info/sct",
        "http://www.nlm.nih.gov/research/umls/rxnorm"
    ]
    
    if system not in valid_systems:
        return False, f"Invalid code system: {system}"
    
    # In production, this would query a terminology server
    return True, "Code is valid"
```

#### Clinical Plausibility Validation
```python
def validate_clinical_plausibility(patient_data, condition_data):
    issues = []
    
    # Check age appropriateness for condition
    if "breast cancer" in condition_data.get("condition", "").lower():
        min_age = patient_data.get("demographics", {}).get("age", {}).get("min", 0)
        if min_age < 15:
            issues.append("Minimum age may be too low for breast cancer")
    
    # Check gender appropriateness
    condition = condition_data.get("condition", "").lower()
    gender = patient_data.get("demographics", {}).get("gender", [])
    
    if "breast cancer" in condition and "male" in gender:
        issues.append("Male patients included for breast cancer trial - verify appropriateness")
    
    return issues
```

## Quality Metrics

### Completeness Metrics
- Percentage of required mCODE elements present
- Coverage of patient demographics
- Condition coding completeness
- Treatment information availability

### Accuracy Metrics
- Confidence scores for extracted elements
- Validation pass/fail rates
- Manual review requirements
- Mapping accuracy rates

### Performance Metrics
- Processing time per clinical trial
- Resource utilization
- Error rates
- Throughput capacity

## Validation Reporting

### Validation Report Structure
```json
{
  "validation_summary": {
    "total_resources": 15,
    "passed_validation": 13,
    "failed_validation": 2,
    "validation_score": 0.87
  },
  "validation_details": [
    {
      "resource_id": "condition-1",
      "resource_type": "Condition",
      "issues": [
        {
          "severity": "warning",
          "message": "Missing body site information",
          "rule": "mcode-body-site-required"
        }
      ]
    }
  ],
  "quality_metrics": {
    "completeness": 0.92,
    "accuracy": 0.88,
    "consistency": 0.95
  }
}
```

### Human-Readable Report
```markdown
# Validation Report for NCT00000001

## Summary
- Total resources processed: 15
- Passed validation: 13 (87%)
- Validation score: 87/100

## Issues Found
1. **Warning**: Missing body site information in Condition resource
   - Recommendation: Add SNOMED CT body site code

2. **Error**: Invalid date format in Patient resource
   - Required format: YYYY-MM-DD
   - Found: MM/DD/YYYY

## Quality Metrics
- Completeness: 92%
- Accuracy: 88%
- Consistency: 95%
```

## Error Handling

### Error Categories
1. **Format Errors** - Invalid JSON/XML structure
2. **Validation Errors** - FHIR profile violations
3. **Data Errors** - Inconsistent or impossible values
4. **System Errors** - Processing failures

### Error Response Formats

#### JSON Error Response
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid date format in Patient resource",
    "details": {
      "resource": "Patient",
      "field": "birthDate",
      "expected_format": "YYYY-MM-DD"
    }
  }
}
```

#### XML Error Response
```xml
<OperationOutcome xmlns="http://hl7.org/fhir">
  <issue>
    <severity value="error"/>
    <code value="invalid"/>
    <details>
      <text value="Invalid date format in Patient resource"/>
    </details>
  </issue>
</OperationOutcome>
```

## Export Options

### File Export Formats
- JSON (.json)
- XML (.xml)
- CSV (.csv)
- PDF (.pdf) for reports
- ZIP archives for batch exports

### Streaming Export
- Support for large dataset exports
- Progress indicators
- Resumable export capabilities
- Memory-efficient processing

## Integration with External Systems

### API Response Formats
- RESTful JSON responses
- FHIR-compliant operation outcomes
- Standardized error codes
- Pagination support

### Data Exchange Standards
- HL7 FHIR R4 compliance
- mCODE Implementation Guide adherence
- Standard HTTP status codes
- OAuth 2.0 authentication support

## Configuration Options

### Output Customization
- Selective resource inclusion
- Format-specific options (pretty print, compact)
- Validation level selection
- Quality threshold settings

### Validation Settings
- Strict vs. lenient validation modes
- Custom validation rules
- External terminology server configuration
- Validation timeout settings

## Performance Considerations

### Memory Management
- Streaming output for large datasets
- Efficient serialization algorithms
- Memory usage monitoring
- Garbage collection optimization

### Processing Optimization
- Parallel validation processing
- Caching of validation results
- Batch processing capabilities
- Resource-specific optimization