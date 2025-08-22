# mCODE Mapping and Translation Engine Design

## Overview
The mCODE Mapping and Translation Engine is responsible for converting extracted clinical trial criteria into standardized mCODE data elements. This component bridges the gap between unstructured clinical text and structured mCODE representations.

## Engine Architecture

### Core Components

#### 1. Concept Extractor
- Identifies medical concepts from parsed text
- Uses medical terminology dictionaries
- Applies NLP techniques for entity recognition
- Handles abbreviations and synonyms

#### 2. Code Mapper
- Maps identified concepts to standard codes
- References ICD-10-CM, CPT, LOINC, RxNorm
- Handles code hierarchies and relationships
- Manages cross-walks between coding systems

#### 3. mCODE Element Generator
- Creates mCODE-specific data elements
- Applies mCODE implementation guide rules
- Ensures compliance with FHIR profiles
- Handles required and optional elements

#### 4. Validation Engine
- Validates generated mCODE elements
- Checks for completeness and consistency
- Ensures adherence to mCODE standards
- Provides quality metrics and error reporting

## Concept Extraction

### Medical Entity Recognition

#### Condition Detection
```python
# Patterns for identifying medical conditions
condition_patterns = [
    r"\b(diagnosis|history|presence) of ([\w\s]+)\b",
    r"\b(?:diagnosed with|suffering from) ([\w\s]+)\b",
    r"\b(?:cancer|tumor|carcinoma|malignancy) of ([\w\s]+)\b"
]
```

#### Procedure Detection
```python
# Patterns for identifying procedures
procedure_patterns = [
    r"\b(?:underwent|received|history of) ([\w\s]+(?:surgery|therapy|treatment))\b",
    r"\b(?:radiation|chemotherapy|radiograph|ct scan|mri)\b"
]
```

#### Medication Detection
```python
# Patterns for identifying medications
medication_patterns = [
    r"\b(?:taking|taking|on|received) ([\w\s]+(?:inib|nib|ciclib|vir|prazole))\b",
    r"\b(?:medication|drug|treatment) ([\w\s]+)\b"
]
```

### Integration with Medical Terminology Services

#### UMLS Integration
- Query UMLS for concept identification
- Use UMLS Semantic Types for categorization
- Handle multiple language variants
- Access rich synonym and relationship data

#### Terminology Mapping
```python
# Example mapping structure
terminology_map = {
    "source": "Clinical Trial Criteria",
    "concept": "Breast Cancer",
    "umls_cui": "C0006142",
    "preferred_term": "Malignant neoplasm of breast",
    "semantic_type": "Neoplastic Process",
    "mapped_codes": {
        "ICD10CM": "C50.911",
        "ICDO3": "C50.9",
        "SNOMEDCT": "254837009"
    }
}
```

## Code Mapping Strategies

### Direct Mapping
- Use established cross-walks between coding systems
- Apply deterministic mapping rules
- Handle code hierarchies and generalizations
- Manage multiple valid mapping options

### Fuzzy Matching
- Handle variations in terminology
- Apply string similarity algorithms
- Use phonetic matching for common variations
- Implement context-aware matching

### Manual Override
- Allow expert curation of mappings
- Provide confidence scoring for automated mappings
- Track mapping decisions for audit purposes
- Support custom mapping rules

## mCODE Element Generation

### Patient Demographics Mapping
```json
{
  "resourceType": "Patient",
  "gender": "male|female|other|unknown",
  "birthDate": "1980-01-01",
  "extension": [
    {
      "url": "http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-ethnicity",
      "valueCodeableConcept": {
        "coding": [
          {
            "system": "http://hl7.org/fhir/us/mcode/CodeSystem/mcode-ethnicity",
            "code": "hispanic-or-latino",
            "display": "Hispanic or Latino"
          }
        ]
      }
    }
  ]
}
```

### Cancer Condition Mapping
```json
{
  "resourceType": "Condition",
  "code": {
    "coding": [
      {
        "system": "http://hl7.org/fhir/sid/icd-10-cm",
        "code": "C50.911",
        "display": "Malignant neoplasm of upper-outer quadrant of right female breast"
      }
    ]
  },
  "bodySite": {
    "coding": [
      {
        "system": "http://snomed.info/sct",
        "code": "76752008",
        "display": "Breast"
      }
    ]
  }
}
```

### Treatment Mapping
```json
{
  "resourceType": "MedicationStatement",
  "medicationCodeableConcept": {
    "coding": [
      {
        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
        "code": "123456",
        "display": "Paclitaxel"
      }
    ]
  }
}
```

## Validation Rules

### Completeness Checks
- Required mCODE elements present
- All referenced codes are valid
- No conflicting information
- Age/gender restrictions consistent
- Special handling for trial-specific elements:
  - LHRHRECEPTOR biomarker (LOINC LP417352-6)
  - Stage IV cancer (SNOMED 399555006)

### Consistency Checks
- No contradictory conditions
- Logical relationships between elements
- Temporal consistency of events
- Proper use of value sets

### Standards Compliance
- Adherence to mCODE Implementation Guide
- Valid FHIR resource structures
- Correct use of extensions
- Proper coding system URIs

## Error Handling and Quality Metrics

### Error Types
1. **Unmappable Concepts** - Terms that cannot be mapped to codes
2. **Ambiguous Mappings** - Terms with multiple possible mappings
3. **Inconsistent Data** - Conflicting information in criteria
4. **Missing Information** - Required elements not found

### Quality Metrics
- Mapping success rate
- Confidence scores for mappings
- Coverage of mCODE elements
- Processing time per trial

### Feedback Mechanisms
- Confidence scoring for automated mappings
- Manual review queue for low-confidence mappings
- Learning from expert corrections
- Continuous improvement of mapping rules

## Integration with External Services

### UMLS Terminology Services
- Concept identification and normalization
- Semantic type categorization
- Cross-reference mapping
- Variant and synonym identification

### LOINC/Terminology Servers
- Lab test code validation
- Observation code mapping
- Value set compliance checking
- Terminology updates and maintenance

### Custom Mapping Services
- Organization-specific mappings
- Local terminology preferences
- Custom value sets
- Institutional best practices

## Performance Considerations

### Caching Strategy
- Cache frequently used mappings
- Store mapping confidence scores
- Maintain mapping history for learning
- Handle cache invalidation

### Parallel Processing
- Batch process multiple concepts
- Parallel API calls where possible
- Asynchronous processing for slow services
- Progress tracking for long operations

### Memory Management
- Efficient data structures for mappings
- Streaming for large terminology datasets
- Memory pooling for repeated operations
- Garbage collection optimization