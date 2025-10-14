# mCODE Ontology Implementation Guide

## Overview

The mCODE Translator implements comprehensive support for the **mCODE (Minimal Common Oncology Data Elements) STU4 (4.0.0)** specification, providing standardized oncology data elements for clinical research and healthcare interoperability.

## mCODE Specification

### What is mCODE?

mCODE (Minimal Common Oncology Data Elements) is a standardized set of FHIR profiles and extensions designed to facilitate the structured exchange of cancer-related data across healthcare systems. Developed by the HL7 community, mCODE enables:

- **Interoperable Cancer Data**: Standardized representation of cancer patient information
- **Clinical Research**: Consistent data structures for clinical trials and research
- **Healthcare Analytics**: Structured data for population health and outcomes research
- **Regulatory Compliance**: Standardized reporting for oncology data requirements

### Current Version: STU4 (4.0.0)

The implementation supports mCODE STU4 (4.0.0), published in 2024, which includes:

- **40+ FHIR Profiles**: Comprehensive oncology data structures
- **14 Extensions**: mCODE-specific data elements
- **70+ Value Sets**: Controlled terminologies for oncology
- **1 Code System**: mCODE-specific codes
- **6 Thematic Groups**: Organized data categories

## Implementation Architecture

### Core Components

#### 1. mCODE Models (`src/shared/mcode_models.py`)

Complete Pydantic models for all mCODE profiles with validation and type safety:

```python
# Patient Profile
class McodePatient(FHIRPatient):
    """mCODE Patient profile with required extensions."""
    extension: Optional[List[Union[BirthSexExtension, USCoreRaceExtension]]] = Field(None)

# Cancer Condition Profile
class CancerCondition(FHIRCondition):
    """mCODE Cancer Condition with histology and laterality."""
    extension: Optional[List[Union[HistologyMorphologyBehaviorExtension, LateralityExtension]]] = Field(None)
```

#### 2. Versioning System (`src/shared/mcode_versioning.py`)

Multi-version support with compatibility checking:

```python
# Version management
version_manager = McodeVersionManager()
profile_url = version_manager.get_profile_url(McodeProfile.CANCER_CONDITION, McodeVersion.STU4)

# Compatibility checking
compatibility = version_manager.check_compatibility(McodeVersion.STU3, McodeVersion.STU4)
```

#### 3. Validation Framework (`src/shared/data_quality_validator.py`)

Comprehensive validation for mCODE compliance:

```python
validator = McodeValidator()
result = validator.validate_cancer_condition(cancer_condition)
```

## mCODE Profiles Implementation

### Patient Information Profiles

#### Core Patient Demographics

```python
# Administrative Gender (required)
patient = McodePatient(
    resourceType="Patient",
    gender=AdministrativeGender.FEMALE,  # Required
    birthDate="1975-06-15"
)

# Birth Sex Extension
birth_sex_ext = BirthSexExtension(valueCode=BirthSex.F)

# US Core Race Extension
race_ext = USCoreRaceExtension(
    valueCodeableConcept=FHIRCodeableConcept(
        coding=[{"system": "urn:oid:2.16.840.1.113883.6.238", "code": "2106-3", "display": "White"}]
    )
)
```

#### Extensions

- **Birth Sex**: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-birth-sex`
- **US Core Race**: `http://hl7.org/fhir/us/core/StructureDefinition/us-core-race`
- **US Core Ethnicity**: `http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity`

### Disease Characterization

#### Cancer Condition

Primary cancer diagnosis with comprehensive metadata:

```python
cancer_condition = CancerCondition(
    resourceType="Condition",
    subject=FHIRReference(reference="Patient/123"),
    clinicalStatus=FHIRCodeableConcept(
        coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
    ),
    category=[FHIRCodeableConcept(
        coding=[{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]
    )],
    code=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "254837009", "display": "Malignant neoplasm of breast"}]
    )
)
```

#### Extensions

- **Histology Morphology Behavior**: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-histology-morphology-behavior`
- **Laterality**: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-laterality`
- **Related Condition**: `http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-related-condition`

#### Cancer Staging

TNM staging observations with structured components:

```python
tnm_stage = TNMStageGroup(
    resourceType="Observation",
    status="final",
    code=FHIRCodeableConcept(
        coding=[{"system": "http://loinc.org", "code": "21908-9", "display": "Stage group"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    valueCodeableConcept=FHIRCodeableConcept(
        coding=[{"system": "urn:oid:2.16.840.1.113883.3.26.1.1", "code": "IIIA"}]
    )
)
```

### Assessment Profiles

#### Tumor Marker Tests

Biomarker testing with standardized results:

```python
er_test = TumorMarkerTest(
    resourceType="Observation",
    status="final",
    category=[FHIRCodeableConcept(
        coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]
    )],
    code=FHIRCodeableConcept(
        coding=[{"system": "http://loinc.org", "code": "85310-0", "display": "Estrogen receptor Ag [Presence] in Breast cancer specimen by Immune stain"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    valueCodeableConcept=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "10828004", "display": "Positive"}]
    )
)
```

#### Performance Status

ECOG and Karnofsky performance assessments:

```python
ecog_status = ECOGPerformanceStatusObservation(
    resourceType="Observation",
    status="final",
    code=FHIRCodeableConcept(
        coding=[{"system": "http://loinc.org", "code": "89247-1", "display": "ECOG Performance Status score"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    valueCodeableConcept=FHIRCodeableConcept(
        coding=[{"system": "urn:oid:2.16.840.1.113883.3.26.1.1", "code": "1", "display": "Restricted in physically strenuous activity"}]
    )
)
```

### Treatment Profiles

#### Cancer-Related Medications

Structured medication administration:

```python
chemo_medication = CancerRelatedMedicationStatement(
    resourceType="MedicationStatement",
    status="active",
    category=FHIRCodeableConcept(
        coding=[{"system": "http://terminology.hl7.org/CodeSystem/medication-statement-category", "code": "patientspecified"}]
    ),
    medicationCodeableConcept=FHIRCodeableConcept(
        coding=[{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "1736854", "display": "paclitaxel"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    effectivePeriod=FHIRPeriod(start="2024-01-15", end="2024-04-15")
)
```

#### Surgical Procedures

Cancer-related surgical interventions:

```python
mastectomy = CancerRelatedSurgicalProcedure(
    resourceType="Procedure",
    status="completed",
    category=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "387713003", "display": "Surgical procedure"}]
    ),
    code=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "392021009", "display": "Mastectomy"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    performedDateTime="2024-02-01"
)
```

#### Radiation Procedures

Radiation therapy documentation:

```python
radiation_tx = CancerRelatedRadiationProcedure(
    resourceType="Procedure",
    status="completed",
    category=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "108290001", "display": "Radiation oncology AND/OR radiotherapy"}]
    ),
    code=FHIRCodeableConcept(
        coding=[{"system": "http://snomed.info/sct", "code": "45643008", "display": "Teleradiotherapy procedure"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    performedPeriod=FHIRPeriod(start="2024-02-15", end="2024-03-15")
)
```

## Genomics Support (STU2+)

### Cancer Genetic Variants

```python
genetic_variant = CancerGeneticVariant(
    resourceType="Observation",
    status="final",
    category=[FHIRCodeableConcept(
        coding=[{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]
    )],
    code=FHIRCodeableConcept(
        coding=[{"system": "http://loinc.org", "code": "69548-6", "display": "Genetic variant assessment"}]
    ),
    subject=FHIRReference(reference="Patient/123"),
    component=[
        {
            "code": {"coding": [{"system": "http://loinc.org", "code": "81290-9", "display": "BRCA1 gene mutations found [Identifier] in Blood or Tissue by Molecular genetics method"}]},
            "valueCodeableConcept": {"coding": [{"system": "http://www.ncbi.nlm.nih.gov/clinvar", "code": "17661"}]}
        }
    ]
)
```

## Validation and Quality Assurance

### Profile Validation

Each mCODE profile includes comprehensive validation:

```python
# Validate cancer condition
validator = McodeValidator()
validation_result = validator.validate_cancer_condition(cancer_condition)

if validation_result["valid"]:
    print("Cancer condition is mCODE compliant")
else:
    print("Validation issues:", validation_result["issues"])
```

### Ontology Completeness Validation

Automated completeness checking against sample data:

```python
# Run ontology validation
validator = McodeOntologyValidator()
report = validator.validate_ontology_completeness("data/patients.ndjson", "data/trials.ndjson")

print(f"Patient coverage: {report.patient_demographics_coverage:.1%}")
print(f"Cancer condition coverage: {report.cancer_condition_coverage:.1%}")
```

## Version Management

### Multi-Version Support

The implementation supports multiple mCODE versions with automatic compatibility:

```python
# Check version compatibility
compatibility = check_version_compatibility("3.0.0", "4.0.0")
if compatibility.compatible:
    print("Migration possible:", compatibility.migration_notes)
```

### Profile URL Generation

Automatic canonical URL generation for different versions:

```python
# Get STU4 profile URL
url = get_profile_url(McodeProfile.CANCER_CONDITION, McodeVersion.STU4)
# Returns: https://mcodeinitiative.org/fhir/stu4/StructureDefinition/CancerCondition
```

## Integration with Processing Pipeline

### mCODE Mapping in LLM Processing

The system integrates mCODE ontology with AI processing:

```python
# Process clinical text to mCODE elements
pipeline = McodePipeline(engine="llm", model_name="deepseek-coder")
result = pipeline.process(trial_data)

# Access mCODE mappings
for mapping in result.mcode_mappings:
    print(f"Element: {mapping.element_type}, Confidence: {mapping.confidence_score:.1%}")
```

### FHIR Bundle Generation

Convert processed data to FHIR bundles:

```python
# Create FHIR bundle from mCODE elements
bundle = create_fhir_bundle_from_mcode(patient, cancer_condition, staging)
print(f"Bundle contains {len(bundle['entry'])} resources")
```

## Best Practices

### Data Quality Guidelines

1. **Always validate** mCODE profiles before storage
2. **Use appropriate versions** for your use case
3. **Include required extensions** for complete profiles
4. **Validate against value sets** for controlled terminologies
5. **Check profile URLs** for version compatibility

### Implementation Recommendations

1. **Version Pinning**: Specify mCODE version in your configuration
2. **Validation Hooks**: Integrate validation into your processing pipeline
3. **Extension Management**: Handle optional extensions gracefully
4. **Code System Updates**: Stay current with terminology updates
5. **Testing**: Use provided validation scripts for quality assurance

## Troubleshooting

### Common Issues

#### Profile Validation Errors

```python
# Symptom: ValidationError on required fields
# Solution: Ensure all required fields are present
cancer_condition = CancerCondition(
    subject=FHIRReference(reference="Patient/123"),  # Required
    category=[...],  # Required
    code=...  # Required
)
```

#### Extension Issues

```python
# Symptom: Extension not recognized
# Solution: Use correct extension URLs
extension = HistologyMorphologyBehaviorExtension(
    url="http://hl7.org/fhir/us/mcode/StructureDefinition/mcode-histology-morphology-behavior",
    valueCodeableConcept=...
)
```

#### Version Compatibility

```python
# Symptom: Profile not available in version
# Solution: Check version compatibility
if not version_manager.is_version_supported(McodeVersion.STU4):
    print("STU4 not supported in current implementation")
```

## Resources

- **mCODE Specification**: https://mcodeinitiative.org/
- **HL7 FHIR mCODE IG**: https://hl7.org/fhir/us/mcode/
- **Implementation Guide**: https://build.fhir.org/ig/HL7/fhir-mCODE-ig/
- **Value Sets**: https://terminology.hl7.org/mCODE.html

This comprehensive implementation provides a solid foundation for oncology data standardization and interoperability using the mCODE ontology.