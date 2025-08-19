# mCODE Extraction and Mapping Pipeline

## Overview

This document provides a comprehensive overview of the mCODE extraction and mapping pipeline implemented in the mCODE Translator. The pipeline processes clinical trial eligibility criteria to extract structured mCODE data elements that can be used for patient matching and other clinical applications.

## Pipeline Architecture

The mCODE extraction and mapping pipeline consists of several interconnected components that work together to transform unstructured clinical trial eligibility criteria into structured mCODE FHIR resources:

```
Clinical Trial Eligibility Criteria
           ↓
    [NLP Engine Processing]
           ↓
   Extracted Entities & Features
           ↓
   [Code Extraction Module]
           ↓
      Standard Codes
           ↓
   [mCODE Mapping Engine]
           ↓
   mCODE Data Elements
           ↓
[Structured Data Generator]
           ↓
   FHIR mCODE Resources
```

## Component Details

### 1. NLP Engine Processing

The first step in the pipeline is to process the eligibility criteria text using one of the available NLP engines:

#### Available NLP Engines

1. **LLM-based Engine** (`llm_nlp_engine.py`)
   - Uses DeepSeek API for advanced natural language understanding
   - Specialized for breast cancer criteria extraction
   - Handles complex criteria with high accuracy
   - Outputs structured JSON with validation

2. **spaCy-based Engine** (`spacy_nlp_engine.py`)
   - Uses medical NLP models (en_core_sci_md)
   - Modular pipeline architecture
   - Supports general clinical trial criteria
   - Includes custom pattern matching

3. **Regex-based Engine** (`regex_nlp_engine.py`)
   - Simple pattern matching approach
   - Fast processing with minimal dependencies
   - Good for well-structured criteria
   - Easy to extend with new patterns

#### NLP Output Structure

The NLP engines produce structured output containing:

```json
{
  "entities": [
    {
      "text": "breast cancer",
      "type": "condition",
      "confidence": 0.95,
      "start": 45,
      "end": 57
    }
  ],
  "features": {
    "demographics": {
      "age": {"min": 18, "max": 75},
      "gender": ["female"]
    },
    "cancer_characteristics": {
      "stage": "II",
      "tumor_size": ">=2cm",
      "metastasis_sites": []
    }
  }
}
```

### 2. Code Extraction Module

The code extraction module identifies and validates standard medical codes from the parsed text and NLP output:

#### Functionality

- **Code Identification**: Recognizes code references in parsed text
- **Code Validation**: Validates extracted codes against standard terminologies
- **Code Mapping**: Maps between different coding systems
- **Confidence Scoring**: Assigns confidence scores to extracted codes

#### Supported Coding Systems

- **ICD-10-CM**: International Classification of Diseases
- **CPT**: Current Procedural Terminology
- **LOINC**: Logical Observation Identifiers Names and Codes
- **RxNorm**: Normalized names for clinical drugs

#### Code Extraction Output

```json
{
  "extracted_codes": {
    "ICD10CM": [
      {
        "text": "C50.911",
        "code": "C50.911",
        "confidence": 0.95
      }
    ],
    "CPT": [],
    "LOINC": [],
    "RxNorm": []
  }
}
```

### 3. mCODE Mapping Engine

The mCODE mapping engine converts the extracted entities and codes into standardized mCODE data elements:

#### Core Functionality

- **Concept Mapping**: Maps identified concepts to mCODE elements
- **Code Translation**: Translates standard codes to mCODE-compliant formats
- **Element Generation**: Creates mCODE-specific data elements
- **Validation**: Ensures compliance with mCODE standards

#### mCODE Elements Generated

- **Patient Demographics**: Age, gender, ethnicity
- **Cancer Conditions**: Primary cancer, secondary cancers
- **Tumor Characteristics**: Stage, grade, biomarkers
- **Treatment History**: Medications, procedures, radiation
- **Laboratory Values**: Biomarker test results

#### Mapping Engine Output

```json
{
  "mapped_elements": [
    {
      "mcode_element": "CancerDiseaseStatus",
      "primary_code": {
        "system": "http://hl7.org/fhir/sid/icd-10-cm",
        "code": "C50.911"
      },
      "mapped_codes": {
        "ICD10CM": "C50.911",
        "SNOMEDCT": "254837009"
      }
    }
  ]
}
```

### 4. Structured Data Generator

The structured data generator creates FHIR resources from the mapped mCODE elements:

#### FHIR Resource Generation

- **Patient Resources**: Demographic information
- **Condition Resources**: Medical conditions and diagnoses
- **Procedure Resources**: Treatments and interventions
- **MedicationStatement Resources**: Drug therapies
- **Observation Resources**: Lab values and vital signs

#### Bundle Creation

All generated resources are bundled together in a FHIR Bundle:

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
              "code": "C50.911"
            }
          ]
        }
      }
    }
  ]
}
```

## Pipeline Integration

### Data Flow

1. **Input**: Clinical trial eligibility criteria text
2. **NLP Processing**: Extract entities and features
3. **Code Extraction**: Identify and validate standard codes
4. **mCODE Mapping**: Convert to mCODE elements
5. **Resource Generation**: Create FHIR resources
6. **Output**: Structured mCODE FHIR bundle

### Error Handling

The pipeline includes comprehensive error handling:

- **NLP Processing Failures**: Fallback to alternative engines
- **Code Validation Errors**: Flag invalid codes for review
- **Mapping Issues**: Handle unmappable concepts gracefully
- **Validation Failures**: Provide detailed error reports

## Usage Examples

### Command Line Interface

```bash
# Process a specific clinical trial with mCODE extraction
python src/data_fetcher/fetcher.py --nct-id NCT03929822 --process-criteria

# Export results to JSON file
python src/data_fetcher/fetcher.py --nct-id NCT03929822 --process-criteria --export study_with_mcode.json
```

### Programmatic Usage

```python
from src.data_fetcher.fetcher import get_full_study
from src.nlp_engine.llm_nlp_engine import LLMNLPEngine
from src.code_extraction.code_extraction import CodeExtractionModule
from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
from src.structured_data_generator.structured_data_generator import StructuredDataGenerator

# Get study data
study = get_full_study("NCT03929822")

# Extract eligibility criteria
eligibility_criteria = study["protocolSection"]["eligibilityModule"]["eligibilityCriteria"]

# Process through pipeline
nlp_engine = LLMNLPEngine()
nlp_result = nlp_engine.process_criteria(eligibility_criteria)

code_extractor = CodeExtractionModule()
code_result = code_extractor.process_criteria_for_codes(eligibility_criteria, nlp_result.entities)

mapper = MCODEMappingEngine()
mapped_elements = mapper.map_entities_to_mcode(nlp_result.entities + code_result.extracted_codes)

generator = StructuredDataGenerator()
structured_data = generator.generate_mcode_resources(mapped_elements, nlp_result.features.demographics)
```

## Quality Assurance

### Validation Process

The pipeline includes multiple validation steps:

1. **Format Validation**: Ensure proper data structure
2. **Code Validation**: Verify code existence and format
3. **mCODE Compliance**: Check adherence to mCODE standards
4. **Consistency Checks**: Identify conflicting information

### Confidence Scoring

Each step in the pipeline assigns confidence scores:

- **NLP Confidence**: 0.0 - 1.0 based on extraction quality
- **Code Confidence**: 0.0 - 1.0 based on validation results
- **Mapping Confidence**: 0.0 - 1.0 based on mapping quality
- **Overall Confidence**: Combined score for the entire pipeline

## Performance Considerations

### Caching Strategy

- **NLP Results**: Cache API responses for LLM engine
- **Code Lookups**: Cache validated code information
- **Mappings**: Cache successful mCODE mappings
- **Generated Resources**: Cache final FHIR resources

### Batch Processing

- **Parallel NLP Processing**: Process multiple criteria simultaneously
- **Bulk Code Validation**: Validate codes in batches
- **Batch Resource Generation**: Create multiple resources at once

## Future Enhancements

### Planned Improvements

1. **Enhanced NLP Models**: Better handling of complex criteria
2. **Expanded Code Systems**: Support for additional coding systems
3. **Improved Mapping Algorithms**: More accurate concept mapping
4. **Active Learning**: Continuous improvement with expert feedback

### Integration Opportunities

1. **EHR Integration**: Direct connection to electronic health records
2. **Terminology Services**: Real-time access to updated code systems
3. **Machine Learning**: Advanced models for better extraction accuracy
4. **Clinical Decision Support**: Integration with CDS systems