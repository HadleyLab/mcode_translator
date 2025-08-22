# MCODE Extraction Integration Testing

This document describes the integration tests for mCODE extraction from clinical trial eligibility criteria, specifically for NCT01698281.

## Overview

The integration tests validate the complete mCODE extraction pipeline, including:

1. **ClinicalTrials.gov API integration** - Fetching study data
2. **NLP processing** - Extracting entities and features from eligibility criteria
3. **Code extraction** - Identifying medical codes from text
4. **mCODE mapping** - Converting extracted data to mCODE standards
5. **Structured data generation** - Creating FHIR resources
6. **Validation** - Ensuring mCODE compliance

## Test Files

### 1. `test_nct01698281_mcode_integration.py`
**Purpose**: Complete end-to-end integration test that runs the actual extraction pipeline.

**Features**:
- Tests study data retrieval from ClinicalTrials.gov API
- Validates NLP processing of eligibility criteria
- Tests code extraction functionality
- Validates mCODE mapping and transformation
- Tests structured FHIR data generation
- Validates mCODE compliance checking

### 2. `test_mcode_extraction_validation.py`
**Purpose**: Validates the structure and content of mCODE extraction results.

**Features**:
- Validates NLP features structure (biomarkers, demographics, etc.)
- Tests code extraction structure and metadata
- Validates FHIR bundle structure
- Tests validation results structure
- Specific content validation for NCT01698281:
  - Triple negative biomarkers (ER-, PR-, HER2-)
  - LHRH receptor positive status
  - Female-only demographic constraints
  - Age ≥ 18 requirement
  - Metastatic stage validation
  - Treatment history constraints

### 3. `test_mcode_pipeline_integration.py`
**Purpose**: Complete pipeline testing with actual CLI execution and result validation.

**Features**:
- Tests complete CLI execution pipeline
- Validates output file creation and structure
- Tests result consistency across runs
- Validates mCODE compliance scores
- Saves validated results for reference

## Running the Tests

### Prerequisites
- Python 3.7+
- Dependencies installed: `pip install -r requirements.txt`
- mcode_translator environment activated

### Running All Integration Tests
```bash
cd tests/integration
python -m pytest -v
```

### Running Specific Test Files
```bash
# Run validation tests only
python -m pytest test_mcode_extraction_validation.py -v

# Run pipeline tests only
python -m pytest test_mcode_pipeline_integration.py -v
```

### Running Individual Test Methods
```bash
# Run specific test methods
python -m pytest test_mcode_extraction_validation.py::TestMCODEExtractionValidation::test_triple_negative_biomarkers -v
```

## Test Data - NCT01698281

The tests focus on **NCT01698281** - "Phase 2 Trial of AEZS-108 in Chemotherapy Refractory Triple Negative, LHRH-positive Metastatic Breast Cancer"

### Key Characteristics Validated:
- **Biomarkers**: ER-negative, PR-negative, HER2-negative, LHRH-receptor positive
- **Demographics**: Female only, age ≥ 18 years
- **Cancer Stage**: Metastatic (Stage IV)
- **Treatment History**: Prior chemotherapy constraints
- **Performance Status**: ECOG ≤ 2

### Expected mCODE Output:
- Patient resource with gender and demographic extensions
- Validation indicating mCODE compliance
- Structured FHIR bundle with cancer patient profile

## Test Results

After running the extraction pipeline, the following results are validated:

### NLP Features Extraction
- ✅ Genomic variants detection
- ✅ Biomarker status extraction (ER, PR, HER2, LHRHRECEPTOR)
- ✅ Cancer characteristics (stage, metastasis)
- ✅ Treatment history constraints
- ✅ Performance status (ECOG)
- ✅ Demographic constraints

### Code Extraction
- ✅ Medical code identification
- ✅ Coding system detection (ICD10CM, CPT, LOINC, RxNorm)
- ✅ Metadata tracking

### mCODE Mapping
- ✅ Entity to mCODE element mapping
- ✅ Compliance validation
- ✅ Structured data generation

### FHIR Resource Generation
- ✅ Patient resource creation
- ✅ mCODE cancer patient profile
- ✅ Bundle structure validation
- ✅ Compliance scoring

## Validation Criteria

### Success Criteria:
- All tests pass without errors
- mCODE compliance score ≥ 0.5
- No critical validation errors
- Structured data contains required resources
- Biomarker status correctly identified
- Demographic constraints properly extracted

### Warning Conditions:
- Missing optional fields (acceptable)
- Non-standard value set usage (monitored)
- Partial code extraction (tracked in metadata)

## Usage Examples

### 1. Running Complete Extraction
```bash
python src/data_fetcher/fetcher.py --nct-id NCT01698281 --process-criteria --export results.json
```

### 2. Validating Results
```python
from tests.integration.test_mcode_extraction_validation import TestMCODEExtractionValidation

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Validate structure
validator = TestMCODEExtractionValidation()
validator.test_nlp_features_structure(results['mcodeResults'])
validator.test_triple_negative_biomarkers(results['mcodeResults'])
```

### 3. Monitoring Pipeline Health
```bash
# Run integration tests regularly
python -m pytest tests/integration/ -v --tb=short

# Check for regressions
python -m pytest tests/integration/test_mcode_pipeline_integration.py::TestMCODEPipelineIntegration::test_mcode_results_consistency -v
```

## Troubleshooting

### Common Issues:
1. **API Connection Errors**: Check internet connection and ClinicalTrials.gov API status
2. **Module Import Errors**: Ensure all dependencies are installed and environment is activated
3. **Validation Failures**: Check if ClinicalTrials.gov data structure has changed
4. **NLP Processing Errors**: Verify NLP engine configuration and API keys

### Debugging:
- Enable debug logging: `export LOG_LEVEL=DEBUG`
- Check test output files in `test_results/` directory
- Review validation warnings for specific issues

## Maintenance

### Regular Updates:
- Update test data if ClinicalTrials.gov API changes
- Review mCODE standards for compliance updates
- Monitor NLP engine performance and accuracy
- Update validation criteria as extraction improves

### Adding New Tests:
1. Create new test file in `tests/integration/`
2. Follow existing patterns and structure
3. Include comprehensive validation
4. Add to documentation
5. Run existing tests to ensure no regressions

## Conclusion

The integration tests provide comprehensive validation of the mCODE extraction pipeline, ensuring reliable and accurate transformation of clinical trial eligibility criteria into standardized mCODE format. The tests validate both the technical implementation and the clinical accuracy of the extracted data.