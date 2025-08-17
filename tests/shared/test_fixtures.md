# Shared Test Fixtures for mCODE Translator

This document outlines the shared test fixtures that will be used across the mCODE Translator project to eliminate redundancy and promote code reuse.

## Overview

Shared test fixtures provide reusable setup and teardown functionality for common testing scenarios. These fixtures will be implemented using pytest and will be available to all test modules in the project.

## Available Fixtures

### 1. Core Component Fixtures

#### `code_extractor`
Provides a configured CodeExtractionModule instance for testing code extraction functionality.

```python
@pytest.fixture
def code_extractor():
    """Provide a CodeExtractionModule instance for testing"""
    return CodeExtractionModule()
```

#### `mcode_mapper`
Provides a configured MCODEMappingEngine instance for testing mCODE mapping functionality.

```python
@pytest.fixture
def mcode_mapper():
    """Provide an MCODEMappingEngine instance for testing"""
    return MCODEMappingEngine()
```

#### `nlp_engine`
Provides different NLP engine instances for testing.

```python
@pytest.fixture
def regex_nlp_engine():
    """Provide a RegexNLPEngine instance for testing"""
    return RegexNLPEngine()

@pytest.fixture
def spacy_nlp_engine():
    """Provide a SpacyNLPEngine instance for testing"""
    return SpacyNLPEngine()

@pytest.fixture
def llm_nlp_engine():
    """Provide an LLMNLPEngine instance for testing"""
    return LLMNLPEngine()
```

### 2. Data Fixtures

#### `sample_eligibility_criteria`
Provides sample eligibility criteria text for testing.

```python
@pytest.fixture
def sample_eligibility_criteria():
    """Provide sample eligibility criteria text for testing"""
    return """
    INCLUSION CRITERIA:
    - Histologically confirmed diagnosis of breast cancer (ICD-10-CM: C50.911)
    - Eastern Cooperative Oncology Group (ECOG) performance status of 0 or 1
    - Adequate organ function as defined by laboratory values
    - Female patients aged 18-75 years
    
    EXCLUSION CRITERIA:
    - Pregnant or nursing women
    - History of other malignancies within 5 years
    - Active infection requiring systemic therapy
    """
```

#### `sample_mcode_bundle`
Provides a sample mCODE bundle for testing.

```python
@pytest.fixture
def sample_mcode_bundle():
    """Provide a sample mCODE bundle for testing"""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "gender": "female",
                    "birthDate": "1970-01-01"
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
                                "display": "Malignant neoplasm of breast"
                            }
                        ]
                    }
                }
            }
        ]
    }
```

#### `biomarker_positive_bundle`
Provides an mCODE bundle with positive biomarkers for testing.

```python
@pytest.fixture
def biomarker_positive_bundle():
    """Provide an mCODE bundle with positive biomarkers for testing"""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "LP417347-6"}]  # ER
                    },
                    "valueCodeableConcept": {
                        "coding": [{"code": "LA6576-8"}]  # Positive
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "LP417351-8"}]  # HER2
                    },
                    "valueCodeableConcept": {
                        "coding": [{"code": "LA6576-8"}]  # Positive
                    }
                }
            }
        ]
    }
```

#### `biomarker_negative_bundle`
Provides an mCODE bundle with negative biomarkers for testing.

```python
@pytest.fixture
def biomarker_negative_bundle():
    """Provide an mCODE bundle with negative biomarkers for testing"""
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "LP417347-6"}]  # ER
                    },
                    "valueCodeableConcept": {
                        "coding": [{"code": "LA6577-6"}]  # Negative
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "LP417348-4"}]  # PR
                    },
                    "valueCodeableConcept": {
                        "coding": [{"code": "LA6577-6"}]  # Negative
                    }
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {
                        "coding": [{"code": "LP417351-8"}]  # HER2
                    },
                    "valueCodeableConcept": {
                        "coding": [{"code": "LA6577-6"}]  # Negative
                    }
                }
            }
        ]
    }
```

### 3. Clinical Trials API Fixtures

#### `mock_clinical_trials_api`
Provides a mock ClinicalTrials.gov API for testing.

```python
@pytest.fixture
def mock_clinical_trials_api():
    """Provide a mock ClinicalTrials.gov API for testing"""
    with patch('src.data_fetcher.fetcher.ClinicalTrials') as mock_client:
        mock_client.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]},
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        yield mock_client
```

#### `mock_cache_manager`
Provides a mock CacheManager for testing.

```python
@pytest.fixture
def mock_cache_manager():
    """Provide a mock CacheManager for testing"""
    with patch('src.data_fetcher.fetcher.CacheManager') as mock_cache:
        mock_cache.return_value.get.return_value = None
        yield mock_cache
```

### 4. Patient Demographics Fixtures

#### `sample_patient_demographics`
Provides sample patient demographics for testing.

```python
@pytest.fixture
def sample_patient_demographics():
    """Provide sample patient demographics for testing"""
    return {
        "gender": "female",
        "age": "55",
        "birthDate": "1970-01-01"
    }
```

### 5. Test Data Generation Fixtures

#### `test_data_generator`
Provides a TestDataGenerators instance for creating test data.

```python
@pytest.fixture
def test_data_generator():
    """Provide a TestDataGenerators instance for creating test data"""
    return TestDataGenerators()
```

## Usage Examples

### Using Core Component Fixtures

```python
def test_code_extraction(code_extractor, sample_eligibility_criteria):
    """Test code extraction with shared fixture"""
    codes = code_extractor.process_criteria_for_codes(sample_eligibility_criteria)
    assert len(codes['extracted_codes']) > 0
```

### Using Data Fixtures

```python
def test_mcode_mapping(mcode_mapper, sample_mcode_bundle):
    """Test mCODE mapping with shared fixture"""
    result = mcode_mapper.validate_mcode_compliance(sample_mcode_bundle)
    assert result['valid'] == True
```

### Using Clinical Trials API Fixtures

```python
def test_api_search(mock_clinical_trials_api, mock_cache_manager):
    """Test API search with mocked fixtures"""
    result = search_trials("cancer", max_results=2)
    assert isinstance(result, dict)
    assert "StudyFields" in result
```

## Benefits of Shared Fixtures

1. **Reduced Code Duplication**: Common setup code is defined once and reused
2. **Consistent Test Data**: All tests use the same sample data for consistency
3. **Easier Maintenance**: Changes to fixtures propagate to all tests that use them
4. **Faster Test Execution**: Optimized fixtures reduce setup time
5. **Improved Test Reliability**: Consistent fixtures reduce test flakiness