# Global Pytest Configuration and Fixtures

This document outlines the global pytest configuration and shared fixtures for the mCODE Translator project.

## Overview

The `conftest.py` file provides global configuration and shared fixtures that are available to all test modules in the project. This centralized approach eliminates redundancy and promotes consistency across tests.

## Global Configuration

### Pytest Configuration

```python
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "component: mark test as component test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add any test collection modifications here
    pass

def pytest_runtest_setup(item):
    """Setup before each test"""
    # Add any test setup logic here
    pass
```

## Shared Fixtures

### 1. Core Component Fixtures

#### `code_extractor`
Provides a configured CodeExtractionModule instance for testing.

```python
import pytest
from src.code_extraction import CodeExtractionModule

@pytest.fixture
def code_extractor():
    """Provide a CodeExtractionModule instance for testing"""
    return CodeExtractionModule()
```

#### `mcode_mapper`
Provides a configured MCODEMappingEngine instance for testing.

```python
import pytest
from src.mcode_mapping_engine import MCODEMappingEngine

@pytest.fixture
def mcode_mapper():
    """Provide an MCODEMappingEngine instance for testing"""
    return MCODEMappingEngine()
```

#### `structured_data_generator`
Provides a configured StructuredDataGenerator instance for testing.

```python
import pytest
from src.structured_data_generator import StructuredDataGenerator

@pytest.fixture
def structured_data_generator():
    """Provide a StructuredDataGenerator instance for testing"""
    return StructuredDataGenerator()
```

#### `criteria_parser`
Provides a configured CriteriaParser instance for testing.

```python
import pytest
from src.criteria_parser import CriteriaParser

@pytest.fixture
def criteria_parser():
    """Provide a CriteriaParser instance for testing"""
    return CriteriaParser()
```

### 2. NLP Engine Fixtures

#### NLP Engine Instances

```python
import pytest
from src.regex_nlp_engine import RegexNLPEngine
from src.spacy_nlp_engine import SpacyNLPEngine
from src.llm_nlp_engine import LLMNLPEngine

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

### 3. Data Fixtures

#### `sample_eligibility_criteria`
Provides sample eligibility criteria text for testing.

```python
import pytest

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
import pytest

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
import pytest

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
import pytest

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

### 4. Patient Demographics Fixtures

#### `sample_patient_demographics`
Provides sample patient demographics for testing.

```python
import pytest

@pytest.fixture
def sample_patient_demographics():
    """Provide sample patient demographics for testing"""
    return {
        "gender": "female",
        "age": "55",
        "birthDate": "1970-01-01"
    }
```

### 5. Component-Specific Fixtures

#### `sample_breast_cancer_profile`
Provides a BreastCancerProfile instance for testing.

```python
import pytest
from src.breast_cancer_profile import BreastCancerProfile

@pytest.fixture
def sample_breast_cancer_profile():
    """Provide a BreastCancerProfile instance for testing"""
    return BreastCancerProfile()
```

## Test Data Generation Fixtures

### `test_data_generator`
Provides a TestDataGenerators instance for creating test data.

```python
import pytest
from tests.shared.test_data_generators import ClinicalTrialDataGenerator, PatientDataGenerator

@pytest.fixture
def clinical_trial_generator():
    """Provide a ClinicalTrialDataGenerator instance for creating test data"""
    return ClinicalTrialDataGenerator()

@pytest.fixture
def patient_data_generator():
    """Provide a PatientDataGenerator instance for creating test data"""
    return PatientDataGenerator()
```

## Mock Object Fixtures

### Clinical Trials API Mock

```python
import pytest
from unittest.mock import patch
from tests.shared.test_data_generators import MockClinicalTrialsAPI, MockCacheManager

@pytest.fixture
def mock_clinical_trials_api():
    """Provide a mock ClinicalTrials.gov API for testing"""
    with patch('src.data_fetcher.fetcher.ClinicalTrials') as mock_client:
        mock_client.return_value.get_study_fields.return_value = {
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT12345678",
                            "briefTitle": "Test Study 1"
                        }
                    }
                },
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        yield mock_client

@pytest.fixture
def mock_cache_manager():
    """Provide a mock CacheManager for testing"""
    with patch('src.data_fetcher.fetcher.CacheManager') as mock_cache:
        mock_cache.return_value.get.return_value = None
        yield mock_cache
```

## Usage Examples

### Using Core Component Fixtures

```python
def test_code_extraction(code_extractor, sample_eligibility_criteria):
    """Test code extraction with shared fixture"""
    codes = code_extractor.process_criteria_for_codes(sample_eligibility_criteria)
    assert len(codes['extracted_codes']) > 0

def test_mcode_mapping(mcode_mapper, sample_mcode_bundle):
    """Test mCODE mapping with shared fixture"""
    result = mcode_mapper.validate_mcode_compliance(sample_mcode_bundle)
    assert result['valid'] == True
```

### Using NLP Engine Fixtures

```python
def test_nlp_engine_comparison(regex_nlp_engine, spacy_nlp_engine, sample_eligibility_criteria):
    """Compare different NLP engines"""
    regex_result = regex_nlp_engine.process_criteria(sample_eligibility_criteria)
    spacy_result = spacy_nlp_engine.process_criteria(sample_eligibility_criteria)
    
    assert 'entities' in regex_result
    assert 'entities' in spacy_result
```

### Using Mock Object Fixtures

```python
def test_api_with_mock(mock_clinical_trials_api, mock_cache_manager):
    """Test API functionality with mock objects"""
    from src.data_fetcher.fetcher import search_trials
    
    result = search_trials("cancer", max_results=2)
    
    assert "studies" in result
    assert len(result["studies"]) == 2
    mock_clinical_trials_api.return_value.get_study_fields.assert_called()
```

## Benefits of Global Configuration and Fixtures

1. **Centralized Management**: All fixtures are defined in one place for easy maintenance
2. **Consistent Availability**: Fixtures are automatically available to all test modules
3. **Reduced Duplication**: No need to redefine common fixtures in multiple files
4. **Improved Performance**: Fixtures can be cached and reused across tests
5. **Enhanced Readability**: Tests are cleaner and more focused on actual test logic
6. **Better Organization**: Logical grouping of related fixtures
7. **Scalability**: Easy to add new fixtures as the project grows