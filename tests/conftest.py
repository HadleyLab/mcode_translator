"""
Global pytest configuration and fixtures for the mCODE Translator project.
This module provides shared fixtures and configuration for all tests.
"""

import pytest
import sys
import os

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Core Component Fixtures

@pytest.fixture
def code_extractor():
    """Provide a CodeExtractionModule instance for testing"""
    from src.code_extraction.code_extraction import CodeExtractionModule
    return CodeExtractionModule()


@pytest.fixture
def mcode_mapper():
    """Provide an MCODEMappingEngine instance for testing"""
    from src.mcode_mapper.mcode_mapping_engine import MCODEMappingEngine
    return MCODEMappingEngine()


@pytest.fixture
def structured_data_generator():
    """Provide a StructuredDataGenerator instance for testing"""
    from src.structured_data_generator.structured_data_generator import StructuredDataGenerator
    return StructuredDataGenerator()


@pytest.fixture
def criteria_parser():
    """Provide a CriteriaParser instance for testing"""
    from src.criteria_parser.criteria_parser import CriteriaParser
    return CriteriaParser()


# NLP Engine Fixtures

@pytest.fixture
def regex_nlp_engine():
    """Provide a RegexNLPEngine instance for testing"""
    from src.nlp_engine.regex_nlp_engine import RegexNLPEngine
    return RegexNLPEngine()


@pytest.fixture
def spacy_nlp_engine():
    """Provide a SpacyNLPEngine instance for testing"""
    from src.nlp_engine.spacy_nlp_engine import SpacyNLPEngine
    return SpacyNLPEngine()


@pytest.fixture
def llm_nlp_engine():
    """Provide an LLMNLPEngine instance for testing"""
    from src.nlp_engine.llm_nlp_engine import LLMNLPEngine
    return LLMNLPEngine()


# Data Fixtures

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


# Patient Demographics Fixtures

@pytest.fixture
def sample_patient_demographics():
    """Provide sample patient demographics for testing"""
    return {
        "gender": "female",
        "age": "55",
        "birthDate": "1970-01-01"
    }


# Component-Specific Fixtures

@pytest.fixture
def sample_breast_cancer_profile():
    """Provide a BreastCancerProfile instance for testing"""
    from src.pipeline.breast_cancer_profile import BreastCancerProfile
    return BreastCancerProfile()


# Mock Object Fixtures

@pytest.fixture
def mock_clinical_trials_api():
    """Provide a mock ClinicalTrials.gov API for testing"""
    from unittest.mock import patch
    with patch('src.data_fetcher.fetcher.ClinicalTrials') as mock_client:
        mock_client.return_value.get_study_fields.return_value = {
            "StudyFields": [
                {"NCTId": ["NCT12345678"], "BriefTitle": ["Test Study 1"]},
                {"NCTId": ["NCT87654321"], "BriefTitle": ["Test Study 2"]}
            ]
        }
        yield mock_client

@pytest.fixture
def mock_cache_manager():
    """Provide a mock CacheManager for testing"""
    from unittest.mock import patch
    with patch('src.data_fetcher.fetcher.CacheManager') as mock_cache:
        mock_cache.return_value.get.return_value = None
        yield mock_cache


# Test Data Generation Fixtures

@pytest.fixture
def clinical_trial_generator():
    """Provide a ClinicalTrialDataGenerator instance for creating test data"""
    from tests.shared.test_components import ClinicalTrialDataGenerator
    return ClinicalTrialDataGenerator()


@pytest.fixture
def patient_data_generator():
    """Provide a PatientDataGenerator instance for creating test data"""
    from tests.shared.test_components import PatientDataGenerator
    return PatientDataGenerator()


# Pytest Configuration

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