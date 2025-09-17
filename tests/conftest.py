"""
Shared test fixtures and configuration for mcode_translator tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Generator

# Setup color logging for all tests
from src.utils.logging_config import setup_logging, get_logger
setup_logging(level="DEBUG")

# Import data factories
try:
    from tests.data.factories import (
        TrialFactory,
        PatientFactory,
        McodeFactory,
        TestDataManager,
        test_data_manager
    )
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from data.factories import (
            TrialFactory,
            PatientFactory,
            McodeFactory,
            TestDataManager,
            test_data_manager
        )
    except ImportError:
        # Create dummy classes if factories don't exist
        class TrialFactory:
            @staticmethod
            def create_breast_cancer_trial():
                return {"nct_id": "NCT12345678", "title": "Breast Cancer Trial"}

            @staticmethod
            def create_lung_cancer_trial():
                return {"nct_id": "NCT87654321", "title": "Lung Cancer Trial"}

            @staticmethod
            def create_invalid_trial(field):
                return {}

            @staticmethod
            def create_large_trial():
                return {"nct_id": "NCT99999999", "title": "Large Trial", "conditions": ["Cancer"] * 100}

        class PatientFactory:
            @staticmethod
            def create_breast_cancer_patient():
                return {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Patient", "id": "12345"}}]}

            @staticmethod
            def create_lung_cancer_patient():
                return {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Patient", "id": "67890"}}]}

            @staticmethod
            def create_invalid_patient(field):
                return {}

            @staticmethod
            def create_patient_bundle(count):
                return {"resourceType": "Bundle", "entry": [{"resource": {"resourceType": "Patient", "id": f"patient_{i}"}} for i in range(count)]}

        class McodeFactory:
            @staticmethod
            def create_mcode_element():
                return {"element": "PrimaryCancerCondition", "code": "12345"}

        class TestDataManager:
            def cleanup(self):
                pass

        test_data_manager = TestDataManager()

# Test data fixtures
@pytest.fixture
def trial_factory():
    """Factory for creating trial test data."""
    return TrialFactory()

@pytest.fixture
def patient_factory():
    """Factory for creating patient test data."""
    return PatientFactory()

@pytest.fixture
def mcode_factory():
    """Factory for creating mCODE test data."""
    return McodeFactory()

@pytest.fixture
def test_data_manager_fixture():
    """Test data manager for cleanup."""
    return test_data_manager

@pytest.fixture
def breast_cancer_trial():
    """Sample breast cancer trial."""
    return TrialFactory.create_breast_cancer_trial()

@pytest.fixture
def lung_cancer_trial():
    """Sample lung cancer trial."""
    return TrialFactory.create_lung_cancer_trial()

@pytest.fixture
def breast_cancer_patient():
    """Sample breast cancer patient."""
    return PatientFactory.create_breast_cancer_patient()

@pytest.fixture
def lung_cancer_patient():
    """Sample lung cancer patient."""
    return PatientFactory.create_lung_cancer_patient()

@pytest.fixture
def invalid_trial():
    """Invalid trial for error testing."""
    return TrialFactory.create_invalid_trial("nct_id")

@pytest.fixture
def invalid_patient():
    """Invalid patient for error testing."""
    return PatientFactory.create_invalid_patient("gender")

@pytest.fixture
def large_trial():
    """Large trial for performance testing."""
    return TrialFactory.create_large_trial()

@pytest.fixture
def large_patient_bundle():
    """Large patient bundle for performance testing."""
    return PatientFactory.create_patient_bundle(50)
@pytest.fixture
def sample_trial_data() -> Dict[str, Any]:
    """Sample clinical trial data for testing."""
    return {
        "nct_id": "NCT12345678",
        "title": "Sample Clinical Trial",
        "eligibility": {
            "criteria": "Inclusion: Age >= 18\nExclusion: Prior chemotherapy"
        },
        "conditions": ["Breast Cancer"],
        "phases": ["Phase 2"],
        "interventions": [{"type": "Drug", "name": "Sample Drug"}]
    }

@pytest.fixture
def sample_patient_data() -> Dict[str, Any]:
    """Sample patient data for testing."""
    return {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "12345",
                    "gender": "female",
                    "birthDate": "1980-01-01"
                }
            }
        ]
    }

@pytest.fixture
def mock_api_response() -> Dict[str, Any]:
    """Mock API response for testing."""
    return {
        "status": "success",
        "data": {
            "nct_id": "NCT12345678",
            "title": "Mock Trial"
        }
    }

@pytest.fixture
def temp_cache_dir() -> Generator[str, None, None]:
    """Temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response for testing."""
    return {
        "mcode_elements": [
            {
                "element": "PrimaryCancerCondition",
                "system": "http://snomed.info/sct",
                "code": "254837009",
                "display": "Malignant neoplasm of breast"
            }
        ],
        "metadata": {
            "confidence": 0.95,
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            }
        }
    }

@pytest.fixture
def mock_config() -> Mock:
    """Mock configuration object."""
    config = Mock()
    config.get_api_key.return_value = "test-api-key"
    config.get_base_url.return_value = "https://api.example.com"
    config.get_model_name.return_value = "test-model"
    config.get_temperature.return_value = 0.7
    config.get_max_tokens.return_value = 1000
    return config

@pytest.fixture
def mock_core_memory_client() -> Mock:
    """Mock Core Memory client."""
    client = Mock()
    client.ingest.return_value = {"status": "success", "id": "test-id"}
    client.search.return_value = {"results": []}
    return client

@pytest.fixture
def test_logger():
    """Logger fixture for tests that need logging."""
    return get_logger("test_logger")

# Test data factories
def create_test_trial(nct_id: str = "NCT12345678", **kwargs) -> Dict[str, Any]:
    """Factory for creating test trial data."""
    base_trial = {
        "nct_id": nct_id,
        "title": kwargs.get("title", "Test Trial"),
        "status": kwargs.get("status", "Recruiting"),
        "phases": kwargs.get("phases", ["Phase 2"]),
        "conditions": kwargs.get("conditions", ["Cancer"]),
        "eligibility": {
            "criteria": kwargs.get("criteria", "Age >= 18")
        }
    }
    base_trial.update(kwargs)
    return base_trial

def create_test_patient(patient_id: str = "12345", **kwargs) -> Dict[str, Any]:
    """Factory for creating test patient data."""
    base_patient = {
        "resourceType": "Bundle",
        "id": patient_id,
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": patient_id,
                    "gender": kwargs.get("gender", "female"),
                    "birthDate": kwargs.get("birth_date", "1980-01-01")
                }
            }
        ]
    }
    base_patient.update(kwargs)
    return base_patient

# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "mock: mark test as using mocks (default)")
    config.addinivalue_line("markers", "live: mark test as using real data/services")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")

# Environment variable for enabling live tests
LIVE_TESTS_ENABLED = os.getenv("ENABLE_LIVE_TESTS", "false").lower() == "true"

def pytest_collection_modifyitems(config, items):
    """Skip live tests unless explicitly enabled."""
    if not LIVE_TESTS_ENABLED:
        skip_live = pytest.mark.skip(reason="Live tests disabled. Set ENABLE_LIVE_TESTS=true to run.")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)