"""
Shared test fixtures and configuration for mcode_translator tests.
"""
import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Generator
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from heysol_api_client/.env relative to project root
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'heysol_api_client', '.env')
    load_dotenv(env_file)
    print(f"Loaded environment from: {env_file}")
except ImportError:
    # python-dotenv not available, skip loading .env file
    print("python-dotenv not available, skipping .env file loading")
except Exception as e:
    print(f"Error loading .env file: {e}")

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
def temp_cache_dir() -> Generator[str, None, None]:
    """Temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir




@pytest.fixture
def mock_memory_storage():
    """Mock memory storage for testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.store.return_value = True
    mock.retrieve.return_value = {"test": "data"}
    return mock


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from unittest.mock import MagicMock
    return MagicMock()
@pytest.fixture
def test_logger():
    """Logger fixture for tests that need logging."""
    return get_logger("test_logger")





# Test data factories
def create_test_trial(nct_id: str = "NCT12345678", **kwargs) -> Dict[str, Any]:
    """Factory for creating test trial data.

    Args:
        nct_id: ClinicalTrials.gov ID
        **kwargs: Additional trial fields to override defaults

    Returns:
        Dictionary containing trial data
    """
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
    """Factory for creating test patient data.

    Args:
        patient_id: Patient identifier
        **kwargs: Additional patient fields to override defaults

    Returns:
        Dictionary containing patient bundle data
    """
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
    config.addinivalue_line("markers", "live: mark test as using real data/services")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "performance: mark test as performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")

# Environment variable for enabling live tests (enabled by default)
LIVE_TESTS_ENABLED = os.getenv("ENABLE_LIVE_TESTS", "true").lower() == "true"

def pytest_collection_modifyitems(config, items):
    """Skip live tests unless explicitly enabled."""
    if not LIVE_TESTS_ENABLED:
        skip_live = pytest.mark.skip(reason="Live tests disabled. Set ENABLE_LIVE_TESTS=true to run.")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)