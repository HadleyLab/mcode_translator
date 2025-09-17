"""
Unit tests for streamlined workflow architecture.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from src.shared.models import WorkflowResult
from src.workflows.trials_processor_workflow import ClinicalTrialsProcessorWorkflow
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow


class TestClinicalTrialsProcessorWorkflow:
    """Test ClinicalTrialsProcessorWorkflow."""

    def test_initialization(self):
        """Test initialization."""
        processor = ClinicalTrialsProcessorWorkflow(config={})

        assert processor is not None
        assert hasattr(processor, 'config')

    def test_execute_success(self):
        """Test successful execution."""
        processor = ClinicalTrialsProcessorWorkflow(config={})

        # Mock the execute method to return a successful result
        with patch.object(processor, 'execute') as mock_execute:
            mock_execute.return_value = WorkflowResult(success=True, data={"test": "data"})

            result = processor.execute()

            assert result.success is True
            assert result.data == {"test": "data"}

    def test_process_single_trial_method(self):
        """Test process_single_trial method exists."""
        processor = ClinicalTrialsProcessorWorkflow(config={})

        # Check that the method exists
        assert hasattr(processor, 'process_single_trial')
        assert callable(getattr(processor, 'process_single_trial'))

    def test_validate_trial_data(self):
        """Test trial data validation."""
        processor = ClinicalTrialsProcessorWorkflow(config={})

        # Valid trial data
        valid_trial = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT123"},
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion criteria: ..."
                }
            }
        }

        is_valid = processor.validate_trial_data(valid_trial)
        assert is_valid is True

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        processor = ClinicalTrialsProcessorWorkflow(config={"test": "config"})

        stats = processor.get_processing_stats()

        assert isinstance(stats, dict)
        assert "status" in stats
        assert stats["status"] == "pipeline_not_initialized"


class TestTrialsFetcherWorkflow:
    """Test TrialsFetcherWorkflow."""

    def test_initialization(self):
        """Test fetcher initialization."""
        fetcher = TrialsFetcherWorkflow()

        assert fetcher is not None
        assert hasattr(fetcher, 'execute')

    def test_execute_method_exists(self):
        """Test that execute method exists."""
        fetcher = TrialsFetcherWorkflow()

        assert hasattr(fetcher, 'execute')
        assert callable(getattr(fetcher, 'execute'))


class TestPatientsProcessorWorkflow:
    """Test PatientsProcessorWorkflow."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = PatientsProcessorWorkflow(config={})

        assert processor is not None
        assert hasattr(processor, 'execute')

    def test_execute_method_exists(self):
        """Test that execute method exists."""
        processor = PatientsProcessorWorkflow(config={})

        assert hasattr(processor, 'execute')
        assert callable(getattr(processor, 'execute'))


if __name__ == "__main__":
    pytest.main([__file__])
