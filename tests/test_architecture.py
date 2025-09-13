"""
Test the refactored mCODE translator architecture.

This module tests the new modular architecture to ensure all components
work together correctly.
"""

from unittest.mock import Mock, patch

import pytest

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config
from src.workflows.base_workflow import WorkflowResult
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow
from src.workflows.trials_processor_workflow import TrialsProcessorWorkflow


class TestArchitecture:
    """Test the overall architecture components."""

    def test_workflow_base_class(self):
        """Test that base workflow provides expected functionality."""
        config = Config()
        workflow = TrialsFetcherWorkflow(config)

        # Test basic attributes
        assert hasattr(workflow, "config")
        assert hasattr(workflow, "logger")
        assert hasattr(workflow, "execute")

        # Test result creation
        result = workflow._create_result(True, {"test": "data"})
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.data == {"test": "data"}

    def test_fetcher_workflows(self):
        """Test that fetcher workflows don't store to core memory."""
        config = Config()

        # Trials fetcher
        trials_fetcher = TrialsFetcherWorkflow(config)
        assert isinstance(trials_fetcher, TrialsFetcherWorkflow)

        # Patients fetcher
        patients_fetcher = PatientsFetcherWorkflow(config)
        assert isinstance(patients_fetcher, PatientsFetcherWorkflow)

    def test_processor_workflows(self):
        """Test that processor workflows can handle core memory storage."""
        config = Config()

        # Mock memory storage
        mock_memory = Mock()

        # Trials processor
        trials_processor = TrialsProcessorWorkflow(config, mock_memory)
        assert isinstance(trials_processor, TrialsProcessorWorkflow)
        assert trials_processor.memory_storage == mock_memory

        # Patients processor
        patients_processor = PatientsProcessorWorkflow(config, mock_memory)
        assert isinstance(patients_processor, PatientsProcessorWorkflow)
        assert patients_processor.memory_storage == mock_memory

    def test_optimizer_workflow(self):
        """Test that optimizer workflow doesn't use core memory."""
        config = Config()

        optimizer = TrialsOptimizerWorkflow(config)
        assert isinstance(optimizer, TrialsOptimizerWorkflow)

        # Should not have memory storage
        assert not hasattr(optimizer, "memory_storage")

    @patch("src.utils.core_memory_client.CoreMemoryClient")
    def test_memory_storage(self, mock_client):
        """Test mCODE memory storage interface."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        storage = McodeMemoryStorage("test_key", "test_source")

        # Test trial storage
        result = storage.store_trial_mcode_summary("NCT123", {"test": "data"})
        assert result is True
        mock_client_instance.ingest.assert_called_once()

        # Test patient storage
        mock_client_instance.reset_mock()
        result = storage.store_patient_mcode_summary("patient_123", {"test": "data"})
        assert result is True
        mock_client_instance.ingest.assert_called_once()

    def test_cli_utils(self):
        """Test shared CLI utilities."""
        # Test argument patterns
        assert hasattr(McodeCLI, "add_core_args")
        assert hasattr(McodeCLI, "add_memory_args")
        assert hasattr(McodeCLI, "add_fetcher_args")
        assert hasattr(McodeCLI, "add_processor_args")
        assert hasattr(McodeCLI, "add_optimizer_args")

    def test_workflow_inheritance(self):
        """Test that workflows properly inherit from base classes."""
        from src.workflows.base_workflow import (BaseWorkflow, FetcherWorkflow,
                                                 ProcessorWorkflow)

        config = Config()

        # Test inheritance hierarchy
        trials_fetcher = TrialsFetcherWorkflow(config)
        assert isinstance(trials_fetcher, FetcherWorkflow)
        assert isinstance(trials_fetcher, BaseWorkflow)

        trials_processor = TrialsProcessorWorkflow(config)
        assert isinstance(trials_processor, ProcessorWorkflow)
        assert isinstance(trials_processor, BaseWorkflow)

        patients_fetcher = PatientsFetcherWorkflow(config)
        assert isinstance(patients_fetcher, FetcherWorkflow)
        assert isinstance(patients_fetcher, BaseWorkflow)

        patients_processor = PatientsProcessorWorkflow(config)
        assert isinstance(patients_processor, ProcessorWorkflow)
        assert isinstance(patients_processor, BaseWorkflow)

        optimizer = TrialsOptimizerWorkflow(config)
        assert isinstance(optimizer, BaseWorkflow)
        # Optimizer doesn't inherit from Fetcher or Processor

    def test_workflow_error_handling(self):
        """Test that workflows handle errors properly."""
        config = Config()
        workflow = TrialsFetcherWorkflow(config)

        # Test error result creation
        result = workflow._handle_error(ValueError("Test error"), "test context")
        assert isinstance(result, WorkflowResult)
        assert result.success is False
        assert "Test error" in result.error_message
        assert result.metadata["error_type"] == "ValueError"

    def test_memory_storage_summaries(self):
        """Test that memory storage creates proper natural language summaries."""
        with patch("src.utils.core_memory_client.CoreMemoryClient") as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance

            storage = McodeMemoryStorage("test_key")

            # Test trial summary creation
            trial_data = {
                "mcode_mappings": [
                    {
                        "mcode_element": "CancerCondition",
                        "value": "Breast Cancer",
                        "system": "http://snomed.info/sct",
                        "code": "254837009",
                    }
                ],
                "metadata": {"brief_title": "Test Trial", "sponsor": "Test Sponsor"},
            }

            storage.store_trial_mcode_summary("NCT123", trial_data)

            # Verify the call was made with natural language summary
            call_args = mock_client_instance.ingest.call_args[0][0]
            assert "Clinical Trial NCT123" in call_args
            assert "Test Trial" in call_args
            assert "Test Sponsor" in call_args
            assert "Breast Cancer" in call_args
            assert "[SNOMED:254837009]" in call_args


if __name__ == "__main__":
    pytest.main([__file__])
