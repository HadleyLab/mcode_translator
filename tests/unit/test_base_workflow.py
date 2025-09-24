"""
Comprehensive unit tests for BaseWorkflow and workflow hierarchy.
Tests cover initialization, CORE memory integration, error handling,
and abstract method implementations.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.workflows.base_workflow import (
    BaseWorkflow,
    FetcherWorkflow,
    ProcessorWorkflow,
    TrialsProcessorWorkflow,
    PatientsProcessorWorkflow,
    SummarizerWorkflow,
    TrialsSummarizerWorkflow,
    PatientsSummarizerWorkflow,
    WorkflowError,
)
from src.shared.models import WorkflowResult, ProcessingMetadata


class ConcreteTestWorkflow(BaseWorkflow):
    """Concrete implementation of BaseWorkflow for testing."""

    @property
    def memory_space(self) -> str:
        return "test_space"

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"executed": True})


class ConcreteFetcherWorkflow(FetcherWorkflow):
    """Concrete implementation of FetcherWorkflow for testing."""

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"fetched": True})


class ConcreteTrialsProcessorWorkflow(TrialsProcessorWorkflow):
    """Concrete implementation of TrialsProcessorWorkflow for testing."""

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"trials_processed": True})


class ConcretePatientsProcessorWorkflow(PatientsProcessorWorkflow):
    """Concrete implementation of PatientsProcessorWorkflow for testing."""

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"patients_processed": True})


class ConcreteTrialsSummarizerWorkflow(TrialsSummarizerWorkflow):
    """Concrete implementation of TrialsSummarizerWorkflow for testing."""

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"trials_summarized": True})


class ConcretePatientsSummarizerWorkflow(PatientsSummarizerWorkflow):
    """Concrete implementation of PatientsSummarizerWorkflow for testing."""

    def execute(self, **kwargs) -> WorkflowResult:
        return self._create_result(success=True, data={"patients_summarized": True})


class TestBaseWorkflow:
    """Test BaseWorkflow core functionality."""

    @pytest.fixture
    def workflow_with_memory(self, mock_memory_storage):
        """Create workflow with mocked memory storage."""
        return ConcreteTestWorkflow(memory_storage=mock_memory_storage)

    def test_init_with_config_and_memory(self, mock_config, mock_memory_storage):
        """Test initialization with custom config and memory storage."""
        workflow = ConcreteTestWorkflow(
            config=mock_config, memory_storage=mock_memory_storage
        )

        assert workflow.config == mock_config
        assert workflow.memory_storage == mock_memory_storage

    def test_init_with_default_config(self, mock_memory_storage):
        """Test initialization with default config."""
        with patch("src.workflows.base_workflow.Config") as mock_config_class:
            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            workflow = ConcreteTestWorkflow(memory_storage=mock_memory_storage)

            mock_config_class.assert_called_once()
            assert workflow.config == mock_config_instance

    def test_init_with_memory_disabled(self):
        """Test initialization with memory disabled."""
        workflow = ConcreteTestWorkflow(memory_storage=False)

        assert workflow.memory_storage is None

    def test_init_with_default_memory(self):
        """Test initialization with default memory storage."""
        with patch(
            "src.workflows.base_workflow.McodeMemoryStorage"
        ) as mock_storage_class:
            mock_storage_instance = MagicMock()
            mock_storage_class.return_value = mock_storage_instance

            workflow = ConcreteTestWorkflow()

            mock_storage_class.assert_called_once()
            assert workflow.memory_storage == mock_storage_instance

    def test_memory_space_property(self):
        """Test that memory_space property works."""
        workflow = ConcreteTestWorkflow()

        assert workflow.memory_space == "test_space"

    def test_execute_method(self):
        """Test that execute method works."""
        workflow = ConcreteTestWorkflow()

        result = workflow.execute()

        assert result.success is True
        assert result.data == {"executed": True}

    def test_validate_inputs_default_true(self):
        """Test default validate_inputs returns True."""
        workflow = ConcreteTestWorkflow()

        assert workflow.validate_inputs(test_param="value") is True

    @patch("src.workflows.base_workflow.get_logger")
    def test_logger_initialization(self, mock_get_logger):
        """Test logger is properly initialized."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        workflow = ConcreteTestWorkflow()

        mock_get_logger.assert_called_once_with("ConcreteTestWorkflow")
        assert workflow.logger == mock_logger


class TestBaseWorkflowMemoryIntegration:
    """Test CORE memory integration functionality."""

    @pytest.fixture
    def workflow_with_memory(self, mock_memory_storage):
        """Create workflow with mocked memory storage."""
        return ConcreteTestWorkflow(memory_storage=mock_memory_storage)

    def test_store_to_core_memory_success(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test successful storage to CORE memory."""
        test_data = {"key": "value"}
        test_metadata = {"meta": "data"}

        result = workflow_with_memory.store_to_core_memory(
            "test_key", test_data, test_metadata
        )

        assert result is True
        mock_memory_storage.store.assert_called_once()
        call_args = mock_memory_storage.store.call_args[0]
        stored_data = call_args[1]

        assert stored_data["data"] == test_data
        assert stored_data["metadata"] == test_metadata
        assert stored_data["workflow_type"] == "ConcreteTestWorkflow"
        assert stored_data["memory_space"] == "test_space"
        assert "timestamp" in stored_data

    def test_store_to_core_memory_without_metadata(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test storage without metadata."""
        test_data = {"key": "value"}

        result = workflow_with_memory.store_to_core_memory("test_key", test_data)

        assert result is True
        stored_data = mock_memory_storage.store.call_args[0][1]
        assert stored_data["metadata"] == {}

    def test_store_to_core_memory_failure(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test storage failure handling."""
        mock_memory_storage.store.return_value = False

        result = workflow_with_memory.store_to_core_memory("test_key", {"data": "test"})

        assert result is False

    def test_store_to_core_memory_exception(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test storage exception handling."""
        mock_memory_storage.store.side_effect = Exception("Storage error")

        result = workflow_with_memory.store_to_core_memory("test_key", {"data": "test"})

        assert result is False

    def test_store_to_core_memory_disabled(self):
        """Test storage when memory is disabled."""
        workflow = ConcreteTestWorkflow(memory_storage=False)

        result = workflow.store_to_core_memory("test_key", {"data": "test"})

        assert result is False

    def test_retrieve_from_core_memory_success(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test successful retrieval from CORE memory."""
        result = workflow_with_memory.retrieve_from_core_memory("test_key")

        assert result == {"test": "data"}
        mock_memory_storage.retrieve.assert_called_once_with("test_space:test_key")

    def test_retrieve_from_core_memory_not_found(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test retrieval when data not found."""
        mock_memory_storage.retrieve.return_value = None

        result = workflow_with_memory.retrieve_from_core_memory("test_key")

        assert result is None

    def test_retrieve_from_core_memory_exception(
        self, workflow_with_memory, mock_memory_storage
    ):
        """Test retrieval exception handling."""
        mock_memory_storage.retrieve.side_effect = Exception("Retrieval error")

        result = workflow_with_memory.retrieve_from_core_memory("test_key")

        assert result is None

    def test_retrieve_from_core_memory_disabled(self):
        """Test retrieval when memory is disabled."""
        workflow = ConcreteTestWorkflow(memory_storage=False)

        result = workflow.retrieve_from_core_memory("test_key")

        assert result is None

    def test_get_timestamp(self):
        """Test timestamp generation."""
        workflow = ConcreteTestWorkflow()

        timestamp = workflow._get_timestamp()

        # Should be ISO format string
        assert isinstance(timestamp, str)
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


class TestWorkflowResultCreation:
    """Test workflow result creation methods."""

    @pytest.fixture
    def workflow(self):
        """Create basic workflow for testing."""
        return ConcreteTestWorkflow()

    def test_create_result_success(self, workflow):
        """Test creating successful result."""
        result = workflow._create_result(
            success=True, data={"test": "data"}, metadata={"meta": "info"}
        )

        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert result.data == {"test": "data"}
        assert result.error_message is None
        assert result.metadata == {"meta": "info"}

    def test_create_result_failure(self, workflow):
        """Test creating failed result."""
        result = workflow._create_result(
            success=False, error_message="Test error", metadata={"error_code": 500}
        )

        assert result.success is False
        assert result.data == {}
        assert result.error_message == "Test error"
        assert result.metadata == {"error_code": 500}

    def test_create_result_with_processing_metadata(self, workflow):
        """Test creating result with ProcessingMetadata object."""
        processing_meta = ProcessingMetadata(
            engine_type="test_engine",
            entities_count=10,
            mapped_count=5,
            processing_time_seconds=1.5,
            model_used="test_model",
            prompt_used="test_prompt",
            token_usage=None,
        )

        result = workflow._create_result(
            success=True, data={"processed": True}, metadata=processing_meta
        )

        assert result.success is True
        assert result.metadata["engine_type"] == "test_engine"
        assert result.metadata["entities_count"] == 10
        assert result.metadata["processing_time_seconds"] == 1.5

    def test_create_result_empty_data_defaults(self, workflow):
        """Test result creation with empty data defaults."""
        result = workflow._create_result(success=True)

        assert result.data == {}
        assert result.metadata == {}

    def test_handle_error_simple(self, workflow):
        """Test simple error handling."""
        error = ValueError("Test error")

        result = workflow._handle_error(error)

        assert result.success is False
        assert result.error_message == "Test error"
        assert result.metadata == {"error_type": "ValueError"}

    def test_handle_error_with_context(self, workflow):
        """Test error handling with context."""
        error = RuntimeError("Connection failed")

        result = workflow._handle_error(error, "Database operation")

        assert result.success is False
        assert result.error_message == "Database operation: Connection failed"
        assert result.metadata == {"error_type": "RuntimeError"}


class TestWorkflowHierarchy:
    """Test workflow class hierarchy and properties."""

    def test_fetcher_workflow_memory_space(self):
        """Test FetcherWorkflow memory space."""
        workflow = ConcreteFetcherWorkflow()
        assert workflow.memory_space == "raw_data"

    def test_trials_processor_workflow_memory_space(self):
        """Test TrialsProcessorWorkflow memory space."""
        workflow = ConcreteTrialsProcessorWorkflow()
        assert workflow.memory_space == "trials"

    def test_patients_processor_workflow_memory_space(self):
        """Test PatientsProcessorWorkflow memory space."""
        workflow = ConcretePatientsProcessorWorkflow()
        assert workflow.memory_space == "patients"

    def test_trials_summarizer_workflow_memory_space(self):
        """Test TrialsSummarizerWorkflow memory space."""
        workflow = ConcreteTrialsSummarizerWorkflow()
        assert workflow.memory_space == "trials_summaries"

    def test_patients_summarizer_workflow_memory_space(self):
        """Test PatientsSummarizerWorkflow memory space."""
        workflow = ConcretePatientsSummarizerWorkflow()
        assert workflow.memory_space == "patients_summaries"

    def test_inheritance_hierarchy(self):
        """Test that inheritance hierarchy is correct."""
        # Test that classes inherit properly
        assert issubclass(FetcherWorkflow, BaseWorkflow)
        assert issubclass(ProcessorWorkflow, BaseWorkflow)
        assert issubclass(SummarizerWorkflow, BaseWorkflow)

        assert issubclass(TrialsProcessorWorkflow, ProcessorWorkflow)
        assert issubclass(PatientsProcessorWorkflow, ProcessorWorkflow)
        assert issubclass(TrialsSummarizerWorkflow, SummarizerWorkflow)
        assert issubclass(PatientsSummarizerWorkflow, SummarizerWorkflow)


class TestWorkflowError:
    """Test custom WorkflowError exception."""

    def test_workflow_error_inheritance(self):
        """Test WorkflowError inherits from Exception."""
        assert issubclass(WorkflowError, Exception)

    def test_workflow_error_creation(self):
        """Test WorkflowError can be created and raised."""
        error = WorkflowError("Test workflow error")

        assert str(error) == "Test workflow error"

        with pytest.raises(WorkflowError, match="Test workflow error"):
            raise error


class TestWorkflowIntegrationScenarios:
    """Test integration scenarios for workflow functionality."""

    def test_memory_integration_workflow(self, mock_memory_storage):
        """Test complete memory integration workflow."""
        workflow = ConcreteTestWorkflow(memory_storage=mock_memory_storage)

        # Store data
        store_result = workflow.store_to_core_memory(
            "integration_key", {"integration": "data"}, {"test": "metadata"}
        )
        assert store_result is True

        # Retrieve data
        retrieve_result = workflow.retrieve_from_core_memory("integration_key")
        assert retrieve_result is not None

        # Verify calls
        assert mock_memory_storage.store.called
        assert mock_memory_storage.retrieve.called

    @patch("src.workflows.base_workflow.BaseWorkflow._get_timestamp")
    def test_timestamp_in_storage(self, mock_get_timestamp, mock_memory_storage):
        """Test that timestamp is included in storage data."""
        mock_get_timestamp.return_value = "2024-01-01T12:00:00"

        workflow = ConcreteTestWorkflow(memory_storage=mock_memory_storage)

        workflow.store_to_core_memory("timestamp_key", {"data": "test"})

        stored_data = mock_memory_storage.store.call_args[0][1]
        assert stored_data["timestamp"] == "2024-01-01T12:00:00"

    def test_error_handling_integration(self):
        """Test error handling integration with result creation."""
        workflow = ConcreteTestWorkflow()

        # Simulate an operation that fails
        try:
            raise ValueError("Simulated operation failure")
        except Exception as e:
            result = workflow._handle_error(e, "Test operation")

        assert result.success is False
        assert "Simulated operation failure" in result.error_message
        assert result.metadata["error_type"] == "ValueError"


# Parametrized tests for different workflow types
@pytest.mark.parametrize(
    "workflow_class,memory_space",
    [
        (ConcreteFetcherWorkflow, "raw_data"),
        (ConcreteTrialsProcessorWorkflow, "trials"),
        (ConcretePatientsProcessorWorkflow, "patients"),
        (ConcreteTrialsSummarizerWorkflow, "trials_summaries"),
        (ConcretePatientsSummarizerWorkflow, "patients_summaries"),
    ],
)
def test_workflow_memory_spaces(workflow_class, memory_space):
    """Test that all workflow types have correct memory spaces."""
    workflow = workflow_class()
    assert workflow.memory_space == memory_space


@pytest.mark.parametrize(
    "workflow_class",
    [
        FetcherWorkflow,
        ProcessorWorkflow,
        TrialsProcessorWorkflow,
        PatientsProcessorWorkflow,
        SummarizerWorkflow,
        TrialsSummarizerWorkflow,
        PatientsSummarizerWorkflow,
    ],
)
def test_workflow_inheritance_from_base(workflow_class):
    """Test that all workflow classes inherit from BaseWorkflow."""
    assert issubclass(workflow_class, BaseWorkflow)


# Edge cases and error conditions
class TestWorkflowEdgeCases:
    """Test edge cases and error conditions."""

    def test_memory_storage_none_operations(self):
        """Test operations when memory storage is None."""
        workflow = ConcreteTestWorkflow(memory_storage=None)

        # These should not crash and should return appropriate defaults
        store_result = workflow.store_to_core_memory("key", {"data": "test"})
        assert store_result is False

        retrieve_result = workflow.retrieve_from_core_memory("key")
        assert retrieve_result is None

    def test_create_result_with_none_values(self):
        """Test result creation with None values."""
        workflow = ConcreteTestWorkflow()

        result = workflow._create_result(
            success=True, data=None, error_message=None, metadata=None
        )

        assert result.success is True
        assert result.data == {}
        assert result.error_message is None
        assert result.metadata == {}

    def test_handle_error_with_empty_context(self):
        """Test error handling with empty context string."""
        workflow = ConcreteTestWorkflow()
        error = Exception("Test")

        result = workflow._handle_error(error, "")

        assert result.error_message == "Test"

    def test_handle_error_with_none_context(self):
        """Test error handling with None context."""
        workflow = ConcreteTestWorkflow()
        error = Exception("Test")

        result = workflow._handle_error(error, "")

        assert result.error_message == "Test"

    def test_validate_inputs_override(self):
        """Test that subclasses can override validate_inputs."""

        class CustomWorkflow(ConcreteTestWorkflow):
            def validate_inputs(self, **kwargs):
                return kwargs.get("valid", False)

        workflow = CustomWorkflow()

        assert workflow.validate_inputs(valid=True) is True
        assert workflow.validate_inputs(valid=False) is False
        assert workflow.validate_inputs() is False
