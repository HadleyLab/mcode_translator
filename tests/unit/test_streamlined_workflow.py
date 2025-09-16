"""
Unit tests for streamlined workflow architecture.
"""

from unittest.mock import MagicMock, Mock

import pytest

from src.shared.models import WorkflowResult
from src.workflows.streamlined_workflow import (StreamlinedTrialProcessor,
                                                StreamlinedWorkflowCoordinator)


class TestStreamlinedTrialProcessor:
    """Test StreamlinedTrialProcessor."""

    def test_initialization_with_injected_pipeline(self):
        """Test initialization with injected pipeline."""
        mock_pipeline = Mock()
        processor = StreamlinedTrialProcessor(pipeline=mock_pipeline)

        assert processor.pipeline == mock_pipeline

    def test_initialization_without_pipeline(self):
        """Test initialization without pipeline (should create one)."""
        # Test that a real pipeline is created when none is provided
        processor = StreamlinedTrialProcessor()

        # Verify that a pipeline was created
        assert hasattr(processor, "pipeline")
        assert processor.pipeline is not None
        assert hasattr(processor.pipeline, "process_trials_batch")
        assert hasattr(processor.pipeline, "validator")
        assert processor.pipeline.storage is not None

    def test_process_single_trial_success(self):
        """Test successful single trial processing."""
        mock_pipeline = Mock()
        mock_result = WorkflowResult(success=True, data={"test": "data"})
        mock_pipeline.process_trial.return_value = mock_result

        processor = StreamlinedTrialProcessor(pipeline=mock_pipeline)
        trial_data = {"protocolSection": {"identificationModule": {"nctId": "NCT123"}}}

        result = processor.process_single_trial(trial_data)

        assert result.success is True
        assert result.data == {"test": "data"}
        mock_pipeline.process_trial.assert_called_once_with(
            trial_data=trial_data, validate=True, store_results=True
        )

    def test_process_single_trial_failure(self):
        """Test failed single trial processing."""
        mock_pipeline = Mock()
        mock_result = WorkflowResult(success=False, error_message="Processing failed")
        mock_pipeline.process_trial.return_value = mock_result

        processor = StreamlinedTrialProcessor(pipeline=mock_pipeline)
        trial_data = {"protocolSection": {"identificationModule": {"nctId": "NCT123"}}}

        result = processor.process_single_trial(trial_data)

        assert result.success is False
        assert result.error_message == "Processing failed"

    def test_process_multiple_trials(self):
        """Test processing multiple trials."""
        mock_pipeline = Mock()
        mock_result = WorkflowResult(
            success=True,
            data={"results": ["result1", "result2"]},
            metadata={"successful": 2, "total_trials": 2},
        )
        mock_pipeline.process_trials_batch.return_value = mock_result

        processor = StreamlinedTrialProcessor(pipeline=mock_pipeline)
        trials_data = [
            {"protocolSection": {"identificationModule": {"nctId": "NCT123"}}},
            {"protocolSection": {"identificationModule": {"nctId": "NCT456"}}},
        ]

        result = processor.process_multiple_trials(trials_data)

        assert result.success is True
        assert len(result.data["results"]) == 2
        mock_pipeline.process_trials_batch.assert_called_once_with(
            trials_data=trials_data, validate=True, store_results=True
        )

    def test_get_processing_stats(self):
        """Test getting processing statistics."""
        mock_pipeline = Mock()
        mock_pipeline.storage = Mock()

        processor = StreamlinedTrialProcessor(
            pipeline=mock_pipeline, config={"test": "config"}
        )

        stats = processor.get_processing_stats()

        assert stats["pipeline_type"] == "unified"
        assert stats["has_validator"] is True
        assert stats["has_processor"] is True
        assert stats["has_storage"] is True
        assert stats["config"] == {"test": "config"}


class TestStreamlinedWorkflowCoordinator:
    """Test StreamlinedWorkflowCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        # Mock the dependency container to avoid model loading issues
        import src.core.dependency_container as dc_module

        original_create_trial_pipeline = dc_module.create_trial_pipeline

        mock_pipeline = Mock()
        dc_module.create_trial_pipeline = Mock(return_value=mock_pipeline)

        try:
            config = {"trial_processor": {"include_storage": False}}
            coordinator = StreamlinedWorkflowCoordinator(config=config)

            assert hasattr(coordinator, "trial_processor")
            assert coordinator.config == config
        finally:
            # Restore original function
            dc_module.create_trial_pipeline = original_create_trial_pipeline

    def test_process_clinical_trials_workflow_success(self):
        """Test successful workflow execution."""
        # Mock the dependency container to avoid model loading issues
        import src.core.dependency_container as dc_module

        original_create_trial_pipeline = dc_module.create_trial_pipeline

        mock_pipeline = Mock()
        mock_result = WorkflowResult(
            success=True,
            data={"trials": ["trial1", "trial2"]},
            metadata={"successful": 2, "total_trials": 2, "success_rate": 1.0},
        )
        mock_pipeline.process_trials_batch.return_value = mock_result
        dc_module.create_trial_pipeline = Mock(return_value=mock_pipeline)

        try:
            mock_processor = Mock()
            mock_processor.process_multiple_trials.return_value = mock_result

            coordinator = StreamlinedWorkflowCoordinator()
            coordinator.trial_processor = mock_processor

            trials_data = [
                {"protocolSection": {"identificationModule": {"nctId": "NCT123"}}},
                {"protocolSection": {"identificationModule": {"nctId": "NCT456"}}},
            ]

            result = coordinator.process_clinical_trials_workflow(trials_data)

            assert result.success is True
            assert result.metadata["successful"] == 2
            assert result.metadata["total_trials"] == 2
        finally:
            # Restore original function
            dc_module.create_trial_pipeline = original_create_trial_pipeline

    def test_process_clinical_trials_workflow_failure(self):
        """Test failed workflow execution."""
        # Mock the dependency container to avoid model loading issues
        import src.core.dependency_container as dc_module

        original_create_trial_pipeline = dc_module.create_trial_pipeline

        mock_pipeline = Mock()
        dc_module.create_trial_pipeline = Mock(return_value=mock_pipeline)

        try:
            mock_processor = Mock()
            mock_result = WorkflowResult(success=False, error_message="Workflow failed")
            mock_processor.process_multiple_trials.return_value = mock_result

            coordinator = StreamlinedWorkflowCoordinator()
            coordinator.trial_processor = mock_processor

            trials_data = [{"invalid": "data"}]

            result = coordinator.process_clinical_trials_workflow(trials_data)

            assert result.success is False
            assert result.error_message == "Workflow failed"
        finally:
            # Restore original function
            dc_module.create_trial_pipeline = original_create_trial_pipeline

    def test_get_workflow_stats(self):
        """Test getting workflow statistics."""
        # Mock the dependency container to avoid model loading issues
        import src.core.dependency_container as dc_module

        original_create_trial_pipeline = dc_module.create_trial_pipeline

        mock_pipeline = Mock()
        dc_module.create_trial_pipeline = Mock(return_value=mock_pipeline)

        try:
            mock_processor = Mock()
            mock_processor.get_processing_stats.return_value = {"processor": "stats"}

            coordinator = StreamlinedWorkflowCoordinator(config={"test": "config"})
            coordinator.trial_processor = mock_processor

            stats = coordinator.get_workflow_stats()

            assert stats["coordinator_type"] == "streamlined"
            assert stats["trial_processor"] == {"processor": "stats"}
            assert stats["config"] == {"test": "config"}
        finally:
            # Restore original function
            dc_module.create_trial_pipeline = original_create_trial_pipeline


if __name__ == "__main__":
    pytest.main([__file__])
