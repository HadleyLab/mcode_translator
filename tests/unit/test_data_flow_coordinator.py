"""
Unit tests for DataFlowCoordinator with comprehensive coverage.
"""
import pytest
from unittest.mock import Mock, patch

from src.core.data_flow_coordinator import (
    DataFlowCoordinator,
    create_data_flow_coordinator,
    process_clinical_trials_flow
)
from src.shared.models import WorkflowResult


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for testing."""
    pipeline = Mock()
    pipeline.process.return_value = Mock(
        success=True,
        mcode_mappings=[],
        validation_results={},
        error_message=None
    )
    return pipeline


@pytest.fixture
def sample_trial_ids():
    """Sample trial IDs for testing."""
    return ["NCT12345678", "NCT87654321", "NCT99999999"]


@pytest.fixture
def sample_trial_data():
    """Sample trial data for testing."""
    return [
        {
            "nct_id": "NCT12345678",
            "title": "Sample Trial 1",
            "conditions": ["Breast Cancer"]
        },
        {
            "nct_id": "NCT87654321",
            "title": "Sample Trial 2",
            "conditions": ["Lung Cancer"]
        }
    ]


class TestDataFlowCoordinator:
    """Test DataFlowCoordinator functionality."""

    @patch('src.core.data_flow_coordinator.create_trial_pipeline')
    def test_init_without_pipeline(self, mock_create_pipeline, mock_pipeline):
        """Test initialization without pipeline creates default."""
        mock_create_pipeline.return_value = mock_pipeline

        coordinator = DataFlowCoordinator()

        assert coordinator.pipeline == mock_pipeline
        assert coordinator.config == {}
        mock_create_pipeline.assert_called_once_with(
            processor_config={},
            include_storage=False
        )

    def test_init_with_pipeline(self, mock_pipeline):
        """Test initialization with provided pipeline."""
        config = {"test": "config"}

        coordinator = DataFlowCoordinator(
            pipeline=mock_pipeline, config=config
        )

        assert coordinator.pipeline == mock_pipeline
        assert coordinator.config == config

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_process_clinical_trials_complete_flow_success(
        self, mock_batch_fetch, mock_pipeline,
        sample_trial_ids, sample_trial_data
    ):
        """Test successful complete flow processing."""
        # Mock successful fetch
        mock_batch_fetch.return_value = {
            "NCT12345678": sample_trial_data[0],
            "NCT87654321": sample_trial_data[1],
            "NCT99999999": {"error": "Not found"}
        }

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator.process_clinical_trials_complete_flow(sample_trial_ids)

        assert result.success is True
        assert len(result.data["fetched_trials"]) == 2
        assert result.data["summary"]["total_requested"] == 3
        assert result.data["summary"]["total_fetched"] == 2
        assert result.data["summary"]["total_successful"] == 2  # Mock pipeline succeeds

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_process_clinical_trials_complete_flow_fetch_failure(
        self, mock_batch_fetch, mock_pipeline, sample_trial_ids
    ):
        """Test complete flow with fetch failure."""
        # Mock failed fetch
        mock_batch_fetch.return_value = {
            "NCT12345678": {"error": "API Error"},
            "NCT87654321": {"error": "Timeout"},
            "NCT99999999": {"error": "Not found"}
        }

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator.process_clinical_trials_complete_flow(sample_trial_ids)

        assert result.success is False
        assert result.error_message == "No trials could be fetched"
        assert result.data == {}

    def test_process_clinical_trials_complete_flow_empty_trials(self, mock_pipeline):
        """Test complete flow with empty trial list."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator.process_clinical_trials_complete_flow([])

        assert result.success is False
        assert "No trials could be fetched" in result.error_message

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_process_clinical_trials_complete_flow_processing_failure(
        self, mock_batch_fetch, sample_trial_ids, sample_trial_data
    ):
        """Test complete flow with processing failure."""
        # Mock successful fetch
        mock_batch_fetch.return_value = {
            "NCT12345678": sample_trial_data[0],
            "NCT87654321": sample_trial_data[1]
        }

        # Mock pipeline failure
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("Processing failed")

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator.process_clinical_trials_complete_flow(sample_trial_ids)

        assert result.success is False  # All processing failed
        assert len(result.data["fetched_trials"]) == 2
        assert result.data["summary"]["total_successful"] == 0  # All processing failed

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_fetch_trial_data_success(self, mock_batch_fetch, sample_trial_ids, sample_trial_data):
        """Test successful trial data fetching."""
        mock_batch_fetch.return_value = {
            "NCT12345678": sample_trial_data[0],
            "NCT87654321": sample_trial_data[1],
            "NCT99999999": {"error": "Not found"}
        }

        coordinator = DataFlowCoordinator()

        result = coordinator._fetch_trial_data(sample_trial_ids)

        assert result.success is True
        assert len(result.data) == 2
        assert result.metadata["total_requested"] == 3
        assert result.metadata["total_fetched"] == 2
        assert len(result.metadata["failed_fetches"]) == 1

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_fetch_trial_data_all_fail(self, mock_batch_fetch, sample_trial_ids):
        """Test trial data fetching when all fetches fail."""
        mock_batch_fetch.return_value = {
            "NCT12345678": {"error": "API Error"},
            "NCT87654321": {"error": "Timeout"}
        }

        coordinator = DataFlowCoordinator()

        result = coordinator._fetch_trial_data(sample_trial_ids)

        assert result.success is False
        assert result.error_message == "No trials could be fetched"
        assert len(result.data) == 0

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_fetch_trial_data_exception(self, mock_batch_fetch, sample_trial_ids):
        """Test trial data fetching with exception."""
        mock_batch_fetch.side_effect = Exception("Network error")

        coordinator = DataFlowCoordinator()

        result = coordinator._fetch_trial_data(sample_trial_ids)

        assert result.success is False
        assert "Data fetching failed: Network error" in result.error_message

    def test_process_trials_in_batches_success(self, mock_pipeline, sample_trial_data):
        """Test successful batch processing."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(sample_trial_data)

        assert result.success is True
        assert len(result.data) == 1  # One batch
        assert result.data[0]["batch_number"] == 1
        assert result.data[0]["successful"] == 2
        assert result.metadata["total_successful"] == 2

    def test_process_trials_in_batches_empty_data(self, mock_pipeline):
        """Test batch processing with empty data."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches([])

        assert result.success is False
        assert "No trial data to process" in result.error_message

    def test_process_trials_in_batches_with_failures(self, sample_trial_data):
        """Test batch processing with some processing failures."""
        mock_pipeline = Mock()
        # First call succeeds, second fails
        mock_pipeline.process.side_effect = [
            Mock(success=True, mcode_mappings=[], validation_results={}, error_message=None),
            Exception("Processing failed")
        ]

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(sample_trial_data)

        assert result.success is True  # At least one succeeded
        assert result.metadata["total_successful"] == 1
        assert result.metadata["total_failed"] == 1

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_process_trials_in_batches_different_sizes(self, mock_pipeline, sample_trial_data, batch_size):
        """Test batch processing with different batch sizes."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(sample_trial_data, batch_size=batch_size)

        assert result.success is True
        total_batches = (len(sample_trial_data) + batch_size - 1) // batch_size
        assert len(result.data) == total_batches

    def test_generate_flow_summary(self, sample_trial_ids):
        """Test flow summary generation."""
        coordinator = DataFlowCoordinator()

        fetch_result = WorkflowResult(
            success=True,
            data=[{"nct_id": "NCT12345678"}],
            metadata={"total_fetched": 1, "failed_fetches": []}
        )

        processing_result = WorkflowResult(
            success=True,
            data=[],
            metadata={"total_processed": 1, "total_successful": 1, "total_failed": 0}
        )

        summary = coordinator._generate_flow_summary(sample_trial_ids, fetch_result, processing_result)

        assert summary["total_requested"] == 3
        assert summary["total_fetched"] == 1
        assert summary["total_processed"] == 1
        assert summary["total_successful"] == 1
        assert summary["overall_success_rate"] == 1/3

    def test_generate_flow_summary_edge_cases(self):
        """Test flow summary generation with edge cases."""
        coordinator = DataFlowCoordinator()

        # Empty trial IDs
        fetch_result = WorkflowResult(success=True, data=[], metadata={"total_fetched": 0})
        processing_result = WorkflowResult(success=True, data=[], metadata={"total_processed": 0, "total_successful": 0})

        summary = coordinator._generate_flow_summary([], fetch_result, processing_result)

        assert summary["total_requested"] == 0
        assert summary["fetch_success_rate"] == 0.0
        assert summary["processing_success_rate"] == 0.0
        assert summary["overall_success_rate"] == 0.0

    def test_get_flow_statistics(self):
        """Test getting flow statistics."""
        config = {"test": "config"}
        coordinator = DataFlowCoordinator(config=config)

        stats = coordinator.get_flow_statistics()

        assert stats["coordinator_type"] == "data_flow_coordinator"
        assert stats["pipeline_type"] == "simplified"
        assert stats["has_processor"] is True
        assert stats["has_storage"] is False
        assert stats["config"] == config
        assert stats["capabilities"]["batch_processing"] is True

    @patch('src.core.data_flow_coordinator.create_data_flow_coordinator')
    def test_process_clinical_trials_flow_convenience(self, mock_create_coordinator, sample_trial_ids):
        """Test convenience function for processing trials."""
        mock_coordinator = Mock()
        mock_result = Mock()
        mock_coordinator.process_clinical_trials_complete_flow.return_value = mock_result
        mock_create_coordinator.return_value = mock_coordinator

        result = process_clinical_trials_flow(sample_trial_ids)

        assert result == mock_result
        mock_create_coordinator.assert_called_once_with(config=None)
        mock_coordinator.process_clinical_trials_complete_flow.assert_called_once_with(sample_trial_ids)

    def test_create_data_flow_coordinator_convenience(self):
        """Test convenience function for creating coordinator."""
        config = {"test": "config"}

        coordinator = create_data_flow_coordinator(config)

        assert isinstance(coordinator, DataFlowCoordinator)
        assert coordinator.config == config


class TestDataFlowCoordinatorIntegration:
    """Integration tests for DataFlowCoordinator with real dependencies."""

    @pytest.mark.integration
    def test_full_workflow_integration(self, sample_trial_ids):
        """Test full workflow with mocked external calls."""
        with patch('src.core.data_flow_coordinator.get_full_studies_batch') as mock_batch, \
             patch('src.core.data_flow_coordinator.create_trial_pipeline') as mock_create_pipeline:

            # Mock successful fetch
            mock_batch.return_value = {
                trial_id: {"nct_id": trial_id, "title": f"Trial {trial_id}"}
                for trial_id in sample_trial_ids
            }

            # Mock pipeline
            mock_pipeline = Mock()
            mock_pipeline.process.return_value = Mock(
                success=True, mcode_mappings=[], validation_results={}
            )
            mock_create_pipeline.return_value = mock_pipeline

            coordinator = DataFlowCoordinator()
            result = coordinator.process_clinical_trials_complete_flow(sample_trial_ids)

            assert result.success is True
            assert len(result.data["fetched_trials"]) == 3
            assert result.data["summary"]["total_successful"] == 3


class TestDataFlowCoordinatorErrorHandling:
    """Test error handling scenarios."""

    def test_pipeline_process_exception_handling(self, sample_trial_data):
        """Test handling of pipeline processing exceptions."""
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = [Exception("Process error"), Mock(success=True)]

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(sample_trial_data)

        # Should continue processing despite first failure
        assert result.success is True
        assert result.metadata["total_successful"] == 1
        assert result.metadata["total_failed"] == 1

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_fetch_with_mixed_results(self, mock_batch_fetch, sample_trial_ids):
        """Test fetching with mix of success and failure."""
        mock_batch_fetch.return_value = {
            "NCT12345678": {"nct_id": "NCT12345678", "title": "Success Trial"},
            "NCT87654321": {"error": "API Error"},
            "NCT99999999": {"nct_id": "NCT99999999", "title": "Another Success"}
        }

        coordinator = DataFlowCoordinator()

        result = coordinator._fetch_trial_data(sample_trial_ids)

        assert result.success is True
        assert len(result.data) == 2
        assert len(result.metadata["failed_fetches"]) == 1

    def test_empty_batch_processing(self, mock_pipeline):
        """Test processing with empty batches."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        # Test with None input
        result = coordinator._process_trials_in_batches(None)
        assert result.success is False

        # Test with None after checking
        result = coordinator._process_trials_in_batches([])
        assert result.success is False


class TestDataFlowCoordinatorPerformance:
    """Performance-related tests."""

    @pytest.mark.performance
    def test_large_batch_processing(self, mock_pipeline):
        """Test processing large number of trials."""
        large_trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}"}
            for i in range(100)
        ]

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(large_trial_data, batch_size=10)

        assert result.success is True
        assert len(result.data) == 10  # 100 trials / 10 batch_size = 10 batches
        assert result.metadata["total_processed"] == 100

    @pytest.mark.performance
    def test_memory_efficiency_with_batches(self, mock_pipeline):
        """Test that batching doesn't cause memory issues."""
        trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}", "large_field": "x" * 1000}
            for i in range(50)
        ]

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        result = coordinator._process_trials_in_batches(trial_data, batch_size=5)

        assert result.success is True
        assert result.metadata["total_processed"] == 50