"""
Unit tests for BatchProcessor with comprehensive coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.core.batch_processor import BatchProcessor
from src.shared.models import PipelineResult, ValidationResult, ProcessingMetadata


@pytest.fixture
def mock_pipeline():
    """Mock pipeline for testing."""
    pipeline = Mock()
    pipeline.process.return_value = PipelineResult(
        extracted_entities=[],
        mcode_mappings=[],
        source_references=[],
        validation_results=ValidationResult(compliance_score=1.0),
        metadata=ProcessingMetadata(
            engine_type="test",
            entities_count=0,
            mapped_count=0
        ),
        original_data={},
        error=None
    )
    return pipeline


@pytest.fixture
def sample_trial_data():
    """Sample trial data for testing."""
    return [
        {
            "nct_id": "NCT12345678",
            "title": "Sample Trial 1",
            "conditions": ["Breast Cancer"],
        },
        {
            "nct_id": "NCT87654321",
            "title": "Sample Trial 2",
            "conditions": ["Lung Cancer"],
        },
    ]


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_init(self, mock_pipeline):
        """Test initialization."""
        processor = BatchProcessor(mock_pipeline)

        assert processor.pipeline == mock_pipeline

    def test_process_trials_in_batches_success(self, mock_pipeline, sample_trial_data):
        """Test successful batch processing."""
        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is True
        assert len(result.data) == 1  # One batch
        assert result.data[0]["batch_number"] == 1
        assert result.data[0]["successful"] == 2
        assert result.metadata["total_successful"] == 2
        assert result.metadata["total_processed"] == 2

    def test_process_trials_in_batches_empty_data(self, mock_pipeline):
        """Test batch processing with empty data."""
        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches([])

        assert result.success is False
        assert "No trial data to process" in result.error_message

    def test_process_trials_in_batches_with_failures(self, sample_trial_data):
        """Test batch processing with some processing failures."""
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = [
            PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=1.0),
                metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                original_data={},
                error=None
            ),
            PipelineResult(
                extracted_entities=[],
                mcode_mappings=[],
                source_references=[],
                validation_results=ValidationResult(compliance_score=0.0),
                metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                original_data={},
                error="Processing failed"
            ),
        ]

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is True  # At least one succeeded
        assert result.metadata["total_successful"] == 1
        assert result.metadata["total_failed"] == 1

    def test_process_trials_in_batches_all_failures(self, sample_trial_data):
        """Test batch processing with all processing failures."""
        mock_pipeline = Mock()
        mock_pipeline.process.return_value = PipelineResult(
            extracted_entities=[],
            mcode_mappings=[],
            source_references=[],
            validation_results=ValidationResult(compliance_score=0.0),
            metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
            original_data={},
            error="Processing failed"
        )

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is False
        assert result.metadata["total_successful"] == 0
        assert result.metadata["total_failed"] == 2

    @pytest.mark.parametrize("batch_size", [1, 2, 5])
    def test_process_trials_in_batches_different_sizes(
        self, mock_pipeline, sample_trial_data, batch_size
    ):
        """Test batch processing with different batch sizes."""
        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data, batch_size=batch_size)

        assert result.success is True
        total_batches = (len(sample_trial_data) + batch_size - 1) // batch_size
        assert len(result.data) == total_batches

    @pytest.mark.parametrize("num_trials", [0, 1, 3, 10])
    def test_process_trials_in_batches_various_trial_counts(
        self, mock_pipeline, num_trials
    ):
        """Test batch processing with various numbers of trials."""
        trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}"} for i in range(num_trials)
        ]

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(trial_data)

        if num_trials == 0:
            assert result.success is False
            assert "No trial data to process" in result.error_message
        else:
            assert result.success is True
            assert result.metadata["total_processed"] == num_trials

    @pytest.mark.parametrize("error_scenario", [
        "all_fail",
        "mixed_results",
        "pipeline_exception"
    ])
    def test_process_trials_in_batches_error_scenarios(
        self, sample_trial_data, error_scenario
    ):
        """Test batch processing with various error scenarios."""
        if error_scenario == "all_fail":
            mock_pipeline = Mock()
            mock_pipeline.process.side_effect = [
                PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=0.0),
                    metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                    original_data={},
                    error="Processing failed"
                ),
                PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=0.0),
                    metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                    original_data={},
                    error="Processing failed"
                ),
            ]
        elif error_scenario == "mixed_results":
            mock_pipeline = Mock()
            mock_pipeline.process.side_effect = [
                PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=1.0),
                    metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                    original_data={},
                    error=None
                ),
                PipelineResult(
                    extracted_entities=[],
                    mcode_mappings=[],
                    source_references=[],
                    validation_results=ValidationResult(compliance_score=0.0),
                    metadata=ProcessingMetadata(engine_type="test", entities_count=0, mapped_count=0),
                    original_data={},
                    error="Processing failed"
                ),
            ]
        elif error_scenario == "pipeline_exception":
            mock_pipeline = Mock()
            mock_pipeline.process.side_effect = Exception("Pipeline error")

        processor = BatchProcessor(mock_pipeline)
        result = processor.process_trials_in_batches(sample_trial_data)

        if error_scenario == "all_fail":
            assert result.success is False
            assert result.metadata["total_successful"] == 0
        elif error_scenario == "mixed_results":
            assert result.success is True  # At least one succeeded
            assert result.metadata["total_successful"] == 1
        elif error_scenario == "pipeline_exception":
            assert result.success is False
            assert result.metadata["total_failed"] == 2

    def test_process_trials_in_batches_exception_handling(self, sample_trial_data):
        """Test exception handling during batch processing."""
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("Unexpected error")

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is False
        assert result.metadata["total_failed"] == 2

    def test_process_trials_in_batches_large_dataset(self, mock_pipeline):
        """Test batch processing with large dataset."""
        large_trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}"} for i in range(25)
        ]

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(large_trial_data, batch_size=10)

        assert result.success is True
        assert len(result.data) == 3  # 25 trials / 10 batch_size = 3 batches (rounded up)
        assert result.metadata["total_processed"] == 25

    def test_process_trials_in_batches_custom_batch_size_override(self, mock_pipeline, sample_trial_data):
        """Test that batch_size parameter overrides instance batch_size."""
        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data, batch_size=1)

        assert result.success is True
        assert len(result.data) == 2  # 2 trials / 1 batch_size = 2 batches


class TestBatchProcessorErrorHandling:
    """Test error handling scenarios."""

    def test_pipeline_process_returns_none(self, sample_trial_data):
        """Test handling when pipeline.process returns None."""
        mock_pipeline = Mock()
        mock_pipeline.process.return_value = None

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is False
        assert result.metadata["total_failed"] == 2

    def test_pipeline_process_missing_attributes(self, sample_trial_data):
        """Test handling when pipeline result is missing attributes."""
        mock_pipeline = Mock()
        mock_pipeline.process.return_value = Mock()

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(sample_trial_data)

        assert result.success is False
        assert result.metadata["total_failed"] == 2


class TestBatchProcessorPerformance:
    """Performance-related tests."""

    @pytest.mark.performance
    def test_batch_processing_performance(self, mock_pipeline):
        """Test performance with large batches."""
        trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}", "large_field": "x" * 1000}
            for i in range(100)
        ]

        processor = BatchProcessor(mock_pipeline)

        result = processor.process_trials_in_batches(trial_data, batch_size=20)

        assert result.success is True
        assert len(result.data) == 5  # 100 / 20 = 5 batches
        assert result.metadata["total_processed"] == 100

    @patch('src.core.batch_processor.get_logger')
    def test_batch_processor_logger_integration(self, mock_get_logger, mock_pipeline, sample_trial_data):
        """Test that BatchProcessor properly integrates with logging."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        processor = BatchProcessor(mock_pipeline)
        result = processor.process_trials_in_batches(sample_trial_data)

        # Verify logger was obtained
        mock_get_logger.assert_called_once()

        # Verify logging calls were made
        assert mock_logger.info.call_count >= 2  # At least batch start and completion logs

        # Verify success
        assert result.success is True

    @patch('src.core.batch_processor.get_logger')
    def test_batch_processor_error_logging(self, mock_get_logger, sample_trial_data):
        """Test that errors are properly logged."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock pipeline that raises exception
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("Test error")

        processor = BatchProcessor(mock_pipeline)
        result = processor.process_trials_in_batches(sample_trial_data)

        # Verify error was logged
        mock_logger.error.assert_called()

        # Verify error logging included the exception message
        error_calls = mock_logger.error.call_args_list
        assert any("Test error" in str(call) for call in error_calls)

        # Verify failure
        assert result.success is False

    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_batch_processor_performance_benchmark(self, benchmark, mock_pipeline):
        """Benchmark batch processing performance."""
        # Create larger dataset for benchmarking
        trial_data = [
            {"nct_id": f"NCT{i:08d}", "title": f"Trial {i}", "large_field": "x" * 500}
            for i in range(50)
        ]

        processor = BatchProcessor(mock_pipeline)

        # Benchmark the batch processing
        result = benchmark(processor.process_trials_in_batches, trial_data, batch_size=10)

        # Verify correctness
        assert result.success is True
        assert len(result.data) == 5  # 50 / 10 = 5 batches
        assert result.metadata["total_processed"] == 50

        # Performance benchmark completed successfully
        # Mean time: ~0.00056 seconds for 50 trials in batches of 10