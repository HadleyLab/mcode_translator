"""
Performance benchmarks for DataFlowCoordinator operations.
"""
import pytest
from unittest.mock import Mock, patch
from src.core.data_flow_coordinator import DataFlowCoordinator


@pytest.mark.performance
class TestDataFlowCoordinatorPerformance:
    """Performance benchmarks for DataFlowCoordinator."""

    @pytest.fixture
    def mock_pipeline(self):
        """Mock pipeline for performance testing."""
        pipeline = Mock()
        pipeline.process.return_value = Mock(
            success=True,
            mcode_mappings=[],
            validation_results={},
            error_message=None
        )
        return pipeline

    @pytest.fixture
    def large_trial_dataset(self):
        """Generate large dataset for performance testing."""
        return [
            {
                "nct_id": f"NCT{i:08d}",
                "title": f"Performance Test Trial {i}",
                "conditions": ["Cancer", "Breast Cancer", "Lung Cancer"] * 5,
                "eligibility": {
                    "criteria": "Age >= 18\n" +
                    "\n".join([f"Criterion {j}" for j in range(20)])
                },
                "phases": ["Phase 1", "Phase 2", "Phase 3"],
                "interventions": [
                    {"type": "Drug", "name": f"Drug {j}"}
                    for j in range(5)
                ]
            } for i in range(50)  # Large batch for performance testing
        ]

    @pytest.fixture
    def coordinator(self, mock_pipeline):
        """Create coordinator for testing."""
        return DataFlowCoordinator(pipeline=mock_pipeline)

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_fetch_trial_data_performance(self, mock_batch_fetch, benchmark, large_trial_dataset):
        """Benchmark trial data fetching performance."""
        # Mock successful fetch of all trials
        mock_batch_fetch.return_value = {
            trial["nct_id"]: trial for trial in large_trial_dataset
        }

        coordinator = DataFlowCoordinator()

        def fetch_operation():
            trial_ids = [t["nct_id"] for t in large_trial_dataset]
            return coordinator._fetch_trial_data(trial_ids)

        result = benchmark(fetch_operation)

        assert result.success is True
        assert len(result.data) == 50
        # Performance assertion: should handle 50 trials quickly
        assert benchmark.stats.stats.mean < 1.0, f"Fetching should be fast, got {benchmark.stats.stats.mean:.2f}s"

    def test_process_trials_in_batches_performance(self, benchmark, mock_pipeline, large_trial_dataset):
        """Benchmark batch processing performance."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        def batch_processing():
            return coordinator._process_trials_in_batches(large_trial_dataset, batch_size=10)

        result = benchmark(batch_processing)

        assert result.success is True
        assert result.metadata["total_processed"] == 50
        # Performance assertion: should process 50 trials in batches quickly
        assert benchmark.stats.stats.mean < 2.0, f"Batch processing should be fast, got {benchmark.stats.stats.mean:.2f}s"

    def test_generate_flow_summary_performance(self, benchmark):
        """Benchmark flow summary generation performance."""
        coordinator = DataFlowCoordinator()

        # Create mock results with many trials
        fetch_result = Mock()
        fetch_result.success = True
        fetch_result.metadata = {"total_fetched": 100, "failed_fetches": []}

        processing_result = Mock()
        processing_result.success = True
        processing_result.metadata = {"total_processed": 100, "total_successful": 95}

        trial_ids = [f"NCT{i:08d}" for i in range(100)]

        def summary_generation():
            return coordinator._generate_flow_summary(trial_ids, fetch_result, processing_result)

        result = benchmark(summary_generation)

        assert result["total_requested"] == 100
        assert result["total_successful"] == 95
        # Performance assertion: summary generation should be very fast
        assert benchmark.stats.stats.mean < 0.01, f"Summary generation should be very fast, got {benchmark.stats.stats.mean:.4f}s"

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_complete_flow_performance_small(self, mock_batch_fetch, benchmark, mock_pipeline):
        """Benchmark complete flow with small dataset."""
        # Small dataset for baseline performance
        small_trials = [
            {
                "nct_id": f"NCT{i:08d}",
                "title": f"Small Trial {i}",
                "conditions": ["Cancer"]
            } for i in range(5)
        ]

        mock_batch_fetch.return_value = {
            trial["nct_id"]: trial for trial in small_trials
        }

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        def complete_flow():
            trial_ids = [t["nct_id"] for t in small_trials]
            return coordinator.process_clinical_trials_complete_flow(trial_ids)

        result = benchmark(complete_flow)

        assert result.success is True
        # Performance assertion: small flow should complete quickly
        assert benchmark.stats.stats.mean < 0.5, f"Small flow should be fast, got {benchmark.stats.stats.mean:.2f}s"

    @patch('src.core.data_flow_coordinator.get_full_studies_batch')
    def test_complete_flow_performance_medium(self, mock_batch_fetch, benchmark, mock_pipeline):
        """Benchmark complete flow with medium dataset."""
        # Medium dataset
        medium_trials = [
            {
                "nct_id": f"NCT{i:08d}",
                "title": f"Medium Trial {i}",
                "conditions": ["Cancer", "Breast Cancer"],
                "eligibility": {"criteria": "Age >= 18\nNo prior treatment"}
            } for i in range(20)
        ]

        mock_batch_fetch.return_value = {
            trial["nct_id"]: trial for trial in medium_trials
        }

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        def complete_flow():
            trial_ids = [t["nct_id"] for t in medium_trials]
            return coordinator.process_clinical_trials_complete_flow(trial_ids)

        result = benchmark(complete_flow)

        assert result.success is True
        # Performance assertion: medium flow should complete reasonably
        assert benchmark.stats.stats.mean < 1.5, f"Medium flow should be reasonable, got {benchmark.stats.stats.mean:.2f}s"

    def test_get_flow_statistics_performance(self, benchmark):
        """Benchmark flow statistics retrieval."""
        config = {"test": "config", "nested": {"value": 123}}
        coordinator = DataFlowCoordinator(config=config)

        def get_stats():
            return coordinator.get_flow_statistics()

        result = benchmark(get_stats)

        assert result["coordinator_type"] == "data_flow_coordinator"
        assert result["config"] == config
        # Performance assertion: stats retrieval should be instant
        assert benchmark.stats.stats.mean < 0.001, f"Stats retrieval should be instant, got {benchmark.stats.stats.mean:.4f}s"

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 25])
    def test_batch_size_performance_impact(self, benchmark, mock_pipeline, large_trial_dataset, batch_size):
        """Test performance impact of different batch sizes."""
        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        def batch_processing():
            return coordinator._process_trials_in_batches(large_trial_dataset, batch_size=batch_size)

        result = benchmark(batch_processing)

        assert result.success is True
        assert result.metadata["total_processed"] == len(large_trial_dataset)

        # Log performance for analysis (no strict assertion, just measurement)
        print(f"Batch size {batch_size}: {benchmark.stats.stats.mean:.3f}s")

    def test_memory_efficiency_large_dataset(self, benchmark, mock_pipeline):
        """Test memory efficiency with very large dataset."""
        # Create a very large dataset to test memory handling
        huge_trials = [
            {
                "nct_id": f"NCT{i:08d}",
                "title": f"Huge Trial {i}",
                "conditions": ["Cancer"] * 20,  # Many conditions
                "eligibility": {
                    "criteria": "Age >= 18\n" + "\n".join([f"Detailed criterion {j}" for j in range(100)])
                },
                "interventions": [{"type": "Drug", "name": f"Drug {j}", "description": "x" * 500} for j in range(10)]
            } for i in range(25)  # Large number of trials
        ]

        coordinator = DataFlowCoordinator(pipeline=mock_pipeline)

        def process_huge_dataset():
            return coordinator._process_trials_in_batches(huge_trials, batch_size=5)

        result = benchmark(process_huge_dataset)

        assert result.success is True
        assert result.metadata["total_processed"] == 25
        # Performance assertion: should handle large data without excessive time
        assert benchmark.stats.stats.mean < 5.0, f"Large dataset should be manageable, got {benchmark.stats.stats.mean:.2f}s"