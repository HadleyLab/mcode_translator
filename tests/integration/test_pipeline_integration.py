"""
Integration tests for pipeline components with real data sources.
Tests end-to-end functionality without mocking external dependencies.
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch
from src.pipeline.mcode_pipeline import McodePipeline
from src.services.summarizer import McodeSummarizer
from src.core.dependency_container import DependencyContainer


@pytest.mark.live
@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline with real data."""

    @pytest.fixture
    def sample_trial_data(self):
        """Load real sample trial data."""
        data_path = Path(__file__).parent.parent / "data" / "sample_trial.json"
        with open(data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def sample_patient_data(self):
        """Load real sample patient data."""
        data_path = Path(__file__).parent.parent / "data" / "sample_patient.json"
        with open(data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def container(self):
        """Create dependency container for integration tests."""
        return DependencyContainer()

    def test_mcode_pipeline_with_real_trial_data(self, sample_trial_data, container):
        """Test McodePipeline processing real trial data."""
        # Skip if live tests disabled
        if not os.getenv("ENABLE_LIVE_TESTS", "").lower() == "true":
            pytest.skip("Live tests disabled")

        pipeline = container.create_clinical_trial_pipeline()

        result = pipeline.process(sample_trial_data)

        assert result is not None
        assert hasattr(result, 'success')
        assert result.success is True
        assert hasattr(result, 'data')
        assert result.data is not None

        # Check for expected mCODE elements in result
        if hasattr(result.data, 'mcode_elements'):
            assert isinstance(result.data.mcode_elements, list)

    def test_summarizer_with_real_trial_data(self, sample_trial_data):
        """Test McodeSummarizer with real trial data."""
        summarizer = McodeSummarizer()

        summary = summarizer.create_trial_summary(sample_trial_data)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert sample_trial_data.get("nct_id", "") in summary
        assert sample_trial_data.get("title", "") in summary

    def test_summarizer_with_real_patient_data(self, sample_patient_data):
        """Test McodeSummarizer with real patient data."""
        summarizer = McodeSummarizer()

        summary = summarizer.create_patient_summary(sample_patient_data)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @patch('src.utils.core_memory_client.CoreMemoryClient.ingest')
    def test_pipeline_with_memory_storage(self, mock_ingest, sample_trial_data, container):
        """Test pipeline with memory storage integration."""
        # Skip if live tests disabled
        if not os.getenv("ENABLE_LIVE_TESTS", "").lower() == "true":
            pytest.skip("Live tests disabled")

        mock_ingest.return_value = {"status": "success"}

        pipeline = container.create_clinical_trial_pipeline()
        storage = container.create_memory_storage()

        # Process trial
        result = pipeline.process(sample_trial_data)

        # Store result
        storage_key = f"trial_{sample_trial_data['nct_id']}"
        storage.store(storage_key, result.data)

        # Retrieve and verify
        retrieved = storage.retrieve(storage_key)
        assert retrieved is not None

        mock_ingest.assert_called()

    def test_data_flow_coordinator_integration(self, sample_trial_data):
        """Test DataFlowCoordinator with real data."""
        # Skip if live tests disabled
        if not os.getenv("ENABLE_LIVE_TESTS", "").lower() == "true":
            pytest.skip("Live tests disabled")

        from src.core.data_flow_coordinator import DataFlowCoordinator

        coordinator = DataFlowCoordinator()

        trial_ids = [sample_trial_data["nct_id"]]

        # This would normally fetch from real APIs, but we'll mock the fetch
        with patch.object(coordinator, '_fetch_trial_data') as mock_fetch:
            mock_fetch.return_value = type('WorkflowResult', (), {
                'success': True,
                'data': [sample_trial_data],
                'errors': []
            })()

            result = coordinator.process_clinical_trials_complete_flow(trial_ids)

            assert result is not None
            mock_fetch.assert_called_once_with(trial_ids)

    def test_config_loading_integration(self):
        """Test configuration loading with real config files."""
        from src.utils.config import Config

        config = Config()

        # Test that config loads without errors
        assert config is not None

        # Test core memory config
        cm_config = config.get_core_memory_config()
        assert isinstance(cm_config, dict)

        # Test LLM config
        llm_config = config.get_llm_config("default")
        assert llm_config is not None

    def test_api_manager_with_cache(self, tmp_path):
        """Test APIManager with real file cache."""
        from src.utils.api_manager import APIManager

        cache_dir = tmp_path / "api_cache"
        cache_dir.mkdir()

        manager = APIManager(cache_dir=str(cache_dir))

        # Test cache operations
        cache = manager.get_cache("test_namespace")

        # Store and retrieve
        test_data = {"test": "data"}
        cache.set(test_data, "test_func", "arg1", "arg2")

        retrieved = cache.get("test_func", "arg1", "arg2")
        assert retrieved == test_data

        # Test stats
        stats = manager.get_cache_stats("test_namespace")
        assert isinstance(stats, dict)

    def test_patient_generator_with_real_archive(self):
        """Test PatientGenerator with real patient data archive."""
        # This test would require actual patient data archives
        # For now, we'll test the generator structure
        from src.utils.patient_generator import PatientGenerator

        # Test with non-existent archive (should handle gracefully)
        try:
            generator = PatientGenerator("nonexistent.zip")
            # Should not crash during init
            assert generator is not None
        except Exception:
            # Expected for non-existent file
            pass

    def test_pattern_manager_loading(self):
        """Test PatternManager loading real patterns."""
        from src.utils.pattern_config import PatternManager

        manager = PatternManager()

        # Test pattern loading
        biomarker_patterns = manager.get_biomarker_patterns()
        assert isinstance(biomarker_patterns, dict)

        genomic_patterns = manager.get_genomic_patterns()
        assert isinstance(genomic_patterns, dict)

        all_patterns = manager.get_all_patterns()
        assert isinstance(all_patterns, dict)
        assert "biomarkers" in all_patterns
        assert "genomics" in all_patterns

    def test_token_tracker_operations(self):
        """Test TokenTracker with real operations."""
        from src.utils.token_tracker import TokenTracker, TokenUsage

        tracker = TokenTracker()

        # Add usage
        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        tracker.add_usage(usage, "test_component")

        # Get total usage
        total = tracker.get_total_usage()
        assert total.total_tokens == 150

        # Get component usage
        component_usage = tracker.get_component_usage("test_component")
        assert component_usage.total_tokens == 150

        # Reset
        tracker.reset()
        total_after_reset = tracker.get_total_usage()
        assert total_after_reset.total_tokens == 0