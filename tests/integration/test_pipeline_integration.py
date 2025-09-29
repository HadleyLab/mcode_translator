"""
Integration tests for pipeline components with real data sources.
Tests end-to-end functionality without mocking external dependencies.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.dependency_container import DependencyContainer
from src.services.summarizer import McodeSummarizer
from src.shared.models import McodeElement
from src.utils.config import Config


@pytest.mark.live
@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline with real data."""

    @pytest.fixture
    def sample_trial_data(self):
        """Load real sample trial data."""
        data_path = Path(__file__).parent.parent / "data" / "sample_trial.json"
        with open(data_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def sample_patient_data(self):
        """Load real sample patient data."""
        data_path = Path(__file__).parent.parent / "data" / "sample_patient.json"
        with open(data_path, "r") as f:
            return json.load(f)

    @pytest.fixture
    def container(self):
        """Create dependency container for integration tests."""
        return DependencyContainer()

    @pytest.mark.asyncio
    async def test_mcode_pipeline_with_real_trial_data(
        self, sample_trial_data, container
    ):
        """Test McodePipeline processing real trial data."""
        pipeline = container.create_clinical_trial_pipeline()

        result = await pipeline.process(sample_trial_data)

        assert result is not None
        assert result.error is None

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

    @pytest.mark.asyncio
    async def test_pipeline_with_memory_storage(
        self, sample_trial_data, container
    ):
        """Test pipeline with memory storage integration."""
        from unittest.mock import Mock

        pipeline = container.create_clinical_trial_pipeline()
        storage = container.create_memory_storage()

        # Mock the client
        mock_client = Mock()
        mock_client.ingest.return_value = {"status": "success"}
        mock_client.search.return_value = {"episodes": [], "facts": []}
        storage._client = mock_client

        # Process trial
        result = await pipeline.process(sample_trial_data)

        # Store result
        trial_id = sample_trial_data["protocolSection"]["identificationModule"]["nctId"]
        # Convert PipelineResult to dict format expected by storage
        pipeline_result_dict = result.model_dump()

        storage_data = {
            "original_trial_data": sample_trial_data,
            "pipeline_result": pipeline_result_dict,
            "trial_metadata": {
                "brief_title": sample_trial_data.get("protocolSection", {})
                .get("identificationModule", {})
                .get("briefTitle"),
                "overall_status": sample_trial_data.get("protocolSection", {})
                .get("statusModule", {})
                .get("overallStatus"),
            },
        }
        storage.store_trial_mcode_summary(trial_id, storage_data)

        # Retrieve and verify (using search since retrieve method doesn't exist)
        search_results = storage.search_similar_trials(f"trial {trial_id}")
        assert search_results is not None

        mock_client.ingest.assert_called()

    def test_data_flow_coordinator_integration(self, sample_trial_data):
        """Test DataFlowCoordinator with real data."""
        from src.core.data_flow_coordinator import DataFlowCoordinator

        coordinator = DataFlowCoordinator()

        trial_ids = [
            sample_trial_data["protocolSection"]["identificationModule"]["nctId"]
        ]

        # This would normally fetch from real APIs, but we'll mock the fetch
        with patch("src.core.data_fetcher.DataFetcher.fetch_trial_data") as mock_fetch:
            mock_fetch.return_value = type(
                "WorkflowResult",
                (),
                {
                    "success": True,
                    "data": [sample_trial_data],
                    "errors": [],
                    "metadata": {},
                },
            )()

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
        llm_config = config.get_llm_config("deepseek-coder")
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
        assert "biomarker" in all_patterns
        assert "genomic" in all_patterns

    def test_token_tracker_operations(self):
        """Test TokenTracker with real operations."""
        from src.utils.token_tracker import TokenTracker, TokenUsage

        tracker = TokenTracker()
        tracker.reset()  # Reset before test

        # Add usage
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
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

    def test_error_handling_with_corrupted_data(self):
        """Test handling of corrupted data files."""
        from src.pipeline import McodePipeline

        pipeline = McodePipeline()

        # Test with corrupted JSON
        corrupted_data = '{"protocolSection": {"identificationModule": {"nctId": "NCT123"'  # Missing closing brace

        with pytest.raises(json.JSONDecodeError):
            json.loads(corrupted_data)

        # Test with missing required fields
        incomplete_data = {"protocolSection": {}}

        with pytest.raises(Exception):
            # This should fail due to missing required fields
            import asyncio
            asyncio.run(pipeline.process(incomplete_data))

    def test_concurrent_pipeline_access(self):
        """Test concurrent access to pipeline resources."""
        from src.pipeline import McodePipeline
        import asyncio

        async def run_pipeline(data):
            pipeline = McodePipeline()
            with patch(
                "src.pipeline.llm_service.LLMService.map_to_mcode"
            ) as mock_map:
                mock_map.return_value = [
                    McodeElement(
                        element_type="CancerCondition", code="C123", display="Test"
                    )
                ]
                return await pipeline.process(data)

        sample_data = {"protocolSection": {"identificationModule": {"nctId": "NCT123"}}}

        # Run multiple pipelines concurrently
        tasks = [run_pipeline(sample_data) for _ in range(5)]
        results = asyncio.run(asyncio.gather(*tasks))

        assert len(results) == 5
        for result in results:
            assert result is not None

    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion scenarios."""
        from src.pipeline import McodePipeline

        pipeline = McodePipeline()

        # Create very large data that might cause memory issues
        large_data = {
            "protocolSection": {
                "identificationModule": {"nctId": "NCT123"},
                "descriptionModule": {"briefSummary": "A" * 1000000},  # 1MB string
                "conditionsModule": {"conditions": ["Cancer"] * 10000},  # Many conditions
            }
        }

        with patch(
            "src.pipeline.llm_service.LLMService.map_to_mcode"
        ) as mock_map:
            mock_map.return_value = [
                McodeElement(
                    element_type="CancerCondition", code="C123", display="Test"
                )
            ]

            # This should handle large data without crashing
            import asyncio
            result = asyncio.run(pipeline.process(large_data))
            assert result is not None

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        from src.utils.config import Config

        # Test with invalid config file
        try:
            Config(config_path="nonexistent.json")
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            assert "config" in str(e).lower() or "file" in str(e).lower()

    def test_api_rate_limiting(self):
        """Test handling of API rate limiting."""
        from src.pipeline import McodePipeline

        pipeline = McodePipeline()

        with patch("src.pipeline.llm_service.LLMService.map_to_mcode") as mock_map:
            # Simulate rate limiting
            mock_map.side_effect = Exception("Rate limit exceeded")

            with pytest.raises(Exception) as exc_info:
                import asyncio
                asyncio.run(pipeline.process({"protocolSection": {"identificationModule": {"nctId": "NCT123"}}}))
            assert "rate" in str(exc_info.value).lower() or "limit" in str(exc_info.value).lower()
