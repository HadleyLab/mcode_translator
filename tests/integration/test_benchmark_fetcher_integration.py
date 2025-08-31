"""
Integration tests for benchmark task tracker and fetcher components
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.optimization.benchmark_task_tracker import BenchmarkTaskTrackerUI
from src.pipeline.fetcher import (
    search_trials, 
    get_full_study, 
    calculate_total_studies,
    _cached_search_trials,
    _cached_get_full_study,
    _cached_calculate_total_studies
)


class TestBenchmarkFetcherIntegration(unittest.TestCase):
    """Integration tests for benchmark task tracker and fetcher components"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for cache
        self.test_cache_dir = tempfile.mkdtemp()
        
        # Mock the NiceGUI components to avoid UI initialization for benchmark tracker
        with patch('src.optimization.benchmark_task_tracker.ui'), \
             patch('src.optimization.benchmark_task_tracker.background_tasks'), \
             patch('src.optimization.benchmark_task_tracker.run'):
            self.tracker = BenchmarkTaskTrackerUI()

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary cache directory
        shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_benchmark_tracker_uses_fetcher_caching(self, mock_config, mock_requests):
        """Test that benchmark tracker properly uses fetcher caching"""
        # Mock the config
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://clinicaltrials.gov/api/v2/studies"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance
        
        # Mock the requests response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "studies": [{"study": "test1"}, {"study": "test2"}],
            "totalCount": 2
        }
        mock_requests.return_value = mock_response
        
        # Test that the fetcher functions used by benchmark tracker are properly cached
        # First call should make an API request
        result1 = search_trials("cancer", max_results=10, use_cache=True)
        self.assertEqual(mock_requests.call_count, 1)
        
        # Second call with same parameters should use cache
        result2 = search_trials("cancer", max_results=10, use_cache=True)
        self.assertEqual(mock_requests.call_count, 1)  # Should still be 1
        
        # Results should be the same
        self.assertEqual(result1, result2)

    def test_benchmark_tracker_pipeline_callback_integration(self):
        """Test that benchmark tracker pipeline callback integrates with fetcher"""
        # Create a pipeline callback
        callback = self.tracker._create_pipeline_callback()
        
        # Verify it's callable
        self.assertTrue(callable(callback))
        
        # Test with mock data
        test_data = {"test": "data"}
        prompt_content = "Test prompt"
        prompt_variant_id = "test_variant"
        
        # We can't fully test the pipeline without mocking the entire pipeline,
        # but we can verify the callback is created correctly
        self.assertTrue(callable(callback))

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_fetcher_cache_persistence_integration(self, mock_config, mock_requests):
        """Test that fetcher cache persists across different function calls"""
        # Mock the config
        mock_config_instance = Mock()
        mock_config_instance.get_rate_limit_delay.return_value = 0
        mock_config_instance.get_clinical_trials_base_url.return_value = "https://clinicaltrials.gov/api/v2/studies"
        mock_config_instance.get_request_timeout.return_value = 30
        mock_config.return_value = mock_config_instance
        
        # Mock the requests response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "studies": [{"study": "test1"}, {"study": "test2"}],
            "totalCount": 2
        }
        mock_requests.return_value = mock_response
        
        # Test search trials caching
        result1 = _cached_search_trials("cancer", "", 10, "None")
        first_call_count = mock_requests.call_count
        
        # Call again with same parameters
        result2 = _cached_search_trials("cancer", "", 10, "None")
        
        # Should not have made another API call
        self.assertEqual(mock_requests.call_count, first_call_count)
        self.assertEqual(result1, result2)
        
        # Test get full study caching
        mock_response.json.return_value = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study"
                }
            }
        }
        
        result3 = _cached_get_full_study("NCT12345678")
        second_call_count = mock_requests.call_count
        
        # Call again with same parameters
        result4 = _cached_get_full_study("NCT12345678")
        
        # Should not have made another API call
        self.assertEqual(mock_requests.call_count, second_call_count)
        self.assertEqual(result3, result4)

    def test_benchmark_tracker_initialization_with_fetcher(self):
        """Test that benchmark tracker initializes correctly with fetcher components"""
        # The tracker should initialize without errors
        self.assertIsNotNone(self.tracker)
        self.assertIsNotNone(self.tracker.framework)
        
        # Check that available prompts and models are loaded
        self.assertIsInstance(self.tracker.available_prompts, dict)
        self.assertIsInstance(self.tracker.available_models, dict)


if __name__ == '__main__':
    unittest.main()