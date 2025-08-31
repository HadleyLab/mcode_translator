"""
Unit tests for the fetcher module with diskcache implementation
"""
import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import cache first to clear it
from src.pipeline.fetcher import cache
# Clear the cache before running tests
cache.clear()

from src.pipeline.fetcher import (
    _cached_search_trials,
    _cached_get_full_study,
    _cached_calculate_total_studies,
    search_trials,
    get_full_study,
    calculate_total_studies
)


class TestFetcherDiskCache(unittest.TestCase):
    """Test cases for fetcher module with diskcache"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear the cache to ensure clean state for each test
        cache.clear()
        # Create a temporary directory for cache
        self.test_cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        # Clean up the temporary cache directory
        shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    def test_cached_search_trials_decorator(self):
        """Test that _cached_search_trials uses diskcache memoize decorator"""
        # Check that the function has the memoize attribute
        self.assertTrue(hasattr(_cached_search_trials, '__cache_key__'))
        self.assertTrue(callable(_cached_search_trials))

    def test_cached_get_full_study_decorator(self):
        """Test that _cached_get_full_study uses diskcache memoize decorator"""
        # Check that the function has the memoize attribute
        self.assertTrue(hasattr(_cached_get_full_study, '__cache_key__'))
        self.assertTrue(callable(_cached_get_full_study))

    def test_cached_calculate_total_studies_decorator(self):
        """Test that _cached_calculate_total_studies uses diskcache memoize decorator"""
        # Check that the function has the memoize attribute
        self.assertTrue(hasattr(_cached_calculate_total_studies, '__cache_key__'))
        self.assertTrue(callable(_cached_calculate_total_studies))

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_search_trials_caching(self, mock_config, mock_requests):
        """Test that search_trials uses caching correctly"""
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
        
        # First call should make an API request
        result1 = search_trials("cancer", max_results=10, use_cache=True)
        
        # Verify API was called
        self.assertEqual(mock_requests.call_count, 1)
        
        # Second call with same parameters should use cache
        result2 = search_trials("cancer", max_results=10, use_cache=True)
        
        # API should still only be called once
        self.assertEqual(mock_requests.call_count, 1)
        
        # Results should be the same
        self.assertEqual(result1, result2)

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_search_trials_without_cache(self, mock_config, mock_requests):
        """Test that search_trials can bypass cache"""
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
        
        # First call should make an API request
        result1 = search_trials("cancer", max_results=10, use_cache=False)
        
        # Verify API was called
        self.assertEqual(mock_requests.call_count, 1)
        
        # Second call with same parameters but cache disabled should make another API request
        result2 = search_trials("cancer", max_results=10, use_cache=False)
        
        # API should be called twice
        self.assertEqual(mock_requests.call_count, 2)
        
        # Results should be the same
        self.assertEqual(result1, result2)

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_get_full_study_caching(self, mock_config, mock_requests):
        """Test that get_full_study uses caching correctly"""
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
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "briefTitle": "Test Study"
                }
            }
        }
        mock_requests.return_value = mock_response
        
        # First call should make an API request
        result1 = get_full_study("NCT12345678", use_cache=True)
        
        # Verify API was called
        self.assertEqual(mock_requests.call_count, 1)
        
        # Second call with same parameters should use cache
        result2 = get_full_study("NCT12345678", use_cache=True)
        
        # API should still only be called once
        self.assertEqual(mock_requests.call_count, 1)
        
        # Results should be the same
        self.assertEqual(result1, result2)

    @patch('src.pipeline.fetcher.requests.get')
    @patch('src.pipeline.fetcher.Config')
    def test_calculate_total_studies_caching(self, mock_config, mock_requests):
        """Test that calculate_total_studies uses caching correctly"""
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
            "totalCount": 150
        }
        mock_requests.return_value = mock_response
        
        # First call should make an API request
        result1 = calculate_total_studies("cancer", page_size=10, use_cache=True)
        
        # Verify API was called
        self.assertEqual(mock_requests.call_count, 1)
        
        # Second call with same parameters should use cache
        result2 = calculate_total_studies("cancer", page_size=10, use_cache=True)
        
        # API should still only be called once
        self.assertEqual(mock_requests.call_count, 1)
        
        # Results should be the same
        self.assertEqual(result1, result2)

    def test_cache_directory_creation(self):
        """Test that cache directory is created"""
        # Check that cache directory exists
        self.assertTrue(os.path.exists('./cache/fetcher_cache'))
        
        # Check that it's a directory
        self.assertTrue(os.path.isdir('./cache/fetcher_cache'))

    def test_cache_persistence(self):
        """Test that cache persists between function calls"""
        # This test would require running the function twice and checking
        # that the second call doesn't make an API request
        # We'll test this more thoroughly in integration tests
        pass


if __name__ == '__main__':
    unittest.main()