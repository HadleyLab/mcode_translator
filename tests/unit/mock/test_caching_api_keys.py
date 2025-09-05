#!/usr/bin/env python3
"""
Mock unit tests for API caching with different API keys.
Focuses on testing that API keys are properly included in cache operations.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from src.utils.api_manager import UnifiedAPIManager, APICache
from src.utils.model_loader import ModelLoader, ModelConfig
from src.utils.prompt_loader import PromptLoader
from src.pipeline.llm_base import LlmBase


class TestCachingAPIKeys(unittest.TestCase):
    """Mock tests for API caching with different API keys"""
    
    def setUp(self):
        """Set up test environment"""
        self.api_manager = UnifiedAPIManager()
        
    def test_api_key_in_cache_key_generation(self):
        """Test that API keys are included in cache key generation"""
        # Get the llm cache instance
        llm_cache = self.api_manager.get_cache("llm")
        
        # Test data with different API keys
        test_data_1 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-test-key-1"
        }
        
        test_data_2 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-test-key-2"  # Different key
        }
        
        test_data_3 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-test-key-1"  # Same as first
        }
        
        # Generate cache keys
        key1 = llm_cache._generate_cache_key(test_data_1)
        key2 = llm_cache._generate_cache_key(test_data_2)
        key3 = llm_cache._generate_cache_key(test_data_3)
        
        # Different API keys should produce different cache keys
        self.assertNotEqual(key1, key2, "Different API keys should produce different cache keys")
        
        # Same API key should produce same cache key
        self.assertEqual(key1, key3, "Same API key should produce same cache key")
    
    def test_model_config_with_different_api_keys(self):
        """Test that ModelConfig properly handles different API keys"""
        # Create model config with first API key
        config1 = ModelConfig(
            name="test-model",
            model_type="TEST",
            model_identifier="test-model-id",
            api_key_env_var="TEST_API_KEY",
            base_url="https://api.test.com/v1",
            default_parameters={"temperature": 0.1}
        )
        
        # Mock environment for first API key
        with patch.dict(os.environ, {'TEST_API_KEY': 'test_key_1'}):
            api_key_1 = config1.api_key
        
        # Mock environment for second API key
        with patch.dict(os.environ, {'TEST_API_KEY': 'test_key_2'}):
            api_key_2 = config1.api_key
        
        # API keys should be different
        self.assertNotEqual(api_key_1, api_key_2)
        self.assertEqual(api_key_1, 'test_key_1')
        self.assertEqual(api_key_2, 'test_key_2')
    
    def test_cache_isolation_between_api_keys(self):
        """Test that cache properly isolates entries for different API keys"""
        # Get the llm cache instance
        llm_cache = self.api_manager.get_cache("llm")
        
        # Clear cache for clean test
        llm_cache.clear()
        
        # Test data with different API keys
        test_data_1 = {
            "function_name": "test_function",
            "args": ("arg1",),
            "kwargs": {"kwarg1": "value1"},
            "api_key": "sk-api-key-1"
        }
        
        test_data_2 = {
            "function_name": "test_function",
            "args": ("arg1",),
            "kwargs": {"kwarg1": "value1"},
            "api_key": "sk-api-key-2"  # Different key
        }
        
        # Store different results for each API key
        result1 = {"response": "result for api key 1"}
        result2 = {"response": "result for api key 2"}
        
        llm_cache.set_by_key(result1, test_data_1, ttl=60)
        llm_cache.set_by_key(result2, test_data_2, ttl=60)
        
        # Retrieve results
        cached_result1 = llm_cache.get_by_key(test_data_1)
        cached_result2 = llm_cache.get_by_key(test_data_2)
        
        # Results should be different
        self.assertNotEqual(cached_result1, cached_result2)
        self.assertEqual(cached_result1["response"], "result for api key 1")
        self.assertEqual(cached_result2["response"], "result for api key 2")


if __name__ == '__main__':
    unittest.main()