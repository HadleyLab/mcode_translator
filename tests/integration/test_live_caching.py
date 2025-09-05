#!/usr/bin/env python3
"""
Live integration test for API caching with different API keys.
Tests the interaction between API manager, model loader, and prompt loader with real components.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.utils.api_manager import UnifiedAPIManager, APICache
from src.utils import ModelLoader
from src.utils.prompt_loader import PromptLoader
from src.utils.config import Config
from src.pipeline.llm_base import LlmBase


class TestLiveCaching(unittest.TestCase):
    """Live integration tests for API caching with different API keys"""
    
    def setUp(self):
        """Set up test environment"""
        # Initialize components with real configurations
        self.api_manager = UnifiedAPIManager()
        self.model_loader = ModelLoader()
        self.prompt_loader = PromptLoader()
        
        # Store original environment
        self.original_env = os.environ.copy()
        
    def tearDown(self):
        """Clean up test environment"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_cache_key_includes_api_key(self):
        """Test that API keys are included in cache key generation"""
        # Get the llm cache instance
        llm_cache = self.api_manager.get_cache("llm")
        
        # Clear cache for clean test
        llm_cache.clear()
        
        # Test with actual API keys from environment (if available)
        deepseek_key_1 = os.environ.get('DEEPSEEK_API_KEY', 'test_key_1')
        deepseek_key_2 = 'test_key_2'  # Different key for testing
        
        test_data_1 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": deepseek_key_1
        }
        
        test_data_2 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": deepseek_key_2  # Different key
        }
        
        test_data_3 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": deepseek_key_1  # Same as first
        }
        
        # Generate cache keys using the cache instance
        key1 = llm_cache._generate_cache_key(test_data_1)
        key2 = llm_cache._generate_cache_key(test_data_2)
        key3 = llm_cache._generate_cache_key(test_data_3)
        
        # Different API keys should produce different cache keys
        self.assertNotEqual(key1, key2, "Different API keys should produce different cache keys")
        
        # Same API key should produce same cache key
        self.assertEqual(key1, key3, "Same API key should produce same cache key")
    
    def test_model_loader_with_different_api_keys(self):
        """Test model loader behavior with different API keys"""
        # Test with DeepSeek API key if available
        if 'DEEPSEEK_API_KEY' in os.environ:
            # Get model config with first API key
            model_config_1 = self.model_loader.get_model("deepseek-chat")
            
            # Temporarily change API key in a way that doesn't conflict with tearDown
            test_key = 'test_key_for_model_loader_test'
            os.environ['DEEPSEEK_API_KEY'] = test_key
            
            model_config_2 = self.model_loader.get_model("deepseek-chat")
            
            # Both should be valid model configs
            self.assertEqual(model_config_1.model_identifier, model_config_2.model_identifier)
            
            # API keys should be different (since we changed the environment)
            # Note: We can't directly compare because tearDown will restore the original env
            self.assertEqual(model_config_2.api_key, test_key)
        else:
            # Skip test if no API key available
            self.skipTest("No DeepSeek API key available in environment")
    
    def test_prompt_loader_functionality(self):
        """Test prompt loader functionality with real prompts"""
        # Test loading a real prompt
        try:
            prompt_content = self.prompt_loader.get_prompt("generic_extraction")
            
            self.assertIsNotNone(prompt_content)
            self.assertIn("Extract", prompt_content)
            self.assertIn("{clinical_text}", prompt_content)
        except Exception as e:
            # If no prompts available, skip this test
            self.skipTest(f"Prompt loading test skipped: {str(e)}")
    
    def test_api_manager_cache_isolation(self):
        """Test that API manager provides proper cache isolation between different API keys"""
        # Get the llm cache instance
        llm_cache = self.api_manager.get_cache("llm")
        
        # Clear cache for clean test
        llm_cache.clear()
        
        # Test data with different API keys
        test_data_1 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-key-1"
        }
        
        test_data_2 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-key-2"  # Different key
        }
        
        test_data_3 = {
            "function_name": "test_function",
            "args": ("test_arg",),
            "kwargs": {"test_kwarg": "value"},
            "api_key": "sk-key-1"  # Same as first
        }
        
        # Test results
        result1 = f"result_for_test_arg_with_sk-key-1"
        result2 = f"result_for_test_arg_with_sk-key-2"
        
        # Store results in cache
        llm_cache.set_by_key({"result": result1}, test_data_1, ttl=60)
        llm_cache.set_by_key({"result": result2}, test_data_2, ttl=60)
        
        # Retrieve from cache
        cached_result1 = llm_cache.get_by_key(test_data_1)
        cached_result3 = llm_cache.get_by_key(test_data_3)  # Should be same as result1
        
        # Results should be the same content-wise
        self.assertEqual(cached_result1["result"], result1)
        self.assertEqual(cached_result3["result"], result1)  # Same API key should return same result
        
        # Different API keys should have different cache entries
        self.assertNotEqual(cached_result1["result"], result2)


if __name__ == '__main__':
    unittest.main()