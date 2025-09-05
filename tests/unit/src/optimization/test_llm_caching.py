#!/usr/bin/env python3
"""
Test script to verify LLM caching is working correctly
"""
import sys
import os
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.llm_base import LlmBase
from utils.api_manager import UnifiedAPIManager

class TestLLM(LlmBase):
    """Test implementation of LlmBase for testing caching"""
    
    def process_request(self, *args, **kwargs):
        return {"test": "response"}

def test_llm_caching():
    """Test that LLM calls are properly cached"""
    print("🧪 Testing LLM caching...")
    
    # Clear LLM cache first
    api_manager = UnifiedAPIManager()
    llm_cache = api_manager.get_cache("llm")
    llm_cache.clear()
    print("🗑️  Cleared LLM cache")
    
    # Create test LLM instance
    llm = TestLLM(
        model_name="deepseek-chat",
        temperature=0.1,
        max_tokens=100
    )
    
    # Create test messages and cache key data
    messages = [{"role": "user", "content": "Test message for caching"}]
    cache_key_data = {
        "test": "data",
        "purpose": "caching verification",
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens
    }
    
    # Check if entry exists in cache (should not)
    cached_result = llm_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        print("❌ Unexpected cache hit before first call")
        return False
    
    # Create test result data
    test_result = {
        'response_content': "Test response content",
        'metrics': {
            'duration': 1.0,
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30,
            'success': True,
            'error_type': None
        }
    }
    # Store in cache using the LLM's cache
    llm_cache.set_by_key(test_result, cache_key_data)
    print("💾 Stored test result in cache")
    
    # Try to retrieve from cache
    retrieved_result = llm_cache.get_by_key(cache_key_data)
    if retrieved_result is None:
        print("❌ Cache miss after storing entry")
        return False
    
    
    if retrieved_result != test_result:
        print("❌ Retrieved result doesn't match stored result")
        print(f"   Stored: {test_result}")
        print(f"   Retrieved: {retrieved_result}")
        return False
    
    print("✅ Cache hit - retrieved result matches stored result")
    
    print("🎉 All LLM caching tests passed!")
    return True

if __name__ == "__main__":
    success = test_llm_caching()
    if not success:
        print("\n💥 LLM caching test failed!")
        sys.exit(1)