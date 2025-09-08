#!/usr/bin/env python3
"""
Test script to verify LLM disk-based caching is working correctly
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
    """Test that LLM calls are properly cached to disk"""
    print("🧪 Testing LLM disk-based caching...")
    
    # Clear LLM cache first
    UnifiedAPIManager().clear_cache("llm")
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
        "purpose": "caching verification"
    }
    
    # Create a comprehensive cache data structure
    complete_cache_data = {
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "response_format": llm.response_format,
        "messages": messages,
        **cache_key_data
    }
    
    # Generate cache key
    cache_key = json.dumps(complete_cache_data, sort_keys=True)
    
    # Check if entry exists in cache (should not)
    llm_cache = UnifiedAPIManager().get_cache("llm")
    cached_result = llm_cache.get_by_key(cache_key)
    if cached_result is not None:
        print("❌ Unexpected cache hit before first call")
        return False
    
    # Simulate storing a result in cache (since we're not making real API calls)
    test_result = ("Test response content", {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    })
    
    # Store in cache
    llm_cache.set_by_key(test_result, cache_key)
    print("💾 Stored test result in cache")
    
    # Try to retrieve from cache
    retrieved_result = llm_cache.get_by_key(cache_key)
    if retrieved_result is None:
        print("❌ Cache miss after storing entry")
        return False
    
    if retrieved_result != test_result:
        print("❌ Retrieved result doesn't match stored result")
        print(f"   Stored: {test_result}")
        print(f"   Retrieved: {retrieved_result}")
        return False
    
    print("✅ Cache hit - retrieved result matches stored result")
    
    # Test with _cached_llm_call method
    try:
        result = llm._cached_llm_call(cache_key)
        if result != test_result:
            print("❌ _cached_llm_call returned different result than expected")
            return False
        print("✅ _cached_llm_call correctly retrieved cached result")
    except Exception as e:
        print(f"❌ Error calling _cached_llm_call: {e}")
        return False
    
    # Get cache stats instead of listing keys
    stats = UnifiedAPIManager().get_cache_stats("llm")
    print(f"📋 Cache stats: {stats['llm']['cached_items']} items")
    
    print("🎉 All LLM caching tests passed!")
    return True

if __name__ == "__main__":
    success = test_llm_caching()
    if not success:
        print("\n💥 LLM caching test failed!")
        sys.exit(1)