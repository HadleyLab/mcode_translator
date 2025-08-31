#!/usr/bin/env python3
"""
Simple test script to verify LLM disk-based caching is working correctly
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.cache_manager import cache_manager

def test_llm_caching():
    """Test that LLM calls are properly cached to disk"""
    print("🧪 Testing LLM disk-based caching...")
    
    # Clear LLM cache first
    cache_manager.clear_cache("llm_cache")
    print("🗑️  Cleared LLM cache")
    
    # Check initial cache size
    initial_size = len(cache_manager.llm_cache)
    print(f"📊 Initial cache size: {initial_size}")
    
    # Create test data
    cache_key = "test_cache_key"
    test_result = ("Test response content", {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    })
    
    # Check if entry exists in cache (should not)
    cached_result = cache_manager.llm_cache.get(cache_key)
    if cached_result is not None:
        print("❌ Unexpected cache hit before first call")
        return False
    
    # Store in cache
    cache_manager.llm_cache.set(cache_key, test_result)
    print("💾 Stored test result in cache")
    
    # Check cache size after storing
    after_store_size = len(cache_manager.llm_cache)
    print(f"📊 Cache size after storing: {after_store_size}")
    
    # Try to retrieve from cache
    retrieved_result = cache_manager.llm_cache.get(cache_key)
    if retrieved_result is None:
        print("❌ Cache miss after storing entry")
        return False
    
    if retrieved_result != test_result:
        print("❌ Retrieved result doesn't match stored result")
        print(f"   Stored: {test_result}")
        print(f"   Retrieved: {retrieved_result}")
        return False
    
    print("✅ Cache hit - retrieved result matches stored result")
    
    # List cache keys
    keys = cache_manager.list_cache_keys("llm_cache")
    print(f"📋 Cache contains {len(keys)} keys")
    if len(keys) > 0:
        print(f"   Sample key: {keys[0][:50]}...")
    
    print("🎉 All LLM caching tests passed!")
    return True

if __name__ == "__main__":
    success = test_llm_caching()
    if not success:
        print("\n💥 LLM caching test failed!")
        sys.exit(1)