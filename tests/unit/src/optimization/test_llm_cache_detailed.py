#!/usr/bin/env python3
"""
Detailed test to understand LLM caching behavior
"""
import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.api_manager import UnifiedAPIManager, APICache
from pipeline.llm_base import LlmBase

class MockLLM:
    """Simple mock LLM implementation without LlmBase inheritance"""
    
    def __init__(self, model_name=None, temperature=None, max_tokens=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

def test_llm_caching_detailed():
    """Detailed test of LLM caching"""
    print("🔬 Detailed LLM Caching Test")
    print("=" * 50)
    
    # Clear cache first
    llm_cache = APICache(".llm_cache")
    llm_cache.clear()
    print("🗑️  Cleared LLM cache")
    
    # Create test LLM instance data (no inheritance needed)
    model_name = "deepseek-chat"
    temperature = 0.1
    max_tokens = 100
    
    # Create test cache key data
    cache_key_data = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt": "Test prompt for caching",
        "task": "entity_extraction"
    }
    
    print(f"🔑 Cache key data: {cache_key_data}")
    
    # Generate cache key
    cache_key = llm_cache._generate_cache_key(cache_key_data)
    print(f"🔑 Generated cache key: {cache_key}")
    
    # Check if cached (should be none)
    cached_result = llm_cache.get_by_key(cache_key_data)
    print(f"🔍 Initial cache check: {cached_result}")
    
    # Create test result
    test_result = {
        'response_content': '{"test": "cached response"}',
        'metrics': {
            'duration': 0.5,
            'prompt_tokens': 10,
            'completion_tokens': 20,
            'total_tokens': 30
        }
    }
    
    # Store in cache
    print("💾 Storing test result in cache...")
    llm_cache.set_by_key(test_result, cache_key_data)
    
    # Check cache files
    cache_files = os.listdir(".llm_cache")
    print(f"📁 Cache files after storing: {len(cache_files)}")
    
    # Try to retrieve from cache
    print("🔍 Retrieving from cache...")
    retrieved_result = llm_cache.get_by_key(cache_key_data)
    
    if retrieved_result is not None:
        print("✅ Cache hit - retrieved result from cache")
        print(f"   Stored: {test_result}")
        print(f"   Retrieved: {retrieved_result}")
        
        # Check if they match
        if retrieved_result == test_result:
            print("✅ Retrieved result matches stored result")
        else:
            print("❌ Retrieved result doesn't match stored result")
            assert False, "Retrieved result doesn't match stored result"
    else:
        print("❌ Cache miss - no result retrieved")
        assert False, "Cache miss - no result retrieved"
    
    print("✅ Detailed LLM caching test passed")

def test_cache_expiration():
    """Test cache expiration"""
    print("\n⏰ Cache Expiration Test")
    print("=" * 30)
    
    # Clear cache
    llm_cache = APICache(".llm_cache")
    llm_cache.clear()
    
    # Create test data with short TTL
    cache_key_data = {"test": "expiration"}
    test_result = {"data": "expires quickly"}
    
    # Store with short TTL (1 second)
    cache_key = llm_cache._generate_cache_key(cache_key_data)
    llm_cache._set_by_key(test_result, cache_key, "test_func", (), {}, ttl=1)  # 1 second TTL
    
    # Immediately check cache
    result1 = llm_cache.get_by_key(cache_key_data)
    print(f"🔍 Immediate cache check: {result1 is not None}")
    
    # Wait for expiration
    print("⏳ Waiting for cache to expire (2 seconds)...")
    time.sleep(2)
    
    # Check cache again
    result2 = llm_cache.get_by_key(cache_key_data)
    print(f"🔍 Cache check after expiration: {result2 is not None}")
    
    if result1 is not None and result2 is None:
        print("✅ Cache expiration working correctly")
    else:
        print("❌ Cache expiration not working")
        assert False, "Cache expiration not working"
    
    print("✅ Cache expiration test passed")

if __name__ == "__main__":
    try:
        test_llm_caching_detailed()
        test_cache_expiration()
        print("\n🎉 All detailed caching tests passed!")
    except Exception as e:
        print(f"\n💥 Error during detailed testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)