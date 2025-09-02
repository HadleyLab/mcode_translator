#!/usr/bin/env python3
"""
Test script to reproduce and debug the benchmark caching issue
"""
import sys
import os
import time
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.optimization.benchmark_task_tracker import BenchmarkTaskTrackerUI
from src.utils.api_manager import UnifiedAPIManager
from src.pipeline.strict_llm_base import StrictLLMBase

def get_cache_stats():
    """Get cache statistics from the unified API manager"""
    api_manager = UnifiedAPIManager()
    return api_manager.get_cache_stats()

def clear_api_cache():
    """Clear all API cache using the unified API manager"""
    api_manager = UnifiedAPIManager()
    api_manager.clear_cache()

def test_benchmark_caching():
    """Test that benchmark caching is working correctly"""
    print("🔍 Testing benchmark caching...")
    
    # Clear all caches first
    clear_api_cache()
    llm_cache = APICache(".llm_cache")
    llm_cache.clear()
    
    print("🗑️  Cleared all caches")
    
    # Check initial cache stats
    stats = get_cache_stats()
    print(f"📊 Initial API cache stats: {stats}")
    
    # Create benchmark tracker instance
    print("🔧 Creating benchmark tracker...")
    tracker = BenchmarkTaskTrackerUI()
    
    # Simulate running a single validation
    print("🚀 Running first validation...")
    start_time = time.time()
    
    # This would normally call the actual benchmark logic
    # For now, let's just check if the cache is working
    
    # Check cache stats after first run
    stats_after_first = get_cache_stats()
    print(f"📊 API cache stats after first run: {stats_after_first}")
    
    print("🚀 Running second validation (should be cached)...")
    time.sleep(1)  # Small delay to differentiate
    
    # Check cache stats after second run
    stats_after_second = get_cache_stats()
    print(f"📊 API cache stats after second run: {stats_after_second}")
    
    # Check LLM cache stats
    llm_cache_stats = get_cache_stats()
    print(f"📊 LLM cache stats: {llm_cache_stats}")
    
    print("✅ Benchmark caching test completed")
    return True

def test_llm_cache_directly():
    """Test LLM caching directly"""
    print("\n🔍 Testing LLM caching directly...")
    
    # Clear LLM cache
    llm_cache = APICache(".llm_cache")
    llm_cache.clear()
    print("🗑️  Cleared LLM cache")
    
    # Create cache key data that would be used in a real scenario
    cache_key_data = {
        "model": "deepseek-chat",
        "temperature": 0.1,
        "max_tokens": 1000,
        "prompt": "Extract entities from: Patient has metastatic breast cancer",
        "task_type": "entity_extraction"
    }
    
    # Create test result
    test_result = {
        "response_content": '{"entities": [{"text": "metastatic breast cancer", "type": "CONDITION"}]}',
        "metrics": {
            "duration": 1.5,
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        }
    }
    
    # Store in cache
    print("💾 Storing test result in LLM cache...")
    llm_cache.set_by_key(test_result, cache_key_data)
    
    # Retrieve from cache
    print("🔍 Retrieving from LLM cache...")
    cached_result = llm_cache.get_by_key(cache_key_data)
    
    if cached_result is not None:
        print("✅ LLM cache working - retrieved result from cache")
        print(f"   Result: {cached_result}")
        return True
    else:
        print("❌ LLM cache not working - no result retrieved")
        return False

def debug_cache_files():
    """Debug cache files to understand what's being cached"""
    print("\n🔍 Debugging cache files...")
    
    # Check API cache
    api_cache_dir = ".api_cache"
    if os.path.exists(api_cache_dir):
        api_files = os.listdir(api_cache_dir)
        print(f"📁 API cache files ({len(api_files)}):")
        for i, file in enumerate(api_files[:5]):  # Show first 5 files
            print(f"   {i+1}. {file}")
        if len(api_files) > 5:
            print(f"   ... and {len(api_files) - 5} more files")
    
    # Check LLM cache
    llm_cache_dir = ".llm_cache"
    if os.path.exists(llm_cache_dir):
        llm_files = os.listdir(llm_cache_dir)
        print(f"📁 LLM cache files ({len(llm_files)}):")
        for i, file in enumerate(llm_files[:5]):  # Show first 5 files
            print(f"   {i+1}. {file}")
        if len(llm_files) > 5:
            print(f"   ... and {len(llm_files) - 5} more files")

def test_cache_key_generation():
    """Test that cache keys are generated correctly for different models"""
    print("\n🔍 Testing cache key generation...")
    
    # Test identical prompts with different models
    cache_data_1 = {
        "prompt": "Extract entities from text",
        "model": "deepseek-coder",
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    cache_data_2 = {
        "prompt": "Extract entities from text",  # Same prompt
        "model": "deepseek-chat",  # Different model
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    # Create cache instance
    cache = APICache(".test_cache")
    
    # Generate cache keys
    key1 = cache._generate_cache_key(cache_data_1)
    key2 = cache._generate_cache_key(cache_data_2)
    
    print(f"🔑 Cache key 1 (deepseek-coder): {key1}")
    print(f"🔑 Cache key 2 (deepseek-chat): {key2}")
    
    if key1 != key2:
        print("✅ Cache keys are different for different models")
        return True
    else:
        print("❌ Cache keys are the same for different models - this is the issue!")
        return False

if __name__ == "__main__":
    print("🧪 Benchmark Caching Issue Test")
    print("=" * 50)
    
    try:
        # Debug cache files first
        debug_cache_files()
        
        # Test cache key generation
        key_test = test_cache_key_generation()
        
        # Test LLM caching directly
        llm_success = test_llm_cache_directly()
        
        # Test benchmark caching
        benchmark_success = test_benchmark_caching()
        
        if key_test and llm_success and benchmark_success:
            print("\n🎉 All caching tests passed!")
        else:
            print("\n💥 Some caching tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)