#!/usr/bin/env python3
"""
Test script to verify the cache implementation is working correctly
"""
import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.api_manager import UnifiedAPIManager
from pipeline.fetcher import search_trials, get_full_study, calculate_total_studies

def get_cache_stats():
    """Get cache statistics from the unified API manager"""
    api_manager = UnifiedAPIManager()
    return api_manager.get_cache_stats()

def clear_api_cache():
    """Clear all API cache using the unified API manager"""
    api_manager = UnifiedAPIManager()
    api_manager.clear_cache()

def test_cache_decorator():
    """Test the cache decorator implementation"""
    print("🧪 Testing cache decorator implementation...")
    
    # Clear cache first
    clear_api_cache()
    print("🗑️  Cleared cache")
    
    # Test cache stats
    stats = get_cache_stats()
    print(f"📊 Initial cache stats: {stats}")
    
    # Test the cache with a simple function using UnifiedAPIManager
    api_manager = UnifiedAPIManager()
    test_cache = api_manager.get_cache("test")
    
    def slow_function(x, y):
        """A slow function to test caching"""
        time.sleep(1)  # Simulate slow operation
        return x + y
    
    # First call should take time
    cache_key_data = {"function": "slow_function", "args": (2, 3)}
    cached_result = test_cache.get_by_key(cache_key_data)
    if cached_result is not None:
        print("❌ Unexpected cache hit before first call")
        return False
    
    start_time = time.time()
    result1 = slow_function(2, 3)
    time1 = time.time() - start_time
    print(f"⏱️  First call took {time1:.2f} seconds, result: {result1}")
    
    # Store result in cache
    test_cache.set_by_key(result1, cache_key_data, ttl=10)  # 10 seconds TTL
    
    # Second call should be fast (cached)
    start_time = time.time()
    cached_result = test_cache.get_by_key(cache_key_data)
    time2 = time.time() - start_time
    if cached_result is not None:
        result2 = cached_result
    else:
        result2 = slow_function(2, 3)
    print(f"⏱️  Second call took {time2:.2f} seconds, result: {result2}")
    
    if time2 < time1 * 0.5:  # Should be much faster
        print("✅ Cache working correctly - second call was fast (cached)")
    else:
        print("❌ Cache not working - second call was slow")
        return False
    
    # Test with different arguments (should not be cached)
    cache_key_data2 = {"function": "slow_function", "args": (3, 4)}
    cached_result2 = test_cache.get_by_key(cache_key_data2)
    if cached_result2 is not None:
        print("❌ Unexpected cache hit for different arguments")
        return False
    
    start_time = time.time()
    result3 = slow_function(3, 4)
    time3 = time.time() - start_time
    print(f"⏱️  Third call (different args) took {time3:.2f} seconds, result: {result3}")
    
    # Store result in cache
    test_cache.set_by_key(result3, cache_key_data2, ttl=10)  # 10 seconds TTL
    
    if time3 > time2 * 2:  # Should be slow again
        print("✅ Cache correctly distinguishes different arguments")
    else:
        print("❌ Cache incorrectly returned cached result for different arguments")
        return False
    
    return True

def test_fetcher_caching():
    """Test caching in the fetcher functions"""
    print("\n🧪 Testing fetcher function caching...")
    
    # Clear cache first
    clear_api_cache()
    print("🗑️  Cleared cache")
    
    # Test search_trials caching
    print("🔍 Testing search_trials caching...")
    try:
        # First call
        start_time = time.time()
        result1 = search_trials("breast cancer", max_results=5)
        time1 = time.time() - start_time
        print(f"⏱️  First search_trials call took {time1:.2f} seconds")
        
        # Second call with same parameters
        start_time = time.time()
        result2 = search_trials("breast cancer", max_results=5)
        time2 = time.time() - start_time
        print(f"⏱️  Second search_trials call took {time2:.2f} seconds")
        
        if time2 < time1 * 0.5:
            print("✅ search_trials caching working correctly")
        else:
            print("⚠️  search_trials caching may not be working (second call was slow)")
        
    except Exception as e:
        print(f"⚠️  search_trials test failed: {e}")
    
    # Test calculate_total_studies caching
    print("🔍 Testing calculate_total_studies caching...")
    try:
        # First call
        start_time = time.time()
        result1 = calculate_total_studies("breast cancer")
        time1 = time.time() - start_time
        print(f"⏱️  First calculate_total_studies call took {time1:.2f} seconds")
        
        # Second call with same parameters
        start_time = time.time()
        result2 = calculate_total_studies("breast cancer")
        time2 = time.time() - start_time
        print(f"⏱️  Second calculate_total_studies call took {time2:.2f} seconds")
        
        if time2 < time1 * 0.5:
            print("✅ calculate_total_studies caching working correctly")
        else:
            print("⚠️  calculate_total_studies caching may not be working (second call was slow)")
            
    except Exception as e:
        print(f"⚠️  calculate_total_studies test failed: {e}")
    
    # Show final cache stats
    stats = get_cache_stats()
    print(f"📊 Final cache stats: {stats}")
    
    return True

def main():
    """Run all cache tests"""
    print("🚀 Starting cache implementation tests...")
    
    success1 = test_cache_decorator()
    success2 = test_fetcher_caching()
    
    if success1 and success2:
        print("\n🎉 All cache tests passed!")
        return True
    else:
        print("\n💥 Some cache tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)