#!/usr/bin/env python3
"""
Test script to verify the cache fix for the ttl parameter issue
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.cache_decorator import APICache
import json

def test_cache_ttl_fix():
    """Test that the set_by_key method accepts a ttl parameter"""
    print("🧪 Testing cache TTL fix...")
    
    # Create cache instance
    cache = APICache(".test_cache")
    
    # Test data
    test_data = {"test": "data"}
    cache_key_data = {"key": "test"}
    
    try:
        # This should work now with our fix
        cache.set_by_key(test_data, cache_key_data, ttl=3600)
        print("✅ set_by_key with ttl parameter works correctly")
        
        # Verify we can retrieve the data
        retrieved_data = cache.get_by_key(cache_key_data)
        if retrieved_data == test_data:
            print("✅ Data correctly cached and retrieved")
            return True
        else:
            print("❌ Data mismatch in cache")
            return False
            
    except Exception as e:
        print(f"❌ Error testing cache TTL fix: {str(e)}")
        return False
    finally:
        # Clean up test cache
        cache.clear()

if __name__ == "__main__":
    success = test_cache_ttl_fix()
    if success:
        print("\n🎉 Cache TTL fix test passed!")
    else:
        print("\n💥 Cache TTL fix test failed!")
        sys.exit(1)