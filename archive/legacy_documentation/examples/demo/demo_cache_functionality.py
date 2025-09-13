#!/usr/bin/env python3
"""
Demo script to show cache functionality in action
"""
import json
import os
import sys
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.fetcher import (calculate_total_studies, get_full_study,
                              search_trials)
from utils.api_manager import UnifiedAPIManager


def get_cache_stats():
    """Get cache statistics from the unified API manager"""
    api_manager = UnifiedAPIManager()
    return api_manager.get_cache_stats()

def clear_api_cache():
    """Clear all API cache using the unified API manager"""
    api_manager = UnifiedAPIManager()
    api_manager.clear_cache()

def demo_cache_functionality():
    """Demonstrate cache functionality"""
    print("🚀 Cache Functionality Demo")
    print("=" * 50)
    
    # Clear cache first
    clear_api_cache()
    print("🗑️  Cache cleared")
    
    # Show initial cache stats
    stats = get_cache_stats()
    clinical_trials_stats = stats.get("clinical_trials", {})
    print(f"📊 Initial cache stats: {clinical_trials_stats.get('cached_items', 0)} items, {clinical_trials_stats.get('total_size_bytes', 0)} bytes")
    
    print("\n🔍 Testing search_trials caching...")
    
    # First call - should be slow
    print("⏱️  First call to search_trials (no cache)...")
    start_time = time.time()
    result1 = search_trials("lung cancer", max_results=3)
    time1 = time.time() - start_time
    print(f"   Took {time1:.2f} seconds")
    print(f"   Found {len(result1.get('studies', []))} studies")
    
    # Second call with same parameters - should be fast
    print("⏱️  Second call to search_trials (cached)...")
    start_time = time.time()
    result2 = search_trials("lung cancer", max_results=3)
    time2 = time.time() - start_time
    print(f"   Took {time2:.2f} seconds")
    print(f"   Found {len(result2.get('studies', []))} studies")
    
    if time2 < time1 * 0.5:
        print("✅ Caching working - second call was much faster!")
    else:
        print("⚠️  Caching may not be working as expected")
    
    print("\n🔍 Testing calculate_total_studies caching...")
    
    # First call - should be slow
    print("⏱️  First call to calculate_total_studies (no cache)...")
    start_time = time.time()
    result1 = calculate_total_studies("lung cancer")
    time1 = time.time() - start_time
    print(f"   Took {time1:.2f} seconds")
    print(f"   Found {result1.get('total_studies', 0):,} studies")
    
    # Second call with same parameters - should be fast
    print("⏱️  Second call to calculate_total_studies (cached)...")
    start_time = time.time()
    result2 = calculate_total_studies("lung cancer")
    time2 = time.time() - start_time
    print(f"   Took {time2:.2f} seconds")
    print(f"   Found {result2.get('total_studies', 0):,} studies")
    
    if time2 < time1 * 0.5:
        print("✅ Caching working - second call was much faster!")
    else:
        print("⚠️  Caching may not be working as expected")
    
    # Show final cache stats
    stats = get_cache_stats()
    clinical_trials_stats = stats.get("clinical_trials", {})
    print(f"\n📊 Final cache stats: {clinical_trials_stats.get('cached_items', 0)} items, {clinical_trials_stats.get('total_size_bytes', 0)} bytes")
    
    print("\n✨ Cache functionality demo completed successfully!")

def main():
    """Main function"""
    try:
        demo_cache_functionality()
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)