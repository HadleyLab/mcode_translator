#!/usr/bin/env python3
"""
Demo script to show LLM caching functionality
"""
import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.llm_base import LlmBase
from utils.api_manager import UnifiedAPIManager

def get_cache_stats():
    """Get cache statistics from the unified API manager"""
    api_manager = UnifiedAPIManager()
    return api_manager.get_cache_stats()

class DemoLLM(LlmBase):
    """Demo implementation of LlmBase for showing caching"""
    
    def process_request(self, *args, **kwargs):
        return {"demo": "response"}

def demo_llm_caching():
    """Demonstrate LLM caching functionality"""
    print("🚀 LLM Caching Demo")
    print("=" * 50)
    
    # Clear LLM cache first
    api_manager = UnifiedAPIManager()
    llm_cache = api_manager.get_cache("llm")
    llm_cache.clear()
    print("🗑️  Cleared LLM cache")
    
    # Show initial cache stats
    stats = get_cache_stats()
    llm_stats = stats.get("llm", {})
    print(f"📊 Initial cache stats: {llm_stats.get('cached_items', 0)} items, {llm_stats.get('total_size_bytes', 0)} bytes")
    
    # Create demo LLM instance
    llm = DemoLLM(
        model_name="deepseek-chat",
        temperature=0.1,
        max_tokens=100
    )
    
    print(f"\n🔧 Created LLM instance with model: {llm.model_name}")
    
    # Create test messages and cache key data
    messages = [{"role": "user", "content": "Extract entities from this clinical text: Patient has metastatic breast cancer."}]
    cache_key_data = {
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "messages": messages
    }
    
    # First call - simulate slow LLM call
    print("\n⏱️  First LLM call (no cache)...")
    start_time = time.time()
    
    # Simulate storing a result in cache (since we're not making real API calls)
    test_result = {
        'response_content': '{"entities": [{"text": "metastatic breast cancer", "type": "DIAGNOSIS", "confidence": 0.95}]}',
        'metrics': {
            'duration': 2.5,
            'prompt_tokens': 50,
            'completion_tokens': 30,
            'total_tokens': 80,
            'success': True,
            'error_type': None
        }
    }
    # Store in cache
    llm_cache.set_by_key(test_result, cache_key_data)
    time1 = time.time() - start_time
    print(f"   Took {time1:.2f} seconds (simulated)")
    print(f"   Stored result in cache")
    
    # Second call - should be fast (cached)
    print("⏱️  Second LLM call (cached)...")
    start_time = time.time()
    # Try to retrieve from cache
    cached_result = llm_cache.get_by_key(cache_key_data)
    time2 = time.time() - start_time
    print(f"   Took {time2:.4f} seconds (cached)")
    
    
    
    if cached_result is not None:
        print("✅ Cache hit - retrieved result from cache")
        print(f"   Entities found: {len(json.loads(cached_result['response_content']).get('entities', []))}")
    else:
        print("❌ Cache miss - no result found")
        return False
    
    # Show cache stats
    stats = get_cache_stats()
    llm_stats = stats.get("llm", {})
    print(f"\n📊 Final cache stats: {llm_stats.get('cached_items', 0)} items, {llm_stats.get('total_size_bytes', 0)} bytes")
    
    print("\n✨ LLM caching demo completed successfully!")
    return True

def main():
    """Main function"""
    try:
        demo_llm_caching()
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