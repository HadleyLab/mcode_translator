#!/usr/bin/env python3
"""
Test script to reproduce and verify cache isolation issue in StrictLLMBase
using REAL LLM calls with DeepSeek models
"""

import sys
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.nlp_engine import StrictNlpExtractor

def test_cache_isolation_real():
    """Test that different model instances don't share cache hits using real LLM calls"""
    print("🧪 Testing cache isolation between different model instances with REAL LLM calls...")
    
    # Check if API key is available
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        print("   Please set DEEPSEEK_API_KEY in your .env file")
        return False
    
    print("✅ DEEPSEEK_API_KEY found")
    
    try:
        # Create first NLP extractor instance with specific configuration
        print("🔧 Creating first NLP extractor instance (deepseek-coder, temp=0.1)...")
        extractor1 = StrictNlpExtractor(
            model_name="deepseek-coder",
            temperature=0.1,
            max_tokens=500  # Increased for complete JSON responses
        )
        
        # Create second NLP extractor instance with different configuration
        print("🔧 Creating second NLP extractor instance (deepseek-chat, temp=0.5)...")
        extractor2 = StrictNlpExtractor(
            model_name="deepseek-chat",
            temperature=0.5,
            max_tokens=500  # Increased for complete JSON responses
        )
        
        # Test clinical text
        clinical_text = "Patient has metastatic breast cancer with liver metastases."
        
        print("🚀 Making first call with extractor1 (deepseek-coder)...")
        start_time = time.time()
        result1 = extractor1.extract_entities(clinical_text)
        time1 = time.time() - start_time
        print(f"   Entities extracted: {len(result1.entities)}")
        print(f"   Time: {time1:.2f}s")
        
        print("🚀 Making second call with extractor2 (deepseek-chat)...")
        start_time = time.time()
        result2 = extractor2.extract_entities(clinical_text)
        time2 = time.time() - start_time
        print(f"   Entities extracted: {len(result2.entities)}")
        print(f"   Time: {time2:.2f}s")
        
        # Check if cache was hit (should be cache miss for different instances)
        # If cache is shared, the second call would be much faster (cache hit)
        # But it should be similar time since instances have different configs
        
        print(f"📊 Time comparison:")
        print(f"   Extractor1 (deepseek-coder): {time1:.2f}s")
        print(f"   Extractor2 (deepseek-chat): {time2:.2f}s")
        
        # If cache is working correctly, both calls should take similar time
        # If cache is broken and shared, second call would be much faster
        time_ratio = time2 / time1 if time1 > 0 else 0
        
        if time_ratio < 0.3:  # Second call is 3x faster (likely cache hit)
            print("❌ CACHE ISSUE DETECTED: Second call was likely a cache hit!")
            print("   Different model instances are sharing cache entries")
            print("   This indicates the cache key doesn't include model configuration")
            return False
        elif 0.7 <= time_ratio <= 1.3:  # Similar timing (cache miss for both)
            print("✅ CACHE ISOLATION WORKING: Both calls made actual API requests")
            print("   Different model configurations are properly isolated")
            return True
        else:
            print(f"❓ UNEXPECTED: Time ratio {time_ratio:.2f}")
            print("   This could be due to network variability or other factors")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_same_model_different_params():
    """Test cache isolation with same model but different parameters"""
    print("\n🧪 Testing cache isolation with same model but different parameters...")
    
    try:
        # Create instances with same model but different temperatures
        extractor_low_temp = StrictNlpExtractor(
            model_name="deepseek-chat",
            temperature=0.1,
            max_tokens=500  # Increased for complete JSON responses
        )
        
        extractor_high_temp = StrictNlpExtractor(
            model_name="deepseek-chat",
            temperature=0.9,
            max_tokens=500  # Increased for complete JSON responses
        )
        
        clinical_text = "Patient has metastatic breast cancer."
        
        print("🚀 Making call with low temperature (0.1)...")
        start_time = time.time()
        result1 = extractor_low_temp.extract_entities(clinical_text)
        time1 = time.time() - start_time
        print(f"   Entities: {len(result1.entities)}, Time: {time1:.2f}s")
        
        print("🚀 Making call with high temperature (0.9)...")
        start_time = time.time()
        result2 = extractor_high_temp.extract_entities(clinical_text)
        time2 = time.time() - start_time
        print(f"   Entities: {len(result2.entities)}, Time: {time2:.2f}s")
        
        time_ratio = time2 / time1 if time1 > 0 else 0
        
        if time_ratio < 0.3:
            print("❌ CACHE ISSUE: Temperature changes not isolated!")
            return False
        else:
            print("✅ Temperature isolation working")
            return True
            
    except Exception as e:
        print(f"❌ Error in temperature test: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing cache isolation with real DeepSeek API calls")
    print("   This will make actual API calls and may incur costs")
    print("   Press Ctrl+C to cancel within 5 seconds...")
    
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n❌ Test cancelled")
        sys.exit(1)
    
    try:
        success1 = test_cache_isolation_real()
        success2 = test_same_model_different_params()
        
        if success1 and success2:
            print("\n🎉 All cache isolation tests passed!")
        else:
            print("\n💥 Cache isolation tests failed - issue detected!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)