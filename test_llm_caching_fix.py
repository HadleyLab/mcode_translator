#!/usr/bin/env python3
"""
Test script to verify the LLM caching fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.strict_llm_base import LLMCallMetrics

def test_llm_metrics_serialization():
    """Test that LLMCallMetrics can be serialized and deserialized correctly"""
    print("🧪 Testing LLMCallMetrics serialization...")
    
    # Create a metrics object
    metrics = LLMCallMetrics(
        duration=1.23,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
        success=True,
        error_type=None
    )
    
    # Serialize to dict
    metrics_dict = metrics.to_dict()
    print(f"✅ Serialized metrics: {metrics_dict}")
    
    # Deserialize back to object
    deserialized_metrics = LLMCallMetrics(
        duration=metrics_dict['duration'],
        prompt_tokens=metrics_dict['prompt_tokens'],
        completion_tokens=metrics_dict['completion_tokens'],
        total_tokens=metrics_dict['total_tokens'],
        success=metrics_dict['success'],
        error_type=metrics_dict['error_type']
    )
    
    # Check that all values match
    assert metrics.duration == deserialized_metrics.duration
    assert metrics.prompt_tokens == deserialized_metrics.prompt_tokens
    assert metrics.completion_tokens == deserialized_metrics.completion_tokens
    assert metrics.total_tokens == deserialized_metrics.total_tokens
    assert metrics.success == deserialized_metrics.success
    assert metrics.error_type == deserialized_metrics.error_type
    
    print("✅ Serialization/deserialization test passed")
    return True

if __name__ == "__main__":
    try:
        success = test_llm_metrics_serialization()
        if success:
            print("\n🎉 LLM caching fix verification passed!")
        else:
            print("\n💥 LLM caching fix verification failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 LLM caching fix verification failed with error: {e}")
        sys.exit(1)