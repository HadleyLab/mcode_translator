#!/usr/bin/env python3
"""
Test script to verify that token usage is properly tracked even with caching.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.strict_llm_base import StrictLLMBase
from utils.token_tracker import global_token_tracker


class TestLLM(StrictLLMBase):
    """Concrete implementation of StrictLLMBase for testing."""
    
    def process_request(self, *args, **kwargs):
        """Implementation of abstract method."""
        pass


def test_token_usage_with_caching():
    """Test that token usage is properly tracked even with caching."""
    # Reset the global token tracker
    global_token_tracker.reset()
    
    # Create an instance of TestLLM
    llm = TestLLM(response_format=None)  # Don't use JSON response format
    
    # Define test messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # Define cache key data
    cache_key_data = {
        "messages": messages,
        "model": llm.model_name,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens
    }
    
    # First call - should make an actual API call
    print("Making first API call...")
    response1, metrics1 = llm._call_llm_api(messages, cache_key_data)
    print(f"First call - Duration: {metrics1.duration:.2f}s, Tokens: {metrics1.total_tokens}")
    
    # Get the total token usage from the global tracker
    total_usage1 = global_token_tracker.get_total_usage()
    print(f"Total token usage after first call: {total_usage1.total_tokens}")
    
    # Second call - should use cached response
    print("\nMaking second API call (should use cache)...")
    response2, metrics2 = llm._call_llm_api(messages, cache_key_data)
    print(f"Second call - Duration: {metrics2.duration:.2f}s, Tokens: {metrics2.total_tokens}")
    
    # Get the total token usage from the global tracker
    total_usage2 = global_token_tracker.get_total_usage()
    print(f"Total token usage after second call: {total_usage2.total_tokens}")
    
    # Verify that the responses are the same
    assert response1 == response2, "Cached response should be the same as the original"
    
    # Verify that token usage is properly tracked for both calls
    assert metrics1.total_tokens > 0, "First call should have token usage"
    assert metrics2.total_tokens > 0, "Second call should also have token usage (from cache)"
    assert metrics1.total_tokens == metrics2.total_tokens, "Both calls should have the same token usage"
    
    # Verify that the global token tracker has the correct total
    expected_total = metrics1.total_tokens + metrics2.total_tokens
    assert total_usage2.total_tokens == expected_total, f"Global tracker should have {expected_total} tokens, but has {total_usage2.total_tokens}"
    
    print("\n✅ All tests passed! Token usage is properly tracked even with caching.")


if __name__ == "__main__":
    test_token_usage_with_caching()