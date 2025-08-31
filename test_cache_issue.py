#!/usr/bin/env python3
"""
Test script to reproduce and verify cache isolation issue in StrictLLMBase
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pipeline.strict_llm_base import StrictLLMBase
from unittest.mock import Mock, patch
import json

class MockStrictLLMBase(StrictLLMBase):
    """Mock implementation for testing cache functionality"""
    
    def process_request(self, *args, **kwargs):
        return {"test": "response"}

def test_cache_isolation():
    """Test that different model instances don't share cache hits"""
    print("🧪 Testing cache isolation between different model instances...")
    
    # Mock the OpenAI client to avoid actual API calls
    with patch('pipeline.strict_llm_base.openai.OpenAI') as mock_openai:
        # Create mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content='{"test": "response"}'))]
        )
        mock_openai.return_value = mock_client
        
        # Create first model instance with specific configuration
        print("🔧 Creating first model instance (deepseek-coder, temp=0.1)...")
        model1 = MockStrictLLMBase(
            model_name="deepseek-coder",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Create second model instance with different configuration
        print("🔧 Creating second model instance (deepseek-chat, temp=0.5)...")
        model2 = MockStrictLLMBase(
            model_name="deepseek-chat", 
            temperature=0.5,
            max_tokens=2000
        )
        
        # Test messages
        messages = [{"role": "user", "content": "Test message"}]
        cache_key_data = {"test": "data"}
        
        print("🚀 Making first call with model1...")
        response1, metrics1 = model1._call_llm_api(messages, cache_key_data)
        
        print("🚀 Making second call with model2...")
        response2, metrics2 = model2._call_llm_api(messages, cache_key_data)
        
        # Check if cache was hit (should be cache miss for different instances)
        print(f"📊 Model1 call count: {mock_client.chat.completions.create.call_count}")
        
        # The issue: if cache is shared, the second call would be a cache hit
        # But it should be a cache miss since instances have different configs
        if mock_client.chat.completions.create.call_count == 1:
            print("❌ CACHE ISSUE DETECTED: Second call was a cache hit!")
            print("   Different model instances are sharing cache entries")
            return False
        elif mock_client.chat.completions.create.call_count == 2:
            print("✅ CACHE ISOLATION WORKING: Both calls made actual API requests")
            return True
        else:
            print(f"❓ UNEXPECTED: {mock_client.chat.completions.create.call_count} API calls")
            return False

if __name__ == "__main__":
    success = test_cache_isolation()
    if success:
        print("\n🎉 Cache isolation test passed!")
    else:
        print("\n💥 Cache isolation test failed - issue detected!")
        sys.exit(1)