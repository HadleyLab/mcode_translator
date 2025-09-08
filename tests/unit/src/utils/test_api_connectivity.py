#!/usr/bin/env python3
"""
Test script to verify DeepSeek API connectivity for the mCODE translator project.
"""

import os
import sys
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.config import Config

print("🧪 Testing DeepSeek API Connectivity")
print("=" * 50)

# Check if API key is available from config file
config = Config()
try:
    api_key = config.get_api_key()
    base_url = config.get_base_url()
    model_name = config.get_model_name()
    max_tokens = config.get_max_tokens()
    temperature = config.get_temperature()
except Exception as e:
    print(f"❌ Configuration error: {e}")
    print("Please ensure config.json is properly configured")
    sys.exit(1)

print(f"API key present: {bool(api_key)}")
print(f"API key length: {len(api_key) if api_key else 0}")

if not api_key:
    print("❌ No API key found. Please configure API key in config.json")
    sys.exit(1)

# Test basic OpenAI client initialization
try:
    import openai
    
    # Configure OpenAI client for DeepSeek using config values
    client = openai.OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    print("✅ OpenAI client configured successfully for DeepSeek")
    
    # Test a simple API call
    print("Testing API connectivity with a simple completion...")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello, are you working?"}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    if response.choices and response.choices[0].message.content:
        print("✅ API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Token usage: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
    else:
        print("❌ API call returned empty response")
        
except Exception as e:
    print(f"❌ API test failed: {e}")
    print("This could be due to:")
    print("  - Invalid API key")
    print("  - Network connectivity issues")
    print("  - API rate limits")
    print("  - DeepSeek API service issues")

print("\n" + "=" * 50)
print("API connectivity test completed")