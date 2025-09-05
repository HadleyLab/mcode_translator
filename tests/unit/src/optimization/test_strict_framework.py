#!/usr/bin/env python3
"""
Test script to verify the strict framework works with valid configurations
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.optimization.prompt_optimization_framework import PromptOptimizationFramework, APIConfig, PromptVariant, PromptType

def test_valid_config_validation():
    """Test that valid API configurations pass validation"""
    print("🧪 Testing valid API configuration validation...")
    
    framework = PromptOptimizationFramework()
    
    # Test valid API config
    valid_config = APIConfig(
        name="test_valid",
        base_url="https://api.example.com/v1",
        api_key="sk-valid-test-key-1234567890abcdef",
        model="test-model"
    )
    
    try:
        framework.add_api_config(valid_config)
        print("✅ Valid API config passed validation")
        return True
    except Exception as e:
        print(f"❌ Valid API config failed validation: {e}")
        return False

def test_invalid_config_validation():
    """Test that invalid API configurations fail validation"""
    print("\n🧪 Testing invalid API configuration validation...")
    
    framework = PromptOptimizationFramework()
    
    # Test various invalid patterns
    invalid_configs = [
        ("empty_key", "", "https://api.example.com/v1"),
        ("placeholder_key", "your-api-key-here", "https://api.example.com/v1"),
        ("fake_key", "fake-api-key-123", "https://api.example.com/v1"),
        ("test_key", "test-api-key-456", "https://api.example.com/v1"),
        ("invalid_url", "sk-valid-key", "invalid-url")
    ]
    
    all_failed = True
    
    for name, api_key, base_url in invalid_configs:
        try:
            config = APIConfig(name=name, base_url=base_url, api_key=api_key)
            framework.add_api_config(config)
            print(f"❌ Invalid config '{name}' should have failed but didn't")
            all_failed = False
        except ValueError as e:
            print(f"✅ Invalid config '{name}' correctly failed: {str(e)[:80]}...")
        except Exception as e:
            print(f"❌ Invalid config '{name}' failed with unexpected error: {e}")
            all_failed = False
    
    return all_failed

def test_config_loading_from_file():
    """Test loading valid configurations from file"""
    print("\n🧪 Testing configuration loading from valid file...")
    
    framework = PromptOptimizationFramework()
    
    # Load the valid config file we created
    config_file = "examples/config/valid_api_configs.json"
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        for config_dict in config_data['api_configurations']:
            config = APIConfig(
                name=config_dict['name'],
                base_url=config_dict['base_url'],
                api_key=config_dict['api_key'],
                model=config_dict['model'],
                temperature=config_dict.get('temperature', 0.2),
                max_tokens=config_dict.get('max_tokens', 4000),
                timeout=config_dict.get('timeout', 30)
            )
            framework.add_api_config(config)
        
        print("✅ Valid config file loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load valid config file: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 STRICT FRAMEWORK VALIDATION TESTS")
    print("=" * 60)
    
    # Run tests
    test1 = test_valid_config_validation()
    test2 = test_invalid_config_validation() 
    test3 = test_config_loading_from_file()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Valid config validation: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Invalid config validation: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Config file loading: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    all_passed = test1 and test2 and test3
    print(f"\nOverall result: {'🎉 ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())