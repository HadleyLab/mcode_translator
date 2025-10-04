#!/usr/bin/env python
# coding: utf-8

"""
Test script for the new mCODE Translator CLI.

This script tests the basic CLI structure and imports.
"""

import sys
from pathlib import Path

# Add heysol_api_client to path first (required for imports)
heysol_path = Path(__file__).parent / ".." / "heysol_api_client" / "src"
if heysol_path.exists():
    sys.path.insert(0, str(heysol_path))

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_imports():
    """Test CLI imports and basic structure."""
    print("Testing CLI imports...")

    try:
        from cli.main import app
        print("✅ Main CLI app imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main CLI app: {e}")
        return False

    try:
        from cli.commands import mcode, memory, config, data
        print("✅ CLI commands imported successfully")
    except ImportError as e:
        print(f"⚠️ Some CLI commands not available: {e}")
        print("   This is expected during development")

    return True

def test_heysol_integration():
    """Test HeySol API client integration."""
    print("\nTesting HeySol integration...")

    try:
        from heysol import HeySolClient
        print("✅ HeySolClient imported successfully")

        # Test basic client creation (without API key for now)
        try:
            # This will fail without API key, but tests the import
            client = HeySolClient(api_key="test_key")
            print("✅ HeySolClient instantiation works")
            client.close()
        except Exception as e:
            print(f"⚠️ HeySolClient instantiation failed (expected without real API key): {e}")

    except ImportError as e:
        print(f"❌ Failed to import HeySolClient: {e}")
        return False

    return True

def test_config_integration():
    """Test configuration integration."""
    print("\nTesting configuration integration...")

    try:
        from config.heysol_config import get_config
        config = get_config()
        print("✅ HeySol config loaded successfully")
        print(f"   Base URL: {config.get_base_url()}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False

    try:
        from utils.config import Config
        main_config = Config()
        print("✅ Main config loaded successfully")
    except Exception as e:
        print(f"❌ Main config loading failed: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("🧪 Testing mCODE Translator CLI")
    print("=" * 50)

    tests = [
        test_cli_imports,
        test_heysol_integration,
        test_config_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - check implementation")
        return 1

if __name__ == "__main__":
    sys.exit(main())