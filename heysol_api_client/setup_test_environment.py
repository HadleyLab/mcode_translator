"""
Setup test environment for HeySol API client testing.

This script helps set up the environment variables and configuration
needed for testing the HeySol API client.
"""

import os
from pathlib import Path


def setup_environment():
    """Setup environment variables for testing."""
    print("Setting up test environment for HeySol API client...")

    # Check for existing API key
    api_key = os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY")

    if not api_key:
        print("\n⚠️  No API key found in environment variables.")
        api_key = input("Please enter your HeySol API key: ").strip()

        if api_key:
            # Determine which variable name to use
            use_coreai = input("Use COREAI_API_KEY (y) or CORE_MEMORY_API_KEY (n)? [y]: ").strip().lower()
            var_name = "COREAI_API_KEY" if use_coreai != "n" else "CORE_MEMORY_API_KEY"

            print(f"Setting {var_name}={api_key[:10]}...")
            os.environ[var_name] = api_key
        else:
            print("❌ No API key provided. Cannot continue with live testing.")
            return False

    # Set other environment variables
    base_url = os.getenv("COREAI_BASE_URL", "https://core.heysol.ai/api/v1/mcp")
    source = os.getenv("COREAI_SOURCE", "heysol-python-client")

    print(f"Using base URL: {base_url}")
    print(f"Using source: {source}")

    # Create .env file for future use
    env_content = f"""# HeySol API Configuration
COREAI_API_KEY={api_key}
COREAI_BASE_URL={base_url}
COREAI_SOURCE={source}
COREAI_LOG_LEVEL=DEBUG
COREAI_LOG_TO_FILE=true
COREAI_LOG_FILE_PATH=heysol_test.log
"""

    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"✅ Environment setup complete. Configuration saved to {env_file}")
    return True


def print_test_instructions():
    """Print instructions for running tests."""
    print("\n" + "="*60)
    print("TESTING INSTRUCTIONS")
    print("="*60)
    print("1. Basic Fix Verification:")
    print("   python tests/test_basic_fixes.py")
    print()
    print("2. Comprehensive Live API Testing:")
    print("   python tests/test_live_api_debug.py")
    print()
    print("3. Individual Test Files:")
    print("   python -m pytest tests/test_authentication.py -v")
    print("   python -m pytest tests/test_endpoints_valid.py -v")
    print("   python -m pytest tests/test_endpoints_invalid.py -v")
    print()
    print("4. All Tests:")
    print("   python -m pytest tests/ -v")
    print()
    print("5. Debug Logs:")
    print("   - Check heysol_debug.log for detailed client logs")
    print("   - Check live_api_test.log for test execution logs")
    print("="*60)


if __name__ == "__main__":
    if setup_environment():
        print_test_instructions()
    else:
        print("❌ Environment setup failed.")