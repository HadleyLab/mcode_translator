#!/usr/bin/env python3
"""
Simple OAuth2 test script to verify the client initialization works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol import HeySolClient

def test_oauth2_initialization():
    """Test OAuth2 client initialization."""
    print("Testing OAuth2 client initialization...")

    try:
        # Create client with OAuth2
        client = HeySolClient(use_oauth2=True)
        print("✅ OAuth2 client created successfully")

        # Check if we have valid tokens
        has_tokens = client._has_valid_oauth2_tokens()
        print(f"✅ Has valid OAuth2 tokens: {has_tokens}")

        # Check if MCP session is initialized
        has_session = client.session_id is not None
        print(f"✅ MCP session initialized: {has_session}")

        print("✅ OAuth2 client initialization test passed!")
        return True

    except Exception as e:
        print(f"❌ OAuth2 client initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = test_oauth2_initialization()
    sys.exit(0 if success else 1)