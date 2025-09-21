#!/usr/bin/env python3
"""
Test non-interactive OAuth2 authentication using client credentials flow.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol import HeySolClient
from heysol.oauth2 import OAuth2ClientCredentialsAuthenticator

def test_non_interactive_oauth2():
    """Test non-interactive OAuth2 authentication."""
    print("Testing non-interactive OAuth2 authentication...")

    try:
        # Create client with OAuth2
        client = HeySolClient(use_oauth2=True)
        print("✅ OAuth2 client created successfully")

        # Check if we have valid tokens
        has_tokens = client._has_valid_oauth2_tokens()
        has_session = client.session_id is not None
        print(f"✅ Initial state - Tokens: {has_tokens}, Session: {has_session}")

        if has_tokens:
            print("✅ Non-interactive OAuth2 authentication successful!")
            return True
        else:
            print("❌ Non-interactive OAuth2 authentication failed - no valid tokens")
            return False

    except Exception as e:
        print(f"❌ Non-interactive OAuth2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ingest_and_delete():
    """Test ingesting and deleting a message."""
    print("\nTesting ingest and delete operations...")

    try:
        # Create client
        client = HeySolClient(use_oauth2=True)

        # Check authentication
        if not client._has_valid_oauth2_tokens():
            print("❌ Cannot test operations - no valid OAuth2 tokens")
            return False

        # Initialize MCP session if needed
        if not client.session_id:
            try:
                client.initialize_mcp_session()
                print("✅ MCP session initialized")
            except Exception as e:
                print(f"❌ Failed to initialize MCP session: {e}")
                return False

        # Create test space
        space_id = client.get_or_create_space("test_oauth2", "Test space for OAuth2")
        print(f"✅ Created/found test space: {space_id}")

        # Ingest test message
        import time
        test_message = f"OAuth2 test message - {os.environ.get('USER', 'user')} - {time.time()}"
        result = client.ingest(test_message, space_id=space_id, tags=["oauth2", "test"])
        log_id = result.get("log_id") or result.get("id")

        if log_id:
            print(f"✅ Ingested test message with ID: {log_id}")
        else:
            print("❌ Failed to get log ID from ingestion")
            return False

        # Delete test message
        delete_result = client.delete_log_entry(log_id)
        if delete_result.get("deleted"):
            print(f"✅ Deleted test message: {log_id}")
        else:
            print("❌ Failed to delete test message")
            return False

        # Clean up test space
        try:
            client.delete_space(space_id)
            print(f"✅ Cleaned up test space: {space_id}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clean up test space: {e}")

        print("✅ Ingest and delete test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Ingest and delete test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test non-interactive authentication
    auth_success = test_non_interactive_oauth2()

    if auth_success:
        # Test operations
        ops_success = test_ingest_and_delete()

        if ops_success:
            print("\n🎉 All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Operations test failed")
            sys.exit(1)
    else:
        print("\n❌ Authentication test failed")
        sys.exit(1)