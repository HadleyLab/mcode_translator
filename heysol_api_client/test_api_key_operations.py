#!/usr/bin/env python3
"""
Test HeySol API operations using API key instead of OAuth2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol import HeySolClient

def test_api_key_operations():
    """Test HeySol API operations using API key."""
    print("Testing HeySol API operations with API key...")

    try:
        # Create client with API key (not OAuth2) - avoid OAuth2 detection
        import os
        # Temporarily clear OAuth2 environment variables
        oauth2_client_id = os.environ.pop('COREAI_OAUTH2_CLIENT_ID', None)
        oauth2_client_secret = os.environ.pop('COREAI_OAUTH2_CLIENT_SECRET', None)

        try:
            client = HeySolClient(api_key="rc_pat_wscfrnty2k1k3w6ammjizbsht2rzq8ywnpq1qawr")
        finally:
            # Restore OAuth2 environment variables
            if oauth2_client_id:
                os.environ['COREAI_OAUTH2_CLIENT_ID'] = oauth2_client_id
            if oauth2_client_secret:
                os.environ['COREAI_OAUTH2_CLIENT_SECRET'] = oauth2_client_secret
        print("✅ API key client created successfully")

        # Initialize MCP session
        if not client.session_id:
            client.initialize_mcp_session()
            print("✅ MCP session initialized")

        # Get user profile
        try:
            profile = client.get_user_profile()
            print(f"✅ User profile retrieved: {profile.get('name', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  User profile retrieval failed: {e}")

        # Create test space
        space_id = client.get_or_create_space("test_api_key", "Test space for API key operations")
        print(f"✅ Created/found test space: {space_id}")

        # Ingest test message
        import time
        test_message = f"API key test message - {os.environ.get('USER', 'user')} - {time.time()}"
        result = client.ingest(test_message, space_id=space_id, tags=["api_key", "test"])
        log_id = result.get("log_id") or result.get("id")

        if log_id:
            print(f"✅ Ingested test message with ID: {log_id}")
        else:
            print("❌ Failed to get log ID from ingestion")
            return False

        # Get the message back
        try:
            logs = client.get_ingestion_logs(space_id=space_id, limit=5)
            print(f"✅ Retrieved {len(logs)} recent logs")
        except Exception as e:
            print(f"⚠️  Log retrieval failed: {e}")

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

        print("✅ API key operations test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ API key operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_api_key_operations()
    sys.exit(0 if success else 1)