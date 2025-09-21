#!/usr/bin/env python3
"""
Simple OAuth2 flow test - authenticate then test basic operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol import HeySolClient

def test_oauth2_flow():
    """Test OAuth2 authentication and basic operations."""
    print("Testing OAuth2 authentication and basic operations...")

    try:
        # Create client with OAuth2
        client = HeySolClient(use_oauth2=True)
        print("✅ OAuth2 client created successfully")

        # Check initial state
        has_tokens = client._has_valid_oauth2_tokens()
        has_session = client.session_id is not None
        print(f"✅ Initial state - Tokens: {has_tokens}, Session: {has_session}")

        if not has_tokens:
            print("\n🔐 Starting OAuth2 authentication...")
            print("This will open a browser for Google authentication.")
            print("Please complete the authentication in the browser.")

            # Perform OAuth2 authentication
            success = client.authorize_oauth2_interactive()
            print(f"✅ OAuth2 authentication result: {success}")

            if not success:
                print("❌ OAuth2 authentication failed")
                return False

        # Check state after authentication
        has_tokens = client._has_valid_oauth2_tokens()
        has_session = client.session_id is not None
        print(f"✅ After authentication - Tokens: {has_tokens}, Session: {has_session}")

        if not has_tokens:
            print("❌ No valid tokens after authentication")
            return False

        # Test basic operations
        print("\n🧪 Testing basic operations...")

        # Get user profile
        try:
            profile = client.get_user_profile()
            print(f"✅ User profile retrieved: {profile.get('name', 'Unknown')}")
        except Exception as e:
            print(f"⚠️  User profile retrieval failed: {e}")

        # Get spaces
        try:
            spaces = client.get_spaces()
            print(f"✅ Retrieved {len(spaces)} spaces")
        except Exception as e:
            print(f"⚠️  Spaces retrieval failed: {e}")

        print("✅ OAuth2 flow test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ OAuth2 flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_oauth2_flow()
    sys.exit(0 if success else 1)