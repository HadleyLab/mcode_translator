#!/usr/bin/env python3
"""
Test interactive OAuth2 authentication.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol import HeySolClient

def test_interactive_oauth2():
    """Test interactive OAuth2 authentication."""
    print("Testing interactive OAuth2 authentication...")

    try:
        # Create client with OAuth2
        client = HeySolClient(use_oauth2=True)
        print("✅ OAuth2 client created successfully")

        # Check initial state
        has_tokens = client._has_valid_oauth2_tokens()
        has_session = client.session_id is not None
        print(f"✅ Initial state - Tokens: {has_tokens}, Session: {has_session}")

        # Perform interactive OAuth2 authentication
        print("\n🔐 Starting interactive OAuth2 authentication...")
        print("This will open a browser window for Google OAuth2 authentication.")
        print("Please complete the authentication in the browser.")

        success = client.authorize_oauth2_interactive()
        print(f"✅ OAuth2 authentication result: {success}")

        if success:
            # Check state after authentication
            has_tokens = client._has_valid_oauth2_tokens()
            has_session = client.session_id is not None
            print(f"✅ After authentication - Tokens: {has_tokens}, Session: {has_session}")

            if has_tokens and has_session:
                print("✅ Interactive OAuth2 authentication test passed!")
                return True
            else:
                print("❌ OAuth2 authentication completed but missing tokens or session")
                return False
        else:
            print("❌ OAuth2 authentication failed")
            return False

    except Exception as e:
        print(f"❌ Interactive OAuth2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_interactive_oauth2()
    sys.exit(0 if success else 1)