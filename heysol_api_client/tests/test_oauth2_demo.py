"""
OAuth2 Demo for HeySol API client.

This script demonstrates the OAuth2 setup and shows how to use interactive OAuth2
authentication with the HeySol API client.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.oauth2_interactive import InteractiveOAuth2Authenticator

def main():
    """Main function to demonstrate OAuth2 setup."""
    print("\n" + "="*70)
    print("HEYSOL OAUTH2 AUTHENTICATION DEMO")
    print("="*70)

    # Check OAuth2 configuration
    client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
    client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

    print("\n🔧 OAUTH2 CONFIGURATION:")
    print(f"Client ID: {client_id[:20]}...{client_id[-10:] if client_id else 'NOT SET'}")
    print(f"Client Secret: {'[SET]' if client_secret else '[NOT SET]'}")

    if not client_id:
        print("\n❌ ERROR: COREAI_OAUTH2_CLIENT_ID not found!")
        print("Please set COREAI_OAUTH2_CLIENT_ID in your environment variables.")
        return

    print("\n✅ OAuth2 configuration looks good!")

    # Create OAuth2 authenticator
    print("\n🏗️ CREATING OAUTH2 AUTHENTICATOR...")
    try:
        config = HeySolConfig(
            oauth2_client_id=client_id,
            oauth2_client_secret=client_secret,
        )

        oauth2_auth = InteractiveOAuth2Authenticator(config)
        print("✅ Interactive OAuth2 authenticator created successfully!")

        # Show authorization URL
        auth_url = oauth2_auth.build_authorization_url()
        print("\n🔗 AUTHORIZATION URL:")
        print(f"URL: {auth_url}")

        print("\n📋 HOW TO USE INTERACTIVE OAUTH2:")
        print("1. Run this command in your terminal:")
        print("   python -c \"from heysol.oauth2_interactive import InteractiveOAuth2Authenticator; from heysol.config import HeySolConfig; import os; auth = InteractiveOAuth2Authenticator(HeySolConfig(oauth2_client_id=os.getenv('COREAI_OAUTH2_CLIENT_ID'))); auth.authorize_interactive()\"")
        print("\n2. A browser will open automatically")
        print("3. Sign in with your Google account")
        print("4. Grant permissions to HeySol")
        print("5. The callback will be handled automatically")
        print("6. You'll get OAuth2 tokens for API access")

        print("\n💡 ALTERNATIVE: Use the client directly")
        print("   client = HeySolClient(use_oauth2=True)")
        print("   client.authorize_oauth2_interactive()")

        print("\n🎯 WHAT THIS ENABLES:")
        print("✅ Interactive browser-based authentication")
        print("✅ Automatic callback handling")
        print("✅ OAuth2 token management")
        print("✅ Full API access with OAuth2")
        print("✅ Delete functions and log management")
        print("✅ Production-ready authentication flow")

        print("\n🚀 READY TO USE!")
        print("The HeySol API client now supports interactive OAuth2 authentication!")
        print("Just run the authorization command above to get started.")

    except Exception as e:
        print(f"\n❌ ERROR: Failed to create OAuth2 authenticator: {e}")
        return

    print("\n" + "="*70)


if __name__ == "__main__":
    main()