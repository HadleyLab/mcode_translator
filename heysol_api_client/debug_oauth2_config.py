#!/usr/bin/env python3
"""
Debug OAuth2 configuration to help troubleshoot authentication issues.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from heysol.config import HeySolConfig
from heysol.oauth2 import InteractiveOAuth2Authenticator

def debug_oauth2_config():
    """Debug OAuth2 configuration and show what values are being used."""
    print("🔍 OAuth2 Configuration Debug")
    print("=" * 50)

    # Load configuration
    config = HeySolConfig.from_env()

    print("Environment Variables:")
    print(f"  COREAI_OAUTH2_CLIENT_ID: {os.getenv('COREAI_OAUTH2_CLIENT_ID', 'NOT SET')}")
    print(f"  COREAI_OAUTH2_CLIENT_SECRET: {'*' * len(os.getenv('COREAI_OAUTH2_CLIENT_SECRET', '')) if os.getenv('COREAI_OAUTH2_CLIENT_SECRET') else 'NOT SET'}")
    print(f"  COREAI_OAUTH2_REDIRECT_URI: {os.getenv('COREAI_OAUTH2_REDIRECT_URI', 'NOT SET')}")
    print(f"  COREAI_OAUTH2_SCOPE: {os.getenv('COREAI_OAUTH2_SCOPE', 'NOT SET')}")

    print("\nConfiguration Object:")
    print(f"  config.oauth2_client_id: {config.oauth2_client_id}")
    print(f"  config.oauth2_client_secret: {'*' * len(config.oauth2_client_secret or '')}")
    print(f"  config.oauth2_redirect_uri: {config.oauth2_redirect_uri}")

    # Create authenticator to see what redirect URI it uses
    try:
        auth = InteractiveOAuth2Authenticator(config)
        print("\nInteractiveOAuth2Authenticator:")
        print(f"  auth.redirect_uri: {auth.redirect_uri}")
        print(f"  auth.client_id: {auth.client_id}")
        print(f"  auth.authorization_url: {auth.authorization_url}")
        print(f"  auth.token_url: {auth.token_url}")

        # Build authorization URL to show what will be used
        auth_url = auth.build_authorization_url()
        print(f"\nAuthorization URL: {auth_url}")

    except Exception as e:
        print(f"❌ Error creating authenticator: {e}")

    print("\n" + "=" * 50)
    print("📋 Google OAuth2 App Configuration Checklist:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Navigate to 'APIs & Services' > 'Credentials'")
    print("3. Find your OAuth2 client ID")
    print("4. Under 'Authorized redirect URIs', add:")
    print(f"   {auth.redirect_uri if 'auth' in locals() else 'http://localhost:8080/callback'}")
    print("5. Save the changes")
    print("6. Try the OAuth2 authentication again")

if __name__ == "__main__":
    debug_oauth2_config()