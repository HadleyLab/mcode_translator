#!/usr/bin/env python3
"""
OAuth2 Setup Guide for HeySol API Client

This script provides instructions and validation for setting up OAuth2 authentication
with the HeySol API client. It helps users create their own OAuth2 credentials
and test the OAuth2 flow.
"""

import os
import sys
import webbrowser
from pathlib import Path

def check_oauth2_credentials():
    """Check if OAuth2 credentials are properly configured."""
    client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
    client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

    print("🔐 Checking OAuth2 Credentials...")
    print(f"Client ID: {'✅ Set' if client_id else '❌ Not set'}")
    print(f"Client Secret: {'✅ Set' if client_secret else '❌ Not set'}")

    if not client_id or not client_secret:
        print("\n⚠️  OAuth2 credentials not found!")
        return False

    return True

def show_setup_instructions():
    """Display OAuth2 setup instructions."""
    print("\n📋 OAuth2 Setup Instructions:")
    print("=" * 50)
    print("To use OAuth2 authentication with HeySol API:")
    print()
    print("1. Go to Google Cloud Console:")
    print("   https://console.cloud.google.com/")
    print()
    print("2. Create a new project or select existing:")
    print("   - Click 'Select Project' → 'New Project'")
    print("   - Enter project name (e.g., 'heysol-api-client')")
    print()
    print("3. Enable required APIs:")
    print("   - Go to 'APIs & Services' → 'Library'")
    print("   - Search for and enable 'Google+ API'")
    print()
    print("4. Create OAuth2 credentials:")
    print("   - Go to 'APIs & Services' → 'Credentials'")
    print("   - Click '+ CREATE CREDENTIALS' → 'OAuth 2.0 Client IDs'")
    print("   - Application type: 'Web application'")
    print("   - Name: 'HeySol API Client'")
    print("   - Authorized redirect URIs:")
    print("     http://localhost:8080/callback")
    print("   - Click 'Create'")
    print()
    print("5. Set environment variables:")
    print("   export COREAI_OAUTH2_CLIENT_ID='your-client-id'")
    print("   export COREAI_OAUTH2_CLIENT_SECRET='your-client-secret'")
    print()
    print("6. Test the setup:")
    print("   python oauth2_setup_guide.py")

def test_oauth2_url():
    """Test OAuth2 URL generation."""
    client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")

    if not client_id:
        print("\n❌ Cannot test OAuth2 URL - no client ID found")
        return

    redirect_uri = "http://localhost:8080/callback"
    auth_url = (
        "https://accounts.google.com/oauth/authorize"
        + f"?client_id={client_id}"
        + "&redirect_uri=" + redirect_uri
        + "&scope=openid%20profile%20email"
        + "&response_type=code"
        + "&access_type=offline"
        + "&prompt=consent"
    )

    print("\n🔗 Generated OAuth2 URL:")
    print(f"   {auth_url}")
    print()
    print("📋 Next steps:")
    print("   1. Copy the URL above")
    print("   2. Paste it into your browser")
    print("   3. Complete Google OAuth2 authentication")
    print("   4. You should be redirected to localhost:8080/callback")
    print("   5. The authorization code will be in the URL parameters")

    # Try to open browser
    try:
        webbrowser.open(auth_url)
        print("\n✅ Browser opened automatically!")
    except Exception as e:
        print(f"\n⚠️  Could not open browser: {e}")

def main():
    """Main function."""
    print("🚀 HeySol API Client - OAuth2 Setup Guide")
    print("=" * 50)

    # Check credentials
    credentials_valid = check_oauth2_credentials()

    if credentials_valid:
        print("\n✅ OAuth2 credentials found!")
        test_oauth2_url()
    else:
        print("\n❌ OAuth2 credentials not configured")
        show_setup_instructions()

    print("\n📞 Need Help?")
    print("   - Check HeySol API documentation")
    print("   - Review the OAuth2 implementation in heysol/oauth2.py")
    print("   - Run the OAuth2 demo notebook for testing")

if __name__ == "__main__":
    main()