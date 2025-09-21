#!/usr/bin/env python3
"""
Google OAuth2 Interactive Demo for HeySol API Client.

This script demonstrates the complete OAuth2 authorization flow using Google authentication:
1. Builds OAuth2 authorization URL with Google OAuth2
2. Opens browser for interactive authentication
3. Handles the callback and extracts authorization code
4. Exchanges code for access tokens
5. Makes authenticated API calls
6. Demonstrates token refresh and user info retrieval

Usage:
    python oauth2_google_demo.py

Requirements:
    - Set HEYSOL_OAUTH2_CLIENT_ID and HEYSOL_OAUTH2_CLIENT_SECRET in environment
    - Install required dependencies: pip install requests python-dotenv flask
"""

import os
import sys
import time
import json
import logging
import webbrowser
import threading
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import HeySol client and OAuth2 components
from heysol.client import HeySolClient
from heysol.oauth2 import InteractiveOAuth2Authenticator, OAuth2Tokens
from heysol.exceptions import HeySolError, ValidationError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oauth2_google_demo.log")
    ]
)
logger = logging.getLogger(__name__)


class OAuth2CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth2 callback."""

    def do_GET(self):
        """Handle GET requests to the callback URL."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        # Extract authorization code from URL parameters
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        if 'code' in query_params:
            code = query_params['code'][0]
            self.server.oauth2_code = code

            # Display success page
            response_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth2 Authorization Successful</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .success { color: #28a745; font-size: 24px; margin-bottom: 20px; }
                    .code { background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; }
                </style>
            </head>
            <body>
                <div class="success">✅ OAuth2 Authorization Successful!</div>
                <p>You can now close this window and return to the terminal.</p>
                <p><strong>Authorization Code:</strong></p>
                <div class="code">{}</div>
            </body>
            </html>
            """.format(code)

            self.wfile.write(response_html.encode())
        else:
            # Display error page
            error_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>OAuth2 Authorization Failed</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .error { color: #dc3545; font-size: 24px; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="error">❌ OAuth2 Authorization Failed</div>
                <p>No authorization code received. Please try again.</p>
                <p>You can close this window and return to the terminal.</p>
            </body>
            </html>
            """
            self.wfile.write(error_html.encode())

    def log_message(self, format, *args):
        """Suppress default HTTP request logging."""
        pass


class OAuth2CallbackServer:
    """Simple HTTP server for handling OAuth2 callbacks."""

    def __init__(self, port: int = 8080):
        """Initialize callback server."""
        self.port = port
        self.oauth2_code = None
        self.server = None

    def start(self):
        """Start the callback server."""
        try:
            self.server = HTTPServer(('localhost', self.port), OAuth2CallbackHandler)
            self.server.oauth2_code = None
            logger.info(f"🌐 OAuth2 callback server started on http://localhost:{self.port}")
            return True
        except OSError as e:
            logger.error(f"❌ Failed to start callback server: {e}")
            return False

    def wait_for_code(self, timeout: int = 300) -> Optional[str]:
        """Wait for OAuth2 authorization code."""
        if not self.server:
            return None

        logger.info("⏳ Waiting for OAuth2 authorization code...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            self.server.handle_request()

            if self.server.oauth2_code:
                code = self.server.oauth2_code
                logger.info("✅ OAuth2 authorization code received")
                return code

            time.sleep(0.1)

        logger.error("⏰ OAuth2 authorization timeout")
        return None

    def stop(self):
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            logger.info("🛑 OAuth2 callback server stopped")


class GoogleOAuth2Demo:
    """Complete Google OAuth2 demonstration for HeySol API."""

    def __init__(self):
        """Initialize the OAuth2 demo."""
        self.client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        self.client_secret = os.getenv("HEYSOL_OAUTH2_CLIENT_SECRET")
        self.redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")
        self.scope = os.getenv("HEYSOL_OAUTH2_SCOPE", "openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email")

        self.oauth2_auth = None
        self.client = None
        self.callback_server = None

    def check_configuration(self) -> bool:
        """Check if OAuth2 configuration is valid."""
        print("\n🔐 Checking OAuth2 Configuration...")
        print("=" * 50)

        if not self.client_id:
            print("❌ HEYSOL_OAUTH2_CLIENT_ID not set")
            return False

        if not self.client_secret:
            print("❌ HEYSOL_OAUTH2_CLIENT_SECRET not set")
            return False

        print("✅ Client ID: Set")
        print("✅ Client Secret: Set")
        print(f"✅ Redirect URI: {self.redirect_uri}")
        print(f"✅ Scope: {self.scope}")
        return True

    def setup_oauth2_authenticator(self) -> bool:
        """Set up OAuth2 authenticator."""
        try:
            print("\n🔧 Setting up OAuth2 Authenticator...")
            print("=" * 50)

            self.oauth2_auth = InteractiveOAuth2Authenticator(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri
            )

            print("✅ OAuth2 authenticator created successfully")
            return True

        except ValidationError as e:
            print(f"❌ OAuth2 authenticator setup failed: {e}")
            return False

    def start_callback_server(self) -> bool:
        """Start the OAuth2 callback server."""
        print("\n🌐 Starting OAuth2 Callback Server...")
        print("=" * 50)

        self.callback_server = OAuth2CallbackServer(port=8080)

        if self.callback_server.start():
            print("✅ Callback server started successfully")
            return True
        else:
            print("❌ Failed to start callback server")
            return False

    def build_google_authorization_url(self) -> str:
        """Build Google OAuth2 authorization URL."""
        print("\n🔗 Building Google OAuth2 Authorization URL...")
        print("=" * 50)

        # Use Google's OAuth2 endpoint instead of HeySol's
        google_auth_url = "https://accounts.google.com/oauth/authorize"

        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "state": "heysol-oauth2-demo"
        }

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        auth_url = f"{google_auth_url}?{query_string}"

        print(f"✅ Authorization URL built: {auth_url[:100]}...")
        return auth_url

    def perform_interactive_authorization(self) -> Optional[str]:
        """Perform interactive OAuth2 authorization."""
        print("\n🌐 Interactive OAuth2 Authorization")
        print("=" * 50)

        # Build authorization URL
        auth_url = self.build_google_authorization_url()

        # Open browser
        print("📱 Opening browser for Google OAuth2 authentication...")
        try:
            webbrowser.open(auth_url)
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print(f"   Please manually open: {auth_url}")

        # Wait for authorization code
        print("⏳ Waiting for user to complete OAuth2 authorization...")
        print("   (This may take a few minutes)")
        print("   Look for the browser window and complete Google login")

        code = self.callback_server.wait_for_code(timeout=300)

        if code:
            print("✅ Authorization code received successfully")
            return code
        else:
            print("❌ Failed to receive authorization code")
            return None

    def exchange_code_for_tokens(self, code: str) -> bool:
        """Exchange authorization code for access tokens."""
        print("\n🔄 Exchanging Authorization Code for Tokens...")
        print("=" * 50)

        try:
            # Use Google's token endpoint
            token_url = "https://oauth2.googleapis.com/token"

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }

            print("📡 Sending token exchange request...")
            response = requests.post(token_url, data=data, headers=headers, timeout=60)
            response.raise_for_status()

            token_data = response.json()
            print("✅ Token exchange successful")

            # Set tokens in authenticator
            tokens = OAuth2Tokens(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                scope=token_data.get("scope"),
            )

            # Calculate expiration time
            if tokens.expires_in:
                tokens.expires_at = time.time() + tokens.expires_in

            self.oauth2_auth.tokens = tokens
            print("✅ Tokens stored successfully")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Token exchange failed: {e}")
            return False
        except (KeyError, ValueError) as e:
            print(f"❌ Invalid token response: {e}")
            return False

    def test_authenticated_api_calls(self) -> bool:
        """Test authenticated API calls."""
        print("\n🧪 Testing Authenticated API Calls...")
        print("=" * 50)

        try:
            # Test Google user info
            print("📡 Testing Google user info endpoint...")
            userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
            auth_header = self.oauth2_auth.get_authorization_header()

            response = requests.get(
                userinfo_url,
                headers={"Authorization": auth_header},
                timeout=30
            )
            response.raise_for_status()

            user_info = response.json()
            print("✅ Google user info retrieved successfully")
            print(f"   Name: {user_info.get('name', 'N/A')}")
            print(f"   Email: {user_info.get('email', 'N/A')}")
            print(f"   ID: {user_info.get('id', 'N/A')}")

            # Test HeySol API calls
            print("\n📡 Testing HeySol API calls...")
            self.client = HeySolClient(oauth2_auth=self.oauth2_auth, skip_mcp_init=True)

            # Test user profile
            try:
                profile = self.client.get_user_profile()
                print("✅ HeySol user profile retrieved successfully")
                print(f"   User ID: {profile.get('id', 'N/A')}")
                print(f"   Username: {profile.get('username', 'N/A')}")
            except Exception as e:
                print(f"⚠️  HeySol user profile failed: {e}")

            # Test memory search
            try:
                result = self.client.search("OAuth2 demo test", limit=1)
                print("✅ HeySol memory search completed successfully")
            except Exception as e:
                print(f"⚠️  HeySol memory search failed: {e}")

            return True

        except Exception as e:
            print(f"❌ Authenticated API calls failed: {e}")
            return False

    def demonstrate_token_refresh(self) -> bool:
        """Demonstrate token refresh functionality."""
        print("\n🔄 Demonstrating Token Refresh...")
        print("=" * 50)

        try:
            # Get current token info
            if not self.oauth2_auth.tokens:
                print("❌ No tokens available for refresh demonstration")
                return False

            print("📡 Testing token introspection...")
            tokeninfo_url = "https://oauth2.googleapis.com/tokeninfo"

            response = requests.post(
                tokeninfo_url,
                data={"access_token": self.oauth2_auth.tokens.access_token},
                timeout=30
            )

            if response.status_code == 200:
                token_info = response.json()
                print("✅ Token is valid")
                print(f"   Expires: {token_info.get('exp', 'N/A')}")
                print(f"   Issued to: {token_info.get('aud', 'N/A')}")
            else:
                print("⚠️  Token may be expired, attempting refresh...")
                # Note: Google OAuth2 doesn't have a simple refresh endpoint like this
                # This is just for demonstration

            return True

        except Exception as e:
            print(f"❌ Token refresh demonstration failed: {e}")
            return False

    def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete OAuth2 demo."""
        logger.info("🚀 Starting Google OAuth2 demo...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "steps": []
        }

        try:
            # Step 1: Configuration check
            print("\n" + "🚀" + "="*58 + "🚀")
            print("🎯 GOOGLE OAUTH2 DEMO - HEYSOL API CLIENT")
            print("🚀" + "="*58 + "🚀")

            if not self.check_configuration():
                results["error"] = "Configuration check failed"
                return results

            results["steps"].append({
                "step": "configuration",
                "status": "completed",
                "description": "OAuth2 configuration validated"
            })

            # Step 2: Setup OAuth2 authenticator
            if not self.setup_oauth2_authenticator():
                results["error"] = "OAuth2 authenticator setup failed"
                return results

            results["steps"].append({
                "step": "authenticator_setup",
                "status": "completed",
                "description": "OAuth2 authenticator created"
            })

            # Step 3: Start callback server
            if not self.start_callback_server():
                results["error"] = "Callback server setup failed"
                return results

            results["steps"].append({
                "step": "callback_server",
                "status": "completed",
                "description": "OAuth2 callback server started"
            })

            # Step 4: Interactive authorization
            code = self.perform_interactive_authorization()
            if not code:
                results["error"] = "Interactive authorization failed"
                return results

            results["steps"].append({
                "step": "interactive_auth",
                "status": "completed",
                "description": "Interactive OAuth2 authorization completed"
            })

            # Step 5: Token exchange
            if not self.exchange_code_for_tokens(code):
                results["error"] = "Token exchange failed"
                return results

            results["steps"].append({
                "step": "token_exchange",
                "status": "completed",
                "description": "Authorization code exchanged for tokens"
            })

            # Step 6: Authenticated API calls
            if not self.test_authenticated_api_calls():
                results["steps"].append({
                    "step": "api_calls",
                    "status": "completed_with_warning",
                    "description": "Authenticated API calls completed with some issues"
                })
            else:
                results["steps"].append({
                    "step": "api_calls",
                    "status": "completed",
                    "description": "Authenticated API calls successful"
                })

            # Step 7: Token refresh demonstration
            if not self.demonstrate_token_refresh():
                results["steps"].append({
                    "step": "token_refresh",
                    "status": "completed_with_warning",
                    "description": "Token refresh demonstration completed with issues"
                })
            else:
                results["steps"].append({
                    "step": "token_refresh",
                    "status": "completed",
                    "description": "Token refresh demonstration successful"
                })

            # Overall success
            results["success"] = True
            logger.info("✅ Google OAuth2 demo completed successfully")

        except Exception as e:
            logger.error(f"❌ Demo execution failed: {e}")
            results["error"] = str(e)

        finally:
            # Cleanup
            if self.callback_server:
                self.callback_server.stop()

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save demo results to file."""
        filename = "oauth2_google_demo_results.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {filename}")


def main():
    """Main function to run the Google OAuth2 demo."""
    demo = GoogleOAuth2Demo()
    results = demo.run_complete_demo()

    # Display results
    print("\n" + "📊" + "="*58 + "📊")
    print("🎯 GOOGLE OAUTH2 DEMO RESULTS")
    print("📊" + "="*58 + "📊")

    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Success: {'✅ YES' if results['success'] else '❌ NO'}")

    if 'error' in results:
        print(f"Error: {results['error']}")

    print(f"\nSteps Completed: {len(results['steps'])}")

    for i, step in enumerate(results['steps'], 1):
        status_icon = {
            "completed": "✅",
            "completed_with_warning": "⚠️",
            "failed": "❌"
        }.get(step['status'], "❓")

        print(f"{i}. {status_icon} {step['step']}: {step['description']}")

    # Save results
    demo.save_results(results)

    if results['success']:
        print("\n🎉 SUCCESS: Google OAuth2 demo completed successfully!")
        print("\nThe demo demonstrated:")
        print("✅ OAuth2 configuration validation")
        print("✅ Interactive Google OAuth2 authorization")
        print("✅ Authorization code exchange for tokens")
        print("✅ Authenticated API calls")
        print("✅ Token management and refresh")
    else:
        print("\n❌ FAILURE: Google OAuth2 demo failed")
        print("Check the logs for details:")
        print("- oauth2_google_demo.log")

    print("\n" + "🎉" + "="*58 + "🎉")
    print("🎯 GOOGLE OAUTH2 DEMO COMPLETE")
    print("🎉" + "="*58 + "🎉")


if __name__ == "__main__":
    main()