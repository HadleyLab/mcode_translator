#!/usr/bin/env python3
"""
HeySol API Client - OAuth2 Log Operations Demo

This script demonstrates the complete OAuth2 authentication flow with the HeySol API client:
1. OAuth2 browser authentication with Google
2. Log ingestion with OAuth2 tokens
3. Log deletion with OAuth2 tokens
4. Automatic token refresh and error handling
5. Complete cleanup of test data

Usage:
    python oauth2_log_demo.py

Requirements:
    - Set COREAI_OAUTH2_CLIENT_ID and COREAI_OAUTH2_CLIENT_SECRET environment variables
    - Install required dependencies: pip install requests python-dotenv
    - Have a valid Google account for OAuth2 authentication
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import unified OAuth2 implementation
from heysol.oauth2 import (
    create_oauth2_demo_runner,
    validate_oauth2_setup,
    AuthenticationError,
    ValidationError,
    HeySolError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oauth2_log_demo.log")
    ]
)
logger = logging.getLogger(__name__)


class OAuth2LogDemo:
    """Complete OAuth2 log operations demo using centralized utilities."""

    def __init__(self):
        """Initialize the demo class."""
        self.demo_runner = None

    def check_oauth2_config(self) -> bool:
        """Check if OAuth2 credentials are properly configured."""
        try:
            validate_oauth2_setup()
            return True
        except AuthenticationError as e:
            print(f"❌ OAuth2 configuration error: {e}")
            return False

    def display_oauth2_info(self, client):
        """Display OAuth2 user and token information."""
        print("\n" + "="*60)
        print("🔐 OAUTH2 AUTHENTICATION DETAILS")
        print("="*60)

        try:
            # Get user info
            print("\n👤 User Information:")
            try:
                user_info = client.get_oauth2_user_info()
                if user_info:
                    print(f"   Name: {user_info.get('name', 'N/A')}")
                    print(f"   Email: {user_info.get('email', 'N/A')}")
                    print(f"   User ID: {user_info.get('id', 'N/A')}")
                    print(f"   Picture: {user_info.get('picture', 'N/A')}")
                else:
                    print("   ❌ Could not retrieve user information")
            except AuthenticationError as e:
                print(f"   ❌ Authentication error: {e}")
            except Exception as e:
                print(f"   ❌ Error getting user info: {e}")

            # Get token info
            print("\n🔑 Token Information:")
            try:
                token_info = client.introspect_oauth2_token()
                if token_info:
                    print(f"   Active: {token_info.get('active', 'N/A')}")
                    print(f"   Client ID: {token_info.get('client_id', 'N/A')}")
                    print(f"   Scope: {token_info.get('scope', 'N/A')}")
                    print(f"   Expires: {token_info.get('exp', 'N/A')}")
                    print(f"   Issued At: {token_info.get('iat', 'N/A')}")
                    print(f"   Token Type: {token_info.get('token_type', 'N/A')}")
                else:
                    print("   ❌ Could not retrieve token information")
            except AuthenticationError as e:
                print(f"   ❌ Authentication error: {e}")
            except Exception as e:
                print(f"   ❌ Error getting token info: {e}")

            # Get user profile
            print("\n📊 HeySol Profile:")
            try:
                profile = client.get_user_profile()
                if profile:
                    print(f"   User ID: {profile.get('id', 'N/A')}")
                    print(f"   Username: {profile.get('username', 'N/A')}")
                    print(f"   Email: {profile.get('email', 'N/A')}")
                    print(f"   Created: {profile.get('created_at', 'N/A')}")
                else:
                    print("   ❌ Could not retrieve HeySol profile")
            except AuthenticationError as e:
                print(f"   ❌ Authentication error: {e}")
            except Exception as e:
                print(f"   ❌ Error getting profile: {e}")

        except Exception as e:
            print(f"   ❌ Error displaying OAuth2 info: {e}")

        print("="*60)

    def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete OAuth2 log operations demo using centralized utilities."""
        logger.info("🚀 Starting complete OAuth2 log operations demo...")

        try:
            # Use centralized demo runner
            self.demo_runner = create_oauth2_demo_runner()
            results = self.demo_runner.run_complete_demo()

            # Display OAuth2 information
            print("\n" + "="*60)
            print("STEP 3: Display OAuth2 Information")
            print("="*60)

            client = self.demo_runner.client_manager.get_client()
            self.display_oauth2_info(client)

            return results

        except AuthenticationError as e:
            logger.error(f"❌ Authentication failed: {e}")
            return {
                "timestamp": time.time(),
                "success": False,
                "error": f"Authentication failed: {e}",
                "steps": []
            }
        except ValidationError as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "timestamp": time.time(),
                "success": False,
                "error": f"Validation failed: {e}",
                "steps": []
            }
        except HeySolError as e:
            logger.error(f"❌ Demo execution failed: {e}")
            return {
                "timestamp": time.time(),
                "success": False,
                "error": f"Demo execution failed: {e}",
                "steps": []
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return {
                "timestamp": time.time(),
                "success": False,
                "error": f"Unexpected error: {e}",
                "steps": []
            }


def main():
    """Main function to run the OAuth2 log operations demo."""
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🎯 HEYSOL OAUTH2 LOG OPERATIONS DEMO")
    print("🚀" + "="*58 + "🚀")

    # Create demo instance
    demo = OAuth2LogDemo()

    # Check configuration
    if not demo.check_oauth2_config():
        print("\n❌ Cannot run demo: OAuth2 configuration is invalid")
        print("Please set COREAI_OAUTH2_CLIENT_ID and COREAI_OAUTH2_CLIENT_SECRET environment variables.")
        return

    # Run the demo
    results = demo.run_complete_demo()

    # Display results
    print("\n" + "📊" + "="*58 + "📊")
    print("🎯 DEMO RESULTS")
    print("📊" + "="*58 + "📊")

    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Success: {'✅ YES' if results['success'] else '❌ NO'}")

    if 'error' in results:
        print(f"Error: {results['error']}")

    print(f"\nSteps Completed: {len(results['steps'])}")

    for i, step in enumerate(results['steps'], 1):
        status_icon = "✅" if step['status'] == 'completed' else "⚠️" if step['status'] == 'completed_with_warning' else "❌"
        print(f"{i}. {status_icon} {step['step']}: {step['description']}")

    # Save detailed results
    with open("oauth2_log_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: oauth2_log_demo_results.json")
    print(f"📝 Debug logs saved to: oauth2_log_demo.log")

    if results['success']:
        print("\n🎉 SUCCESS: OAuth2 log operations demo completed successfully!")
        print("\nThe HeySol API client now supports:")
        print("✅ Interactive OAuth2 authentication")
        print("✅ Browser-based authorization")
        print("✅ Automatic token management")
        print("✅ Log ingestion with OAuth2")
        print("✅ Log deletion with OAuth2")
        print("✅ Error handling and token refresh")
    else:
        print("\n❌ FAILURE: OAuth2 log operations demo failed")
        print("Check the logs for details:")
        print("- oauth2_log_demo.log")
        print("- oauth2_client_debug.log")

    print("\n" + "🎉" + "="*58 + "🎉")
    print("🎯 DEMO COMPLETE - OAuth2 Log Operations Ready!")
    print("🎉" + "="*58 + "🎉")


if __name__ == "__main__":
    main()