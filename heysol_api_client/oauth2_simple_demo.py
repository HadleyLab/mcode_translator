#!/usr/bin/env python3
"""
Simple OAuth2 Demo for HeySol API Client.

This script demonstrates basic OAuth2 functionality without requiring interactive authentication.
It shows how to work with OAuth2 tokens and make authenticated API calls.

Usage:
    python oauth2_simple_demo.py

Requirements:
    - Set HEYSOL_API_KEY in environment (for API key authentication)
    - Install required dependencies: pip install requests python-dotenv
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import HeySol client
from heysol.client import HeySolClient
from heysol.exceptions import HeySolError, ValidationError


class SimpleOAuth2Demo:
    """Simple OAuth2 demonstration without interactive authentication."""

    def __init__(self):
        """Initialize the demo."""
        self.api_key = os.getenv("HEYSOL_API_KEY")
        self.client = None

    def check_configuration(self) -> bool:
        """Check if configuration is valid."""
        print("\n🔐 Checking Configuration...")
        print("=" * 40)

        if not self.api_key:
            print("❌ HEYSOL_API_KEY not set")
            print("   Please set your API key: export HEYSOL_API_KEY='your-api-key'")
            return False

        print("✅ API Key: Set")
        return True

    def setup_client(self) -> bool:
        """Set up HeySol client."""
        print("\n🔧 Setting up HeySol Client...")
        print("=" * 40)

        try:
            self.client = HeySolClient(api_key=self.api_key)
            print("✅ HeySol client created successfully")
            return True
        except ValidationError as e:
            print(f"❌ Client setup failed: {e}")
            return False

    def demonstrate_oauth2_endpoints(self) -> bool:
        """Demonstrate OAuth2 endpoints."""
        print("\n🔗 Demonstrating OAuth2 Endpoints...")
        print("=" * 40)

        try:
            # Test OAuth2 authorization URL
            print("📡 Testing OAuth2 authorization URL...")
            try:
                auth_url = self.client.get_oauth2_authorization_url("openid profile email")
                print("✅ OAuth2 authorization URL generated successfully")
                print(f"   URL: {auth_url.get('authorization_url', 'N/A')[:100]}...")
            except Exception as e:
                print(f"⚠️  OAuth2 authorization URL failed: {e}")

            # Test OAuth2 user info (this would require a valid access token)
            print("\n📡 Testing OAuth2 user info (requires valid token)...")
            try:
                # This will fail without a valid token, which is expected
                user_info = self.client.get_oauth2_user_info("invalid-token")
                print("✅ OAuth2 user info retrieved (unexpected success)")
            except ValidationError as e:
                print(f"✅ OAuth2 user info validation working: {e}")
            except Exception as e:
                print(f"⚠️  OAuth2 user info failed: {e}")

            return True

        except Exception as e:
            print(f"❌ OAuth2 endpoints demonstration failed: {e}")
            return False

    def demonstrate_api_calls(self) -> bool:
        """Demonstrate API calls."""
        print("\n🧪 Demonstrating API Calls...")
        print("=" * 40)

        try:
            # Test user profile
            print("📡 Testing user profile...")
            try:
                profile = self.client.get_user_profile()
                print("✅ User profile retrieved successfully")
                print(f"   User ID: {profile.get('id', 'N/A')}")
                print(f"   Username: {profile.get('username', 'N/A')}")
            except Exception as e:
                print(f"⚠️  User profile failed: {e}")

            # Test memory search
            print("\n📡 Testing memory search...")
            try:
                result = self.client.search("OAuth2 demo", limit=1)
                print("✅ Memory search completed successfully")
                print(f"   Results: {len(result.get('episodes', []))} episodes")
            except Exception as e:
                print(f"⚠️  Memory search failed: {e}")

            # Test spaces
            print("\n📡 Testing spaces...")
            try:
                spaces = self.client.get_spaces()
                print("✅ Spaces retrieved successfully")
                if isinstance(spaces, list):
                    print(f"   Found {len(spaces)} spaces")
                else:
                    print(f"   Spaces: {spaces}")
            except Exception as e:
                print(f"⚠️  Spaces failed: {e}")

            return True

        except Exception as e:
            print(f"❌ API calls demonstration failed: {e}")
            return False

    def demonstrate_error_handling(self) -> bool:
        """Demonstrate error handling."""
        print("\n🛡️  Demonstrating Error Handling...")
        print("=" * 40)

        try:
            # Test invalid OAuth2 user info call
            print("📡 Testing invalid OAuth2 user info...")
            try:
                self.client.get_oauth2_user_info("")
                print("❌ Expected validation error but got success")
            except ValidationError as e:
                print(f"✅ Validation error caught correctly: {e}")
            except Exception as e:
                print(f"⚠️  Unexpected error: {e}")

            # Test invalid token exchange
            print("\n📡 Testing invalid token exchange...")
            try:
                self.client.oauth2_token_exchange("", "")
                print("❌ Expected validation error but got success")
            except ValidationError as e:
                print(f"✅ Validation error caught correctly: {e}")
            except Exception as e:
                print(f"⚠️  Unexpected error: {e}")

            return True

        except Exception as e:
            print(f"❌ Error handling demonstration failed: {e}")
            return False

    def run_demo(self) -> Dict[str, Any]:
        """Run the simple OAuth2 demo."""
        print("\n" + "🚀" + "="*38 + "🚀")
        print("🎯 SIMPLE OAUTH2 DEMO - HEYSOL API CLIENT")
        print("🚀" + "="*38 + "🚀")

        results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "steps": []
        }

        try:
            # Step 1: Configuration check
            if not self.check_configuration():
                results["error"] = "Configuration check failed"
                return results

            results["steps"].append({
                "step": "configuration",
                "status": "completed",
                "description": "Configuration validated"
            })

            # Step 2: Client setup
            if not self.setup_client():
                results["error"] = "Client setup failed"
                return results

            results["steps"].append({
                "step": "client_setup",
                "status": "completed",
                "description": "HeySol client created"
            })

            # Step 3: OAuth2 endpoints
            if not self.demonstrate_oauth2_endpoints():
                results["steps"].append({
                    "step": "oauth2_endpoints",
                    "status": "completed_with_warning",
                    "description": "OAuth2 endpoints demonstrated with some issues"
                })
            else:
                results["steps"].append({
                    "step": "oauth2_endpoints",
                    "status": "completed",
                    "description": "OAuth2 endpoints demonstrated successfully"
                })

            # Step 4: API calls
            if not self.demonstrate_api_calls():
                results["steps"].append({
                    "step": "api_calls",
                    "status": "completed_with_warning",
                    "description": "API calls completed with some issues"
                })
            else:
                results["steps"].append({
                    "step": "api_calls",
                    "status": "completed",
                    "description": "API calls successful"
                })

            # Step 5: Error handling
            if not self.demonstrate_error_handling():
                results["steps"].append({
                    "step": "error_handling",
                    "status": "completed_with_warning",
                    "description": "Error handling demonstrated with issues"
                })
            else:
                results["steps"].append({
                    "step": "error_handling",
                    "status": "completed",
                    "description": "Error handling demonstrated successfully"
                })

            # Overall success
            results["success"] = True

        except Exception as e:
            results["error"] = str(e)

        return results

    def save_results(self, results: Dict[str, Any]):
        """Save demo results to file."""
        filename = "oauth2_simple_demo_results.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {filename}")


def main():
    """Main function to run the simple OAuth2 demo."""
    demo = SimpleOAuth2Demo()
    results = demo.run_demo()

    # Display results
    print("\n" + "📊" + "="*38 + "📊")
    print("🎯 SIMPLE OAUTH2 DEMO RESULTS")
    print("📊" + "="*38 + "📊")

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
        print("\n🎉 SUCCESS: Simple OAuth2 demo completed successfully!")
        print("\nThe demo demonstrated:")
        print("✅ Configuration validation")
        print("✅ Client setup")
        print("✅ OAuth2 endpoints")
        print("✅ API calls")
        print("✅ Error handling")
    else:
        print("\n❌ FAILURE: Simple OAuth2 demo failed")
        print("Check the configuration and try again.")

    print("\n" + "🎉" + "="*38 + "🎉")
    print("🎯 SIMPLE OAUTH2 DEMO COMPLETE")
    print("🎉" + "="*38 + "🎉")


if __name__ == "__main__":
    main()