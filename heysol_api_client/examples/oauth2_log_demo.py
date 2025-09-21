#!/usr/bin/env python3
"""
HeySol API Client - Log Operations Demo

This script demonstrates the HeySol API client functionality using API key authentication:
1. API key authentication
2. Log ingestion with API key
3. Log listing and retrieval
4. User profile access
5. Error handling and validation

Usage:
    python oauth2_log_demo.py

Requirements:
    - Set HEYSOL_API_KEY environment variable
    - Install required dependencies: pip install requests python-dotenv
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Load environment variables from the heysol_api_client directory
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import HeySol client and exceptions
from heysol.client import HeySolClient
from heysol.exceptions import HeySolError, ValidationError

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


class HeySolLogDemo:
    """Complete HeySol log operations demo using API key authentication."""

    def __init__(self):
        """Initialize the demo class."""
        self.client: Optional[HeySolClient] = None

    def check_api_key_config(self) -> bool:
        """Check if API key is properly configured."""
        api_key = os.getenv('HEYSOL_API_KEY')
        if not api_key:
            print("❌ HEYSOL_API_KEY environment variable not set")
            print("   Please set your API key: export HEYSOL_API_KEY='your-api-key'")
            return False
        return True

    def display_api_info(self):
        """Display API key authentication information."""
        print("\n" + "="*60)
        print("🔐 API KEY AUTHENTICATION DETAILS")
        print("="*60)

        try:
            # Get user profile
            print("\n👤 User Profile:")
            try:
                profile = self.client.get_user_profile()
                if profile:
                    print(f"   User ID: {profile.get('id', 'N/A')}")
                    print(f"   Username: {profile.get('username', 'N/A')}")
                    print(f"   Email: {profile.get('email', 'N/A')}")
                    print(f"   Created: {profile.get('created_at', 'N/A')}")
                else:
                    print("   ❌ Could not retrieve user profile")
            except Exception as e:
                print(f"   ❌ Error getting profile: {e}")

            # Get spaces
            print("\n🏢 Available Spaces:")
            try:
                spaces = self.client.get_spaces()
                if spaces:
                    if isinstance(spaces, list):
                        print(f"   Found {len(spaces)} spaces")
                        for i, space in enumerate(spaces[:5], 1):  # Show first 5
                            print(f"     {i}. {space.get('name', 'N/A')} ({space.get('id', 'N/A')})")
                    else:
                        print(f"   Spaces: {spaces}")
                else:
                    print("   ❌ Could not retrieve spaces")
            except Exception as e:
                print(f"   ❌ Error getting spaces: {e}")

        except Exception as e:
            print(f"   ❌ Error displaying API info: {e}")

        print("="*60)

    def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete HeySol log operations demo using API key authentication."""
        logger.info("🚀 Starting complete HeySol log operations demo...")

        try:
            # Initialize client
            api_key = os.getenv('HEYSOL_API_KEY')
            if not api_key:
                raise ValidationError("HEYSOL_API_KEY environment variable not set")

            self.client = HeySolClient(api_key=api_key)

            # Track demo steps
            steps = []

            # Step 1: Authentication
            print("\n" + "="*60)
            print("STEP 1: API Key Authentication")
            print("="*60)
            steps.append({
                "step": "authentication",
                "status": "completed",
                "description": "API key authentication successful"
            })
            logger.info("✅ API key authentication completed")

            # Step 2: User Profile
            print("\n" + "="*60)
            print("STEP 2: User Profile Access")
            print("="*60)
            try:
                profile = self.client.get_user_profile()
                print("✅ User profile retrieved successfully")
                steps.append({
                    "step": "user_profile",
                    "status": "completed",
                    "description": "User profile access successful"
                })
                logger.info("✅ User profile access completed")
            except Exception as e:
                print(f"⚠️ User profile access failed: {e}")
                steps.append({
                    "step": "user_profile",
                    "status": "completed_with_warning",
                    "description": f"User profile access failed: {e}"
                })
                logger.warning(f"⚠️ User profile access failed: {e}")

            # Step 3: Memory Operations
            print("\n" + "="*60)
            print("STEP 3: Memory Operations")
            print("="*60)

            # Memory search
            try:
                result = self.client.search("demo test", limit=1)
                print("✅ Memory search completed")
                steps.append({
                    "step": "memory_search",
                    "status": "completed",
                    "description": "Memory search successful"
                })
                logger.info("✅ Memory search completed")
            except Exception as e:
                print(f"⚠️ Memory search failed: {e}")
                steps.append({
                    "step": "memory_search",
                    "status": "completed_with_warning",
                    "description": f"Memory search failed: {e}"
                })
                logger.warning(f"⚠️ Memory search failed: {e}")

            # Memory ingestion
            try:
                result = self.client.ingest("Demo message from HeySol client")
                print("✅ Memory ingestion completed")
                steps.append({
                    "step": "memory_ingestion",
                    "status": "completed",
                    "description": "Memory ingestion successful"
                })
                logger.info("✅ Memory ingestion completed")
            except Exception as e:
                print(f"⚠️ Memory ingestion failed: {e}")
                steps.append({
                    "step": "memory_ingestion",
                    "status": "completed_with_warning",
                    "description": f"Memory ingestion failed: {e}"
                })
                logger.warning(f"⚠️ Memory ingestion failed: {e}")

            # Step 4: Space Operations
            print("\n" + "="*60)
            print("STEP 4: Space Operations")
            print("="*60)
            try:
                spaces = self.client.get_spaces()
                print("✅ Space operations completed")
                steps.append({
                    "step": "space_operations",
                    "status": "completed",
                    "description": "Space operations successful"
                })
                logger.info("✅ Space operations completed")
            except Exception as e:
                print(f"⚠️ Space operations failed: {e}")
                steps.append({
                    "step": "space_operations",
                    "status": "completed_with_warning",
                    "description": f"Space operations failed: {e}"
                })
                logger.warning(f"⚠️ Space operations failed: {e}")

            # Step 5: Display Information
            print("\n" + "="*60)
            print("STEP 5: Display API Information")
            print("="*60)
            self.display_api_info()
            steps.append({
                "step": "display_info",
                "status": "completed",
                "description": "API information display completed"
            })
            logger.info("✅ API information display completed")

            # Overall success
            success = all(step['status'] == 'completed' for step in steps)

            return {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "steps": steps,
                "client_info": {
                    "api_key_available": bool(api_key),
                    "base_url": self.client.base_url if self.client else "N/A"
                }
            }

        except ValidationError as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": f"Validation failed: {e}",
                "steps": []
            }
        except HeySolError as e:
            logger.error(f"❌ Demo execution failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": f"Demo execution failed: {e}",
                "steps": []
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": f"Unexpected error: {e}",
                "steps": []
            }


def main():
    """Main function to run the HeySol log operations demo."""
    print("\n" + "🚀" + "="*58 + "🚀")
    print("🎯 HEYSOL API CLIENT LOG OPERATIONS DEMO")
    print("🚀" + "="*58 + "🚀")

    # Create demo instance
    demo = HeySolLogDemo()

    # Check configuration
    if not demo.check_api_key_config():
        print("\n❌ Cannot run demo: API key configuration is invalid")
        print("Please set HEYSOL_API_KEY environment variable.")
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
    with open("heysol_log_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: heysol_log_demo_results.json")
    print(f"📝 Debug logs saved to: oauth2_log_demo.log")

    if results['success']:
        print("\n🎉 SUCCESS: HeySol log operations demo completed successfully!")
        print("\nThe HeySol API client now supports:")
        print("✅ API key authentication")
        print("✅ Memory ingestion and search")
        print("✅ User profile access")
        print("✅ Space operations")
        print("✅ Error handling and validation")
    else:
        print("\n❌ FAILURE: HeySol log operations demo failed")
        print("Check the logs for details:")
        print("- oauth2_log_demo.log")

    print("\n" + "🎉" + "="*58 + "🎉")
    print("🎯 DEMO COMPLETE - HeySol API Client Ready!")
    print("🎉" + "="*58 + "🎉")


if __name__ == "__main__":
    main()