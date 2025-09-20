#!/usr/bin/env python3
"""
Log management example for the HeySol API client.

This script demonstrates log management functionality including
retrieving logs, getting specific logs, and deleting log entries.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the Python path to import the client
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol import HeySolClient, HeySolConfig, HeySolError


def main():
    """Demonstrate HeySol API log management functionality."""
    print("HeySol API Client - Log Management Example")
    print("=" * 50)

    # Load configuration from environment variables
    config = HeySolConfig.from_env()

    # Override with COREAI_API_KEY from .env file if not set
    if not config.api_key:
        config.api_key = os.getenv("COREAI_API_KEY")
        if config.api_key:
            print("✅ Using COREAI_API_KEY from .env file")
        else:
            print("❌ COREAI_API_KEY not found in .env file")
            return

    try:
        # Initialize the client
        print("🔧 Initializing HeySol client...")
        client = HeySolClient(config=config)
        print("✅ Client initialized successfully")

        # Get ingestion logs
        print("\n📋 Getting ingestion logs...")
        try:
            logs = client.get_ingestion_logs(limit=5)
            print(f"✅ Found {len(logs)} recent logs:")

            for i, log in enumerate(logs[:3], 1):  # Show first 3 logs
                print(f"   {i}. ID: {log.get('id', 'Unknown')}")
                print(f"      Status: {log.get('status', 'Unknown')}")
                print(f"      Message: {log.get('message', 'Unknown')[:100]}...")
                print(f"      Created: {log.get('created_at', 'Unknown')}")
                print()

        except HeySolError as e:
            print(f"⚠️  Could not retrieve logs: {e}")

        # Demonstrate log retrieval with filters
        print("\n🔍 Getting logs with filters...")
        try:
            # Try to get logs for a specific space (if we have one)
            spaces = client.get_spaces()
            if spaces:
                space_id = spaces[0].get('id')
                filtered_logs = client.get_ingestion_logs(
                    space_id=space_id,
                    limit=3,
                    status="success"
                )
                print(f"✅ Found {len(filtered_logs)} logs for space {space_id}")
            else:
                print("⚠️  No spaces available for filtering")

        except HeySolError as e:
            print(f"⚠️  Could not retrieve filtered logs: {e}")

        # Demonstrate specific log retrieval
        print("\n📄 Getting specific log (placeholder)...")
        try:
            # This would work with a real log ID
            # log = client.get_specific_log("log-id-here")
            # print(f"✅ Retrieved log: {log.get('id')}")
            print("⚠️  Skipping specific log retrieval (requires real log ID)")
        except HeySolError as e:
            print(f"⚠️  Could not retrieve specific log: {e}")

        # Demonstrate log deletion
        print("\n🗑️  Log deletion (placeholder)...")
        try:
            # This would work with a real log ID
            # result = client.delete_log_entry("log-id-here")
            # print(f"✅ Deleted log: {result.get('id')}")
            print("⚠️  Skipping log deletion (requires real log ID)")
        except HeySolError as e:
            print(f"⚠️  Could not delete log: {e}")

        # Clean up
        client.close()
        print("\n🎉 Log management example completed!")

    except HeySolError as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())