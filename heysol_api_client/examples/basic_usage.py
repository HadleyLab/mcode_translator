#!/usr/bin/env python3
"""
Basic usage example for the HeySol API client.

This script demonstrates basic functionality of the HeySol API client including
authentication, memory ingestion, and search operations.
"""

import asyncio
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
    """Demonstrate basic HeySol API client functionality."""
    print("HeySol API Client - Basic Usage Example")
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

        # Get user profile
        print("\n👤 Getting user profile...")
        try:
            profile = client.get_user_profile()
            print(f"✅ User profile retrieved: {profile.get('name', 'Unknown')}")
        except HeySolError as e:
            print(f"⚠️  Could not retrieve user profile: {e}")

        # Get available spaces
        print("\n📂 Getting available spaces...")
        try:
            spaces = client.get_spaces()
            print(f"✅ Found {len(spaces)} spaces:")
            # Handle different response formats
            if isinstance(spaces, list):
                for space in spaces[:3]:  # Show first 3 spaces
                    if isinstance(space, dict):
                        print(f"   - {space.get('name', 'Unknown')} (ID: {space.get('id', 'Unknown')})")
                    else:
                        print(f"   - {space}")
            else:
                print(f"   Response type: {type(spaces)}")
                print(f"   Response: {spaces}")
        except HeySolError as e:
            print(f"⚠️  Could not retrieve spaces: {e}")

        # Create or get a space for our examples
        print("\n🏗️  Getting or creating example space...")
        space_id = client.get_or_create_space("Python Examples", "Space for Python client examples")
        print(f"✅ Using space ID: {space_id}")

        # Ingest some sample data
        print("\n📝 Ingesting sample data...")
        sample_messages = [
            "The patient presented with advanced breast cancer and required immediate treatment.",
            "Clinical trial NCT12345 shows promising results for targeted therapy in oncology patients.",
            "mCODE standard implementation requires structured data collection for better outcomes.",
        ]

        for i, message in enumerate(sample_messages, 1):
            try:
                result = client.ingest(
                    message=message,
                    space_id=space_id,
                    source="python-example",
                    tags=["example", "clinical", "oncology"]
                )
                print(f"✅ Ingested message {i}/3")
            except HeySolError as e:
                print(f"⚠️  Failed to ingest message {i}: {e}")

        # Search for data
        print("\n🔍 Searching for 'breast cancer'...")
        try:
            search_results = client.search(
                query="breast cancer",
                space_id=space_id,
                limit=5
            )

            episodes = search_results.get("episodes", [])
            facts = search_results.get("facts", [])

            print(f"✅ Search completed: {len(episodes)} episodes, {len(facts)} facts")

            # Display some search results
            for i, episode in enumerate(episodes[:2], 1):
                print(f"   {i}. {episode.get('content', 'No content')[:100]}...")

        except HeySolError as e:
            print(f"⚠️  Search failed: {e}")

        # Demonstrate error handling
        print("\n🛡️  Testing error handling...")
        try:
            client.search(query="")  # Empty query should raise ValidationError
        except HeySolError as e:
            print(f"✅ Error handling works: {type(e).__name__}: {e}")

        # Clean up
        client.close()
        print("\n🎉 Example completed successfully!")

    except HeySolError as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())