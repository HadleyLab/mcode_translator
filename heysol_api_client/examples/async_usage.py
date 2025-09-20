#!/usr/bin/env python3
"""
Async usage example for the HeySol API client.

This script demonstrates async functionality of the HeySol API client including
concurrent operations, bulk ingestion, and async search.
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

from heysol import AsyncHeySolClient, HeySolConfig, HeySolError


async def bulk_ingest_example(client: AsyncHeySolClient, space_id: str) -> None:
    """Demonstrate bulk ingestion of data."""
    print("\n📦 Performing bulk ingestion...")

    # Sample clinical trial data
    clinical_data = [
        {
            "message": "Phase III trial of new immunotherapy shows 45% response rate in advanced melanoma patients",
            "tags": ["clinical-trial", "immunotherapy", "melanoma", "phase-3"]
        },
        {
            "message": "mCODE implementation study demonstrates 30% improvement in data quality metrics",
            "tags": ["mcode", "data-quality", "implementation"]
        },
        {
            "message": "Patient-reported outcomes show significant quality of life improvements with targeted therapy",
            "tags": ["patient-reported", "quality-of-life", "targeted-therapy"]
        },
        {
            "message": "Biomarker analysis reveals predictive signature for treatment response in breast cancer",
            "tags": ["biomarkers", "breast-cancer", "predictive", "analysis"]
        },
        {
            "message": "Real-world evidence study validates clinical trial findings for checkpoint inhibitors",
            "tags": ["real-world-evidence", "checkpoint-inhibitors", "validation"]
        }
    ]

    # Ingest data concurrently
    tasks = []
    for i, data in enumerate(clinical_data, 1):
        task = client.ingest(
            message=data["message"],
            space_id=space_id,
            source="async-example",
            tags=data["tags"]
        )
        tasks.append(task)

    # Wait for all ingestions to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"⚠️  Ingestion {i+1} failed: {result}")
        else:
            print(f"✅ Ingestion {i+1} completed")
            success_count += 1

    print(f"📊 Bulk ingestion completed: {success_count}/{len(clinical_data)} successful")


async def concurrent_search_example(client: AsyncHeySolClient, space_id: str) -> None:
    """Demonstrate concurrent search operations."""
    print("\n🔍 Performing concurrent searches...")

    search_queries = [
        "immunotherapy",
        "breast cancer",
        "clinical trial",
        "patient outcomes",
        "biomarkers"
    ]

    # Perform searches concurrently
    tasks = []
    for query in search_queries:
        task = client.search(
            query=query,
            space_id=space_id,
            limit=3
        )
        tasks.append((query, task))

    # Wait for all searches to complete
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    # Display results
    for (query, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"⚠️  Search for '{query}' failed: {result}")
        else:
            episodes = result.get("episodes", [])
            print(f"✅ Search for '{query}': {len(episodes)} results")


async def main():
    """Demonstrate async HeySol API client functionality."""
    print("HeySol API Client - Async Usage Example")
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

    # Enable async configuration
    config.async_enabled = True
    config.max_async_workers = 5

    async with AsyncHeySolClient(config=config) as client:
        print("🔧 Async client initialized successfully")

        try:
            # Get user profile
            print("\n👤 Getting user profile...")
            profile = await client.get_user_profile()
            print(f"✅ User profile retrieved: {profile.get('name', 'Unknown')}")

            # Get available spaces
            print("\n📂 Getting available spaces...")
            spaces = await client.get_spaces()
            print(f"✅ Found {len(spaces)} spaces")

            # Create or get a space for our examples
            print("\n🏗️  Getting or creating example space...")
            space_id = await client.get_or_create_space(
                "Async Python Examples",
                "Space for async Python client examples"
            )
            print(f"✅ Using space ID: {space_id}")

            # Demonstrate concurrent operations
            await bulk_ingest_example(client, space_id)

            # Wait a moment for data to be processed
            print("\n⏳ Waiting for data processing...")
            await asyncio.sleep(2)

            # Demonstrate concurrent searches
            await concurrent_search_example(client, space_id)

            print("\n🎉 Async example completed successfully!")

        except HeySolError as e:
            print(f"❌ Error: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))