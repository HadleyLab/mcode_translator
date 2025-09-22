#!/usr/bin/env python3
"""
Comprehensive source filtering demonstration for HeySol API client.

This example demonstrates MCP-based source filtering operations including:
- Ingesting data with specific sources
- Searching with source filters
- Getting logs by source
- Deleting logs by source
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol import HeySolClient, HeySolConfig, HeySolError


def main():
    """Demonstrate comprehensive source filtering operations."""
    print("🔍 HeySol API Client - Source Filtering Demo")
    print("=" * 60)

    # Load configuration
    config = HeySolConfig.from_env()

    if not config.api_key:
        print("❌ Error: API key required. Set HEYSOL_API_KEY or COREAI_API_KEY environment variable")
        return 1

    try:
        # Initialize client
        print("🔧 Initializing HeySol client...")
        client = HeySolClient(config=config)
        print("✅ Client initialized successfully")

        # Get or create test space
        print("\n📂 Getting or creating test space...")
        space_id = client.get_or_create_space(
            "Source Filtering Demo",
            "Space for demonstrating source filtering operations"
        )
        print(f"✅ Using space: {space_id}")

        # Demonstrate ingesting data with different sources
        print("\n📝 Ingesting data with different sources...")

        sources_and_messages = [
            ("api", "API request processed successfully at 2025-01-15T10:30:00Z"),
            ("api", "User authentication failed - invalid credentials"),
            ("webhook", "Webhook received from external service"),
            ("manual", "Manual data entry by user"),
            ("api", "Database backup completed successfully"),
            ("integration", "Third-party integration sync completed"),
            ("manual", "User updated profile information"),
        ]

        ingested_items = []

        for source, message in sources_and_messages:
            try:
                result = client.ingest(
                    message=message,
                    space_id=space_id,
                    source=source
                )
                ingested_items.append({
                    "source": source,
                    "message": message,
                    "result": result
                })
                print(f"✅ Ingested: {source} - {message[:50]}...")
            except HeySolError as e:
                print(f"⚠️  Failed to ingest {source}: {e}")

        # Demonstrate searching with source filtering
        print("\n🔍 Searching with source filtering...")

        search_queries = [
            ("*", "All sources"),
            ("api", "API source only"),
            ("manual", "Manual source only"),
        ]

        for query, description in search_queries:
            try:
                print(f"\n📊 Searching for '{query}' ({description}):")
                result = client.search(
                    query=query,
                    space_ids=[space_id],
                    limit=5
                )

                # Apply source filter if specified
                if query != "*":
                    filtered_episodes = [
                        ep for ep in result.get("episodes", [])
                        if ep.get("source") == query
                    ]
                    result["episodes"] = filtered_episodes
                    result["total"] = len(filtered_episodes)

                episodes = result.get("episodes", [])
                print(f"✅ Found {len(episodes)} episodes")

                for i, episode in enumerate(episodes[:3], 1):
                    source = episode.get("source", "unknown")
                    content = episode.get("content", "")[:100]
                    print(f"   {i}. [{source}] {content}...")

            except HeySolError as e:
                print(f"⚠️  Search failed: {e}")

        # Demonstrate MCP-based source filtering operations
        print("\n🔧 MCP-based source filtering operations...")

        # Get logs by source using MCP
        print("\n📋 Getting logs by source using MCP:")
        for source in ["api", "manual", "webhook"]:
            try:
                logs_result = client.get_logs_by_source(
                    source=source,
                    space_id=space_id,
                    limit=3
                )

                logs = logs_result.get("logs", [])
                print(f"✅ Source '{source}': {logs_result.get('total_count', 0)} logs found")

                for log in logs:
                    content = log.get("content", "")[:80]
                    print(f"   - {content}...")

            except HeySolError as e:
                print(f"⚠️  Failed to get logs for source '{source}': {e}")

        # Demonstrate log deletion by source (preview only)
        print("\n🗑️  Log deletion by source (preview - requires --confirm):")

        for source in ["api", "webhook"]:
            try:
                # First show what would be deleted
                logs_result = client.get_logs_by_source(
                    source=source,
                    space_id=space_id,
                    limit=5
                )

                count = logs_result.get("total_count", 0)
                print(f"📊 Source '{source}': {count} logs would be deleted")

                if count > 0:
                    print("   💡 Use: python -m cli logs delete-by-source {source} --space-id {space_id} --confirm")

            except HeySolError as e:
                print(f"⚠️  Failed to preview deletion for source '{source}': {e}")

        # Demonstrate comprehensive source analysis
        print("\n📈 Comprehensive source analysis:")

        try:
            # Get all logs to analyze sources
            all_result = client.search(
                query="*",
                space_ids=[space_id],
                limit=100
            )

            episodes = all_result.get("episodes", [])
            source_counts = {}

            for episode in episodes:
                source = episode.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1

            print("📊 Source distribution:")
            for source, count in sorted(source_counts.items()):
                print(f"   {source}: {count} items")

        except HeySolError as e:
            print(f"⚠️  Failed to analyze source distribution: {e}")

        # Clean up
        client.close()
        print("\n🎉 Source filtering demo completed successfully!")

        return 0

    except HeySolError as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())