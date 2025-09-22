#!/usr/bin/env python3
"""
CLI-based source filtering demonstration.

This example shows how to use the HeySol CLI for source filtering operations.
Run this script to see CLI commands in action.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_cli_command(command: str, description: str) -> bool:
    """Run a CLI command and display results."""
    print(f"\n🔧 {description}")
    print(f"Command: python -m cli {command}")
    print("-" * 50)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "cli"] + command.split(),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.stdout:
            print(result.stdout[:500])  # Limit output length
            if len(result.stdout) > 500:
                print("... (output truncated)")

        if result.stderr:
            print(f"Error: {result.stderr}")

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Failed to run command: {e}")
        return False


def main():
    """Demonstrate CLI source filtering operations."""
    print("🖥️  HeySol CLI - Source Filtering Demo")
    print("=" * 60)
    print("This demo shows CLI commands for source filtering operations.")
    print("Make sure you have set HEYSOL_API_KEY environment variable.\n")

    # Check if API key is available
    if not os.getenv("HEYSOL_API_KEY") and not os.getenv("COREAI_API_KEY"):
        print("❌ Error: Please set HEYSOL_API_KEY or COREAI_API_KEY environment variable")
        return 1

    success_count = 0
    total_commands = 0

    # Basic operations
    print("📋 BASIC OPERATIONS")
    print("=" * 30)

    # 1. Get user profile
    total_commands += 1
    if run_cli_command("profile", "Get user profile"):
        success_count += 1

    # 2. List spaces
    total_commands += 1
    if run_cli_command("spaces list", "List available spaces"):
        success_count += 1

    # 3. Create test space
    total_commands += 1
    if run_cli_command(
        'spaces create "CLI Source Demo" --description "Space for CLI source filtering demo"',
        "Create test space"
    ):
        success_count += 1

    # Source filtering operations
    print("\n🔍 SOURCE FILTERING OPERATIONS")
    print("=" * 40)

    # 4. Ingest data with different sources
    print("\n📝 INGESTING DATA WITH DIFFERENT SOURCES")
    print("-" * 45)

    sources_and_data = [
        ('api', 'API endpoint /users called successfully'),
        ('webhook', 'GitHub webhook received for repository update'),
        ('manual', 'User manually entered patient data'),
        ('integration', 'External system sync completed'),
        ('api', 'Authentication service health check passed'),
    ]

    for source, message in sources_and_data:
        total_commands += 1
        if run_cli_command(
            f'ingest "{message}" --source {source} --space-id "CLI Source Demo"',
            f"Ingest data with source '{source}'"
        ):
            success_count += 1

    # 5. Search with source filtering
    print("\n🔍 SEARCHING WITH SOURCE FILTERING")
    print("-" * 40)

    total_commands += 1
    if run_cli_command(
        'search "*" --space-id "CLI Source Demo" --source-filter api --limit 3',
        "Search with API source filter"
    ):
        success_count += 1

    # 6. Get logs by source using MCP
    print("\n📋 MCP-BASED LOG OPERATIONS")
    print("-" * 35)

    total_commands += 1
    if run_cli_command(
        'logs get-by-source api --space-id "CLI Source Demo" --limit 3',
        "Get logs by API source using MCP"
    ):
        success_count += 1

    total_commands += 1
    if run_cli_command(
        'logs get-by-source manual --space-id "CLI Source Demo" --limit 2',
        "Get logs by manual source using MCP"
    ):
        success_count += 1

    # 7. List MCP tools
    print("\n🛠️  MCP TOOLS")
    print("-" * 15)

    total_commands += 1
    if run_cli_command("tools", "List available MCP tools"):
        success_count += 1

    # 8. Show log deletion command (without executing)
    print("\n🗑️  LOG DELETION (PREVIEW)")
    print("-" * 30)
    print("💡 To delete logs by source, use:")
    print("   python -m cli logs delete-by-source api --space-id 'CLI Source Demo' --confirm")
    print("   python -m cli logs delete-by-source manual --space-id 'CLI Source Demo' --confirm")

    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    print(f"Commands executed: {total_commands}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_commands - success_count}")

    if success_count == total_commands:
        print("🎉 All CLI operations completed successfully!")
        return 0
    else:
        print("⚠️  Some operations failed. Check your API key and network connection.")
        return 1


if __name__ == "__main__":
    sys.exit(main())