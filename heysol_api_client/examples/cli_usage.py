#!/usr/bin/env python3
"""
CLI Usage Example for the HeySol API client.

This script demonstrates how to use the HeySol API client via command line interface.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add the parent directory to the Python path to import the client
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol import HeySolClient


def run_cli_command(command_args):
    """Run a CLI command and return the result."""
    try:
        # Build the command
        cmd = [sys.executable, "-m", "heysol_api_client.cli"] + command_args

        # Set environment variables for API key
        env = os.environ.copy()
        env["HEYSOL_API_KEY"] = "demo-api-key"  # Replace with actual key

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent.parent / "heysol_api_client"
        )

        print(f"Command: {' '.join(cmd)}")
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        if result.stderr:
            print(f"Error:\n{result.stderr}")
        print("-" * 50)

        return result.returncode == 0

    except Exception as e:
        print(f"Error running command: {e}")
        return False


def main():
    """Demonstrate CLI usage."""
    print("HeySol API Client - CLI Usage Example")
    print("=" * 50)

    # Note: These commands will fail without a valid API key
    # They are shown for demonstration purposes

    print("\n1. Get user profile:")
    run_cli_command(["profile"])

    print("\n2. List spaces:")
    run_cli_command(["spaces", "list"])

    print("\n3. Create a space:")
    run_cli_command(["spaces", "create", "CLI Demo Space", "--description", "Space created via CLI"])

    print("\n4. Ingest data:")
    run_cli_command(["ingest", "Sample data from CLI", "--space-id", "demo-space-123"])

    print("\n5. Search memories:")
    run_cli_command(["search", "clinical trial", "--limit", "5"])

    print("\n6. Get logs:")
    run_cli_command(["logs", "--limit", "10"])

    print("\n7. Show help:")
    run_cli_command(["--help"])

    print("\nNote: Replace 'demo-api-key' with your actual HeySol API key to run these commands successfully.")
    print("Set the HEYSOL_API_KEY environment variable or use --api-key option.")


if __name__ == "__main__":
    main()