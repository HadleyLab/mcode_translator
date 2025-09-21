#!/usr/bin/env python3
"""
HeySol API Client - Log Operations CLI

Command-line interface for log operations with the HeySol API client using API key authentication.

Usage:
    python oauth2_log_cli.py [COMMAND] [OPTIONS]

Commands:
    auth        Perform API key authentication
    ingest      Ingest a log entry
    delete      Delete a log entry (requires OAuth2)
    list        List available logs
    info        Show API information
    demo        Run complete demo

Examples:
    python oauth2_log_cli.py auth
    python oauth2_log_cli.py ingest "My log message"
    python oauth2_log_cli.py delete --log-id "log-123"
    python oauth2_log_cli.py list --limit 10
    python oauth2_log_cli.py info
    python oauth2_log_cli.py demo

Environment Variables:
    HEYSOL_API_KEY - Your HeySol API key (required)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

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
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HeySolLogCLI:
    """Command-line interface for HeySol log operations using API key authentication."""

    def __init__(self):
        """Initialize the CLI."""
        self.client: Optional[HeySolClient] = None

    def initialize_client(self) -> bool:
        """Initialize HeySol client with API key."""
        try:
            # Get API key from environment
            api_key = os.getenv('HEYSOL_API_KEY')
            if not api_key:
                print("❌ HEYSOL_API_KEY environment variable not set")
                print("   Please set your API key: export HEYSOL_API_KEY='your-api-key'")
                return False

            # Create client
            self.client = HeySolClient(api_key=api_key)
            return True

        except Exception as e:
            print(f"❌ Failed to initialize client: {e}")
            return False

    def ensure_authenticated(self) -> bool:
        """Ensure client is initialized and ready."""
        if not self.client:
            if not self.initialize_client():
                return False
        return True

    def auth_command(self, args):
        """Handle auth command with API key authentication."""
        print("🔐 API Key Authentication")
        print("=" * 30)

        try:
            if self.ensure_authenticated():
                print("✅ API key authentication successful!")
                return self.info_command(args)
            else:
                print("❌ API key authentication failed!")
                return 1

        except Exception as e:
            print(f"❌ Unexpected error during authentication: {e}")
            return 1

    def ingest_command(self, args):
        """Handle ingest command with strict validation."""
        print("📝 Log Ingestion")
        print("=" * 20)

        try:
            if not self.ensure_authenticated():
                return 1

            if not args.message:
                raise ValidationError("Message is required for ingestion")

            # Use client directly
            result = self.client.ingest(
                message=args.message,
                space_id="oauth2_cli_demo"
            )

            print("✅ Log ingested successfully!")
            print(f"   Message: {args.message}")
            print(f"   Result: {result}")
            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            print("Usage: python oauth2_log_cli.py ingest \"Your message here\"")
            return 1
        except HeySolError as e:
            print(f"❌ Log ingestion failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error during ingestion: {e}")
            return 1

    def delete_command(self, args):
        """Handle delete command with strict error handling."""
        print("🗑️ Log Deletion")
        print("=" * 18)

        try:
            if not self.ensure_authenticated():
                return 1

            if not args.log_id:
                raise ValidationError("Log ID is required for deletion")

            # Use client directly (Note: May require OAuth2 authentication)
            result = self.client.delete_log_entry(log_id=args.log_id)

            print("✅ Log deleted successfully!")
            print(f"   Log ID: {args.log_id}")
            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            print("Usage: python oauth2_log_cli.py delete --log-id \"your-log-id\"")
            return 1
        except HeySolError as e:
            print(f"❌ Log deletion failed: {e}")
            print("   Note: Delete operations may require OAuth2 authentication")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error during deletion: {e}")
            return 1

    def list_command(self, args):
        """Handle list command with strict error handling."""
        print("📋 List Logs")
        print("=" * 14)

        try:
            if not self.ensure_authenticated():
                return 1

            # Use client directly
            logs = self.client.get_ingestion_logs(limit=args.limit or 10)

            if not logs:
                print("📭 No logs found")
                return 0

            print(f"📊 Found {len(logs)} logs:")
            print()

            for i, log in enumerate(logs, 1):
                log_id = log.get('id', 'N/A')
                status = log.get('status', 'N/A')
                created_at = log.get('created_at', 'N/A')
                message = log.get('message', 'N/A')

                print(f"{i}. ID: {log_id}")
                print(f"   Status: {status}")
                print(f"   Created: {created_at}")
                print(f"   Message: {message[:100]}{'...' if len(str(message)) > 100 else ''}")
                print()

            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            return 1
        except HeySolError as e:
            print(f"❌ Log retrieval failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error listing logs: {e}")
            return 1

    def info_command(self, args):
        """Handle info command with API key authentication."""
        print("🔐 API Key Information")
        print("=" * 22)

        try:
            if not self.ensure_authenticated():
                return 1

            # Get user profile
            print("\n👤 User Profile:")
            try:
                profile = self.client.get_user_profile()
                if profile:
                    print(f"   User ID: {profile.get('id', 'N/A')}")
                    print(f"   Username: {profile.get('username', 'N/A')}")
                    print(f"   Email: {profile.get('email', 'N/A')}")
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

            return 0

        except Exception as e:
            print(f"❌ Unexpected error getting info: {e}")
            return 1

    def demo_command(self, args):
        """Handle demo command with API key authentication."""
        print("🚀 Complete HeySol Log Operations Demo")
        print("=" * 40)

        try:
            if not self.ensure_authenticated():
                return 1

            # Run demo operations
            print("\n1. Testing user profile...")
            try:
                profile = self.client.get_user_profile()
                print("   ✅ User profile retrieved")
            except Exception as e:
                print(f"   ❌ User profile failed: {e}")

            print("\n2. Testing memory ingestion...")
            try:
                result = self.client.ingest("Demo message from CLI")
                print("   ✅ Memory ingestion successful")
            except Exception as e:
                print(f"   ❌ Memory ingestion failed: {e}")

            print("\n3. Testing memory search...")
            try:
                result = self.client.search("demo", limit=1)
                print("   ✅ Memory search successful")
            except Exception as e:
                print(f"   ❌ Memory search failed: {e}")

            print("\n4. Testing spaces...")
            try:
                spaces = self.client.get_spaces()
                print("   ✅ Spaces retrieved successfully")
            except Exception as e:
                print(f"   ❌ Spaces retrieval failed: {e}")

            print("\n🎉 Demo completed!")
            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            return 1
        except HeySolError as e:
            print(f"❌ Demo execution failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error running demo: {e}")
            return 1


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="HeySol API Client - Log Operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oauth2_log_cli.py auth
  python oauth2_log_cli.py ingest "My log message"
  python oauth2_log_cli.py delete --log-id "log-123"
  python oauth2_log_cli.py list --limit 10
  python oauth2_log_cli.py info
  python oauth2_log_cli.py demo

Environment Variables:
  HEYSOL_API_KEY - Your HeySol API key (required)
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Auth command
    subparsers.add_parser('auth', help='Perform OAuth2 authentication')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a log entry')
    ingest_parser.add_argument('message', help='Log message to ingest')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a log entry')
    delete_parser.add_argument('--log-id', required=True, help='Log ID to delete')

    # List command
    list_parser = subparsers.add_parser('list', help='List available logs')
    list_parser.add_argument('--limit', type=int, default=10, help='Maximum number of logs to list')

    # Info command
    subparsers.add_parser('info', help='Show OAuth2 information')

    # Demo command
    subparsers.add_parser('demo', help='Run complete demo')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Create CLI instance and run command
    cli = HeySolLogCLI()

    try:
        return getattr(cli, f"{args.command}_command")(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())