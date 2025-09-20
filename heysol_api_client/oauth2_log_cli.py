#!/usr/bin/env python3
"""
HeySol API Client - OAuth2 Log Operations CLI

Command-line interface for OAuth2 log operations with the HeySol API client.

Usage:
    python oauth2_log_cli.py [COMMAND] [OPTIONS]

Commands:
    auth        Perform OAuth2 authentication
    ingest      Ingest a log entry
    delete      Delete a log entry
    list        List available logs
    info        Show OAuth2 information
    demo        Run complete demo

Examples:
    python oauth2_log_cli.py auth
    python oauth2_log_cli.py ingest "My log message"
    python oauth2_log_cli.py delete --log-id "log-123"
    python oauth2_log_cli.py list --limit 10
    python oauth2_log_cli.py info
    python oauth2_log_cli.py demo
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import centralized OAuth2 utilities
from heysol.oauth2_utils import (
    OAuth2ClientManager,
    OAuth2LogOperations,
    validate_oauth2_setup,
    AuthenticationError,
    ValidationError,
    HeySolError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OAuth2LogCLI:
    """Command-line interface for OAuth2 log operations."""

    def __init__(self):
        """Initialize the CLI."""
        self.client_manager: Optional[OAuth2ClientManager] = None
        self.log_ops: Optional[OAuth2LogOperations] = None

    def initialize_oauth2(self) -> bool:
        """Initialize OAuth2 components with strict error handling."""
        try:
            # Validate OAuth2 setup first
            validate_oauth2_setup()

            # Create client manager and log operations
            self.client_manager = OAuth2ClientManager()
            self.log_ops = OAuth2LogOperations(self.client_manager)

            return True

        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to initialize OAuth2: {e}")
            return False

    def ensure_authenticated(self) -> bool:
        """Ensure OAuth2 authentication is valid."""
        if not self.client_manager:
            if not self.initialize_oauth2():
                return False

        try:
            # This will perform authentication if needed
            self.client_manager.ensure_authenticated()
            return True

        except AuthenticationError as e:
            print(f"❌ Authentication failed: {e}")
            return False

    def auth_command(self, args):
        """Handle auth command with strict error handling."""
        print("🔐 OAuth2 Authentication")
        print("=" * 30)

        try:
            if self.ensure_authenticated():
                print("✅ OAuth2 authentication successful!")
                return self.info_command(args)
            else:
                print("❌ OAuth2 authentication failed!")
                return 1

        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
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

            # Use centralized log operations
            result = self.log_ops.ingest_log(
                message=args.message,
                space_name="oauth2_cli_demo"
            )

            log_id = result["log_id"]
            print("✅ Log ingested successfully!")
            print(f"   Log ID: {log_id}")
            print(f"   Message: {args.message}")
            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            print("Usage: python oauth2_log_cli.py ingest \"Your message here\"")
            return 1
        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
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

            # Use centralized log operations
            result = self.log_ops.delete_log(log_id=args.log_id)

            print("✅ Log deleted successfully!")
            print(f"   Log ID: {args.log_id}")
            return 0

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            print("Usage: python oauth2_log_cli.py delete --log-id \"your-log-id\"")
            return 1
        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return 1
        except HeySolError as e:
            print(f"❌ Log deletion failed: {e}")
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

            # Use centralized log operations
            logs = self.log_ops.get_logs(limit=args.limit or 10)

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
        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return 1
        except HeySolError as e:
            print(f"❌ Log retrieval failed: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error listing logs: {e}")
            return 1

    def info_command(self, args):
        """Handle info command with strict error handling."""
        print("🔐 OAuth2 Information")
        print("=" * 22)

        try:
            if not self.ensure_authenticated():
                return 1

            client = self.client_manager.get_client()

            # Get user info
            print("\n👤 User Information:")
            try:
                user_info = client.get_oauth2_user_info()
                if user_info:
                    print(f"   Name: {user_info.get('name', 'N/A')}")
                    print(f"   Email: {user_info.get('email', 'N/A')}")
                    print(f"   User ID: {user_info.get('id', 'N/A')}")
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
                else:
                    print("   ❌ Could not retrieve HeySol profile")
            except AuthenticationError as e:
                print(f"   ❌ Authentication error: {e}")
            except Exception as e:
                print(f"   ❌ Error getting profile: {e}")

            return 0

        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error getting OAuth2 info: {e}")
            return 1

    def demo_command(self, args):
        """Handle demo command with strict error handling."""
        print("🚀 Complete OAuth2 Log Operations Demo")
        print("=" * 40)

        try:
            # Use centralized demo runner
            from heysol.oauth2_utils import create_oauth2_demo_runner

            demo_runner = create_oauth2_demo_runner()
            results = demo_runner.run_complete_demo()

            if results['success']:
                print("\n🎉 Demo completed successfully!")
                return 0
            else:
                print(f"\n❌ Demo failed: {results.get('error', 'Unknown error')}")
                return 1

        except AuthenticationError as e:
            print(f"❌ Authentication error: {e}")
            return 1
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
        description="HeySol API Client - OAuth2 Log Operations CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oauth2_log_cli.py auth
  python oauth2_log_cli.py ingest "My log message"
  python oauth2_log_cli.py delete --log-id "log-123"
  python oauth2_log_cli.py list --limit 10
  python oauth2_log_cli.py info
  python oauth2_log_cli.py demo
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
    cli = OAuth2LogCLI()

    try:
        return getattr(cli, f"{args.command}_command")(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())