#!/usr/bin/env python3
"""
HeySol API Client - Log Management Examples

Demonstrates how to manage ingestion logs including:
- Listing logs with filtering
- Getting specific log details
- Deleting log entries
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Add parent directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol.client import HeySolClient
from heysol.exceptions import HeySolError, ValidationError


class LogManager:
    """Manages HeySol ingestion logs."""

    def __init__(self, api_key: str = None):
        """Initialize the log manager."""
        self.api_key = api_key or os.getenv('HEYSOL_API_KEY')
        if not self.api_key:
            raise ValueError("HEYSOL_API_KEY environment variable is required")

        self.client = HeySolClient(api_key=self.api_key)

    def list_logs(self, space_id: str = None, limit: int = 50, status: str = None,
                  start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """List ingestion logs with optional filtering."""
        print(f"📋 Listing logs (limit: {limit})...")

        try:
            logs = self.client.get_ingestion_logs(
                space_id=space_id,
                limit=limit,
                status=status,
                start_date=start_date,
                end_date=end_date
            )

            print(f"✅ Found {len(logs)} logs")
            for log in logs[:5]:  # Show first 5 logs
                print(f"  - ID: {log.get('id')}, Status: {log.get('status')}")
            if len(logs) > 5:
                print(f"  ... and {len(logs) - 5} more")

            return logs

        except HeySolError as e:
            print(f"❌ Error listing logs: {e}")
            return []

    def get_log_details(self, log_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific log."""
        print(f"🔍 Getting details for log: {log_id}")

        try:
            log_details = self.client.get_specific_log(log_id)
            print("✅ Log details retrieved:")
            print(f"  - Status: {log_details.get('status')}")
            print(f"  - Created: {log_details.get('created_at')}")
            if 'completed_at' in log_details:
                print(f"  - Completed: {log_details.get('completed_at')}")
            return log_details

        except HeySolError as e:
            print(f"❌ Error getting log details: {e}")
            return {}

    def delete_log(self, log_id: str, confirm: bool = True) -> bool:
        """Delete a log entry."""
        print(f"🗑️  Deleting log: {log_id}")

        if not confirm:
            print("❌ Deletion cancelled - confirmation required")
            return False

        try:
            result = self.client.delete_log_entry(log_id)
            print(f"✅ Log deleted successfully: {result}")
            return True

        except ValidationError as e:
            print(f"❌ Validation error: {e}")
            return False
        except HeySolError as e:
            print(f"❌ Error deleting log: {e}")
            return False

    def cleanup_old_logs(self, days_old: int = 30, space_id: str = None,
                         dry_run: bool = True) -> int:
        """Clean up logs older than specified days."""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        print(f"🧹 Cleaning up logs older than {days_old} days (cutoff: {cutoff_date})")

        try:
            logs = self.client.get_ingestion_logs(
                space_id=space_id,
                limit=1000,  # Get many logs to clean up
                start_date="2020-01-01T00:00:00Z",  # Far back to get all old logs
                end_date=cutoff_date
            )

            old_logs = [log for log in logs if log.get('status') in ['completed', 'failed']]
            print(f"📋 Found {len(old_logs)} old logs to potentially delete")

            if dry_run:
                print("🔍 DRY RUN - No actual deletions performed")
                for log in old_logs[:10]:  # Show first 10
                    print(f"  - Would delete: {log.get('id')} (Status: {log.get('status')})")
                if len(old_logs) > 10:
                    print(f"  ... and {len(old_logs) - 10} more")
                return len(old_logs)

            # Actually delete the logs
            deleted_count = 0
            for log in old_logs:
                if self.delete_log(log.get('id'), confirm=False):
                    deleted_count += 1

            print(f"✅ Cleanup complete: {deleted_count}/{len(old_logs)} logs deleted")
            return deleted_count

        except HeySolError as e:
            print(f"❌ Error during cleanup: {e}")
            return 0


def main():
    """Main function demonstrating log management."""
    print("🚀 HeySol API Client - Log Management Demo")
    print("=" * 50)

    # Initialize log manager
    try:
        manager = LogManager()
        print("✅ Log manager initialized successfully")
    except ValueError as e:
        print(f"❌ Failed to initialize: {e}")
        return 1

    # Example 1: List recent logs
    print("\n📋 Example 1: List recent logs")
    logs = manager.list_logs(limit=10)

    # Example 2: Get details of first log (if any)
    if logs:
        print("\n🔍 Example 2: Get log details")
        log_details = manager.get_log_details(logs[0].get('id'))

    # Example 3: Demonstrate log deletion
    print("\n🗑️  Example 3: Log deletion")
    if logs:
        # Create a test log first (if possible)
        try:
            test_data = {"text": "Test log for deletion demo", "source": "demo"}
            result = manager.client.add_data_to_ingestion_queue(test_data)
            print(f"✅ Created test log: {result.get('id')}")

            # Delete the test log
            if 'id' in result:
                manager.delete_log(result['id'])

        except HeySolError as e:
            print(f"❌ Could not create/delete test log: {e}")

    # Example 4: Cleanup old logs (dry run)
    print("\n🧹 Example 4: Cleanup old logs (dry run)")
    old_count = manager.cleanup_old_logs(days_old=7, dry_run=True)

    print("\n🎉 Log management demo complete!")
    return 0


if __name__ == "__main__":
    exit(main())