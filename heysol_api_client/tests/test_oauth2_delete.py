"""
Test OAuth2 authentication and delete functions for HeySol API client.

This script tests the OAuth2 implementation and verifies that delete functions
work properly with OAuth2 authentication.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.oauth2 import OAuth2ClientCredentialsAuthenticator, OAuth2Error
from heysol.exceptions import HeySolError, AuthenticationError

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("oauth2_test.log")
    ]
)
logger = logging.getLogger(__name__)


class OAuth2DeleteTester:
    """Test OAuth2 authentication and delete functions."""

    def __init__(self):
        """Initialize the OAuth2 tester."""
        self.client: Optional[HeySolClient] = None
        self.oauth2_auth: Optional[OAuth2ClientCredentialsAuthenticator] = None
        self.test_results = []

    def setup_oauth2(self) -> bool:
        """Setup OAuth2 client credentials authentication."""
        try:
            logger.info("Setting up OAuth2 client credentials authentication...")

            # Get OAuth2 credentials from environment
            client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
            client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")

            if not client_id or not client_secret:
                logger.error("OAuth2 credentials not found in environment variables")
                logger.info("Please set COREAI_OAUTH2_CLIENT_ID and COREAI_OAUTH2_CLIENT_SECRET")
                return False

            logger.info(f"Using OAuth2 client ID: {client_id[:10]}...")
            logger.info("OAuth2 client secret: [REDACTED]")

            # Create OAuth2 authenticator
            config = HeySolConfig(
                oauth2_client_id=client_id,
                oauth2_client_secret=client_secret,
                log_level="DEBUG",
                log_to_file=True,
                log_file_path="oauth2_debug.log"
            )

            self.oauth2_auth = OAuth2ClientCredentialsAuthenticator(config)
            self.oauth2_auth.authenticate()

            logger.info("OAuth2 authentication successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup OAuth2: {e}")
            return False

    def setup_client_with_oauth2(self) -> bool:
        """Setup HeySol client with OAuth2 authentication."""
        try:
            if not self.oauth2_auth:
                return False

            logger.info("Setting up HeySol client with OAuth2...")

            # Create client with OAuth2
            config = HeySolConfig(
                oauth2_client_id=self.oauth2_auth.config.oauth2_client_id,
                oauth2_client_secret=self.oauth2_auth.config.oauth2_client_secret,
                log_level="DEBUG",
                log_to_file=True,
                log_file_path="oauth2_client_debug.log"
            )

            self.client = HeySolClient(
                config=config,
                use_oauth2=True
            )

            logger.info("HeySol client with OAuth2 setup successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup client with OAuth2: {e}")
            return False

    def test_oauth2_authentication(self) -> Dict[str, Any]:
        """Test OAuth2 authentication."""
        logger.info("Testing OAuth2 authentication...")
        result = {
            "test_name": "oauth2_authentication",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.oauth2_auth:
                result["error"] = "OAuth2 authenticator not initialized"
                return result

            # Test token validity
            token = self.oauth2_auth.get_valid_access_token()
            result["status"] = "passed"
            result["details"] = {
                "token_length": len(token),
                "token_prefix": token[:20] + "..."
            }
            logger.info("OAuth2 authentication test passed")

        except Exception as e:
            result["error"] = f"OAuth2 authentication failed: {e}"
            logger.error(f"OAuth2 authentication test failed: {e}")

        return result

    def test_oauth2_token_introspection(self) -> Dict[str, Any]:
        """Test OAuth2 token introspection."""
        logger.info("Testing OAuth2 token introspection...")
        result = {
            "test_name": "oauth2_token_introspection",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.oauth2_auth:
                result["error"] = "OAuth2 authenticator not initialized"
                return result

            # Introspect the current token
            introspection_data = self.oauth2_auth.introspect_token()
            result["status"] = "passed"
            result["details"] = {
                "introspection_keys": list(introspection_data.keys()),
                "active": introspection_data.get("active", False),
                "client_id": introspection_data.get("client_id"),
                "scope": introspection_data.get("scope")
            }
            logger.info("OAuth2 token introspection test passed")

        except Exception as e:
            result["error"] = f"OAuth2 token introspection failed: {e}"
            logger.error(f"OAuth2 token introspection test failed: {e}")

        return result

    def test_delete_space_with_oauth2(self) -> Dict[str, Any]:
        """Test deleting a space with OAuth2 authentication."""
        logger.info("Testing delete space with OAuth2...")
        result = {
            "test_name": "delete_space_oauth2",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # First create a space to delete
            space_name = f"test_oauth2_delete_space_{int(__import__('time').time())}"
            space_id = self.client.create_space(space_name, "Test space for OAuth2 deletion")

            # Now delete the space
            delete_result = self.client.delete_space(space_id)

            result["status"] = "passed"
            result["details"] = {
                "space_name": space_name,
                "space_id": space_id,
                "delete_result": delete_result
            }
            logger.info(f"Delete space with OAuth2 test passed for space: {space_name}")

        except Exception as e:
            result["error"] = f"Delete space with OAuth2 failed: {e}"
            logger.error(f"Delete space with OAuth2 test failed: {e}")

        return result

    def test_get_ingestion_logs_with_oauth2(self) -> Dict[str, Any]:
        """Test getting ingestion logs with OAuth2 authentication."""
        logger.info("Testing get ingestion logs with OAuth2...")
        result = {
            "test_name": "get_ingestion_logs_oauth2",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # Try to get ingestion logs
            logs = self.client.get_ingestion_logs(limit=5)
            result["status"] = "passed"
            result["details"] = {
                "logs_count": len(logs),
                "logs": logs
            }
            logger.info(f"Get ingestion logs with OAuth2 test passed - found {len(logs)} logs")

        except Exception as e:
            result["error"] = f"Get ingestion logs with OAuth2 failed: {e}"
            logger.error(f"Get ingestion logs with OAuth2 test failed: {e}")

        return result

    def test_delete_log_entry_with_oauth2(self) -> Dict[str, Any]:
        """Test deleting a log entry with OAuth2 authentication."""
        logger.info("Testing delete log entry with OAuth2...")
        result = {
            "test_name": "delete_log_entry_oauth2",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # First try to get some logs
            logs = self.client.get_ingestion_logs(limit=5)

            if not logs:
                result["error"] = "No logs available to delete"
                logger.warning("No logs available for delete log entry test")
                return result

            # Use the first log ID for deletion
            log_id = logs[0].get("id", "placeholder_log_1")
            delete_result = self.client.delete_log_entry(log_id)

            result["status"] = "passed"
            result["details"] = {
                "log_id": log_id,
                "delete_result": delete_result
            }
            logger.info(f"Delete log entry with OAuth2 test passed for log ID: {log_id}")

        except Exception as e:
            result["error"] = f"Delete log entry with OAuth2 failed: {e}"
            logger.error(f"Delete log entry with OAuth2 test failed: {e}")

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all OAuth2 and delete tests."""
        logger.info("Starting comprehensive OAuth2 and delete test suite...")

        # Setup OAuth2 first
        if not self.setup_oauth2():
            return {
                "success": False,
                "error": "Failed to setup OAuth2",
                "tests": []
            }

        # Setup client with OAuth2
        if not self.setup_client_with_oauth2():
            return {
                "success": False,
                "error": "Failed to setup client with OAuth2",
                "tests": []
            }

        tests = [
            self.test_oauth2_authentication,
            self.test_oauth2_token_introspection,
            self.test_delete_space_with_oauth2,
            self.test_get_ingestion_logs_with_oauth2,
            self.test_delete_log_entry_with_oauth2,
        ]

        results = []
        passed = 0
        failed = 0

        for test in tests:
            try:
                test_result = test()
                results.append(test_result)
                if test_result["status"] == "passed":
                    passed += 1
                else:
                    failed += 1
                logger.info(f"Test {test_result['test_name']}: {test_result['status']}")
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with exception: {e}")
                results.append({
                    "test_name": test.__name__,
                    "status": "error",
                    "error": str(e),
                    "details": {}
                })
                failed += 1

        summary = {
            "success": failed == 0,
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "tests": results
        }

        logger.info(f"OAuth2 test suite completed: {passed}/{len(tests)} tests passed")
        return summary

    def save_results(self, results: Dict[str, Any], filename: str = "oauth2_test_results.json"):
        """Save test results to file."""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"OAuth2 test results saved to {filename}")


def main():
    """Main function to run OAuth2 and delete tests."""
    logger.info("Starting HeySol OAuth2 and delete function testing...")

    tester = OAuth2DeleteTester()
    results = tester.run_all_tests()

    # Save results
    tester.save_results(results)

    # Print summary
    print("\n" + "="*70)
    print("HEYSOL OAUTH2 & DELETE FUNCTIONS TEST RESULTS")
    print("="*70)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success: {results['success']}")

    if results['success']:
        print("✅ All OAuth2 and delete tests passed!")
    else:
        print("❌ Some tests failed. Check logs for details.")

    # Print failed tests
    if results['failed'] > 0:
        print("\nFailed Tests:")
        for test in results['tests']:
            if test['status'] != 'passed':
                print(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")

    print(f"\nDetailed results saved to: oauth2_test_results.json")
    print(f"Debug logs saved to: oauth2_test.log, oauth2_debug.log, oauth2_client_debug.log")


if __name__ == "__main__":
    main()