"""
Live API debugging and testing script for HeySol API client.

This script tests the HeySol API client with live API calls to verify
authentication, endpoints, and functionality after applying fixes.
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
from heysol.exceptions import HeySolError, AuthenticationError, APIError

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_api_test.log")
    ]
)
logger = logging.getLogger(__name__)


class LiveAPITester:
    """Test HeySol API client with live API calls."""

    def __init__(self):
        """Initialize the API tester."""
        self.client: Optional[HeySolClient] = None
        self.test_results = []

    def setup_client(self) -> bool:
        """Setup the HeySol API client with configuration."""
        try:
            logger.info("Setting up HeySol API client...")

            # Try to load from environment variables
            api_key = os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY")
            base_url = os.getenv("COREAI_BASE_URL", "https://core.heysol.ai/api/v1/mcp")
            source = os.getenv("COREAI_SOURCE", "heysol-python-client")

            if not api_key:
                logger.error("No API key found in environment variables")
                logger.info("Please set COREAI_API_KEY or CORE_MEMORY_API_KEY environment variable")
                return False

            logger.info(f"Using API key: {api_key[:10]}...")
            logger.info(f"Using base URL: {base_url}")
            logger.info(f"Using source: {source}")

            # Create configuration
            config = HeySolConfig(
                api_key=api_key,
                base_url=base_url,
                source=source,
                log_level="DEBUG",
                log_to_file=True,
                log_file_path="heysol_debug.log"
            )

            # Create client
            self.client = HeySolClient(config=config)
            logger.info("HeySol API client setup successful")
            return True

        except Exception as e:
            logger.error(f"Failed to setup client: {e}")
            return False

    def test_authentication(self) -> Dict[str, Any]:
        """Test authentication with the API."""
        logger.info("Testing authentication...")
        result = {
            "test_name": "authentication",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # Test basic connectivity by initializing session
            # The client initialization already tests authentication
            result["status"] = "passed"
            result["details"] = {
                "session_id": self.client.session_id,
                "available_tools": list(self.client.tools.keys())
            }
            logger.info("Authentication test passed")

        except AuthenticationError as e:
            result["error"] = f"Authentication failed: {e}"
            logger.error(f"Authentication test failed: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
            logger.error(f"Authentication test error: {e}")

        return result

    def test_get_spaces(self) -> Dict[str, Any]:
        """Test getting available spaces."""
        logger.info("Testing get_spaces...")
        result = {
            "test_name": "get_spaces",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            spaces = self.client.get_spaces()
            result["status"] = "passed"
            result["details"] = {
                "spaces_count": len(spaces),
                "spaces": spaces
            }
            logger.info(f"get_spaces test passed - found {len(spaces)} spaces")

        except Exception as e:
            result["error"] = f"get_spaces failed: {e}"
            logger.error(f"get_spaces test failed: {e}")

        return result

    def test_create_space(self) -> Dict[str, Any]:
        """Test creating a new space."""
        logger.info("Testing create_space...")
        result = {
            "test_name": "create_space",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            space_name = f"test_space_{int(__import__('time').time())}"
            space_id = self.client.create_space(space_name, "Test space for API testing")
            result["status"] = "passed"
            result["details"] = {
                "space_name": space_name,
                "space_id": space_id
            }
            logger.info(f"create_space test passed - created space: {space_id}")

        except Exception as e:
            result["error"] = f"create_space failed: {e}"
            logger.error(f"create_space test failed: {e}")

        return result

    def test_memory_ingestion(self) -> Dict[str, Any]:
        """Test memory ingestion."""
        logger.info("Testing memory ingestion...")
        result = {
            "test_name": "memory_ingestion",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            test_message = "This is a test message for API testing"
            ingestion_result = self.client.ingest(test_message)
            result["status"] = "passed"
            result["details"] = {
                "message": test_message,
                "result": ingestion_result
            }
            logger.info("Memory ingestion test passed")

        except Exception as e:
            result["error"] = f"Memory ingestion failed: {e}"
            logger.error(f"Memory ingestion test failed: {e}")

        return result

    def test_memory_search(self) -> Dict[str, Any]:
        """Test memory search."""
        logger.info("Testing memory search...")
        result = {
            "test_name": "memory_search",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            search_query = "test message"
            search_result = self.client.search(search_query, limit=5)
            result["status"] = "passed"
            result["details"] = {
                "query": search_query,
                "episodes_count": len(search_result.get("episodes", [])),
                "facts_count": len(search_result.get("facts", [])),
                "result": search_result
            }
            logger.info("Memory search test passed")

        except Exception as e:
            result["error"] = f"Memory search failed: {e}"
            logger.error(f"Memory search test failed: {e}")

        return result

    def test_user_profile(self) -> Dict[str, Any]:
        """Test getting user profile."""
        logger.info("Testing user profile...")
        result = {
            "test_name": "user_profile",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            profile = self.client.get_user_profile()
            result["status"] = "passed"
            result["details"] = {
                "profile_keys": list(profile.keys()) if isinstance(profile, dict) else "Not a dict",
                "profile": profile
            }
            logger.info("User profile test passed")

        except Exception as e:
            result["error"] = f"User profile failed: {e}"
            logger.error(f"User profile test failed: {e}")

        return result

    def test_get_ingestion_logs(self) -> Dict[str, Any]:
        """Test getting ingestion logs."""
        logger.info("Testing get_ingestion_logs...")
        result = {
            "test_name": "get_ingestion_logs",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            logs = self.client.get_ingestion_logs(limit=10)
            result["status"] = "passed"
            result["details"] = {
                "logs_count": len(logs),
                "logs": logs
            }
            logger.info(f"get_ingestion_logs test passed - found {len(logs)} logs")

        except Exception as e:
            result["error"] = f"get_ingestion_logs failed: {e}"
            logger.error(f"get_ingestion_logs test failed: {e}")

        return result

    def test_get_specific_log(self) -> Dict[str, Any]:
        """Test getting a specific log by ID."""
        logger.info("Testing get_specific_log...")
        result = {
            "test_name": "get_specific_log",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # First get some logs to find a valid log ID
            logs = self.client.get_ingestion_logs(limit=5)

            if not logs:
                result["error"] = "No logs available to test with"
                logger.warning("No logs available for get_specific_log test")
                return result

            # Use the first log ID
            log_id = logs[0].get("id", "placeholder_log_1")
            log_details = self.client.get_specific_log(log_id)

            result["status"] = "passed"
            result["details"] = {
                "log_id": log_id,
                "log_details": log_details
            }
            logger.info(f"get_specific_log test passed for log ID: {log_id}")

        except Exception as e:
            result["error"] = f"get_specific_log failed: {e}"
            logger.error(f"get_specific_log test failed: {e}")

        return result

    def test_delete_log_entry(self) -> Dict[str, Any]:
        """Test deleting a log entry."""
        logger.info("Testing delete_log_entry...")
        result = {
            "test_name": "delete_log_entry",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # First get some logs to find a valid log ID
            logs = self.client.get_ingestion_logs(limit=5)

            if not logs:
                result["error"] = "No logs available to delete"
                logger.warning("No logs available for delete_log_entry test")
                return result

            # Use the first log ID for deletion
            log_id = logs[0].get("id", "placeholder_log_1")
            delete_result = self.client.delete_log_entry(log_id)

            result["status"] = "passed"
            result["details"] = {
                "log_id": log_id,
                "delete_result": delete_result
            }
            logger.info(f"delete_log_entry test passed for log ID: {log_id}")

        except Exception as e:
            result["error"] = f"delete_log_entry failed: {e}"
            logger.error(f"delete_log_entry test failed: {e}")

        return result

    def test_delete_space(self) -> Dict[str, Any]:
        """Test deleting a space."""
        logger.info("Testing delete_space...")
        result = {
            "test_name": "delete_space",
            "status": "failed",
            "error": None,
            "details": {}
        }

        try:
            if not self.client:
                result["error"] = "Client not initialized"
                return result

            # First create a space to delete
            space_name = f"test_delete_space_{int(__import__('time').time())}"
            space_id = self.client.create_space(space_name, "Test space for deletion")

            # Now delete the space
            delete_result = self.client.delete_space(space_id)

            result["status"] = "passed"
            result["details"] = {
                "space_name": space_name,
                "space_id": space_id,
                "delete_result": delete_result
            }
            logger.info(f"delete_space test passed for space: {space_name}")

        except Exception as e:
            result["error"] = f"delete_space failed: {e}"
            logger.error(f"delete_space test failed: {e}")

        return result

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        logger.info("Starting comprehensive API test suite...")

        if not self.setup_client():
            return {
                "success": False,
                "error": "Failed to setup client",
                "tests": []
            }

        tests = [
            self.test_authentication,
            self.test_get_spaces,
            self.test_create_space,
            self.test_memory_ingestion,
            self.test_memory_search,
            self.test_user_profile,
            self.test_get_ingestion_logs,
            self.test_get_specific_log,
            self.test_delete_log_entry,
            self.test_delete_space,
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

        logger.info(f"Test suite completed: {passed}/{len(tests)} tests passed")
        return summary

    def save_results(self, results: Dict[str, Any], filename: str = "live_api_test_results.json"):
        """Save test results to file."""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {filename}")


def main():
    """Main function to run live API tests."""
    logger.info("Starting HeySol API live testing...")

    tester = LiveAPITester()
    results = tester.run_all_tests()

    # Save results
    tester.save_results(results)

    # Print summary
    print("\n" + "="*60)
    print("HEYSOL API LIVE TEST RESULTS")
    print("="*60)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success: {results['success']}")

    if results['success']:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check logs for details.")

    # Print failed tests
    if results['failed'] > 0:
        print("\nFailed Tests:")
        for test in results['tests']:
            if test['status'] != 'passed':
                print(f"  - {test['test_name']}: {test.get('error', 'Unknown error')}")

    print(f"\nDetailed results saved to: live_api_test_results.json")
    print(f"Debug logs saved to: live_api_test.log and heysol_debug.log")


if __name__ == "__main__":
    main()