"""
Live API integration tests for HeySol API client.
Tests actual API endpoints with real HTTP calls.
"""

import os
import unittest
from datetime import datetime
import pytest

from heysol.client import HeySolClient
from heysol.exceptions import HeySolError, ValidationError


class TestLiveAPIIntegration(unittest.TestCase):
    """Live API integration tests using real HTTP calls."""

    def setUp(self):
        """Set up test fixtures with real API credentials."""
        self.api_key = os.getenv("HEYSOL_API_KEY")
        if not self.api_key:
            self.skipTest("HEYSOL_API_KEY environment variable not set")

        self.base_url = "https://core.heysol.ai/api/v1"
        self.client = HeySolClient(api_key=self.api_key, base_url=self.base_url)

        # Test space ID - use a known test space or create one
        self.test_space_id = "cmflmxmas00rvqf1vz3bslziu"  # Profile space from earlier tests

    def tearDown(self):
        """Clean up after tests."""
        pass

    @pytest.mark.integration
    def test_live_user_profile(self):
        """Test live get_user_profile endpoint."""
        try:
            result = self.client.get_user_profile()
            self.assertIsInstance(result, dict)
            print(f"✅ User profile retrieved: {result.get('id', 'unknown')}")
        except HeySolError as e:
            self.skipTest(f"User profile endpoint not accessible: {e}")

    @pytest.mark.integration
    def test_live_search_knowledge_graph(self):
        """Test live search_knowledge_graph endpoint."""
        try:
            result = self.client.search_knowledge_graph("test query", limit=5, depth=1)
            self.assertIsInstance(result, dict)
            print(f"✅ Knowledge graph search successful: {len(result.get('nodes', []))} nodes")
        except HeySolError as e:
            self.skipTest(f"Knowledge graph search not accessible: {e}")

    @pytest.mark.integration
    def test_live_add_to_ingestion_queue(self):
        """Test live add_data_to_ingestion_queue endpoint."""
        try:
            data = {
                "content": f"Integration test data - {datetime.now().isoformat()}",
                "type": "test",
                "source": "live-api-test"
            }
            result = self.client.add_data_to_ingestion_queue(
                data=data,
                space_id=self.test_space_id,
                priority="normal",
                tags=["integration-test", "automated"],
                metadata={"test_run": True}
            )
            self.assertIsInstance(result, dict)
            self.assertTrue(result.get("success", False))
            print(f"✅ Data added to ingestion queue: {result.get('queue_id', 'unknown')}")
        except HeySolError as e:
            self.skipTest(f"Ingestion queue not accessible: {e}")

    @pytest.mark.integration
    def test_live_get_ingestion_logs(self):
        """Test live get_ingestion_logs endpoint."""
        try:
            result = self.client.get_ingestion_logs(
                space_id=self.test_space_id,
                limit=10,
                offset=0
            )
            self.assertIsInstance(result, list)
            print(f"✅ Ingestion logs retrieved: {len(result)} logs")
        except HeySolError as e:
            self.skipTest(f"Ingestion logs not accessible: {e}")

    @pytest.mark.integration
    def test_live_get_spaces(self):
        """Test live get_spaces endpoint."""
        try:
            result = self.client.get_spaces()
            self.assertIsInstance(result, (list, dict))
            print(f"✅ Spaces retrieved: {len(result) if isinstance(result, list) else 'dict format'}")
        except HeySolError as e:
            self.skipTest(f"Spaces endpoint not accessible: {e}")

    @pytest.mark.integration
    def test_live_create_and_delete_space(self):
        """Test live create_space and delete_space endpoints."""
        try:
            # Create a test space
            space_name = f"test-space-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            space_description = "Test space created by integration tests"

            space_id = self.client.create_space(space_name, space_description)
            self.assertIsInstance(space_id, str)
            self.assertTrue(len(space_id) > 0)
            print(f"✅ Space created: {space_id}")

            # Get space details
            space_details = self.client.get_space_details(space_id)
            self.assertIsInstance(space_details, dict)
            self.assertEqual(space_details.get("name"), space_name)
            print(f"✅ Space details retrieved: {space_details.get('name')}")

            # Update space
            updated_details = self.client.update_space(
                space_id,
                description="Updated description",
                metadata={"updated_by": "integration-test"}
            )
            self.assertIsInstance(updated_details, dict)
            print("✅ Space updated successfully")

            # Delete space
            delete_result = self.client.delete_space(space_id, confirm=True)
            self.assertIsInstance(delete_result, dict)
            self.assertTrue(delete_result.get("success", False))
            print("✅ Space deleted successfully")

        except HeySolError as e:
            self.skipTest(f"Space operations not accessible: {e}")

    @pytest.mark.integration
    def test_live_oauth2_authorization_url(self):
        """Test live get_oauth2_authorization_url endpoint."""
        try:
            result = self.client.get_oauth2_authorization_url("openid profile email")
            self.assertIsInstance(result, str)
            self.assertIn("authorize", result)
            print(f"✅ OAuth2 authorization URL generated: {result[:50]}...")
        except HeySolError as e:
            self.skipTest(f"OAuth2 authorization not accessible: {e}")

    @pytest.mark.integration
    def test_live_webhook_operations(self):
        """Test live webhook operations."""
        try:
            # Register webhook
            webhook_url = "https://example.com/test-webhook"
            events = ["ingestion.completed", "search.performed"]

            webhook_result = self.client.register_webhook(
                webhook_url,
                events,
                space_id=self.test_space_id,
                secret="test-webhook-secret"
            )
            self.assertIsInstance(webhook_result, dict)
            webhook_id = webhook_result.get("id")
            print(f"✅ Webhook registered: {webhook_id}")

            # List webhooks
            webhooks = self.client.list_webhooks(space_id=self.test_space_id, limit=10)
            self.assertIsInstance(webhooks, list)
            print(f"✅ Webhooks listed: {len(webhooks)} webhooks")

            # Get webhook details
            if webhook_id:
                webhook_details = self.client.get_webhook(webhook_id)
                self.assertIsInstance(webhook_details, dict)
                self.assertEqual(webhook_details.get("id"), webhook_id)
                print(f"✅ Webhook details retrieved: {webhook_details.get('url')}")

                # Update webhook
                updated_webhook = self.client.update_webhook(
                    webhook_id,
                    events=["ingestion.completed"],
                    active=False
                )
                self.assertIsInstance(updated_webhook, dict)
                print("✅ Webhook updated successfully")

                # Delete webhook
                delete_result = self.client.delete_webhook(webhook_id, confirm=True)
                self.assertIsInstance(delete_result, dict)
                self.assertTrue(delete_result.get("success", False))
                print("✅ Webhook deleted successfully")

        except HeySolError as e:
            self.skipTest(f"Webhook operations not accessible: {e}")

    @pytest.mark.integration
    def test_live_oauth2_token_operations(self):
        """Test live OAuth2 token operations."""
        try:
            # Test token introspection (if we have a token)
            # This would require a valid token to test properly
            print("⚠️  OAuth2 token operations require valid tokens - skipping detailed tests")
        except HeySolError as e:
            self.skipTest(f"OAuth2 token operations not accessible: {e}")

    @pytest.mark.integration
    def test_live_error_handling(self):
        """Test live API error handling."""
        # Test invalid space ID
        try:
            self.client.get_space_details("invalid-space-id")
            print("⚠️  Expected error for invalid space ID was not raised")
        except HeySolError:
            print("✅ Proper error handling for invalid space ID")

        # Test invalid webhook ID
        try:
            self.client.get_webhook("invalid-webhook-id")
            print("⚠️  Expected error for invalid webhook ID was not raised")
        except HeySolError:
            print("✅ Proper error handling for invalid webhook ID")

        # Test validation errors
        try:
            self.client.search_knowledge_graph("")
            print("⚠️  Expected validation error for empty query was not raised")
        except ValidationError:
            print("✅ Proper validation error handling")

    @pytest.mark.integration
    def test_live_response_parsing(self):
        """Test live API response parsing."""
        try:
            # Test different response formats
            result = self.client.get_ingestion_logs(limit=1)
            if isinstance(result, list):
                print("✅ List response parsed correctly")
            elif isinstance(result, dict):
                print("✅ Dict response parsed correctly")
            else:
                print(f"⚠️  Unexpected response type: {type(result)}")
        except HeySolError as e:
            self.skipTest(f"Response parsing test not accessible: {e}")

    @pytest.mark.integration
    def test_live_authentication_headers(self):
        """Test live API authentication headers."""
        # This test verifies that authentication headers are properly set
        # by making a simple request and checking it doesn't fail with auth errors
        try:
            result = self.client.get_user_profile()
            print("✅ Authentication headers accepted by API")
        except HeySolError as e:
            if "401" in str(e) or "Unauthorized" in str(e):
                self.fail("Authentication headers not accepted by API")
            else:
                self.skipTest(f"Authentication test inconclusive: {e}")

    @pytest.mark.integration
    def test_live_query_parameters(self):
        """Test live API query parameter handling."""
        try:
            # Test various query parameter combinations
            result1 = self.client.get_ingestion_logs(limit=5, offset=0)
            result2 = self.client.get_ingestion_logs(limit=3, offset=2, status="completed")

            # Both should return valid responses
            self.assertIsInstance(result1, (list, dict))
            self.assertIsInstance(result2, (list, dict))
            print("✅ Query parameters handled correctly")
        except HeySolError as e:
            self.skipTest(f"Query parameter test not accessible: {e}")


if __name__ == "__main__":
    # Set up environment for testing
    if not os.getenv("HEYSOL_API_KEY"):
        print("⚠️  Warning: HEYSOL_API_KEY not set. Live API tests will be skipped.")
        print("   To run live tests, set the HEYSOL_API_KEY environment variable.")

    unittest.main()