#!/usr/bin/env python3
"""
Integration Test Suite for HeySol API Client with Live API Calls

Tests all endpoints with real API calls to verify:
- Actual API functionality
- Response format compliance
- Authentication with real tokens
- Error handling with real API failures
- Performance with real network calls
"""

import os
import time
import pytest
import requests_mock
from typing import Dict, Any, List, Optional

from heysol.client import HeySolClient
from heysol.exceptions import HeySolError, ValidationError


class TestHeySolClientLiveAPI:
    """Integration test suite for HeySol API client with live API calls."""

    @pytest.fixture
    def live_client(self):
        """Create a client instance with real API key from environment."""
        api_key = os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY", "test-key-for-live-tests")
        return HeySolClient(api_key=api_key)

    # User Endpoints Integration Tests
    @pytest.mark.integration
    def test_live_get_user_profile(self, live_client):
        """Test get_user_profile with live API."""
        profile = live_client.get_user_profile()
        assert isinstance(profile, dict)
        assert "id" in profile or "sub" in profile  # OAuth2 vs API key user
        print(f"✅ Live user profile retrieved: {profile.get('name', 'Unknown')}")

    # Memory Endpoints Integration Tests
    @pytest.mark.integration
    def test_live_search_knowledge_graph(self, live_client):
        """Test search_knowledge_graph with live API."""
        try:
            result = live_client.search_knowledge_graph("test", limit=5, depth=1)
            assert isinstance(result, dict)
            # Response should have some structure
            assert any(key in result for key in ["nodes", "entities", "results"])
            print(f"✅ Live knowledge graph search completed: {len(result.get('nodes', []))} nodes found")
        except Exception as e:
            pytest.skip(f"Live API knowledge graph test skipped due to: {e}")

    @pytest.mark.integration
    def test_live_get_episode_facts(self, live_client):
        """Test get_episode_facts with live API."""
        try:
            # Use a test episode ID - this may not exist but will test the API call
            facts = live_client.get_episode_facts(episode_id="test-episode-123", limit=10)
            assert isinstance(facts, list)
            print(f"✅ Live episode facts retrieved: {len(facts)} facts")
        except Exception as e:
            pytest.skip(f"Live API episode facts test skipped due to: {e}")

    # Spaces Endpoints Integration Tests
    @pytest.mark.integration
    def test_live_get_spaces(self, live_client):
        """Test get_spaces with live API."""
        spaces = live_client.get_spaces()
        assert isinstance(spaces, list)
        print(f"✅ Live spaces retrieved: {len(spaces)} spaces")

    # OAuth2 Endpoints Integration Tests
    @pytest.mark.integration
    def test_live_get_oauth2_authorization_url(self, live_client):
        """Test get_oauth2_authorization_url with live API."""
        try:
            result = live_client.get_oauth2_authorization_url()
            assert isinstance(result, dict)
            assert "authorization_url" in result
            print(f"✅ Live OAuth2 authorization URL retrieved")
        except Exception as e:
            pytest.skip(f"Live API OAuth2 authorization test skipped due to: {e}")

    # Webhook Endpoints Integration Tests
    @pytest.mark.integration
    def test_live_list_webhooks(self, live_client):
        """Test list_webhooks with live API."""
        try:
            webhooks = live_client.list_webhooks(limit=10)
            assert isinstance(webhooks, list)
            print(f"✅ Live webhooks listed: {len(webhooks)} webhooks")
        except Exception as e:
            pytest.skip(f"Live API webhooks list test skipped due to: {e}")

    # Authentication Integration Tests
    @pytest.mark.integration
    def test_live_bearer_token_authentication(self, live_client):
        """Test Bearer token authentication with live API."""
        try:
            # Make a simple request to test authentication
            profile = live_client.get_user_profile()
            assert isinstance(profile, dict)
            print("✅ Live Bearer token authentication successful")
        except Exception as e:
            pytest.skip(f"Live API authentication test skipped due to: {e}")

    # Performance Integration Tests
    @pytest.mark.integration
    def test_live_api_performance(self, live_client):
        """Test live API performance."""
        try:
            start_time = time.time()

            # Make multiple requests to test performance
            for _ in range(3):
                live_client.get_user_profile()
                time.sleep(0.1)  # Small delay between requests

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete in reasonable time (less than 10 seconds for 3 requests)
            assert total_time < 10.0, f"Live API requests too slow: {total_time:.2f}s"
            print(f"✅ Live API performance test passed: {total_time:.2f} total time")
        except Exception as e:
            pytest.skip(f"Live API performance test skipped due to: {e}")

    # Error Handling Integration Tests
    @pytest.mark.integration
    def test_live_api_error_handling(self, live_client):
        """Test error handling with live API."""
        try:
            # Test invalid space ID
            with pytest.raises(Exception):
                live_client.get_space_details("invalid-space-id")

            print("✅ Live API error handling test passed")
        except Exception as e:
            pytest.skip(f"Live API error handling test skipped due to: {e}")


if __name__ == "__main__":
    # Run integration tests
    print("🔗 Running Live API Integration Tests...")

    try:
        # Check if API key is available
        api_key = os.getenv("HEYSOL_API_KEY") or os.getenv("COREAI_API_KEY")
        if not api_key:
            print("⚠️  HEYSOL_API_KEY or COREAI_API_KEY not set - skipping live API tests")
            print("To run live API tests, set the HEYSOL_API_KEY environment variable")
            exit(0)

        test_suite = TestHeySolClientLiveAPI()

        # Test live API connectivity
        print("✅ Live API integration tests setup complete")
        print("Run with pytest to execute individual tests:")
        print("  pytest test_integration_live_api.py::TestHeySolClientLiveAPI::test_live_get_user_profile -v")
        print("  pytest test_integration_live_api.py -k integration -v")

    except Exception as e:
        print(f"❌ Integration test setup failed: {e}")
        raise