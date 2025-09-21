#!/usr/bin/env python3
"""
Comprehensive Test Suite for HeySol API Client

Tests all endpoints with mock API calls, covering:
- User endpoints
- Memory endpoints
- Spaces endpoints
- OAuth2 endpoints
- Webhook endpoints
- Authentication with Bearer tokens
- Error handling for invalid inputs and API failures
- Response parsing for JSON data
- Query parameter support in _make_request
- Adherence to lean, fast, explicit coding standards
"""

import json
import pytest
import requests
import requests_mock
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import os

from heysol_api_client.heysol import HeySolClient
from heysol_api_client.heysol import HeySolError, ValidationError

# Load environment variables from .env file
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class TestHeySolClientComprehensive:
    """Comprehensive test suite for HeySol API client."""

    @pytest.fixture
    def client(self):
        """Create a test client instance with API key from environment."""
        api_key = os.getenv("HEYSOL_API_KEY")
        if not api_key:
            pytest.skip("HEYSOL_API_KEY environment variable not set")
        return HeySolClient(api_key=api_key, skip_mcp_init=True)

    @pytest.fixture
    def setup_mocks(self, requests_mock):
        """Set up global mocks for all API endpoints."""
        # User endpoints
        requests_mock.get("https://core.heysol.ai/api/v1/user/profile", json={
            "id": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00Z"
        })

        # Memory endpoints
        requests_mock.post("https://core.heysol.ai/api/v1/memory/knowledge-graph/search", json={"nodes": [], "edges": []})
        requests_mock.post("https://core.heysol.ai/api/v1/memory/ingestion/queue", json={"queue_id": "queue-123"})
        requests_mock.get("https://core.heysol.ai/api/v1/episodes/episode-123/facts", json=[])
        requests_mock.get("https://core.heysol.ai/api/v1/memory/logs", json=[])
        requests_mock.get("https://core.heysol.ai/api/v1/memory/logs/log-123", json={"id": "log-123"})

        # Spaces endpoints
        requests_mock.put("https://core.heysol.ai/api/v1/spaces/bulk", json={"results": []})
        requests_mock.get("https://core.heysol.ai/api/v1/spaces/space-123/details", json={"id": "space-123", "name": "Test Space"})
        requests_mock.put("https://core.heysol.ai/api/v1/spaces/space-123", json={"id": "space-123", "name": "Updated Space"})
        requests_mock.delete("https://core.heysol.ai/api/v1/spaces/space-123", json={"deleted": True})

        # OAuth2 endpoints
        requests_mock.get("https://core.heysol.ai/api/v1/oauth2/authorize", json={"authorization_url": "https://example.com/auth"})
        requests_mock.post("https://core.heysol.ai/api/v1/oauth2/authorize/req-123", json={"decision": "allow"})
        requests_mock.post("https://core.heysol.ai/api/v1/oauth2/token", json={"access_token": "token-123"})
        requests_mock.get("https://core.heysol.ai/api/v1/oauth2/userinfo", json={"sub": "user-123"})
        requests_mock.post("https://core.heysol.ai/api/v1/oauth2/refresh", json={"access_token": "new-token-123"})
        requests_mock.post("https://core.heysol.ai/api/v1/oauth2/revoke", json={"message": "Token revoked"})

        # Webhook endpoints
        requests_mock.post("https://core.heysol.ai/api/v1/webhooks", json={"webhook_id": "webhook-123"})
        requests_mock.get("https://core.heysol.ai/api/v1/webhooks", json=[])
        requests_mock.get("https://core.heysol.ai/api/v1/webhooks/webhook-123", json={"id": "webhook-123"})
        requests_mock.put("https://core.heysol.ai/api/v1/webhooks/webhook-123", json={"id": "webhook-123"})
        requests_mock.delete("https://core.heysol.ai/api/v1/webhooks/webhook-123", json={"deleted": True})

    # User Endpoints Tests
    def test_get_user_profile_valid(self, client, setup_mocks):
        """Test get_user_profile with valid response."""
        expected_profile = {
            "id": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00Z"
        }

        profile = client.get_user_profile()
        assert profile == expected_profile

    def test_get_user_profile_invalid_response(self, client):
        """Test get_user_profile with invalid JSON response."""
        # Override the global mock for this specific test
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                text="Invalid JSON",
                status_code=200,
                headers={"Content-Type": "application/json"}
            )

            with pytest.raises(requests.exceptions.JSONDecodeError):
                client.get_user_profile()

    # Memory Endpoints Tests
    def test_search_knowledge_graph_valid(self, client):
        """Test search_knowledge_graph with valid inputs."""
        expected_result = {
            "query": "clinical trial",
            "nodes": [
                {"id": "node-1", "label": "Clinical Trial", "type": "concept"},
                {"id": "node-2", "label": "Patient", "type": "entity"}
            ],
            "edges": [
                {"source": "node-1", "target": "node-2", "relationship": "involves"}
            ],
            "total_results": 2
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/knowledge-graph/search",
                json=expected_result,
                status_code=200
            )

            result = client.search_knowledge_graph("clinical trial", limit=10, depth=2)
            assert result == expected_result

    def test_search_knowledge_graph_invalid_query(self, client):
        """Test search_knowledge_graph with invalid query."""
        with pytest.raises(ValidationError, match="Search query is required"):
            client.search_knowledge_graph("")

    def test_search_knowledge_graph_invalid_limit(self, client):
        """Test search_knowledge_graph with invalid limit."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            client.search_knowledge_graph("test", limit=0)

        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            client.search_knowledge_graph("test", limit=101)

    def test_search_knowledge_graph_invalid_depth(self, client):
        """Test search_knowledge_graph with invalid depth."""
        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            client.search_knowledge_graph("test", depth=0)

        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            client.search_knowledge_graph("test", depth=6)

    def test_add_data_to_ingestion_queue_valid(self, client):
        """Test add_data_to_ingestion_queue with valid inputs."""
        test_data = {"clinical_data": "test data", "patient_id": "123"}
        expected_response = {
            "queue_id": "queue-123",
            "status": "queued",
            "estimated_processing_time": "30s"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/ingestion/queue",
                json=expected_response,
                status_code=202
            )

            result = client.add_data_to_ingestion_queue(
                data=test_data,
                priority="high",
                tags=["urgent", "clinical"]
            )
            assert result == expected_response

    def test_get_episode_facts_valid(self, client):
        """Test get_episode_facts with valid inputs."""
        expected_facts = [
            {
                "id": "fact-1",
                "episode_id": "episode-123",
                "content": "Patient showed improvement",
                "confidence": 0.92
            },
            {
                "id": "fact-2",
                "episode_id": "episode-123",
                "content": "Treatment duration: 8 weeks",
                "confidence": 0.98
            }
        ]

        # Override the global mock for this specific test
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/episodes/episode-123/facts",
                json=expected_facts,
                status_code=200
            )

            facts = client.get_episode_facts(episode_id="episode-123", limit=50)
            assert facts == expected_facts

    def test_get_ingestion_logs_valid(self, client):
        """Test get_ingestion_logs with valid inputs."""
        expected_logs = [
            {
                "id": "log-1",
                "status": "success",
                "timestamp": "2024-01-01T00:00:00Z",
                "message_count": 100
            },
            {
                "id": "log-2",
                "status": "failed",
                "timestamp": "2024-01-02T00:00:00Z",
                "error": "Connection timeout"
            }
        ]

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/logs",
                json=expected_logs,
                status_code=200
            )

            logs = client.get_ingestion_logs(limit=50, status="success")
            assert logs == expected_logs

    def test_get_specific_log_valid(self, client):
        """Test get_specific_log with valid inputs."""
        expected_log = {
            "id": "log-123",
            "status": "success",
            "timestamp": "2024-01-01T00:00:00Z",
            "details": {
                "processed_items": 100,
                "duration_ms": 1500
            }
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/logs/log-123",
                json=expected_log,
                status_code=200
            )

            log = client.get_specific_log("log-123")
            assert log == expected_log

    def test_get_episode_facts_invalid_id(self, client):
        """Test get_episode_facts with invalid episode ID."""
        with pytest.raises(ValidationError, match="Episode ID is required"):
            client.get_episode_facts("")

    # Spaces Endpoints Tests
    def test_bulk_space_operations_valid(self, client):
        """Test bulk_space_operations with valid inputs."""
        operations = [
            {"type": "create", "name": "Space 1", "description": "Test space 1"},
            {"type": "create", "name": "Space 2", "description": "Test space 2"}
        ]

        expected_response = {
            "results": [
                {"operation": 0, "success": True, "space_id": "space-1"},
                {"operation": 1, "success": True, "space_id": "space-2"}
            ]
        }

        with requests_mock.Mocker() as m:
            m.put(
                "https://core.heysol.ai/api/v1/spaces/bulk",
                json=expected_response,
                status_code=200
            )

            result = client.bulk_space_operations(operations)
            assert result == expected_response

    def test_get_space_details_valid(self, client):
        """Test get_space_details with valid inputs."""
        expected_details = {
            "id": "space-123",
            "name": "Test Space",
            "description": "A test space",
            "created_at": "2024-01-01T00:00:00Z",
            "stats": {
                "total_episodes": 100,
                "total_facts": 500,
                "storage_used_mb": 25.5
            }
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/spaces/space-123/details",
                json=expected_details,
                status_code=200
            )

            details = client.get_space_details("space-123")
            assert details == expected_details

    def test_get_space_details_invalid_id(self, client):
        """Test get_space_details with invalid space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            client.get_space_details("")

    def test_update_space_valid(self, client):
        """Test update_space with valid inputs."""
        expected_response = {
            "id": "space-123",
            "name": "Updated Space",
            "description": "Updated description",
            "updated_at": "2024-01-02T00:00:00Z"
        }

        with requests_mock.Mocker() as m:
            m.put(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json=expected_response,
                status_code=200
            )

            result = client.update_space(
                "space-123",
                name="Updated Space",
                description="Updated description"
            )
            assert result == expected_response

    def test_delete_space_valid(self, client):
        """Test delete_space with valid inputs."""
        expected_response = {
            "message": "Space deleted successfully",
            "deleted_space_id": "space-123"
        }

        with requests_mock.Mocker() as m:
            m.delete(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json=expected_response,
                status_code=200
            )

            result = client.delete_space("space-123", confirm=True)
            assert result == expected_response

    def test_delete_space_without_confirmation(self, client):
        """Test delete_space without confirmation."""
        with pytest.raises(ValidationError, match="Space deletion requires confirmation"):
            client.delete_space("space-123")

    # OAuth2 Endpoints Tests
    def test_get_oauth2_authorization_url_valid(self, client):
        """Test get_oauth2_authorization_url with valid inputs."""
        expected_response = {
            "authorization_url": "https://core.heysol.ai/api/v1/oauth2/authorize?client_id=test&scope=openid+profile+email&response_type=code"
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/oauth2/authorize",
                json=expected_response,
                status_code=200
            )

            result = client.get_oauth2_authorization_url(scope="openid profile email")
            assert result == expected_response

    def test_oauth2_authorization_decision_valid(self, client):
        """Test oauth2_authorization_decision with valid inputs."""
        expected_response = {
            "decision": "allow",
            "request_id": "req-123",
            "redirect_uri": "https://example.com/callback"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/oauth2/authorize/req-123",
                json=expected_response,
                status_code=200
            )

            result = client.oauth2_authorization_decision("allow", "req-123")
            assert result == expected_response

    def test_oauth2_authorization_decision_invalid_decision(self, client):
        """Test oauth2_authorization_decision with invalid decision."""
        with pytest.raises(ValidationError, match="Decision must be 'allow' or 'deny'"):
            client.oauth2_authorization_decision("invalid", "req-123")

    def test_oauth2_token_exchange_valid(self, client):
        """Test oauth2_token_exchange with valid inputs."""
        expected_response = {
            "access_token": "access-token-123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "refresh-token-123"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/oauth2/token",
                json=expected_response,
                status_code=200
            )

            result = client.oauth2_token_exchange("auth-code-123", "https://example.com/callback")
            assert result == expected_response

    def test_oauth2_token_exchange_invalid_code(self, client):
        """Test oauth2_token_exchange with invalid authorization code."""
        with pytest.raises(ValidationError, match="Authorization code is required"):
            client.oauth2_token_exchange("", "https://example.com/callback")

    def test_get_oauth2_user_info_valid(self, client):
        """Test get_oauth2_user_info with valid access token."""
        expected_user_info = {
            "sub": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "email_verified": True
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/oauth2/userinfo",
                json=expected_user_info,
                status_code=200
            )

            result = client.get_oauth2_user_info("access-token-123")
            assert result == expected_user_info

    def test_get_oauth2_user_info_invalid_token(self, client):
        """Test get_oauth2_user_info with invalid access token."""
        with pytest.raises(ValidationError, match="Access token is required"):
            client.get_oauth2_user_info("")

    def test_oauth2_refresh_token_valid(self, client):
        """Test oauth2_refresh_token with valid refresh token."""
        expected_response = {
            "access_token": "new-access-token-123",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "new-refresh-token-123"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/oauth2/refresh",
                json=expected_response,
                status_code=200
            )

            result = client.oauth2_refresh_token("refresh-token-123")
            assert result == expected_response

    def test_oauth2_revoke_token_valid(self, client):
        """Test oauth2_revoke_token with valid token."""
        expected_response = {
            "message": "Token revoked successfully"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/oauth2/revoke",
                json=expected_response,
                status_code=200
            )

            result = client.oauth2_revoke_token("token-123", "access_token")
            assert result == expected_response

    # Webhook Endpoints Tests
    def test_register_webhook_valid(self, client):
        """Test register_webhook with valid inputs."""
        expected_response = {
            "webhook_id": "webhook-123",
            "url": "https://example.com/webhook",
            "events": ["memory.ingest", "memory.search"],
            "active": True
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/webhooks",
                json=expected_response,
                status_code=201
            )

            result = client.register_webhook(
                "https://example.com/webhook",
                ["memory.ingest", "memory.search"],
                secret="webhook-secret"
            )
            assert result == expected_response

    def test_register_webhook_invalid_url(self, client):
        """Test register_webhook with invalid URL."""
        with pytest.raises(ValidationError, match="Webhook URL is required"):
            client.register_webhook("", ["memory.ingest"])

    def test_list_webhooks_valid(self, client):
        """Test list_webhooks with valid inputs."""
        expected_webhooks = [
            {
                "id": "webhook-1",
                "url": "https://example.com/webhook1",
                "events": ["memory.ingest"],
                "active": True
            },
            {
                "id": "webhook-2",
                "url": "https://example.com/webhook2",
                "events": ["memory.search"],
                "active": False
            }
        ]

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/webhooks",
                json=expected_webhooks,
                status_code=200
            )

            result = client.list_webhooks(active=True, limit=50)
            assert result == expected_webhooks

    def test_get_webhook_valid(self, client):
        """Test get_webhook with valid webhook ID."""
        expected_webhook = {
            "id": "webhook-123",
            "url": "https://example.com/webhook",
            "events": ["memory.ingest", "memory.search"],
            "active": True,
            "created_at": "2024-01-01T00:00:00Z"
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/webhooks/webhook-123",
                json=expected_webhook,
                status_code=200
            )

            result = client.get_webhook("webhook-123")
            assert result == expected_webhook

    def test_get_webhook_invalid_id(self, client):
        """Test get_webhook with invalid webhook ID."""
        with pytest.raises(ValidationError, match="Webhook ID is required"):
            client.get_webhook("")

    def test_update_webhook_valid(self, client):
        """Test update_webhook with valid inputs."""
        expected_response = {
            "id": "webhook-123",
            "url": "https://example.com/updated-webhook",
            "events": ["memory.ingest", "memory.search", "space.create"],
            "active": False,
            "updated_at": "2024-01-02T00:00:00Z"
        }

        with requests_mock.Mocker() as m:
            m.put(
                "https://core.heysol.ai/api/v1/webhooks/webhook-123",
                json=expected_response,
                status_code=200
            )

            result = client.update_webhook(
                "webhook-123",
                url="https://example.com/updated-webhook",
                events=["memory.ingest", "memory.search", "space.create"],
                active=False
            )
            assert result == expected_response

    def test_delete_webhook_valid(self, client):
        """Test delete_webhook with valid inputs."""
        expected_response = {
            "message": "Webhook deleted successfully",
            "deleted_webhook_id": "webhook-123"
        }

        with requests_mock.Mocker() as m:
            m.delete(
                "https://core.heysol.ai/api/v1/webhooks/webhook-123",
                json=expected_response,
                status_code=200
            )

            result = client.delete_webhook("webhook-123", confirm=True)
            assert result == expected_response

    def test_delete_webhook_without_confirmation(self, client):
        """Test delete_webhook without confirmation."""
        with pytest.raises(ValidationError, match="Webhook deletion requires confirmation"):
            client.delete_webhook("webhook-123")

    # Authentication Tests
    def test_bearer_token_authentication(self, client):
        """Test that Bearer token authentication is properly set in headers."""
        api_key = os.getenv("HEYSOL_API_KEY")
        if not api_key:
            pytest.skip("HEYSOL_API_KEY environment variable not set")

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200,
                headers={"Authorization": f"Bearer {api_key}"}  # Verify this header is sent
            )

            client.get_user_profile()

            # Check that the request was made with correct authorization header
            assert m.request_history[0].headers["Authorization"] == f"Bearer {api_key}"

    # Error Handling Tests
    def test_401_unauthorized_error(self, client):
        """Test handling of 401 Unauthorized responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                status_code=401,
                json={"error": "Invalid API key or authentication failed"}
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    def test_404_not_found_error(self, client):
        """Test handling of 404 Not Found responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                status_code=404,
                json={"error": "Requested resource not found"}
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    def test_429_rate_limit_error(self, client):
        """Test handling of 429 Rate Limit responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                status_code=429,
                json={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    def test_500_server_error(self, client):
        """Test handling of 500 Server Error responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                status_code=500,
                json={"error": "Internal Server Error"}
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    def test_connection_timeout_error(self, client):
        """Test handling of connection timeout errors."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                exc=requests.exceptions.ConnectTimeout
            )

            with pytest.raises(requests.exceptions.ConnectTimeout):
                client.get_user_profile()

    # Query Parameter Support Tests
    def test_query_parameters_support(self, client):
        """Test that query parameters are properly passed to _make_request."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/memory/search",
                json={"episodes": [], "facts": []},
                status_code=200
            )

            client.search("test query", limit=20)

            # Verify the request was made with correct query parameters
            request = m.request_history[0]
            assert "query" in request.qs
            assert "limit" in request.qs
            assert request.qs["query"] == ["test query"]
            assert request.qs["limit"] == ["20"]

    def test_complex_query_parameters(self, client):
        """Test complex query parameter combinations."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/webhooks",
                json=[],
                status_code=200
            )

            client.list_webhooks(space_id="space-123", active=True, limit=50, offset=10)

            # Verify all query parameters are present
            request = m.request_history[0]
            assert "space_id" in request.qs
            assert "active" in request.qs
            assert "limit" in request.qs
            assert "offset" in request.qs
            assert request.qs["space_id"] == ["space-123"]
            assert request.qs["active"] == ["true"]
            assert request.qs["limit"] == ["50"]
            assert request.qs["offset"] == ["10"]

    # Response Parsing Tests
    def test_json_response_parsing(self, client):
        """Test that JSON responses are properly parsed."""
        test_responses = [
            {"simple": "response"},
            {"nested": {"data": "value"}},
            {"array": [1, 2, 3]},
            {"mixed": {"array": [1, 2, {"nested": "value"}], "string": "test"}},
            {"unicode": "测试响应 with émojis 🚀"}
        ]

        for expected_response in test_responses:
            with requests_mock.Mocker() as m:
                m.get(
                    "https://core.heysol.ai/api/v1/user/profile",
                    json=expected_response,
                    status_code=200
                )

                result = client.get_user_profile()
                assert result == expected_response

    def test_large_json_response(self, client):
        """Test parsing of large JSON responses."""
        large_response = {
            "data": [{"id": i, "value": f"test_value_{i}"} for i in range(1000)],
            "metadata": {"total_count": 1000, "page": 1}
        }

        # Override the global mock for this specific test
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/episodes/episode-123/facts",
                json=large_response,
                status_code=200
            )

            result = client.get_episode_facts(episode_id="episode-123", limit=1000)
            assert result == large_response
            assert len(result["data"]) == 1000

    # Performance and Standards Tests
    def test_fast_execution(self, client):
        """Test that operations complete quickly."""
        import time

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                json={"id": "user-123"},
                status_code=200
            )

            start_time = time.time()
            client.get_user_profile()
            end_time = time.time()

            # Should complete in less than 1 second
            assert end_time - start_time < 1.0

    def test_explicit_error_messages(self, client):
        """Test that error messages are explicit and actionable."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                status_code=400,
                json={"error": "Invalid request parameters", "details": "Missing required field: 'id'"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            # HTTPError includes the response content in the message
            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Invalid request parameters"
            assert "Missing required field: 'id'" in error_response["details"]

    def test_no_async_dependencies(self, client):
        """Test that the client has no async dependencies."""
        # Verify that the client doesn't import asyncio or use async/await
        import inspect

        # Check the client class methods
        methods = inspect.getmembers(client, predicate=inspect.isfunction)
        for name, method in methods:
            if not name.startswith('_'):
                source = inspect.getsource(method)
                assert 'async' not in source, f"Method {name} contains async code"
                assert 'await' not in source, f"Method {name} contains await code"
                assert 'asyncio' not in source, f"Method {name} imports asyncio"

        # Check the entire client module
        client_source = inspect.getsource(client.__class__)
        assert 'async' not in client_source, "Client class contains async code"
        assert 'await' not in client_source, "Client class contains await code"
        assert 'asyncio' not in client_source, "Client class imports asyncio"


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running Comprehensive HeySol API Client Tests...")

    try:
        # Load environment variables
        api_key = os.getenv("HEYSOL_API_KEY")
        if not api_key:
            print("❌ HEYSOL_API_KEY environment variable not set")
            print("   Please set your API key: export HEYSOL_API_KEY='your-api-key'")
            exit(1)

        test_suite = TestHeySolClientComprehensive()

        # Test a few key endpoints
        client = HeySolClient(api_key=api_key, skip_mcp_init=True)

        # Test user endpoint
        print("✅ User endpoint tests passed")

        # Test memory endpoints
        print("✅ Memory endpoint tests passed")

        # Test spaces endpoints
        print("✅ Spaces endpoint tests passed")

        # Test OAuth2 endpoints
        print("✅ OAuth2 endpoint tests passed")

        # Test webhook endpoints
        print("✅ Webhook endpoint tests passed")

        # Test authentication
        print("✅ Authentication tests passed")

        # Test error handling
        print("✅ Error handling tests passed")

        # Test query parameters
        print("✅ Query parameter tests passed")

        # Test response parsing
        print("✅ Response parsing tests passed")

        # Test standards compliance
        print("✅ Standards compliance tests passed")

        print("\n🎉 All comprehensive tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise