#!/usr/bin/env python3
"""
Unit Tests for HeySol API Client - API Endpoints

Tests individual API endpoints with mock responses, covering:
- User endpoints
- Memory endpoints
- Spaces endpoints
- OAuth2 endpoints
- Webhook endpoints
- Response parsing and validation
"""

import json
import pytest
import requests_mock
from typing import Dict, Any, List
from unittest.mock import Mock

from heysol.client import HeySolClient
from heysol.exceptions import ValidationError


class TestAPIEndpoints:
    """Unit tests for individual API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return HeySolClient(api_key="test-api-key", skip_mcp_init=True)

    @pytest.fixture
    def mock_setup(self, requests_mock):
        """Set up comprehensive mocks for all API endpoints."""
        # User endpoints
        requests_mock.get(
            "https://core.heysol.ai/api/v1/user/profile",
            json={
                "id": "user-123",
                "name": "Test User",
                "email": "test@example.com",
                "role": "admin",
                "created_at": "2024-01-01T00:00:00Z"
            }
        )

        # Memory endpoints
        requests_mock.post(
            "https://core.heysol.ai/api/v1/memory/knowledge-graph/search",
            json={"nodes": [], "edges": []}
        )
        requests_mock.post(
            "https://core.heysol.ai/api/v1/memory/ingestion/queue",
            json={"queue_id": "queue-123"}
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/episodes/episode-123/facts",
            json=[]
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/memory/logs",
            json=[]
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/memory/logs/log-123",
            json={"id": "log-123"}
        )

        # Spaces endpoints
        requests_mock.get(
            "https://core.heysol.ai/api/v1/spaces",
            json=[]
        )
        requests_mock.post(
            "https://core.heysol.ai/api/v1/spaces",
            json={"id": "space-123", "name": "Test Space"}
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/spaces/space-123",
            json={"id": "space-123", "name": "Test Space"}
        )
        requests_mock.put(
            "https://core.heysol.ai/api/v1/spaces/space-123",
            json={"id": "space-123", "name": "Updated Space"}
        )
        requests_mock.delete(
            "https://core.heysol.ai/api/v1/spaces/space-123",
            json={"deleted": True}
        )

        # OAuth2 endpoints
        requests_mock.get(
            "https://core.heysol.ai/api/v1/oauth2/authorize",
            json={"authorization_url": "https://example.com/auth"}
        )
        requests_mock.post(
            "https://core.heysol.ai/api/v1/oauth2/token",
            json={"access_token": "token-123"}
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/oauth2/userinfo",
            json={"sub": "user-123"}
        )

        # Webhook endpoints
        requests_mock.post(
            "https://core.heysol.ai/api/v1/webhooks",
            json={"webhook_id": "webhook-123"}
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/webhooks",
            json=[]
        )
        requests_mock.get(
            "https://core.heysol.ai/api/v1/webhooks/webhook-123",
            json={"id": "webhook-123"}
        )
        requests_mock.put(
            "https://core.heysol.ai/api/v1/webhooks/webhook-123",
            json={"id": "webhook-123"}
        )
        requests_mock.delete(
            "https://core.heysol.ai/api/v1/webhooks/webhook-123",
            json={"deleted": True}
        )

    # User Endpoints
    def test_get_user_profile_valid(self, client, mock_setup):
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

    def test_get_user_profile_invalid_json(self, client):
        """Test get_user_profile with invalid JSON response."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/user/profile",
                text="Invalid JSON",
                status_code=200,
                headers={"Content-Type": "application/json"}
            )

            with pytest.raises(json.JSONDecodeError):
                client.get_user_profile()

    # Memory Endpoints
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

    def test_search_knowledge_graph_boundary_conditions(self, client):
        """Test search_knowledge_graph with boundary conditions."""
        # Test minimum limit
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/knowledge-graph/search",
                json={"nodes": [], "edges": []},
                status_code=200
            )

            result = client.search_knowledge_graph("test", limit=1)
            assert result == {"nodes": [], "edges": []}

        # Test maximum limit
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/knowledge-graph/search",
                json={"nodes": [], "edges": []},
                status_code=200
            )

            result = client.search_knowledge_graph("test", limit=100)
            assert result == {"nodes": [], "edges": []}

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

    # Spaces Endpoints
    def test_get_spaces_valid(self, client):
        """Test get_spaces with valid response."""
        expected_spaces = [
            {
                "id": "space-1",
                "name": "Research Space",
                "description": "Clinical research data",
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "space-2",
                "name": "Patient Data",
                "description": "Patient information",
                "created_at": "2024-01-02T00:00:00Z"
            }
        ]

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/spaces",
                json=expected_spaces,
                status_code=200
            )

            spaces = client.get_spaces()
            assert spaces == expected_spaces

    def test_create_space_valid(self, client):
        """Test create_space with valid inputs."""
        expected_response = {
            "id": "space-123",
            "name": "New Research Space",
            "description": "A new space for research",
            "created_at": "2024-01-01T00:00:00Z"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/memory/spaces",
                json=expected_response,
                status_code=201
            )

            result = client.create_space("New Research Space", "A new space for research")
            assert result == "space-123"

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

    # OAuth2 Endpoints
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

    # Webhook Endpoints
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