"""
Tests for API endpoints with valid inputs in the HeySol API client.
"""

import pytest
import requests_mock
from unittest.mock import Mock, patch
import json

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import HeySolError


class TestEndpointsValid:
    """Test API endpoints with valid inputs."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        config = HeySolConfig(api_key="test-api-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None
            client = HeySolClient(config=config)
            yield client

    @pytest.fixture
    def mock_async_client(self):
        """Create a mock async client for testing."""
        config = HeySolConfig(api_key="test-api-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None
            client = AsyncHeySolClient(config=config)
            yield client

    def test_get_user_profile_valid_sync(self, mock_client):
        """Test get_user_profile with valid response."""
        expected_profile = {
            "id": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "role": "admin"
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json=expected_profile,
                status_code=200
            )

            profile = mock_client.get_user_profile()
            assert profile == expected_profile

    def test_get_user_profile_valid_async(self, mock_async_client):
        """Test get_user_profile with valid response for async client."""
        expected_profile = {
            "id": "user-123",
            "name": "Test User",
            "email": "test@example.com",
            "role": "admin"
        }

        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json=expected_profile,
                status_code=200
            )

            import asyncio
            async def test():
                profile = await mock_async_client.get_user_profile()
                assert profile == expected_profile

            asyncio.run(test())

    def test_get_spaces_valid_sync(self, mock_client):
        """Test get_spaces with valid response."""
        expected_spaces = [
            {
                "id": "space-1",
                "name": "Clinical Trials",
                "description": "Space for clinical trial data",
                "writable": True
            },
            {
                "id": "space-2",
                "name": "Research",
                "description": "Space for research data",
                "writable": True
            }
        ]

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {
                "content": [{"type": "text", "text": json.dumps(expected_spaces)}]
            }

            spaces = mock_client.get_spaces()
            assert spaces == expected_spaces

    def test_get_spaces_valid_async(self, mock_async_client):
        """Test get_spaces with valid response for async client."""
        expected_spaces = [
            {
                "id": "space-1",
                "name": "Clinical Trials",
                "description": "Space for clinical trial data",
                "writable": True
            }
        ]

        with patch.object(mock_async_client, '_call_tool') as mock_call:
            mock_call.return_value = expected_spaces

            import asyncio
            async def test():
                spaces = await mock_async_client.get_spaces()
                assert spaces == expected_spaces

            asyncio.run(test())

    def test_create_space_valid_sync(self, mock_client):
        """Test create_space with valid inputs."""
        expected_response = {
            "space": {
                "id": "space-123",
                "name": "Test Space",
                "description": "A test space"
            }
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces",
                json=expected_response,
                status_code=201
            )

            space_id = mock_client.create_space("Test Space", "A test space")
            assert space_id == "space-123"

    def test_create_space_valid_async(self, mock_async_client):
        """Test create_space with valid inputs for async client."""
        expected_response = {
            "space": {
                "id": "space-123",
                "name": "Test Space",
                "description": "A test space"
            }
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces",
                json=expected_response,
                status_code=201
            )

            import asyncio
            async def test():
                space_id = await mock_async_client.create_space("Test Space", "A test space")
                assert space_id == "space-123"

            asyncio.run(test())

    def test_get_or_create_space_existing_sync(self, mock_client):
        """Test get_or_create_space when space already exists."""
        with patch.object(mock_client, 'get_spaces') as mock_get_spaces:
            mock_get_spaces.return_value = [
                {"id": "space-123", "name": "Existing Space", "writable": True}
            ]

            space_id = mock_client.get_or_create_space("Existing Space")
            assert space_id == "space-123"

    def test_get_or_create_space_new_sync(self, mock_client):
        """Test get_or_create_space when creating new space."""
        with patch.object(mock_client, 'get_spaces') as mock_get_spaces, \
             patch.object(mock_client, 'create_space') as mock_create:

            mock_get_spaces.return_value = []
            mock_create.return_value = "space-456"

            space_id = mock_client.get_or_create_space("New Space", "Description")
            assert space_id == "space-456"
            mock_create.assert_called_once_with("New Space", "Description")

    def test_ingest_valid_sync(self, mock_client):
        """Test ingest with valid inputs."""
        expected_result = {
            "id": "ingest-123",
            "status": "success",
            "message": "Data ingested successfully"
        }

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = expected_result

            result = mock_client.ingest(
                message="Test clinical data",
                space_id="space-123",
                tags=["clinical", "test"]
            )

            assert result == expected_result

    def test_ingest_valid_async(self, mock_async_client):
        """Test ingest with valid inputs for async client."""
        expected_result = {
            "id": "ingest-123",
            "status": "success",
            "message": "Data ingested successfully"
        }

        with patch.object(mock_async_client, '_call_tool') as mock_call:
            mock_call.return_value = expected_result

            import asyncio
            async def test():
                result = await mock_async_client.ingest(
                    message="Test clinical data",
                    space_id="space-123",
                    tags=["clinical", "test"]
                )
                assert result == expected_result

            asyncio.run(test())

    def test_search_valid_sync(self, mock_client):
        """Test search with valid inputs."""
        expected_result = {
            "episodes": [
                {
                    "id": "episode-1",
                    "content": "Clinical trial shows promising results",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            ],
            "facts": [
                {
                    "id": "fact-1",
                    "content": "Drug X shows 80% efficacy",
                    "confidence": 0.95
                }
            ]
        }

        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = expected_result

            result = mock_client.search("clinical trial", limit=10)
            assert result == expected_result

    def test_search_valid_async(self, mock_async_client):
        """Test search with valid inputs for async client."""
        expected_result = {
            "episodes": [
                {
                    "id": "episode-1",
                    "content": "Clinical trial shows promising results",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            ],
            "facts": []
        }

        with patch.object(mock_async_client, '_call_tool') as mock_call:
            mock_call.return_value = expected_result

            import asyncio
            async def test():
                result = await mock_async_client.search("clinical trial", limit=10)
                assert result == expected_result

            asyncio.run(test())

    def test_get_ingestion_logs_valid_sync(self, mock_client):
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
                "https://core.heysol.ai/api/v1/logs",
                json=expected_logs,
                status_code=200
            )

            logs = mock_client.get_ingestion_logs(limit=50)
            assert logs == expected_logs

    def test_get_specific_log_valid_sync(self, mock_client):
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
                "https://core.heysol.ai/api/v1/logs/log-123",
                json=expected_log,
                status_code=200
            )

            log = mock_client.get_specific_log("log-123")
            assert log == expected_log

    def test_delete_log_entry_valid_sync(self, mock_client):
        """Test delete_log_entry with valid inputs."""
        expected_response = {
            "message": "Log entry deleted successfully",
            "deleted_id": "log-123"
        }

        with requests_mock.Mocker() as m:
            m.delete(
                "https://core.heysol.ai/api/v1/logs/log-123",
                json=expected_response,
                status_code=200
            )

            result = mock_client.delete_log_entry("log-123")
            assert result == expected_response

    def test_bulk_space_operations_valid_sync(self, mock_client):
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
            m.post(
                "https://core.heysol.ai/api/v1/spaces/bulk",
                json=expected_response,
                status_code=200
            )

            result = mock_client.bulk_space_operations(operations)
            assert result == expected_response

    def test_get_space_details_valid_sync(self, mock_client):
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
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json=expected_details,
                status_code=200
            )

            details = mock_client.get_space_details("space-123")
            assert details == expected_details

    def test_update_space_valid_sync(self, mock_client):
        """Test update_space with valid inputs."""
        expected_response = {
            "id": "space-123",
            "name": "Updated Space",
            "description": "Updated description",
            "updated_at": "2024-01-02T00:00:00Z"
        }

        with requests_mock.Mocker() as m:
            m.patch(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json=expected_response,
                status_code=200
            )

            result = mock_client.update_space(
                "space-123",
                name="Updated Space",
                description="Updated description"
            )
            assert result == expected_response

    def test_delete_space_valid_sync(self, mock_client):
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

            result = mock_client.delete_space("space-123")
            assert result == expected_response

    def test_add_data_to_ingestion_queue_valid_sync(self, mock_client):
        """Test add_data_to_ingestion_queue with valid inputs."""
        expected_response = {
            "queue_id": "queue-123",
            "status": "queued",
            "estimated_processing_time": "30s"
        }

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/ingestion/queue",
                json=expected_response,
                status_code=202
            )

            result = mock_client.add_data_to_ingestion_queue(
                data={"clinical_data": "test"},
                priority="high",
                tags=["urgent"]
            )
            assert result == expected_response

    def test_get_episode_facts_valid_sync(self, mock_client):
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
                "https://core.heysol.ai/api/v1/episodes/facts",
                json=expected_facts,
                status_code=200
            )

            facts = mock_client.get_episode_facts(episode_id="episode-123", limit=50)
            assert facts == expected_facts

    def test_search_knowledge_graph_valid_sync(self, mock_client):
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
            m.get(
                "https://core.heysol.ai/api/v1/knowledge-graph/search",
                json=expected_result,
                status_code=200
            )

            result = mock_client.search_knowledge_graph("clinical trial", limit=10)
            assert result == expected_result