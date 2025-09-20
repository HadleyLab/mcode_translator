"""
Tests for API endpoints with invalid inputs in the HeySol API client.
"""

import pytest
import requests_mock
from unittest.mock import Mock, patch

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import ValidationError, APIError, NotFoundError


class TestEndpointsInvalid:
    """Test API endpoints with invalid inputs."""

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

    def test_create_space_empty_name_sync(self, mock_client):
        """Test create_space with empty name."""
        with pytest.raises(ValidationError, match="Space name is required"):
            mock_client.create_space("", "Description")

    def test_create_space_empty_name_async(self, mock_async_client):
        """Test create_space with empty name for async client."""
        import asyncio
        async def test():
            with pytest.raises(ValidationError, match="Space name is required"):
                await mock_async_client.create_space("", "Description")

        asyncio.run(test())

    def test_create_space_none_name_sync(self, mock_client):
        """Test create_space with None name."""
        with pytest.raises(ValidationError, match="Space name is required"):
            mock_client.create_space(None, "Description")

    def test_ingest_empty_message_sync(self, mock_client):
        """Test ingest with empty message."""
        with pytest.raises(ValidationError, match="Message is required for ingestion"):
            mock_client.ingest("", space_id="space-123")

    def test_ingest_empty_message_async(self, mock_async_client):
        """Test ingest with empty message for async client."""
        import asyncio
        async def test():
            with pytest.raises(ValidationError, match="Message is required for ingestion"):
                await mock_async_client.ingest("", space_id="space-123")

        asyncio.run(test())

    def test_ingest_none_message_sync(self, mock_client):
        """Test ingest with None message."""
        with pytest.raises(ValidationError, match="Message is required for ingestion"):
            mock_client.ingest(None, space_id="space-123")

    def test_search_empty_query_sync(self, mock_client):
        """Test search with empty query."""
        with pytest.raises(ValidationError, match="Search query is required"):
            mock_client.search("")

    def test_search_empty_query_async(self, mock_async_client):
        """Test search with empty query for async client."""
        import asyncio
        async def test():
            with pytest.raises(ValidationError, match="Search query is required"):
                await mock_async_client.search("")

        asyncio.run(test())

    def test_search_invalid_limit_sync(self, mock_client):
        """Test search with invalid limit."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            mock_client.search("test query", limit=0)

        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            mock_client.search("test query", limit=101)

    def test_search_invalid_limit_async(self, mock_async_client):
        """Test search with invalid limit for async client."""
        import asyncio
        async def test():
            with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
                await mock_async_client.search("test query", limit=0)

        asyncio.run(test())

    def test_get_specific_log_empty_id_sync(self, mock_client):
        """Test get_specific_log with empty log ID."""
        with pytest.raises(ValidationError, match="Log ID is required"):
            mock_client.get_specific_log("")

    def test_get_specific_log_empty_id_async(self, mock_async_client):
        """Test get_specific_log with empty log ID for async client."""
        import asyncio
        async def test():
            with pytest.raises(ValidationError, match="Log ID is required"):
                await mock_async_client.get_specific_log("")

        asyncio.run(test())

    def test_get_specific_log_none_id_sync(self, mock_client):
        """Test get_specific_log with None log ID."""
        with pytest.raises(ValidationError, match="Log ID is required"):
            mock_client.get_specific_log(None)

    def test_delete_log_entry_empty_id_sync(self, mock_client):
        """Test delete_log_entry with empty log ID."""
        with pytest.raises(ValidationError, match="Log ID is required"):
            mock_client.delete_log_entry("")

    def test_delete_log_entry_none_id_sync(self, mock_client):
        """Test delete_log_entry with None log ID."""
        with pytest.raises(ValidationError, match="Log ID is required"):
            mock_client.delete_log_entry(None)

    def test_bulk_space_operations_empty_list_sync(self, mock_client):
        """Test bulk_space_operations with empty operations list."""
        with pytest.raises(ValidationError, match="Operations list cannot be empty"):
            mock_client.bulk_space_operations([])

    def test_bulk_space_operations_none_list_sync(self, mock_client):
        """Test bulk_space_operations with None operations list."""
        with pytest.raises(ValidationError, match="Operations list cannot be empty"):
            mock_client.bulk_space_operations(None)

    def test_get_space_details_empty_id_sync(self, mock_client):
        """Test get_space_details with empty space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            mock_client.get_space_details("")

    def test_get_space_details_none_id_sync(self, mock_client):
        """Test get_space_details with None space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            mock_client.get_space_details(None)

    def test_update_space_empty_id_sync(self, mock_client):
        """Test update_space with empty space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            mock_client.update_space("", name="New Name")

    def test_update_space_no_fields_sync(self, mock_client):
        """Test update_space with no fields to update."""
        with pytest.raises(ValidationError, match="At least one field must be provided for update"):
            mock_client.update_space("space-123")

    def test_delete_space_empty_id_sync(self, mock_client):
        """Test delete_space with empty space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            mock_client.delete_space("")

    def test_delete_space_none_id_sync(self, mock_client):
        """Test delete_space with None space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            mock_client.delete_space(None)

    def test_search_knowledge_graph_empty_query_sync(self, mock_client):
        """Test search_knowledge_graph with empty query."""
        with pytest.raises(ValidationError, match="Search query is required"):
            mock_client.search_knowledge_graph("")

    def test_search_knowledge_graph_invalid_depth_sync(self, mock_client):
        """Test search_knowledge_graph with invalid depth."""
        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            mock_client.search_knowledge_graph("test", depth=0)

        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            mock_client.search_knowledge_graph("test", depth=6)

    def test_search_knowledge_graph_invalid_limit_sync(self, mock_client):
        """Test search_knowledge_graph with invalid limit."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            mock_client.search_knowledge_graph("test", limit=0)

        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            mock_client.search_knowledge_graph("test", limit=101)

    def test_get_spaces_invalid_response_format_sync(self, mock_client):
        """Test get_spaces with invalid response format."""
        with patch.object(mock_client, '_call_tool') as mock_call:
            # Return invalid format that can't be parsed
            mock_call.return_value = {
                "content": [{"type": "text", "text": "invalid json"}]
            }

            spaces = mock_client.get_spaces()
            assert spaces == []  # Should return empty list on parse failure

    def test_create_space_server_error_sync(self, mock_client):
        """Test create_space with server error response."""
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces",
                json={"error": "Internal server error"},
                status_code=500
            )

            with pytest.raises(APIError):
                mock_client.create_space("Test Space")

    def test_get_user_profile_not_found_sync(self, mock_client):
        """Test get_user_profile with 404 response."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Profile not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.get_user_profile()

    def test_ingest_invalid_space_id_sync(self, mock_client):
        """Test ingest with invalid space ID."""
        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.side_effect = APIError("Invalid space ID")

            with pytest.raises(APIError, match="Invalid space ID"):
                mock_client.ingest("Test message", space_id="invalid-space")

    def test_search_invalid_parameters_sync(self, mock_client):
        """Test search with invalid parameter combinations."""
        # Test with invalid date format
        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"episodes": [], "facts": []}

            # Should not raise error for invalid date format (handled by API)
            result = mock_client.search("test", start_time="invalid-date")
            assert "episodes" in result
            assert "facts" in result

    def test_get_ingestion_logs_invalid_response_sync(self, mock_client):
        """Test get_ingestion_logs with invalid response format."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/logs",
                text="invalid json",
                status_code=200
            )

            with pytest.raises(APIError, match="Failed to parse JSON response"):
                mock_client.get_ingestion_logs()

    def test_get_specific_log_not_found_sync(self, mock_client):
        """Test get_specific_log with 404 response."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/logs/log-123",
                json={"error": "Log not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.get_specific_log("log-123")

    def test_delete_log_entry_not_found_sync(self, mock_client):
        """Test delete_log_entry with 404 response."""
        with requests_mock.Mocker() as m:
            m.delete(
                "https://core.heysol.ai/api/v1/logs/log-123",
                json={"error": "Log not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.delete_log_entry("log-123")

    def test_bulk_space_operations_invalid_operation_sync(self, mock_client):
        """Test bulk_space_operations with invalid operation format."""
        operations = [
            {"invalid_field": "value"}  # Missing required 'type' field
        ]

        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces/bulk",
                json={"error": "Invalid operation format"},
                status_code=400
            )

            with pytest.raises(APIError):
                mock_client.bulk_space_operations(operations)

    def test_get_space_details_not_found_sync(self, mock_client):
        """Test get_space_details with 404 response."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json={"error": "Space not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.get_space_details("space-123")

    def test_update_space_not_found_sync(self, mock_client):
        """Test update_space with 404 response."""
        with requests_mock.Mocker() as m:
            m.patch(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json={"error": "Space not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.update_space("space-123", name="New Name")

    def test_delete_space_not_found_sync(self, mock_client):
        """Test delete_space with 404 response."""
        with requests_mock.Mocker() as m:
            m.delete(
                "https://core.heysol.ai/api/v1/spaces/space-123",
                json={"error": "Space not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.delete_space("space-123")

    def test_add_data_to_ingestion_queue_invalid_data_sync(self, mock_client):
        """Test add_data_to_ingestion_queue with invalid data format."""
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/ingestion/queue",
                json={"error": "Invalid data format"},
                status_code=400
            )

            with pytest.raises(APIError):
                mock_client.add_data_to_ingestion_queue("invalid data format")

    def test_get_episode_facts_invalid_response_sync(self, mock_client):
        """Test get_episode_facts with invalid response format."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/episodes/facts",
                text="not json",
                status_code=200
            )

            with pytest.raises(APIError, match="Failed to parse JSON response"):
                mock_client.get_episode_facts()

    def test_search_knowledge_graph_not_found_sync(self, mock_client):
        """Test search_knowledge_graph with 404 response."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/v1/knowledge-graph/search",
                json={"error": "Knowledge graph not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError):
                mock_client.search_knowledge_graph("nonexistent")