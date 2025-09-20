#!/usr/bin/env python3
"""
Comprehensive Endpoint Testing Suite for HeySol API Client

Tests all available endpoints with valid and invalid inputs, error handling,
and response codes.
"""

import os
import json
import time
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import HeySol components
from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import (
    HeySolError,
    AuthenticationError,
    ValidationError,
    NotFoundError,
    RateLimitError,
    APIError
)


class TestEndpointValidation:
    """Test suite for endpoint validation with various inputs."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

        # Mock the _make_request method to avoid actual API calls
        self.mock_response = Mock()
        self.mock_response.status_code = 200
        self.mock_response.headers = {"Content-Type": "application/json"}
        self.mock_response.json.return_value = {"status": "success"}

    def test_get_spaces_valid_inputs(self):
        """Test get_spaces endpoint with valid inputs."""
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value=[]):
                result = self.client.get_spaces()
                assert isinstance(result, list)

    def test_get_spaces_with_pagination(self):
        """Test get_spaces with pagination parameters."""
        # This would test pagination if implemented
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value=[]):
                result = self.client.get_spaces()
                assert isinstance(result, list)

    def test_create_space_valid_inputs(self):
        """Test create_space endpoint with valid inputs."""
        test_space_name = "test_space"
        test_description = "Test space description"

        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"id": "test-space-id"}):
                space_id = self.client.create_space(test_space_name, test_description)
                assert isinstance(space_id, str)
                assert len(space_id) > 0

    def test_create_space_invalid_inputs(self):
        """Test create_space endpoint with invalid inputs."""
        # Test empty name
        with pytest.raises(ValidationError, match="Space name is required"):
            self.client.create_space("", "description")

        # Test None name
        with pytest.raises(ValidationError, match="Space name is required"):
            self.client.create_space(None, "description")

    def test_ingest_valid_inputs(self):
        """Test ingest endpoint with valid inputs."""
        test_message = "Test message for ingestion"
        test_space_id = "test-space-id"

        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test-ingestion-id"}):
                result = self.client.ingest(test_message, space_id=test_space_id)
                assert isinstance(result, dict)
                assert "id" in result

    def test_ingest_invalid_inputs(self):
        """Test ingest endpoint with invalid inputs."""
        # Test empty message
        with pytest.raises(ValidationError, match="Message is required"):
            self.client.ingest("", space_id="test-space-id")

        # Test None message
        with pytest.raises(ValidationError, match="Message is required"):
            self.client.ingest(None, space_id="test-space-id")

    def test_search_valid_inputs(self):
        """Test search endpoint with valid inputs."""
        test_query = "test search query"
        test_limit = 10

        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_mcp_response', return_value={"episodes": [], "facts": []}):
                result = self.client.search(test_query, limit=test_limit)
                assert isinstance(result, dict)
                assert "episodes" in result
                assert "facts" in result

    def test_search_invalid_inputs(self):
        """Test search endpoint with invalid inputs."""
        # Test empty query
        with pytest.raises(ValidationError, match="Search query is required"):
            self.client.search("")

        # Test None query
        with pytest.raises(ValidationError, match="Search query is required"):
            self.client.search(None)

        # Test invalid limit
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            self.client.search("test", limit=0)

        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            self.client.search("test", limit=101)

    def test_get_user_profile_valid(self):
        """Test get_user_profile endpoint."""
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"id": "user-id", "email": "test@example.com"}):
                result = self.client.get_user_profile()
                assert isinstance(result, dict)
                assert "id" in result

    def test_log_operations_valid_inputs(self):
        """Test log operations with valid inputs."""
        test_log_id = "test-log-id"

        # Test get ingestion logs
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value=[]):
                result = self.client.get_ingestion_logs()
                assert isinstance(result, list)

        # Test get specific log
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"id": test_log_id}):
                result = self.client.get_specific_log(test_log_id)
                assert isinstance(result, dict)
                assert result["id"] == test_log_id

        # Test delete log entry
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"deleted": True}):
                result = self.client.delete_log_entry(test_log_id)
                assert isinstance(result, dict)

    def test_log_operations_invalid_inputs(self):
        """Test log operations with invalid inputs."""
        # Test get specific log with empty ID
        with pytest.raises(ValidationError, match="Log ID is required"):
            self.client.get_specific_log("")

        # Test get specific log with None ID
        with pytest.raises(ValidationError, match="Log ID is required"):
            self.client.get_specific_log(None)

        # Test delete log entry with empty ID
        with pytest.raises(ValidationError, match="Log ID is required"):
            self.client.delete_log_entry("")

        # Test delete log entry with None ID
        with pytest.raises(ValidationError, match="Log ID is required"):
            self.client.delete_log_entry(None)

    def test_space_operations_valid_inputs(self):
        """Test space operations with valid inputs."""
        test_space_id = "test-space-id"

        # Test get space details
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"id": test_space_id, "name": "test"}):
                result = self.client.get_space_details(test_space_id)
                assert isinstance(result, dict)
                assert result["id"] == test_space_id

        # Test update space
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"id": test_space_id, "name": "updated"}):
                result = self.client.update_space(test_space_id, name="updated")
                assert isinstance(result, dict)

        # Test delete space
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"deleted": True}):
                result = self.client.delete_space(test_space_id)
                assert isinstance(result, dict)

    def test_space_operations_invalid_inputs(self):
        """Test space operations with invalid inputs."""
        # Test get space details with empty ID
        with pytest.raises(ValidationError, match="Space ID is required"):
            self.client.get_space_details("")

        # Test get space details with None ID
        with pytest.raises(ValidationError, match="Space ID is required"):
            self.client.get_space_details(None)

        # Test update space with no changes
        with pytest.raises(ValidationError, match="At least one field must be provided"):
            self.client.update_space("test-id")

        # Test delete space with empty ID
        with pytest.raises(ValidationError, match="Space ID is required"):
            self.client.delete_space("")

        # Test delete space with None ID
        with pytest.raises(ValidationError, match="Space ID is required"):
            self.client.delete_space(None)

    def test_bulk_operations_valid_inputs(self):
        """Test bulk operations with valid inputs."""
        operations = [
            {"type": "create", "name": "test1"},
            {"type": "create", "name": "test2"}
        ]

        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"results": []}):
                result = self.client.bulk_space_operations(operations)
                assert isinstance(result, dict)

    def test_bulk_operations_invalid_inputs(self):
        """Test bulk operations with invalid inputs."""
        # Test empty operations
        with pytest.raises(ValidationError, match="Operations list cannot be empty"):
            self.client.bulk_space_operations([])

        # Test None operations
        with pytest.raises(ValidationError, match="Operations list cannot be empty"):
            self.client.bulk_space_operations(None)

    def test_knowledge_graph_operations(self):
        """Test knowledge graph operations."""
        test_query = "test knowledge query"

        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value={"entities": [], "relations": []}):
                result = self.client.search_knowledge_graph(test_query)
                assert isinstance(result, dict)

    def test_episode_facts_operations(self):
        """Test episode facts operations."""
        with patch.object(self.client, '_make_request', return_value=self.mock_response):
            with patch.object(self.client, '_parse_json_response', return_value=[]):
                result = self.client.get_episode_facts()
                assert isinstance(result, list)


class TestErrorHandling:
    """Test suite for error handling and response codes."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

    def test_401_unauthorized_error(self):
        """Test 401 Unauthorized error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                self.client.get_spaces()

    def test_404_not_found_error(self):
        """Test 404 Not Found error handling."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(NotFoundError, match="Requested resource not found"):
                self.client.get_space_details("nonexistent-space-id")

    def test_429_rate_limit_error(self):
        """Test 429 Rate Limit error handling."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "Rate limit exceeded"

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                self.client.get_spaces()

    def test_500_server_error(self):
        """Test 500 Server Error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(APIError, match="Server error"):
                self.client.get_spaces()

    def test_400_client_error(self):
        """Test 400 Client Error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(APIError, match="Client error"):
                self.client.get_spaces()

    def test_connection_timeout_error(self):
        """Test connection timeout error handling."""
        with patch.object(self.client.session, 'request', side_effect=requests.exceptions.Timeout):
            with pytest.raises(HeySolError, match="Request timeout"):
                self.client.get_spaces()

    def test_connection_error(self):
        """Test connection error handling."""
        with patch.object(self.client.session, 'request', side_effect=requests.exceptions.ConnectionError):
            with pytest.raises(HeySolError, match="Connection error"):
                self.client.get_spaces()

    def test_json_parse_error(self):
        """Test JSON parsing error handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect=json.JSONDecodeError("Invalid JSON", "", 0)

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with pytest.raises(APIError, match="Failed to parse JSON response"):
                self.client._parse_json_response(mock_response)

    def test_mcp_error_handling(self):
        """Test MCP error response handling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"error": "MCP error occurred"}

        with patch.object(self.client, '_make_request', return_value=mock_response):
            with patch.object(self.client, '_parse_mcp_response', side_effect=APIError("MCP error: MCP error occurred")):
                with pytest.raises(APIError, match="MCP error"):
                    self.client._call_tool("test_tool")


class TestDataSerialization:
    """Test suite for data serialization and deserialization."""

    def setup_method(self):
        """Setup test environment."""
        self.api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.config = HeySolConfig(api_key=self.api_key, log_level="WARNING")
        self.client = HeySolClient(config=self.config, use_oauth2=False)

    def test_json_serialization(self):
        """Test JSON serialization of request data."""
        test_data = {
            "message": "Test message with special chars: éñüñ",
            "spaceId": "test-space-id",
            "priority": "high",
            "tags": ["tag1", "tag2", "tag with spaces"],
            "metadata": {
                "key1": "value1",
                "nested": {"inner": "value"},
                "numbers": [1, 2, 3]
            }
        }

        # Test that data can be JSON serialized
        json_str = json.dumps(test_data)
        parsed_back = json.loads(json_str)

        assert parsed_back["message"] == test_data["message"]
        assert parsed_back["tags"] == test_data["tags"]
        assert parsed_back["metadata"]["nested"]["inner"] == "value"

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        unicode_messages = [
            "Test with émojis 🚀 and ümlauts",
            "测试消息 with Chinese characters",
            "Сообщение with Russian text",
            "🌟 Unicode test: αβγδε",
            "Mixed content: 123 éñüñ 🚀"
        ]

        for message in unicode_messages:
            with patch.object(self.client, '_make_request', return_value=Mock()):
                with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test-id"}):
                    # Should not raise UnicodeEncodeError
                    result = self.client.ingest(message, space_id="test-space")
                    assert isinstance(result, dict)

    def test_large_data_handling(self):
        """Test handling of large data payloads."""
        large_message = "A" * 10000  # 10KB message

        with patch.object(self.client, '_make_request', return_value=Mock()):
            with patch.object(self.client, '_parse_mcp_response', return_value={"id": "test-id"}):
                result = self.client.ingest(large_message, space_id="test-space")
                assert isinstance(result, dict)

    def test_nested_data_structures(self):
        """Test handling of nested data structures."""
        complex_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "deep_value": "deep",
                        "deep_array": [1, 2, {"nested": "value"}]
                    }
                }
            },
            "array_of_objects": [
                {"id": 1, "data": {"nested": "value1"}},
                {"id": 2, "data": {"nested": "value2"}}
            ]
        }

        # Test serialization
        json_str = json.dumps(complex_data)
        parsed_back = json.loads(json_str)

        assert parsed_back["level1"]["level2"]["level3"]["deep_value"] == "deep"
        assert len(parsed_back["array_of_objects"]) == 2
        assert parsed_back["array_of_objects"][0]["data"]["nested"] == "value1"

    def test_boolean_and_null_values(self):
        """Test handling of boolean and null values."""
        test_cases = [
            {"bool_true": True, "bool_false": False, "null_value": None},
            {"mixed": [True, False, None, "string", 123]},
            {"nested": {"bool": True, "null": None, "string": "test"}}
        ]

        for test_data in test_cases:
            json_str = json.dumps(test_data)
            parsed_back = json.loads(json_str)

            assert parsed_back == test_data

    def test_special_characters(self):
        """Test handling of special characters."""
        special_messages = [
            'Message with "quotes" and \'apostrophes\'',
            "Message with \n newlines \t and tabs",
            'Message with \\ backslashes and / forward slashes',
            "Message with {braces} and [brackets]",
            "Message with <html> & special chars"
        ]

        for message in special_messages:
            json_str = json.dumps({"message": message})
            parsed_back = json.loads(json_str)

            assert parsed_back["message"] == message


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running Endpoint Tests...")

    try:
        test_suite = TestEndpointValidation()
        test_suite.setup_method()

        # Test valid inputs
        test_suite.test_get_spaces_valid_inputs()
        print("✅ GET spaces valid inputs test passed")

        test_suite.test_create_space_valid_inputs()
        print("✅ Create space valid inputs test passed")

        test_suite.test_ingest_valid_inputs()
        print("✅ Ingest valid inputs test passed")

        test_suite.test_search_valid_inputs()
        print("✅ Search valid inputs test passed")

        # Test invalid inputs
        test_suite.test_create_space_invalid_inputs()
        print("✅ Create space invalid inputs test passed")

        test_suite.test_ingest_invalid_inputs()
        print("✅ Ingest invalid inputs test passed")

        test_suite.test_search_invalid_inputs()
        print("✅ Search invalid inputs test passed")

        print("\n🎉 All endpoint tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise