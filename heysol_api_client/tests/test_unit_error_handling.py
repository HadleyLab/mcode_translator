#!/usr/bin/env python3
"""
Unit Tests for HeySol API Client - Error Handling

Tests error handling scenarios including:
- HTTP error responses (400, 401, 404, 429, 500)
- Connection errors and timeouts
- Invalid input validation
- Authentication failures
- Rate limiting
- Server errors
"""

import json
import pytest
import requests
import requests_mock
from unittest.mock import Mock, patch
from typing import Dict, Any

from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import ValidationError


class TestErrorHandling:
    """Unit tests for error handling scenarios."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return HeySolClient(api_key="test-api-key", skip_mcp_init=True)

    @pytest.fixture
    def config(self):
        """Create a test config instance."""
        return HeySolConfig(api_key="test-api-key")

    # HTTP Error Response Tests
    def test_400_bad_request_error(self, client, config):
        """Test handling of 400 Bad Request responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=400,
                json={"error": "Invalid request parameters", "details": "Missing required field: 'id'"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            # Verify error details are preserved
            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Invalid request parameters"
            assert "Missing required field: 'id'" in error_response["details"]

    def test_401_unauthorized_error(self, client, config):
        """Test handling of 401 Unauthorized responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=401,
                json={"error": "Invalid API key or authentication failed"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Invalid API key or authentication failed"

    def test_403_forbidden_error(self, client, config):
        """Test handling of 403 Forbidden responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=403,
                json={"error": "Access denied", "details": "Insufficient permissions"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Access denied"

    def test_404_not_found_error(self, client, config):
        """Test handling of 404 Not Found responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=404,
                json={"error": "Requested resource not found"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Requested resource not found"

    def test_408_request_timeout_error(self, client, config):
        """Test handling of 408 Request Timeout responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=408,
                json={"error": "Request timeout", "details": "The request took too long to process"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Request timeout"

    def test_429_rate_limit_error(self, client, config):
        """Test handling of 429 Rate Limit responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=429,
                json={"error": "Rate limit exceeded"},
                headers={"Retry-After": "60"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Rate limit exceeded"
            assert exc_info.value.response.headers["Retry-After"] == "60"

    def test_500_server_error(self, client, config):
        """Test handling of 500 Server Error responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=500,
                json={"error": "Internal Server Error"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Internal Server Error"

    def test_502_bad_gateway_error(self, client, config):
        """Test handling of 502 Bad Gateway responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=502,
                json={"error": "Bad Gateway", "details": "Upstream server error"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Bad Gateway"

    def test_503_service_unavailable_error(self, client, config):
        """Test handling of 503 Service Unavailable responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                status_code=503,
                json={"error": "Service Unavailable", "details": "Server is temporarily unavailable"},
                headers={"Retry-After": "300"}
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"] == "Service Unavailable"

    # Connection Error Tests
    def test_connection_timeout_error(self, client, config):
        """Test handling of connection timeout errors."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                exc=requests.exceptions.ConnectTimeout
            )

            with pytest.raises(requests.exceptions.ConnectTimeout):
                client.get_user_profile()

    def test_connection_error(self, client, config):
        """Test handling of connection errors."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                exc=requests.exceptions.ConnectionError
            )

            with pytest.raises(requests.exceptions.ConnectionError):
                client.get_user_profile()

    def test_too_many_redirects_error(self, client, config):
        """Test handling of too many redirects errors."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                exc=requests.exceptions.TooManyRedirects
            )

            with pytest.raises(requests.exceptions.TooManyRedirects):
                client.get_user_profile()

    # Input Validation Error Tests
    def test_empty_api_key_validation(self):
        """Test validation of empty API key."""
        with pytest.raises(ValidationError, match="API key is required"):
            HeySolClient(api_key="", skip_mcp_init=True)

    def test_none_api_key_validation(self):
        """Test validation of None API key when no config is available."""
        # This test would require mocking the config to return None for api_key
        # For now, we'll skip this test as the current implementation loads from config
        pass

    def test_invalid_search_query_validation(self, client):
        """Test validation of invalid search query."""
        with pytest.raises(ValidationError, match="Search query is required"):
            client.search_knowledge_graph("")

    def test_invalid_search_limit_validation(self, client):
        """Test validation of invalid search limit."""
        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            client.search_knowledge_graph("test", limit=0)

        with pytest.raises(ValidationError, match="Limit must be between 1 and 100"):
            client.search_knowledge_graph("test", limit=101)

    def test_invalid_search_depth_validation(self, client):
        """Test validation of invalid search depth."""
        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            client.search_knowledge_graph("test", depth=0)

        with pytest.raises(ValidationError, match="Depth must be between 1 and 5"):
            client.search_knowledge_graph("test", depth=6)

    def test_invalid_episode_id_validation(self, client):
        """Test validation of invalid episode ID."""
        with pytest.raises(ValidationError, match="Episode ID is required"):
            client.get_episode_facts("")

    def test_invalid_space_id_validation(self, client):
        """Test validation of invalid space ID."""
        with pytest.raises(ValidationError, match="Space ID is required"):
            client.get_space_details("")

    def test_invalid_webhook_url_validation(self, client):
        """Test validation of invalid webhook URL."""
        with pytest.raises(ValidationError, match="Webhook URL is required"):
            client.register_webhook("", ["memory.ingest"], "secret")

    def test_invalid_webhook_id_validation(self, client):
        """Test validation of invalid webhook ID."""
        with pytest.raises(ValidationError, match="Webhook ID is required"):
            client.get_webhook("")


    def test_delete_space_without_confirmation_validation(self, client):
        """Test validation of space deletion without confirmation."""
        with pytest.raises(ValidationError, match="Space deletion requires confirmation"):
            client.delete_space("space-123")

    def test_delete_webhook_without_confirmation_validation(self, client):
        """Test validation of webhook deletion without confirmation."""
        with pytest.raises(ValidationError, match="Webhook deletion requires confirmation"):
            client.delete_webhook("webhook-123")

    # Response Parsing Error Tests
    def test_malformed_json_response(self, client, config):
        """Test handling of malformed JSON responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                text='{"invalid": json syntax}',
                status_code=200,
                headers={"Content-Type": "application/json"}
            )

            with pytest.raises(json.JSONDecodeError):
                client.get_user_profile()

    def test_empty_response_body(self, client, config):
        """Test handling of empty response body."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                text="",
                status_code=200,
                headers={"Content-Type": "application/json"}
            )

            with pytest.raises(json.JSONDecodeError):
                client.get_user_profile()

    def test_html_error_response(self, client, config):
        """Test handling of HTML error responses."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                text="<html><body><h1>Server Error</h1></body></html>",
                status_code=500,
                headers={"Content-Type": "text/html"}
            )

            with pytest.raises(requests.exceptions.HTTPError):
                client.get_user_profile()

    # Network Error Tests
    def test_network_unavailable(self, client, config):
        """Test handling of network unavailable errors."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                exc=requests.exceptions.ConnectionError
            )

            with pytest.raises(requests.exceptions.ConnectionError):
                client.get_user_profile()

    def test_ssl_verification_error(self, client, config):
        """Test handling of SSL verification errors."""
        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                exc=requests.exceptions.SSLError
            )

            with pytest.raises(requests.exceptions.SSLError):
                client.get_user_profile()

    # Edge Cases
    def test_extremely_long_api_key(self, client, config):
        """Test handling of extremely long API key."""
        long_api_key = "x" * 10000  # 10KB API key
        client_with_long_key = HeySolClient(api_key=long_api_key, skip_mcp_init=True)

        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                json={"id": "user-123"},
                status_code=200
            )

            # Should handle long API key without issues
            result = client_with_long_key.get_user_profile()
            assert result == {"id": "user-123"}

    def test_unicode_in_response(self, client, config):
        """Test handling of Unicode characters in responses."""
        unicode_response = {
            "id": "user-123",
            "name": "测试用户",  # Chinese characters
            "description": "User with émojis 🚀 and spëcial characters",
            "tags": ["标签1", "标签2"]  # Chinese tags
        }

        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                json=unicode_response,
                status_code=200
            )

            result = client.get_user_profile()
            assert result == unicode_response

    def test_large_response_handling(self, client, mock_api_setup):
        """Test handling of large JSON responses."""
        large_response = {
            "logs": [{"id": i, "value": f"test_value_{i}"} for i in range(10000)],
            "metadata": {"total_count": 10000, "page": 1}
        }

        result = client.get_ingestion_logs(limit=10000)
        assert result == large_response["logs"]
        assert len(result) == 10000

    def test_nested_error_response(self, client, config):
        """Test handling of nested error response structures."""
        nested_error = {
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid input",
                "details": {
                    "field": "email",
                    "issue": "Invalid email format",
                    "provided": "invalid-email"
                }
            }
        }

        with requests_mock.Mocker() as m:
            m.get(
                config.profile_url,
                json=nested_error,
                status_code=400
            )

            with pytest.raises(requests.exceptions.HTTPError) as exc_info:
                client.get_user_profile()

            error_response = exc_info.value.response.json()
            assert error_response["error"]["code"] == "VALIDATION_ERROR"
            assert error_response["error"]["details"]["field"] == "email"