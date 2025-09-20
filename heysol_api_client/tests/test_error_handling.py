"""
Tests for error handling and response codes in the HeySol API client.
"""

import pytest
import requests_mock
from unittest.mock import Mock, patch
import json

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import (
    AuthenticationError,
    RateLimitError,
    APIError,
    ConnectionError,
    ServerError,
    NotFoundError,
    HeySolError
)


class TestErrorHandling:
    """Test error handling and response codes."""

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

    def test_401_unauthorized_sync(self, mock_client):
        """Test handling of 401 Unauthorized responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Unauthorized"},
                status_code=401
            )

            with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                mock_client.get_user_profile()

    def test_401_unauthorized_async(self, mock_async_client):
        """Test handling of 401 Unauthorized responses for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Unauthorized"},
                status_code=401
            )

            import asyncio
            async def test():
                with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_404_not_found_sync(self, mock_client):
        """Test handling of 404 Not Found responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Not found"},
                status_code=404
            )

            with pytest.raises(NotFoundError, match="Requested resource not found"):
                mock_client.get_user_profile()

    def test_404_not_found_async(self, mock_async_client):
        """Test handling of 404 Not Found responses for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Not found"},
                status_code=404
            )

            import asyncio
            async def test():
                with pytest.raises(NotFoundError, match="Requested resource not found"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_429_rate_limit_sync(self, mock_client):
        """Test handling of 429 Rate Limit responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"}
            )

            with pytest.raises(RateLimitError) as exc_info:
                mock_client.get_user_profile()

            assert exc_info.value.retry_after == 60
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_429_rate_limit_async(self, mock_async_client):
        """Test handling of 429 Rate Limit responses for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "30"}
            )

            import asyncio
            async def test():
                with pytest.raises(RateLimitError) as exc_info:
                    await mock_async_client.get_user_profile()

                assert exc_info.value.retry_after == 30

            asyncio.run(test())

    def test_429_rate_limit_no_retry_after_sync(self, mock_client):
        """Test handling of 429 Rate Limit responses without Retry-After header."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Rate limit exceeded"},
                status_code=429
            )

            with pytest.raises(RateLimitError) as exc_info:
                mock_client.get_user_profile()

            assert exc_info.value.retry_after is None

    def test_500_server_error_sync(self, mock_client):
        """Test handling of 500 Server Error responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Internal server error"},
                status_code=500
            )

            with pytest.raises(ServerError, match="Server error: 500"):
                mock_client.get_user_profile()

    def test_500_server_error_async(self, mock_async_client):
        """Test handling of 500 Server Error responses for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Internal server error"},
                status_code=500
            )

            import asyncio
            async def test():
                with pytest.raises(ServerError, match="Server error: 500"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_502_bad_gateway_sync(self, mock_client):
        """Test handling of 502 Bad Gateway responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Bad gateway"},
                status_code=502
            )

            with pytest.raises(ServerError, match="Server error: 502"):
                mock_client.get_user_profile()

    def test_503_service_unavailable_sync(self, mock_client):
        """Test handling of 503 Service Unavailable responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                json={"error": "Service unavailable"},
                status_code=503
            )

            with pytest.raises(ServerError, match="Server error: 503"):
                mock_client.get_user_profile()

    def test_400_bad_request_sync(self, mock_client):
        """Test handling of 400 Bad Request responses."""
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces",
                json={"error": "Invalid request data"},
                status_code=400
            )

            with pytest.raises(APIError, match="Client error: 400"):
                mock_client.create_space("Test Space")

    def test_422_unprocessable_entity_sync(self, mock_client):
        """Test handling of 422 Unprocessable Entity responses."""
        with requests_mock.Mocker() as m:
            m.post(
                "https://core.heysol.ai/api/v1/spaces",
                json={"error": "Validation failed"},
                status_code=422
            )

            with pytest.raises(APIError, match="Client error: 422"):
                mock_client.create_space("Test Space")

    def test_timeout_error_sync(self, mock_client):
        """Test handling of timeout errors."""
        with patch.object(mock_client.session, 'request') as mock_request:
            mock_request.side_effect = requests_mock.exceptions.Timeout("Connection timed out")

            with pytest.raises(ConnectionError, match="Request timeout"):
                mock_client.get_user_profile()

    def test_timeout_error_async(self, mock_async_client):
        """Test handling of timeout errors for async client."""
        with patch.object(mock_async_client._session, 'request') as mock_request:
            mock_request.side_effect = asyncio.TimeoutError("Connection timed out")

            import asyncio
            async def test():
                with pytest.raises(ConnectionError, match="Request timeout"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_connection_error_sync(self, mock_client):
        """Test handling of connection errors."""
        with patch.object(mock_client.session, 'request') as mock_request:
            mock_request.side_effect = requests_mock.exceptions.ConnectionError("Connection failed")

            with pytest.raises(ConnectionError, match="Connection error"):
                mock_client.get_user_profile()

    def test_connection_error_async(self, mock_async_client):
        """Test handling of connection errors for async client."""
        with patch.object(mock_async_client._session, 'request') as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Connection failed")

            import asyncio
            async def test():
                with pytest.raises(ConnectionError, match="Client error"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_invalid_json_response_sync(self, mock_client):
        """Test handling of invalid JSON responses."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                text="invalid json response",
                status_code=200
            )

            with pytest.raises(APIError, match="Failed to parse JSON response"):
                mock_client.get_user_profile()

    def test_invalid_json_response_async(self, mock_async_client):
        """Test handling of invalid JSON responses for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                text="invalid json response",
                status_code=200
            )

            import asyncio
            async def test():
                with pytest.raises(APIError, match="Failed to parse JSON response"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())

    def test_mcp_error_response_sync(self, mock_client):
        """Test handling of MCP protocol errors."""
        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = {"error": "MCP protocol error"}

            with pytest.raises(APIError, match="MCP error: MCP protocol error"):
                mock_client.get_spaces()

    def test_mcp_error_response_async(self, mock_async_client):
        """Test handling of MCP protocol errors for async client."""
        with patch.object(mock_async_client, '_call_tool') as mock_call:
            mock_call.return_value = {"error": "MCP protocol error"}

            import asyncio
            async def test():
                with pytest.raises(APIError, match="MCP error: MCP protocol error"):
                    await mock_async_client.get_spaces()

            asyncio.run(test())

    def test_tool_not_available_sync(self, mock_client):
        """Test handling when MCP tool is not available."""
        with pytest.raises(APIError, match="Tool 'nonexistent_tool' is not available"):
            mock_client._call_tool("nonexistent_tool")

    def test_tool_not_available_async(self, mock_async_client):
        """Test handling when MCP tool is not available for async client."""
        import asyncio
        async def test():
            with pytest.raises(APIError, match="Tool 'nonexistent_tool' is not available"):
                await mock_async_client._call_tool("nonexistent_tool")

        asyncio.run(test())

    def test_rate_limit_enforcement_sync(self, mock_client):
        """Test that rate limiting is properly enforced."""
        mock_client._rate_limit_remaining = 0

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            mock_client.get_user_profile()

    def test_rate_limit_enforcement_async(self, mock_async_client):
        """Test that rate limiting is properly enforced for async client."""
        mock_async_client._rate_limit_remaining = 0

        import asyncio
        async def test():
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                await mock_async_client.get_user_profile()

        asyncio.run(test())

    def test_rate_limit_reset_sync(self, mock_client):
        """Test that rate limit resets after the time window."""
        import time

        mock_client._rate_limit_remaining = 0
        mock_client._rate_limit_reset_time = time.time() - 1  # Reset time in the past

        # Should allow request after reset
        with patch.object(mock_client.session, 'request') as mock_request:
            mock_request.return_value = Mock(status_code=200, json=lambda: {"name": "Test"})

            try:
                mock_client.get_user_profile()
                # Should not raise RateLimitError
            except RateLimitError:
                pytest.fail("Rate limit should have been reset")

    def test_http_error_handling_sync(self, mock_client):
        """Test handling of general HTTP errors."""
        with patch.object(mock_client.session, 'request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 418  # I'm a teapot
            mock_response.raise_for_status.side_effect = requests_mock.exceptions.HTTPError("418 I'm a teapot")
            mock_request.return_value = mock_response

            with pytest.raises(APIError, match="HTTP error"):
                mock_client.get_user_profile()

    def test_request_exception_handling_sync(self, mock_client):
        """Test handling of general request exceptions."""
        with patch.object(mock_client.session, 'request') as mock_request:
            mock_request.side_effect = requests_mock.exceptions.RequestException("Network error")

            with pytest.raises(ConnectionError, match="Request failed"):
                mock_client.get_user_profile()

    def test_mcp_initialization_failure_sync(self):
        """Test handling of MCP session initialization failure."""
        config = HeySolConfig(api_key="test-key")

        with patch('heysol.client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session.post.side_effect = requests_mock.exceptions.ConnectionError("Connection failed")
            mock_session_class.return_value = mock_session

            with pytest.raises(APIError, match="Failed to initialize MCP session"):
                HeySolClient(config=config)

    def test_mcp_initialization_failure_async(self):
        """Test handling of MCP session initialization failure for async client."""
        config = HeySolConfig(api_key="test-key")

        with patch('heysol.async_client.requests.Session') as mock_session_class:
            mock_session = Mock()
            mock_session.post.side_effect = requests_mock.exceptions.ConnectionError("Connection failed")
            mock_session_class.return_value = mock_session

            with pytest.raises(APIError, match="Failed to initialize MCP session"):
                AsyncHeySolClient(config=config)

    def test_sse_parsing_error_sync(self, mock_client):
        """Test handling of SSE parsing errors."""
        with patch.object(mock_client, '_call_tool') as mock_call:
            mock_call.return_value = None  # Simulate failed SSE parsing

            # This should not crash, should handle gracefully
            result = mock_client.get_spaces()
            assert result == []  # Should return empty list on parsing failure

    def test_sse_parsing_error_async(self, mock_async_client):
        """Test handling of SSE parsing errors for async client."""
        with patch.object(mock_async_client, '_call_tool') as mock_call:
            mock_call.return_value = None  # Simulate failed SSE parsing

            import asyncio
            async def test():
                result = await mock_async_client.get_spaces()
                assert result == []  # Should return empty list on parsing failure

            asyncio.run(test())

    def test_unexpected_content_type_sync(self, mock_client):
        """Test handling of unexpected content types."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                text="plain text response",
                status_code=200,
                headers={"Content-Type": "text/plain"}
            )

            with pytest.raises(APIError, match="Unexpected Content-Type"):
                mock_client.get_user_profile()

    def test_unexpected_content_type_async(self, mock_async_client):
        """Test handling of unexpected content types for async client."""
        with requests_mock.Mocker() as m:
            m.get(
                "https://core.heysol.ai/api/profile",
                text="plain text response",
                status_code=200,
                headers={"Content-Type": "text/plain"}
            )

            import asyncio
            async def test():
                with pytest.raises(APIError, match="Unexpected Content-Type"):
                    await mock_async_client.get_user_profile()

            asyncio.run(test())