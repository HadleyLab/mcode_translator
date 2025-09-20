"""
Tests for authentication mechanisms in the HeySol API client.
"""

import pytest
import requests_mock
from unittest.mock import Mock, patch

from heysol.client import HeySolClient
from heysol.async_client import AsyncHeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import AuthenticationError


class TestAuthentication:
    """Test authentication mechanisms."""

    def test_valid_api_key_sync(self):
        """Test successful authentication with valid API key."""
        config = HeySolConfig(api_key="valid-api-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None
            client = HeySolClient(config=config)

            assert client.config.api_key == "valid-api-key"
            assert "Bearer valid-api-key" in client._headers()["Authorization"]

    def test_valid_api_key_async(self):
        """Test successful authentication with valid API key for async client."""
        config = HeySolConfig(api_key="valid-api-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None
            client = AsyncHeySolClient(config=config)

            assert client.config.api_key == "valid-api-key"
            assert "Bearer valid-api-key" in client._headers()["Authorization"]

    def test_missing_api_key_sync(self):
        """Test that missing API key raises AuthenticationError."""
        config = HeySolConfig(api_key=None)

        with pytest.raises(AuthenticationError, match="API key is required"):
            HeySolClient(config=config)

    def test_missing_api_key_async(self):
        """Test that missing API key raises AuthenticationError for async client."""
        config = HeySolConfig(api_key=None)

        with pytest.raises(AuthenticationError, match="API key is required"):
            AsyncHeySolClient(config=config)

    def test_empty_api_key_sync(self):
        """Test that empty API key raises AuthenticationError."""
        config = HeySolConfig(api_key="")

        with pytest.raises(AuthenticationError, match="API key is required"):
            HeySolClient(config=config)

    def test_empty_api_key_async(self):
        """Test that empty API key raises AuthenticationError for async client."""
        config = HeySolConfig(api_key="")

        with pytest.raises(AuthenticationError, match="API key is required"):
            AsyncHeySolClient(config=config)

    def test_whitespace_api_key_sync(self):
        """Test that whitespace-only API key raises AuthenticationError."""
        config = HeySolConfig(api_key="   ")

        with pytest.raises(AuthenticationError, match="API key is required"):
            HeySolClient(config=config)

    def test_whitespace_api_key_async(self):
        """Test that whitespace-only API key raises AuthenticationError for async client."""
        config = HeySolConfig(api_key="   ")

        with pytest.raises(AuthenticationError, match="API key is required"):
            AsyncHeySolClient(config=config)

    def test_invalid_api_key_response_sync(self):
        """Test handling of invalid API key in API response."""
        config = HeySolConfig(api_key="invalid-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)

            with requests_mock.Mocker() as m:
                m.get(requests_mock.ANY, status_code=401, json={"error": "Invalid API key"})

                with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                    client.get_user_profile()

    def test_invalid_api_key_response_async(self):
        """Test handling of invalid API key in API response for async client."""
        config = HeySolConfig(api_key="invalid-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)

            with requests_mock.Mocker() as m:
                m.get(requests_mock.ANY, status_code=401, json={"error": "Invalid API key"})

                import asyncio
                async def test():
                    with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                        await client.get_user_profile()

                asyncio.run(test())

    def test_expired_token_handling_sync(self):
        """Test handling of expired tokens."""
        config = HeySolConfig(api_key="expired-token")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)

            with requests_mock.Mocker() as m:
                m.get(requests_mock.ANY, status_code=401, json={"error": "Token expired"})

                with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                    client.get_user_profile()

    def test_expired_token_handling_async(self):
        """Test handling of expired tokens for async client."""
        config = HeySolConfig(api_key="expired-token")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)

            with requests_mock.Mocker() as m:
                m.get(requests_mock.ANY, status_code=401, json={"error": "Token expired"})

                import asyncio
                async def test():
                    with pytest.raises(AuthenticationError, match="Invalid API key or authentication failed"):
                        await client.get_user_profile()

                asyncio.run(test())

    def test_malformed_authorization_header_sync(self):
        """Test handling of malformed authorization headers."""
        config = HeySolConfig(api_key="test-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)

            headers = client._headers()
            assert headers["Authorization"] == "Bearer test-key"

            # Test with special characters in API key
            config_special = HeySolConfig(api_key="test@key#123")
            with patch.object(HeySolClient, '_initialize_session') as mock_init:
                mock_init.return_value = None
                client_special = HeySolClient(config=config_special)

                headers_special = client_special._headers()
                assert headers_special["Authorization"] == "Bearer test@key#123"

    def test_malformed_authorization_header_async(self):
        """Test handling of malformed authorization headers for async client."""
        config = HeySolConfig(api_key="test-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)

            headers = client._headers()
            assert headers["Authorization"] == "Bearer test-key"

    def test_session_id_in_headers_sync(self):
        """Test that session ID is included in headers when available."""
        config = HeySolConfig(api_key="test-key")

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)
            client.session_id = "test-session-123"

            headers = client._headers()
            assert headers["Mcp-Session-Id"] == "test-session-123"

    def test_session_id_in_headers_async(self):
        """Test that session ID is included in headers when available for async client."""
        config = HeySolConfig(api_key="test-key")

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)
            client.session_id = "test-session-123"

            headers = client._headers()
            assert headers["Mcp-Session-Id"] == "test-session-123"

    def test_authentication_with_custom_base_url_sync(self):
        """Test authentication with custom base URL."""
        config = HeySolConfig(
            api_key="test-key",
            base_url="https://custom-api.example.com/api/v1/mcp"
        )

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)

            assert client.config.base_url == "https://custom-api.example.com/api/v1/mcp"
            assert "Bearer test-key" in client._headers()["Authorization"]

    def test_authentication_with_custom_base_url_async(self):
        """Test authentication with custom base URL for async client."""
        config = HeySolConfig(
            api_key="test-key",
            base_url="https://custom-api.example.com/api/v1/mcp"
        )

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)

            assert client.config.base_url == "https://custom-api.example.com/api/v1/mcp"
            assert "Bearer test-key" in client._headers()["Authorization"]

    def test_authentication_retry_on_failure_sync(self):
        """Test that authentication failures are retried appropriately."""
        config = HeySolConfig(api_key="test-key", max_retries=2)

        with patch.object(HeySolClient, '_initialize_session') as mock_init:
            mock_init.return_value = None

            client = HeySolClient(config=config)

            with requests_mock.Mocker() as m:
                # First call fails with 401, second succeeds
                m.get(
                    "https://core.heysol.ai/api/profile",
                    [
                        {"status_code": 401, "json": {"error": "Invalid API key"}},
                        {"status_code": 200, "json": {"name": "Test User"}}
                    ]
                )

                # Should succeed after retry
                profile = client.get_user_profile()
                assert profile["name"] == "Test User"

    def test_authentication_retry_on_failure_async(self):
        """Test that authentication failures are retried appropriately for async client."""
        config = HeySolConfig(api_key="test-key", max_retries=2)

        with patch.object(AsyncHeySolClient, '_initialize_session_sync') as mock_init:
            mock_init.return_value = None

            client = AsyncHeySolClient(config=config)

            with requests_mock.Mocker() as m:
                # First call fails with 401, second succeeds
                m.get(
                    "https://core.heysol.ai/api/profile",
                    [
                        {"status_code": 401, "json": {"error": "Invalid API key"}},
                        {"status_code": 200, "json": {"name": "Test User"}}
                    ]
                )

                import asyncio
                async def test():
                    # Should succeed after retry
                    profile = await client.get_user_profile()
                    assert profile["name"] == "Test User"

                asyncio.run(test())