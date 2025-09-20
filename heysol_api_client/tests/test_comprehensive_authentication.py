#!/usr/bin/env python3
"""
Comprehensive Authentication Testing Suite for HeySol API Client

Tests both API key and OAuth2 authentication mechanisms with various scenarios.
"""

import os
import time
import pytest
import requests
from unittest.mock import Mock, patch, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import HeySol components
from heysol.client import HeySolClient
from heysol.config import HeySolConfig
from heysol.exceptions import AuthenticationError, HeySolError
from heysol.oauth2 import OAuth2Authenticator, OAuth2Error


class TestAuthenticationMechanisms:
    """Test suite for authentication mechanisms."""

    def setup_method(self):
        """Setup test environment."""
        self.valid_api_key = os.getenv("COREAI_API_KEY", "test-api-key")
        self.valid_client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID", "test-client-id")
        self.valid_client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET", "test-client-secret")

        # Create test config
        self.config = HeySolConfig(
            api_key=self.valid_api_key,
            oauth2_client_id=self.valid_client_id,
            oauth2_client_secret=self.valid_client_secret,
            log_level="WARNING"  # Reduce noise during testing
        )

    def test_api_key_authentication_success(self):
        """Test successful API key authentication."""
        client = HeySolClient(config=self.config, use_oauth2=False)

        # Verify client is initialized with API key
        assert client.use_oauth2 is False
        assert client.config.api_key == self.valid_api_key
        assert "Authorization" in client._headers()
        assert client._headers()["Authorization"] == f"Bearer {self.valid_api_key}"

    def test_oauth2_authentication_initialization(self):
        """Test OAuth2 authentication initialization."""
        client = HeySolClient(config=self.config, use_oauth2=True)

        # Verify client is initialized with OAuth2
        assert client.use_oauth2 is True
        assert client.oauth2_auth is not None
        assert isinstance(client.oauth2_auth, OAuth2Authenticator)

    def test_oauth2_client_credentials_flow(self):
        """Test OAuth2 client credentials flow."""
        # Mock OAuth2 client credentials authenticator
        with patch('heysol.oauth2.OAuth2ClientCredentialsAuthenticator') as mock_auth:
            mock_instance = Mock()
            mock_instance.get_authorization_header.return_value = "Bearer mock-token"
            mock_auth.return_value = mock_instance

            client = HeySolClient(config=self.config, use_oauth2=True)

            # Verify OAuth2 client credentials authenticator is used
            assert client.oauth2_auth is not None
            assert client._headers()["Authorization"] == "Bearer mock-token"

    def test_missing_api_key_error(self):
        """Test error when API key is missing."""
        config_no_key = HeySolConfig(api_key=None)

        with pytest.raises(AuthenticationError, match="API key is required"):
            HeySolClient(config=config_no_key, use_oauth2=False)

    def test_missing_oauth2_client_id_error(self):
        """Test error when OAuth2 client ID is missing."""
        config_no_oauth2 = HeySolConfig(
            api_key=None,
            oauth2_client_id=None,
            oauth2_client_secret=self.valid_client_secret
        )

        with pytest.raises(AuthenticationError, match="OAuth2 client ID is required"):
            HeySolClient(config=config_no_oauth2, use_oauth2=True)

    def test_invalid_api_key_format(self):
        """Test handling of invalid API key formats."""
        invalid_keys = ["", " ", "invalid-key", None]

        for invalid_key in invalid_keys:
            config = HeySolConfig(api_key=invalid_key)
            client = HeySolClient(config=config, use_oauth2=False)

            # Should still initialize but may fail on requests
            assert client.config.api_key == invalid_key

    def test_oauth2_token_refresh_mechanism(self):
        """Test OAuth2 token refresh functionality."""
        client = HeySolClient(config=self.config, use_oauth2=True)

        # Mock token refresh
        if client.oauth2_auth:
            original_refresh = client.oauth2_auth.refresh_access_token
            client.oauth2_auth.refresh_access_token = Mock(return_value=True)

            # Test refresh call
            result = client.refresh_oauth2_token()
            assert result is True

            # Restore original method
            client.oauth2_auth.refresh_access_token = original_refresh

    def test_authentication_headers_format(self):
        """Test authentication header formats."""
        # Test API key header
        client = HeySolClient(config=self.config, use_oauth2=False)
        headers = client._headers()
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert headers["Authorization"] == f"Bearer {self.valid_api_key}"

        # Test OAuth2 header
        client_oauth2 = HeySolClient(config=self.config, use_oauth2=True)
        oauth2_headers = client_oauth2._headers()
        assert "Authorization" in oauth2_headers
        assert oauth2_headers["Authorization"].startswith("Bearer ")

    def test_session_persistence(self):
        """Test that authentication persists across requests."""
        client = HeySolClient(config=self.config, use_oauth2=False)

        # Make multiple header requests
        headers1 = client._headers()
        headers2 = client._headers()

        # Headers should be consistent
        assert headers1["Authorization"] == headers2["Authorization"]
        assert headers1["User-Agent"] == headers2["User-Agent"]

    def test_environment_variable_loading(self):
        """Test loading authentication from environment variables."""
        # Test API key from env
        config_from_env = HeySolConfig.from_env()
        assert config_from_env.api_key == self.valid_api_key

        # Test OAuth2 from env
        assert config_from_env.oauth2_client_id == self.valid_client_id
        assert config_from_env.oauth2_client_secret == self.valid_client_secret

    def test_authentication_switching(self):
        """Test switching between authentication methods."""
        # Start with API key
        client = HeySolClient(config=self.config, use_oauth2=False)
        assert client.use_oauth2 is False
        assert client._headers()["Authorization"] == f"Bearer {self.valid_api_key}"

        # Switch to OAuth2
        client.use_oauth2 = True
        client._initialize_authentication()
        assert client.use_oauth2 is True
        assert client.oauth2_auth is not None

    def test_concurrent_authentication(self):
        """Test authentication works with concurrent requests."""
        client = HeySolClient(config=self.config, use_oauth2=False)

        # Simulate concurrent header requests
        headers_list = [client._headers() for _ in range(10)]

        # All headers should be identical
        for headers in headers_list:
            assert headers["Authorization"] == f"Bearer {self.valid_api_key}"
            assert headers["User-Agent"] == headers_list[0]["User-Agent"]


class TestOAuth2AdvancedFeatures:
    """Test advanced OAuth2 features."""

    def setup_method(self):
        """Setup test environment."""
        self.config = HeySolConfig(
            oauth2_client_id=self.valid_client_id,
            oauth2_client_secret=self.valid_client_secret,
            log_level="WARNING"
        )

    def test_oauth2_user_info_retrieval(self):
        """Test OAuth2 user information retrieval."""
        client = HeySolClient(config=self.config, use_oauth2=True)

        # Mock user info response
        mock_user_info = {
            "id": "test-user-id",
            "email": "test@example.com",
            "name": "Test User",
            "picture": "https://example.com/avatar.jpg"
        }

        if client.oauth2_auth:
            client.oauth2_auth.get_user_info = Mock(return_value=mock_user_info)

            user_info = client.get_oauth2_user_info()
            assert user_info == mock_user_info
            assert user_info["email"] == "test@example.com"

    def test_oauth2_token_introspection(self):
        """Test OAuth2 token introspection."""
        client = HeySolClient(config=self.config, use_oauth2=True)

        # Mock introspection response
        mock_introspection = {
            "active": True,
            "client_id": self.valid_client_id,
            "scope": "openid profile email",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "token_type": "Bearer"
        }

        if client.oauth2_auth:
            client.oauth2_auth.introspect_token = Mock(return_value=mock_introspection)

            token_info = client.introspect_oauth2_token()
            assert token_info["active"] is True
            assert token_info["client_id"] == self.valid_client_id

    def test_oauth2_error_handling(self):
        """Test OAuth2 error handling."""
        client = HeySolClient(config=self.config, use_oauth2=True)

        # Mock OAuth2 error
        if client.oauth2_auth:
            client.oauth2_auth.get_user_info = Mock(side_effect=OAuth2Error("Invalid token"))

            with pytest.raises(HeySolError, match="Failed to get user info"):
                client.get_oauth2_user_info()

    def test_oauth2_scope_validation(self):
        """Test OAuth2 scope validation."""
        # Test with minimal scope
        config_minimal = HeySolConfig(
            oauth2_client_id=self.valid_client_id,
            oauth2_client_secret=self.valid_client_secret,
            oauth2_scope="openid"
        )

        client = HeySolClient(config=config_minimal, use_oauth2=True)
        assert client.config.oauth2_scope == "openid"

        # Test with extended scope
        config_extended = HeySolConfig(
            oauth2_client_id=self.valid_client_id,
            oauth2_client_secret=self.valid_client_secret,
            oauth2_scope="openid profile email api"
        )

        client_extended = HeySolClient(config=config_extended, use_oauth2=True)
        assert client_extended.config.oauth2_scope == "openid profile email api"


if __name__ == "__main__":
    # Run basic tests
    test_suite = TestAuthenticationMechanisms()
    test_suite.setup_method()

    print("🧪 Running Authentication Tests...")

    try:
        test_suite.test_api_key_authentication_success()
        print("✅ API key authentication test passed")

        test_suite.test_oauth2_authentication_initialization()
        print("✅ OAuth2 authentication initialization test passed")

        test_suite.test_missing_api_key_error()
        print("✅ Missing API key error test passed")

        test_suite.test_missing_oauth2_client_id_error()
        print("✅ Missing OAuth2 client ID error test passed")

        test_suite.test_authentication_headers_format()
        print("✅ Authentication headers format test passed")

        print("\n🎉 All authentication tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise