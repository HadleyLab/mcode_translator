#!/usr/bin/env python3
"""
Lean OAuth2 Test Suite for HeySol API Client

Tests OAuth2 functionality using live APIs including:
- Token validation and expiration checking
- Authorization URL building
- Client initialization and validation
- Security considerations
- Performance testing
"""

import json
import time
import os
import pytest
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from heysol import HeySolClient
from heysol.oauth2 import InteractiveOAuth2Authenticator, OAuth2Tokens
from heysol.exceptions import HeySolError, ValidationError, AuthenticationError


class TestOAuth2Comprehensive:
    """Lean OAuth2 test suite using live APIs."""



    @pytest.fixture
    def expired_token(self):
        """Create an expired token for testing."""
        return OAuth2Tokens(
            access_token="expired-token",
            refresh_token="refresh-token",
            expires_at=time.time() - 3600  # Expired 1 hour ago
        )

    @pytest.fixture
    def valid_token(self):
        """Create a valid token for testing."""
        return OAuth2Tokens(
            access_token="valid-token",
            refresh_token="refresh-token",
            expires_in=3600,
            expires_at=time.time() + 3600  # Expires in 1 hour
        )

    # OAuth2Tokens Tests
    def test_oauth2_tokens_creation(self):
        """Test OAuth2Tokens creation and basic properties."""
        tokens = OAuth2Tokens(
            access_token="test-access-token",
            refresh_token="test-refresh-token",
            token_type="Bearer",
            expires_in=3600,
            scope="openid profile email"
        )

        assert tokens.access_token == "test-access-token"
        assert tokens.refresh_token == "test-refresh-token"
        assert tokens.token_type == "Bearer"
        assert tokens.expires_in == 3600
        assert tokens.scope == "openid profile email"
        # expires_at should be None when not explicitly set
        assert tokens.expires_at is None

    def test_oauth2_tokens_expiration_check(self):
        """Test token expiration checking logic."""
        # Test non-expiring token
        tokens = OAuth2Tokens(access_token="test-token")
        assert not tokens.is_expired()

        # Test expired token
        tokens = OAuth2Tokens(access_token="test-token", expires_at=time.time() - 3600)
        assert tokens.is_expired()

        # Test valid token
        tokens = OAuth2Tokens(access_token="test-token", expires_at=time.time() + 3600)
        assert not tokens.is_expired()

    # InteractiveOAuth2Authenticator Tests
    def test_oauth2_authenticator_initialization(self):
        """Test OAuth2 authenticator initialization."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        client_secret = os.getenv("HEYSOL_OAUTH2_CLIENT_SECRET")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri
        )

        assert auth.client_id == client_id
        assert auth.client_secret == client_secret
        assert auth.redirect_uri == redirect_uri
        assert auth.tokens is None

    def test_oauth2_authenticator_initialization_validation(self):
        """Test OAuth2 authenticator initialization validation."""
        with pytest.raises(ValidationError, match="OAuth2 client ID is required"):
            InteractiveOAuth2Authenticator(client_id="")

    def test_build_authorization_url(self):
        """Test authorization URL building."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )
        url = auth.build_authorization_url()

        assert client_id in url
        assert redirect_uri in url
        assert "openid profile email" in url
        assert "response_type=code" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url

    def test_build_authorization_url_custom_scope(self):
        """Test authorization URL building with custom scope."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )
        url = auth.build_authorization_url(scope="openid profile")

        assert "openid profile" in url
        assert "email+api" not in url




    def test_get_authorization_header_no_tokens(self):
        """Test authorization header with no tokens."""
        auth = InteractiveOAuth2Authenticator("test-client-id")

        with pytest.raises(HeySolError, match="No tokens available"):
            auth.get_authorization_header()

    def test_get_authorization_header_expired_token(self, expired_token):
        """Test authorization header with expired token."""
        auth = InteractiveOAuth2Authenticator("test-client-id")
        auth.tokens = expired_token

        with pytest.raises(HeySolError, match="Access token has expired"):
            auth.get_authorization_header()

    def test_get_authorization_header_valid_token(self, valid_token):
        """Test authorization header with valid token."""
        auth = InteractiveOAuth2Authenticator("test-client-id")
        auth.tokens = valid_token

        header = auth.get_authorization_header()
        assert header == "Bearer valid-token"

    def test_set_tokens_directly(self):
        """Test setting tokens directly."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )

        auth.set_tokens(
            access_token="direct-access-token",
            refresh_token="direct-refresh-token",
            token_type="Bearer",
            expires_in=7200
        )

        assert auth.tokens is not None
        assert auth.tokens.access_token == "direct-access-token"
        assert auth.tokens.refresh_token == "direct-refresh-token"
        assert auth.tokens.expires_in == 7200
        # Check that expires_at is approximately correct (within 1 second)
        expected_expires_at = time.time() + 7200
        assert abs(auth.tokens.expires_at - expected_expires_at) < 1.0

    # HeySolClient OAuth2 Integration Tests
    def test_client_oauth2_initialization(self):
        """Test client initialization with OAuth2 authenticator."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )
        client = HeySolClient(oauth2_auth=auth, skip_mcp_init=True)

        assert client.oauth2_auth == auth
        assert client.api_key is None  # Should not use API key when using OAuth2

    def test_client_oauth2_vs_api_key_validation(self):
        """Test client validation for OAuth2 vs API key."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )

        # Should raise error when both are provided
        with pytest.raises(ValidationError, match="Cannot use both API key and OAuth2 authentication"):
            HeySolClient(api_key="test-key", oauth2_auth=auth)

        # Should raise error when neither is provided
        with pytest.raises(ValidationError, match="Either API key or OAuth2 authenticator is required"):
            HeySolClient(api_key="", skip_mcp_init=True)

    def test_client_oauth2_get_authorization_header(self):
        """Test client OAuth2 authorization header generation."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )
        client = HeySolClient(oauth2_auth=auth, skip_mcp_init=True)

        # Set valid tokens
        auth.set_tokens("test-access-token", expires_in=3600)

        header = client._get_authorization_header()
        assert header == "Bearer test-access-token"


    def test_client_oauth2_user_info_validation(self):
        """Test OAuth2 user info validation."""
        api_key = os.getenv("HEYSOL_API_KEY")
        client = HeySolClient(api_key=api_key, skip_mcp_init=True)  # Use API key for this test

        with pytest.raises(ValidationError, match="Access token is required"):
            client.get_oauth2_user_info("")

    def test_oauth2_token_security_validation(self):
        """Test OAuth2 token security validation."""
        # Test with potentially dangerous tokens
        dangerous_tokens = [
            "token-with-script<script>alert('xss')</script>",
            "token-with-newlines\nmalicious",
            "token-with-nullbytes\x00malicious",
            "token-with-unicode\u0000malicious"
        ]

        for token in dangerous_tokens:
            tokens = OAuth2Tokens(access_token=token)
            # Should not raise exceptions for token content
            assert tokens.access_token == token

    def test_oauth2_refresh_token_security(self):
        """Test OAuth2 refresh token security."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        redirect_uri = os.getenv("HEYSOL_OAUTH2_REDIRECT_URI", "http://localhost:8080/callback")

        auth = InteractiveOAuth2Authenticator(
            client_id=client_id,
            client_secret="test-client-secret",
            redirect_uri=redirect_uri
        )

        # Test refresh token storage and retrieval
        refresh_token = "sensitive-refresh-token-12345"
        auth.set_tokens("access-token", refresh_token=refresh_token)

        assert auth.tokens is not None
        assert auth.tokens.refresh_token == refresh_token

        # Test that refresh token is not exposed in authorization header
        header = auth.get_authorization_header()
        assert refresh_token not in header
        assert header == "Bearer access-token"


    def test_oauth2_authorization_header_performance(self, valid_token):
        """Test authorization header generation performance."""
        client_id = os.getenv("HEYSOL_OAUTH2_CLIENT_ID")
        auth = InteractiveOAuth2Authenticator(client_id)
        auth.tokens = valid_token

        # Measure header generation time
        start_time = time.time()
        for _ in range(1000):
            header = auth.get_authorization_header()
        end_time = time.time()

        # Should be very fast (< 1 second for 1000 operations)
        assert end_time - start_time < 1.0
        assert header == "Bearer valid-token"


