"""
OAuth2 authentication implementation for HeySol API client.

This module provides OAuth2 authentication support for the HeySol API client,
including authorization flows, token management, and automatic token refresh.
"""

import json
import time
import uuid
import hashlib
import secrets
import webbrowser
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlencode, parse_qs, urlparse
import requests
from dataclasses import dataclass

from .config import HeySolConfig
from .exceptions import AuthenticationError, APIError


@dataclass
class OAuth2Tokens:
    """OAuth2 token data structure."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    expires_at: Optional[float] = None
    scope: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the access token is expired."""
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at


class OAuth2Error(Exception):
    """OAuth2-specific error."""
    pass


class OAuth2Authenticator:
    """
    OAuth2 authentication handler for HeySol API.

    Supports authorization code flow with PKCE for enhanced security.
    """

    def __init__(self, config: HeySolConfig):
        """
        Initialize OAuth2 authenticator.

        Args:
            config: HeySol configuration instance
        """
        self.config = config
        self.tokens: Optional[OAuth2Tokens] = None
        self.client_id = config.oauth2_client_id or "heysol-python-client"
        self.client_secret = config.oauth2_client_secret
        self.redirect_uri = config.oauth2_redirect_uri or "https://core.heysol.ai/auth/google/callback"

        # OAuth2 endpoints (Google OAuth2 for HeySol)
        self.authorization_url = "https://accounts.google.com/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.introspection_url = "https://oauth2.googleapis.com/tokeninfo"

    def generate_pkce_challenge(self) -> Tuple[str, str]:
        """
        Generate PKCE code challenge and verifier.

        Returns:
            Tuple of (code_challenge, code_verifier)
        """
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = (
            secrets.token_urlsafe(32)
            .replace("-", "")
            .replace("_", "")
        )[:43]  # Ensure it's 43 characters

        return code_challenge, code_verifier

    def build_authorization_url(
        self,
        state: Optional[str] = None,
        code_challenge: Optional[str] = None,
        scope: str = "openid profile email"
    ) -> str:
        """
        Build OAuth2 authorization URL.

        Args:
            state: Optional state parameter for CSRF protection
            code_challenge: Optional PKCE code challenge
            scope: OAuth2 scope

        Returns:
            Authorization URL
        """
        if state is None:
            state = str(uuid.uuid4())

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "state": state,
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        return f"{self.authorization_url}?{urlencode(params)}"

    def authorize_interactive(self, scope: str = "openid profile email") -> OAuth2Tokens:
        """
        Perform interactive OAuth2 authorization.

        Args:
            scope: OAuth2 scope

        Returns:
            OAuth2Tokens instance

        Raises:
            OAuth2Error: If authorization fails
        """
        # Generate PKCE challenge
        code_challenge, code_verifier = self.generate_pkce_challenge()

        # Build authorization URL
        auth_url = self.build_authorization_url(
            code_challenge=code_challenge,
            scope=scope
        )

        print(f"Please visit this URL to authorize the application: {auth_url}")
        print("After authorization, you'll be redirected to a localhost URL.")
        print("Copy the authorization code from the URL and paste it here.")

        # Open browser for user
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        # Get authorization code from user
        auth_code = input("Enter the authorization code: ").strip()

        if not auth_code:
            raise OAuth2Error("Authorization code is required")

        # Exchange code for tokens
        return self.exchange_code_for_tokens(auth_code, code_verifier)

    def exchange_code_for_tokens(
        self,
        code: str,
        code_verifier: Optional[str] = None
    ) -> OAuth2Tokens:
        """
        Exchange authorization code for access tokens.

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier

        Returns:
            OAuth2Tokens instance

        Raises:
            OAuth2Error: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self.client_id,
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            response = requests.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            token_data = response.json()

            # Parse token response
            tokens = OAuth2Tokens(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                scope=token_data.get("scope"),
            )

            # Calculate expiration time
            if tokens.expires_in:
                tokens.expires_at = time.time() + tokens.expires_in

            self.tokens = tokens
            return tokens

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"Token exchange failed: {e}")
        except (KeyError, ValueError) as e:
            raise OAuth2Error(f"Invalid token response: {e}")

    def refresh_access_token(self) -> Optional[OAuth2Tokens]:
        """
        Refresh access token using refresh token.

        Returns:
            New OAuth2Tokens instance or None if refresh fails

        Raises:
            OAuth2Error: If refresh fails
        """
        if not self.tokens or not self.tokens.refresh_token:
            raise OAuth2Error("No refresh token available")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.tokens.refresh_token,
            "client_id": self.client_id,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            response = requests.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            token_data = response.json()

            # Update tokens
            self.tokens.access_token = token_data["access_token"]
            self.tokens.token_type = token_data.get("token_type", "Bearer")

            if "refresh_token" in token_data:
                self.tokens.refresh_token = token_data["refresh_token"]

            if "expires_in" in token_data:
                self.tokens.expires_in = token_data["expires_in"]
                self.tokens.expires_at = time.time() + token_data["expires_in"]

            return self.tokens

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"Token refresh failed: {e}")
        except (KeyError, ValueError) as e:
            raise OAuth2Error(f"Invalid refresh response: {e}")

    def get_user_info(self) -> Dict[str, Any]:
        """
        Get user information using access token.

        Returns:
            User information dictionary

        Raises:
            OAuth2Error: If user info retrieval fails
        """
        if not self.tokens:
            raise OAuth2Error("No access token available")

        headers = {
            "Authorization": f"Bearer {self.tokens.access_token}",
            "Accept": "application/json",
        }

        try:
            response = requests.get(
                self.user_info_url,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"User info retrieval failed: {e}")

    def introspect_token(self, token: Optional[str] = None) -> Dict[str, Any]:
        """
        Introspect access token.

        Args:
            token: Token to introspect (uses current access token if None)

        Returns:
            Token introspection data

        Raises:
            OAuth2Error: If introspection fails
        """
        if not self.tokens:
            raise OAuth2Error("No access token available")

        target_token = token or self.tokens.access_token

        data = {
            "token": target_token,
            "client_id": self.client_id,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            response = requests.post(
                self.introspection_url,
                data=data,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"Token introspection failed: {e}")

    def get_valid_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token

        Raises:
            OAuth2Error: If no valid token can be obtained
        """
        if not self.tokens:
            raise OAuth2Error("No tokens available")

        if self.tokens.is_expired():
            if self.tokens.refresh_token:
                try:
                    self.refresh_access_token()
                except OAuth2Error:
                    raise OAuth2Error("Token expired and refresh failed")
            else:
                raise OAuth2Error("Token expired and no refresh token available")

        return self.tokens.access_token

    def get_authorization_header(self) -> str:
        """
        Get authorization header for API requests.

        Returns:
            Authorization header value

        Raises:
            OAuth2Error: If no valid token available
        """
        token = self.get_valid_access_token()
        return f"Bearer {token}"


class OAuth2ClientCredentialsAuthenticator:
    """
    OAuth2 client credentials flow authenticator.

    For server-to-server authentication without user interaction.
    """

    def __init__(self, config: HeySolConfig):
        """
        Initialize client credentials authenticator.

        Args:
            config: HeySol configuration instance
        """
        self.config = config
        self.tokens: Optional[OAuth2Tokens] = None
        self.client_id = config.oauth2_client_id
        self.client_secret = config.oauth2_client_secret

        if not self.client_id or not self.client_secret:
            raise OAuth2Error("Client ID and secret required for client credentials flow")

        self.token_url = "https://oauth2.googleapis.com/token"

    def authenticate(self, scope: str = "api") -> OAuth2Tokens:
        """
        Authenticate using client credentials.

        Args:
            scope: OAuth2 scope

        Returns:
            OAuth2Tokens instance

        Raises:
            OAuth2Error: If authentication fails
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": scope,
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            response = requests.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            token_data = response.json()

            tokens = OAuth2Tokens(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                scope=token_data.get("scope"),
            )

            if tokens.expires_in:
                tokens.expires_at = time.time() + tokens.expires_in

            self.tokens = tokens
            return tokens

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"Client credentials authentication failed: {e}")
        except (KeyError, ValueError) as e:
            raise OAuth2Error(f"Invalid token response: {e}")

    def get_valid_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token

        Raises:
            OAuth2Error: If no valid token can be obtained
        """
        if not self.tokens:
            self.authenticate()

        if self.tokens.is_expired():
            self.authenticate()

        return self.tokens.access_token

    def get_authorization_header(self) -> str:
        """
        Get authorization header for API requests.

        Returns:
            Authorization header value

        Raises:
            OAuth2Error: If no valid token available
        """
        token = self.get_valid_access_token()
        return f"Bearer {token}"