"""
Minimal OAuth2 implementation for HeySol API client.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import requests

from .exceptions import HeySolError, ValidationError


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


class InteractiveOAuth2Authenticator:
    """
    Interactive OAuth2 authentication handler for HeySol API.
    """

    def __init__(self, client_id: str, client_secret: Optional[str] = None, redirect_uri: Optional[str] = None):
        """
        Initialize interactive OAuth2 authenticator.

        Args:
            client_id: OAuth2 client ID
            client_secret: OAuth2 client secret
            redirect_uri: OAuth2 redirect URI
        """
        if not client_id:
            raise ValidationError("OAuth2 client ID is required")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri or "http://localhost:8080/callback"
        self.tokens: Optional[OAuth2Tokens] = None

        # OAuth2 endpoints (Google OAuth2 for HeySol)
        self.authorization_url = "https://accounts.google.com/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"

    def build_authorization_url(self, scope: str = "openid profile email") -> str:
        """
        Build OAuth2 authorization URL.

        Args:
            scope: OAuth2 scope

        Returns:
            Authorization URL
        """
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": scope,
            "access_type": "offline",
            "prompt": "consent",
        }

        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.authorization_url}?{query_string}"

    def exchange_code_for_tokens(self, code: str) -> OAuth2Tokens:
        """
        Exchange authorization code for access tokens.

        Args:
            code: Authorization code

        Returns:
            OAuth2Tokens instance

        Raises:
            HeySolError: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
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
                timeout=60
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
            raise HeySolError(f"Token exchange failed: {e}")
        except (KeyError, ValueError) as e:
            raise HeySolError(f"Invalid token response: {e}")

    def get_authorization_header(self) -> str:
        """
        Get authorization header for API requests.

        Returns:
            Authorization header value

        Raises:
            HeySolError: If no valid token available
        """
        if not self.tokens:
            raise HeySolError("No tokens available")

        return f"Bearer {self.tokens.access_token}"