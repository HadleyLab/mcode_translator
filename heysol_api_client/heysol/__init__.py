"""
HeySol API Client Library

A comprehensive Python client for the HeySol API with support for MCP protocol,
authentication, memory management, and robust error handling.
"""

from .client import HeySolClient
from .async_client import AsyncHeySolClient
from .oauth2 import OAuth2Authenticator, OAuth2ClientCredentialsAuthenticator, OAuth2Tokens, OAuth2Error
from .oauth2_interactive import InteractiveOAuth2Authenticator
from .exceptions import HeySolError, AuthenticationError, RateLimitError, APIError
from .config import HeySolConfig

__version__ = "1.0.0"
__all__ = [
    "HeySolClient",
    "AsyncHeySolClient",
    "OAuth2Authenticator",
    "OAuth2ClientCredentialsAuthenticator",
    "InteractiveOAuth2Authenticator",
    "OAuth2Tokens",
    "OAuth2Error",
    "HeySolError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "HeySolConfig",
]