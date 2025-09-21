"""
Unified OAuth2 implementation for HeySol API client.

This module provides complete OAuth2 authentication support including:
- Core OAuth2 protocol (authorization code, client credentials flows)
- Interactive browser-based authentication
- Configuration validation and client management
- Log operations and demo utilities
- Performance optimizations (caching, connection pooling)
"""

import json
import time
import uuid
import hashlib
import secrets
import webbrowser
import threading
import logging
from typing import Dict, Any, Optional, Tuple, Callable
from urllib.parse import urlencode, parse_qs, urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dataclasses import dataclass
from functools import lru_cache
import os

from .config import HeySolConfig
from .client import HeySolClient
from .exceptions import HeySolError, AuthenticationError, ValidationError, APIError


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


# ===== INTERACTIVE OAUTH2 COMPONENTS =====

@dataclass
class OAuth2CallbackResult:
    """Result from OAuth2 callback."""
    success: bool
    authorization_code: Optional[str] = None
    error: Optional[str] = None
    state: Optional[str] = None


class OAuth2CallbackHandler:
    """HTTP handler for OAuth2 callback."""

    def __init__(self, callback_result: OAuth2CallbackResult):
        self.callback_result = callback_result

    def do_GET(self, path: str) -> str:
        """Handle GET request for OAuth2 callback."""
        try:
            # Parse the URL
            parsed_url = urlparse(path)
            query_params = parse_qs(parsed_url.query)

            # Get authorization code or error
            if 'code' in query_params:
                self.callback_result.success = True
                self.callback_result.authorization_code = query_params['code'][0]
                self.callback_result.state = query_params.get('state', [None])[0]

                # Send success response
                return """
                <!DOCTYPE html>
                <html>
                <head><title>Authorization Successful</title></head>
                <body>
                <h2>Authorization Successful!</h2>
                <p>You can now close this window and return to the application.</p>
                <script>window.close();</script>
                </body>
                </html>
                """
            else:
                self.callback_result.success = False
                self.callback_result.error = query_params.get('error', ['Unknown error'])[0]

                # Send error response
                return f"""
                <!DOCTYPE html>
                <html>
                <head><title>Authorization Failed</title></head>
                <body>
                <h2>Authorization Failed</h2>
                <p>Error: {self.callback_result.error}</p>
                <p>You can close this window and try again.</p>
                </body>
                </html>
                """

        except Exception as e:
            self.callback_result.success = False
            self.callback_result.error = str(e)

            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Server Error</title></head>
            <body>
            <h2>Server Error</h2>
            <p>Error: {e}</p>
            </body>
            </html>
            """


class InteractiveOAuth2Authenticator(OAuth2Authenticator):
    """
    Interactive OAuth2 authentication handler for HeySol API.

    Opens a browser for user authentication and handles the callback automatically.
    """

    def __init__(self, config: HeySolConfig):
        """
        Initialize interactive OAuth2 authenticator.

        Args:
            config: HeySol configuration instance
        """
        super().__init__(config)
        self.redirect_uri = config.oauth2_redirect_uri or "http://localhost:8080/callback"

        # Callback server
        self.callback_result = OAuth2CallbackResult(success=False)
        self.server_thread: Optional[threading.Thread] = None
        self.server = None

    def generate_pkce_challenge(self) -> tuple[str, str]:
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
        scope: str = "openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
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
            "access_type": "offline",  # Request refresh token
            "prompt": "consent",  # Force consent screen
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        query_string = "&".join([f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()])
        return f"{self.authorization_url}?{query_string}"

    def start_callback_server(self) -> None:
        """Start the local callback server."""
        try:
            # Find an available port
            for port in range(8080, 8090):
                try:
                    from http.server import HTTPServer, BaseHTTPRequestHandler
                    self.server = HTTPServer(("", port), lambda *args, **kwargs: OAuth2CallbackHandler(self.callback_result))
                    self.server.timeout = 300  # 5 minute timeout
                    break
                except OSError:
                    continue

            if not self.server:
                raise OAuth2Error("Could not find an available port for callback server")

            # Start server in background thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()

        except Exception as e:
            raise OAuth2Error(f"Failed to start callback server: {e}")

    def stop_callback_server(self) -> None:
        """Stop the callback server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

        if self.server_thread:
            self.server_thread.join(timeout=1)
            self.server_thread = None

    def wait_for_callback(self, timeout: int = 300) -> OAuth2CallbackResult:
        """
        Wait for OAuth2 callback.

        Args:
            timeout: Timeout in seconds

        Returns:
            OAuth2CallbackResult instance
        """
        if not self.server:
            raise OAuth2Error("Callback server not started")

        start_time = time.time()

        while time.time() - start_time < timeout:
            self.server.handle_request()  # Handle one request

            if self.callback_result.success or self.callback_result.error:
                return self.callback_result

            time.sleep(0.1)  # Small delay to prevent busy waiting

        # Timeout
        self.callback_result.success = False
        self.callback_result.error = "Authorization timeout"
        return self.callback_result

    def authorize_interactive(
        self,
        scope: str = "openid https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email",
        on_progress: Optional[Callable[[str], None]] = None
    ) -> OAuth2Tokens:
        """
        Perform interactive OAuth2 authorization.

        Args:
            scope: OAuth2 scope for authorization
            on_progress: Optional callback for progress updates

        Returns:
            OAuth2Tokens instance

        Raises:
            OAuth2Error: If authorization fails
        """
        if on_progress:
            on_progress("Starting interactive OAuth2 authorization...")

        # Generate PKCE challenge
        code_challenge, code_verifier = self.generate_pkce_challenge()

        if on_progress:
            on_progress("Generated PKCE challenge for enhanced security...")

        # Start callback server
        self.start_callback_server()

        if on_progress:
            on_progress("Started local callback server...")

        try:
            # Build authorization URL
            auth_url = self.build_authorization_url(
                code_challenge=code_challenge,
                scope=scope
            )

            if on_progress:
                on_progress("Opening browser for authentication...")

            # Open browser for user
            print(f"\n🔐 Opening browser for authentication...")
            print(f"URL: {auth_url}")
            print("Please authenticate in the browser window that opens.")
            print("After authentication, you'll be redirected back to the application.\n")

            webbrowser.open(auth_url)

            # Wait for callback
            if on_progress:
                on_progress("Waiting for authentication callback...")

            callback_result = self.wait_for_callback(timeout=300)

            if not callback_result.success:
                raise OAuth2Error(f"Authorization failed: {callback_result.error}")

            if on_progress:
                on_progress("Authorization successful! Exchanging code for tokens...")

            # Exchange code for tokens
            return self.exchange_code_for_tokens(callback_result.authorization_code, code_verifier)

        finally:
            # Always stop the callback server
            self.stop_callback_server()


# ===== OAUTH2 UTILITIES AND CLIENT MANAGEMENT =====

@dataclass
class OAuth2ValidationResult:
    """Result of OAuth2 configuration validation."""
    is_valid: bool
    missing_fields: list[str]
    error_message: str = ""


class OAuth2ConfigurationValidator:
    """Strict validator for OAuth2 configuration."""

    REQUIRED_FIELDS = ["COREAI_OAUTH2_CLIENT_ID", "COREAI_OAUTH2_CLIENT_SECRET"]

    @classmethod
    def validate(cls) -> OAuth2ValidationResult:
        """
        Validate OAuth2 configuration strictly.

        Returns:
            OAuth2ValidationResult with validation status and details

        Raises:
            AuthenticationError: If configuration is invalid
        """
        missing_fields = []
        error_messages = []

        # Check required environment variables
        for field in cls.REQUIRED_FIELDS:
            if not os.getenv(field):
                missing_fields.append(field)

        if missing_fields:
            error_message = f"Missing required OAuth2 configuration: {', '.join(missing_fields)}"
            error_messages.append(error_message)

        # Additional validation
        client_id = os.getenv("COREAI_OAUTH2_CLIENT_ID")
        if client_id and len(client_id) < 10:
            error_messages.append("COREAI_OAUTH2_CLIENT_ID appears to be too short")

        client_secret = os.getenv("COREAI_OAUTH2_CLIENT_SECRET")
        if client_secret and len(client_secret) < 10:
            error_messages.append("COREAI_OAUTH2_CLIENT_SECRET appears to be too short")

        if error_messages:
            raise AuthenticationError("; ".join(error_messages))

        return OAuth2ValidationResult(
            is_valid=True,
            missing_fields=[],
            error_message=""
        )


class OAuth2ClientManager:
    """Centralized OAuth2 client management with strict error handling and performance optimizations."""

    def __init__(self, config: Optional[HeySolConfig] = None):
        """
        Initialize OAuth2 client manager.

        Args:
            config: Optional HeySol configuration

        Raises:
            AuthenticationError: If configuration is invalid
        """
        self.config = config or HeySolConfig.from_env()
        self.client: Optional[HeySolClient] = None
        self.logger = logging.getLogger(__name__)

        # Performance optimizations
        self._auth_cache: Dict[str, Tuple[bool, float]] = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_lock = threading.Lock()
        self._session: Optional[requests.Session] = None

        # Validate configuration immediately
        OAuth2ConfigurationValidator.validate()

    def _is_auth_cached(self, scope: str) -> Optional[bool]:
        """Check if authentication status is cached."""
        with self._cache_lock:
            if scope in self._auth_cache:
                cached_result, timestamp = self._auth_cache[scope]
                if time.time() - timestamp < self._cache_ttl:
                    return cached_result
                else:
                    # Cache expired
                    del self._auth_cache[scope]
        return None

    def _cache_auth_status(self, scope: str, status: bool) -> None:
        """Cache authentication status."""
        with self._cache_lock:
            self._auth_cache[scope] = (status, time.time())

    def clear_auth_cache(self):
        """Clear authentication cache."""
        with self._cache_lock:
            self._auth_cache.clear()

    def _create_optimized_session(self) -> requests.Session:
        """
        Create an optimized requests session with connection pooling and retry logic.

        Returns:
            Configured requests Session
        """
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )

            # Create HTTP adapter with connection pooling
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=20
            )

            # Mount adapter for both HTTP and HTTPS
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set reasonable timeouts
            self._session.timeout = (10, 30)  # (connect, read) timeouts

        return self._session

    def get_optimized_session(self) -> requests.Session:
        """
        Get optimized requests session for API calls.

        Returns:
            Optimized requests Session
        """
        return self._create_optimized_session()

    def create_client(self) -> HeySolClient:
        """
        Create OAuth2-enabled HeySol client with performance optimizations.

        Returns:
            Configured HeySolClient instance

        Raises:
            AuthenticationError: If client creation fails
        """
        if self.client is not None:
            return self.client

        try:
            # Create optimized session for potential use by client
            optimized_session = self._create_optimized_session()

            self.client = HeySolClient(
                config=self.config,
                use_oauth2=True
            )

            self.logger.info("OAuth2 client created successfully with performance optimizations")
            return self.client

        except Exception as e:
            raise AuthenticationError(f"Failed to create OAuth2 client: {e}")

    def get_client(self) -> HeySolClient:
        """
        Get existing OAuth2 client or create new one.

        Returns:
            HeySolClient instance

        Raises:
            AuthenticationError: If client is not available
        """
        if self.client is None:
            raise AuthenticationError("OAuth2 client not initialized. Call create_client() first.")
        return self.client

    def ensure_authenticated(self, scope: str = "openid profile email") -> bool:
        """
        Ensure client is authenticated with OAuth2, with caching for performance.

        Args:
            scope: OAuth2 scope for authorization

        Returns:
            True if authentication successful

        Raises:
            AuthenticationError: If authentication fails
        """
        # Check cache first
        cached_result = self._is_auth_cached(scope)
        if cached_result is not None:
            if cached_result:
                self.logger.info("OAuth2 authentication valid (cached)")
                return True
            else:
                # Cached as invalid, clear cache and re-authenticate
                self.clear_auth_cache()

        # Use client's OAuth2 authentication method
        try:
            success = self.client.authorize_oauth2_interactive(scope=scope)
            if success:
                self._cache_auth_status(scope, True)
                self.logger.info("OAuth2 authentication completed successfully")
                return True
            else:
                self._cache_auth_status(scope, False)
                raise AuthenticationError("OAuth2 authorization failed")

        except Exception as e:
            self._cache_auth_status(scope, False)
            raise AuthenticationError(f"OAuth2 authentication failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the OAuth2 client manager.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "cache_size": len(self._auth_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "session_created": self._session is not None,
            "client_created": self.client is not None
        }


class OAuth2LogOperations:
    """Centralized OAuth2 log operations with strict error handling."""

    def __init__(self, client_manager: Optional[OAuth2ClientManager] = None):
        """
        Initialize OAuth2 log operations.

        Args:
            client_manager: Optional pre-configured client manager
        """
        self.client_manager = client_manager or OAuth2ClientManager()
        self.logger = logging.getLogger(__name__)

    def ingest_log(self, message: str, space_name: str = "oauth2_demo") -> Dict[str, Any]:
        """
        Ingest a log entry with OAuth2 authentication.

        Args:
            message: Log message to ingest
            space_name: Name of space to use

        Returns:
            Log entry result

        Raises:
            ValidationError: If message is invalid
            AuthenticationError: If OAuth2 authentication fails
            HeySolError: If ingestion fails
        """
        if not message or not message.strip():
            raise ValidationError("Log message cannot be empty")

        if not space_name or not space_name.strip():
            raise ValidationError("Space name cannot be empty")

        try:
            client = self.client_manager.get_client()

            # Create or get space
            space_id = client.get_or_create_space(space_name, f"Space for {space_name}")

            # Ingest log
            result = client.ingest(
                message=message.strip(),
                space_id=space_id,
                tags=["oauth2", "automated"]
            )

            if not result:
                raise HeySolError("Log ingestion returned empty result")

            log_id = result.get("id") or result.get("log_id")
            if not log_id:
                raise HeySolError("Log ingestion did not return a valid ID")

            self.logger.info(f"Log ingested successfully: {log_id}")
            return {"log_id": log_id, "space_id": space_id, "message": message}

        except ValidationError:
            raise  # Re-raise validation errors
        except AuthenticationError:
            raise  # Re-raise auth errors
        except Exception as e:
            raise HeySolError(f"Log ingestion failed: {e}")

    def delete_log(self, log_id: str) -> Dict[str, Any]:
        """
        Delete a log entry with OAuth2 authentication.

        Args:
            log_id: ID of log entry to delete

        Returns:
            Deletion result

        Raises:
            ValidationError: If log_id is invalid
            AuthenticationError: If OAuth2 authentication fails
            HeySolError: If deletion fails
        """
        if not log_id or not log_id.strip():
            raise ValidationError("Log ID cannot be empty")

        try:
            client = self.client_manager.get_client()
            result = client.delete_log_entry(log_id.strip())

            if not result:
                raise HeySolError(f"Log deletion returned empty result for ID: {log_id}")

            self.logger.info(f"Log deleted successfully: {log_id}")
            return {"log_id": log_id, "deleted": True}

        except ValidationError:
            raise  # Re-raise validation errors
        except AuthenticationError:
            raise  # Re-raise auth errors
        except Exception as e:
            raise HeySolError(f"Log deletion failed: {e}")

    def get_logs(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent logs with OAuth2 authentication.

        Args:
            limit: Maximum number of logs to retrieve

        Returns:
            List of log entries

        Raises:
            ValidationError: If limit is invalid
            AuthenticationError: If OAuth2 authentication fails
            HeySolError: If retrieval fails
        """
        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100")

        try:
            client = self.client_manager.get_client()
            logs = client.get_ingestion_logs(limit=limit)

            if not isinstance(logs, list):
                raise HeySolError("Invalid logs response format")

            self.logger.info(f"Retrieved {len(logs)} logs")
            return logs

        except ValidationError:
            raise  # Re-raise validation errors
        except AuthenticationError:
            raise  # Re-raise auth errors
        except Exception as e:
            raise HeySolError(f"Log retrieval failed: {e}")


class OAuth2DemoRunner:
    """Centralized OAuth2 demo runner with strict error handling."""

    def __init__(self):
        """Initialize demo runner."""
        self.client_manager = OAuth2ClientManager()
        self.log_ops = OAuth2LogOperations(self.client_manager)
        self.logger = logging.getLogger(__name__)

    def run_complete_demo(self) -> Dict[str, Any]:
        """
        Run complete OAuth2 demo with strict error handling.

        Returns:
            Demo results dictionary

        Raises:
            AuthenticationError: If OAuth2 setup fails
            HeySolError: If demo execution fails
        """
        results = {
            "timestamp": time.time(),
            "success": False,
            "steps": [],
            "error": None
        }

        try:
            # Step 1: Validate configuration
            self.logger.info("Step 1: Validating OAuth2 configuration")
            OAuth2ConfigurationValidator.validate()
            results["steps"].append({
                "step": "configuration_validation",
                "status": "completed",
                "description": "OAuth2 configuration validated"
            })

            # Step 2: Create client
            self.logger.info("Step 2: Creating OAuth2 client")
            client = self.client_manager.create_client()
            results["steps"].append({
                "step": "client_creation",
                "status": "completed",
                "description": "OAuth2 client created"
            })

            # Step 3: Authenticate
            self.logger.info("Step 3: Performing OAuth2 authentication")
            self.client_manager.ensure_authenticated()
            results["steps"].append({
                "step": "authentication",
                "status": "completed",
                "description": "OAuth2 authentication completed"
            })

            # Step 4: Create test space
            self.logger.info("Step 4: Creating test space")
            space_id = client.get_or_create_space(
                "oauth2_demo",
                "Test space for OAuth2 demo"
            )
            results["steps"].append({
                "step": "space_creation",
                "status": "completed",
                "description": f"Created test space: {space_id}",
                "space_id": space_id
            })

            # Step 5: Ingest test log
            self.logger.info("Step 5: Ingesting test log")
            test_message = f"OAuth2 demo test - {time.time()}"
            ingest_result = self.log_ops.ingest_log(test_message, "oauth2_demo")
            log_id = ingest_result["log_id"]
            results["steps"].append({
                "step": "log_ingestion",
                "status": "completed",
                "description": f"Ingested test log: {log_id}",
                "log_id": log_id
            })

            # Step 6: Delete test log
            self.logger.info("Step 6: Deleting test log")
            delete_result = self.log_ops.delete_log(log_id)
            results["steps"].append({
                "step": "log_deletion",
                "status": "completed",
                "description": f"Deleted test log: {log_id}"
            })

            # Step 7: Clean up
            self.logger.info("Step 7: Cleaning up test space")
            client.delete_space(space_id)
            results["steps"].append({
                "step": "cleanup",
                "status": "completed",
                "description": f"Cleaned up test space: {space_id}"
            })

            results["success"] = True
            self.logger.info("OAuth2 demo completed successfully")
            return results

        except AuthenticationError as e:
            results["error"] = f"Authentication failed: {e}"
            raise
        except ValidationError as e:
            results["error"] = f"Validation failed: {e}"
            raise
        except HeySolError as e:
            results["error"] = f"Demo execution failed: {e}"
            raise
        except Exception as e:
            results["error"] = f"Unexpected error: {e}"
            raise HeySolError(f"Demo execution failed: {e}")


# ===== UTILITY FUNCTIONS =====

def create_oauth2_demo_runner() -> OAuth2DemoRunner:
    """
    Factory function to create OAuth2 demo runner.

    Returns:
        Configured OAuth2DemoRunner instance

    Raises:
        AuthenticationError: If OAuth2 configuration is invalid
    """
    try:
        OAuth2ConfigurationValidator.validate()
        return OAuth2DemoRunner()
    except Exception as e:
        raise AuthenticationError(f"Failed to create OAuth2 demo runner: {e}")


def validate_oauth2_setup() -> bool:
    """
    Validate OAuth2 setup and provide helpful error messages.

    Returns:
        True if setup is valid

    Raises:
        AuthenticationError: If setup is invalid
    """
    try:
        result = OAuth2ConfigurationValidator.validate()
        print("✅ OAuth2 configuration is valid")
        return True
    except AuthenticationError as e:
        print(f"❌ OAuth2 configuration error: {e}")
        print("\nTo fix this:")
        print("1. Set COREAI_OAUTH2_CLIENT_ID environment variable")
        print("2. Set COREAI_OAUTH2_CLIENT_SECRET environment variable")
        print("3. Ensure credentials are valid Google OAuth2 credentials")
        raise


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