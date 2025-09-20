"""
Interactive OAuth2 authentication implementation for HeySol API client.

This module provides an interactive OAuth2 authorization flow that opens a browser
for user authentication and handles the callback automatically.
"""

import json
import time
import uuid
import secrets
import threading
import webbrowser
import http.server
import socketserver
from typing import Dict, Any, Optional, Callable
from urllib.parse import urlparse, parse_qs
import requests
from dataclasses import dataclass

from .config import HeySolConfig
from .oauth2 import OAuth2Tokens, OAuth2Error


@dataclass
class OAuth2CallbackResult:
    """Result from OAuth2 callback."""
    success: bool
    authorization_code: Optional[str] = None
    error: Optional[str] = None
    state: Optional[str] = None


class OAuth2CallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for OAuth2 callback."""

    def __init__(self, callback_result: OAuth2CallbackResult, *args, **kwargs):
        self.callback_result = callback_result
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET request for OAuth2 callback."""
        try:
            # Parse the URL
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            # Get authorization code or error
            if 'code' in query_params:
                self.callback_result.success = True
                self.callback_result.authorization_code = query_params['code'][0]
                self.callback_result.state = query_params.get('state', [None])[0]

                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"""
                <!DOCTYPE html>
                <html>
                <head><title>Authorization Successful</title></head>
                <body>
                <h2>Authorization Successful!</h2>
                <p>You can now close this window and return to the application.</p>
                <script>window.close();</script>
                </body>
                </html>
                """)
            else:
                self.callback_result.success = False
                self.callback_result.error = query_params.get('error', ['Unknown error'])[0]

                # Send error response
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"""
                <!DOCTYPE html>
                <html>
                <head><title>Authorization Failed</title></head>
                <body>
                <h2>Authorization Failed</h2>
                <p>Error: {self.callback_result.error}</p>
                <p>You can close this window and try again.</p>
                </body>
                </html>
                """.encode())

        except Exception as e:
            self.callback_result.success = False
            self.callback_result.error = str(e)

            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Server Error</title></head>
            <body>
            <h2>Server Error</h2>
            <p>Error: {e}</p>
            </body>
            </html>
            """.encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class InteractiveOAuth2Authenticator:
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
        self.config = config
        self.tokens: Optional[OAuth2Tokens] = None
        self.client_id = config.oauth2_client_id or "829721654699-fnnmk21eflekbi1tgr2lc2rms559bbh6.apps.googleusercontent.com"
        self.client_secret = config.oauth2_client_secret
        self.redirect_uri = config.oauth2_redirect_uri or "http://localhost:8080/callback"

        # OAuth2 endpoints (Google OAuth2 for HeySol)
        self.authorization_url = "https://accounts.google.com/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.user_info_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        self.introspection_url = "https://oauth2.googleapis.com/tokeninfo"

        # Callback server
        self.callback_result = OAuth2CallbackResult(success=False)
        self.server_thread: Optional[threading.Thread] = None
        self.server: Optional[socketserver.TCPServer] = None

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
                    with socketserver.TCPServer(("", port), lambda *args, **kwargs: OAuth2CallbackHandler(self.callback_result, *args, **kwargs)) as httpd:
                        self.server = httpd
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
            return self.exchange_code_for_tokens(callback_result.authorization_code, code_verifier, on_progress)

        finally:
            # Always stop the callback server
            self.stop_callback_server()

    def exchange_code_for_tokens(
        self,
        code: str,
        code_verifier: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None
    ) -> OAuth2Tokens:
        """
        Exchange authorization code for access tokens.

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier
            on_progress: Optional callback for progress updates

        Returns:
            OAuth2Tokens instance

        Raises:
            OAuth2Error: If token exchange fails
        """
        if on_progress:
            on_progress("Exchanging authorization code for access tokens...")

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

            if on_progress:
                on_progress("Successfully obtained access tokens!")

            return tokens

        except requests.exceptions.RequestException as e:
            raise OAuth2Error(f"Token exchange failed: {e}")
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