"""
Centralized OAuth2 utilities for HeySol API client.

This module provides consolidated OAuth2 functionality to eliminate duplication
across CLI, demo scripts, and notebooks. It enforces strict error handling
and provides lean, performant OAuth2 operations.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from functools import lru_cache
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HeySolConfig
from .client import HeySolClient
from .exceptions import HeySolError, AuthenticationError, ValidationError
from .oauth2_interactive import InteractiveOAuth2Authenticator


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

            # If client supports session injection, it could use the optimized session
            # For now, the session is available via get_optimized_session()

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

    def _is_auth_cached(self, scope: str) -> Optional[bool]:
        """
        Check if authentication status is cached and valid.

        Args:
            scope: OAuth2 scope

        Returns:
            True if authenticated, False if not, None if cache miss
        """
        with self._cache_lock:
            cache_key = f"auth_{scope}"
            if cache_key in self._auth_cache:
                is_valid, timestamp = self._auth_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return is_valid
                else:
                    # Cache expired, remove it
                    del self._auth_cache[cache_key]
        return None

    def _cache_auth_status(self, scope: str, is_valid: bool):
        """
        Cache authentication status.

        Args:
            scope: OAuth2 scope
            is_valid: Whether authentication is valid
        """
        with self._cache_lock:
            cache_key = f"auth_{scope}"
            self._auth_cache[cache_key] = (is_valid, time.time())

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

        client = self.get_client()

        try:
            # Check if already authenticated
            if hasattr(client, 'oauth2_auth') and client.oauth2_auth:
                try:
                    # Try to get user info to verify authentication
                    client.get_oauth2_user_info()
                    self.logger.info("OAuth2 authentication already valid")
                    self._cache_auth_status(scope, True)
                    return True
                except AuthenticationError:
                    self.logger.info("OAuth2 authentication expired, re-authenticating...")
                    self._cache_auth_status(scope, False)

            # Perform interactive authorization
            success = client.authorize_oauth2_interactive(scope=scope)
            if not success:
                raise AuthenticationError("Interactive OAuth2 authorization failed")

            self.logger.info("OAuth2 authentication completed successfully")
            self._cache_auth_status(scope, True)
            return True

        except Exception as e:
            self._cache_auth_status(scope, False)
            raise AuthenticationError(f"OAuth2 authentication failed: {e}")


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