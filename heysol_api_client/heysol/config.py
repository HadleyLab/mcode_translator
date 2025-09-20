"""
Configuration management for the HeySol API client.

This module provides configuration management for the HeySol API client,
supporting environment variables, configuration files, and programmatic configuration.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, asdict


@dataclass
class HeySolConfig:
    """
    Configuration class for HeySol API client.

    This class holds all configuration options for the HeySol API client,
    with support for loading from environment variables and config files.
    """

    # API Configuration
    api_key: Optional[str] = None
    base_url: str = "https://core.heysol.ai/api/v1/mcp"
    source: str = "heysol-python-client"
    timeout: int = 60

    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_enabled: bool = True

    # Memory Management
    default_spaces: Dict[str, str] = None
    batch_size: int = 10
    max_concurrent_requests: int = 5

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Optional[str] = None

    # Async Configuration
    async_enabled: bool = False
    max_async_workers: int = 10

    # OAuth2 Configuration
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_redirect_uri: Optional[str] = None
    oauth2_use_pkce: bool = True
    oauth2_scope: str = "openid profile email api"

    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.default_spaces is None:
            self.default_spaces = {
                "clinical_trials": "Clinical Trials",
                "patients": "Patients"
            }

    @classmethod
    def from_env(cls) -> "HeySolConfig":
        """
        Create configuration from environment variables.

        Returns:
            HeySolConfig: Configuration instance loaded from environment variables
        """
        def safe_int(value: str, default: int) -> int:
            """Safely convert string to int, returning default on failure."""
            try:
                return int(value)
            except (ValueError, TypeError):
                return default

        def safe_float(value: str, default: float) -> float:
            """Safely convert string to float, returning default on failure."""
            try:
                return float(value)
            except (ValueError, TypeError):
                return default

        return cls(
            api_key=os.getenv("COREAI_API_KEY") or os.getenv("CORE_MEMORY_API_KEY"),
            base_url=os.getenv("COREAI_BASE_URL", "https://core.heysol.ai/api/v1/mcp"),
            source=os.getenv("COREAI_SOURCE", "heysol-python-client"),
            timeout=safe_int(os.getenv("COREAI_TIMEOUT", "60"), 60),
            max_retries=safe_int(os.getenv("COREAI_MAX_RETRIES", "3"), 3),
            retry_delay=safe_float(os.getenv("COREAI_RETRY_DELAY", "1.0"), 1.0),
            backoff_factor=safe_float(os.getenv("COREAI_BACKOFF_FACTOR", "2.0"), 2.0),
            rate_limit_per_minute=safe_int(os.getenv("COREAI_RATE_LIMIT_PER_MINUTE", "60"), 60),
            rate_limit_enabled=os.getenv("COREAI_RATE_LIMIT_ENABLED", "true").lower() == "true",
            log_level=os.getenv("COREAI_LOG_LEVEL", "INFO"),
            log_to_file=os.getenv("COREAI_LOG_TO_FILE", "false").lower() == "true",
            log_file_path=os.getenv("COREAI_LOG_FILE_PATH"),
            async_enabled=os.getenv("COREAI_ASYNC_ENABLED", "false").lower() == "true",
            max_async_workers=safe_int(os.getenv("COREAI_MAX_ASYNC_WORKERS", "10"), 10),
            oauth2_client_id=os.getenv("COREAI_OAUTH2_CLIENT_ID"),
            oauth2_client_secret=os.getenv("COREAI_OAUTH2_CLIENT_SECRET"),
            oauth2_redirect_uri=os.getenv("COREAI_OAUTH2_REDIRECT_URI"),
            oauth2_use_pkce=os.getenv("COREAI_OAUTH2_USE_PKCE", "true").lower() == "true",
            oauth2_scope=os.getenv("COREAI_OAUTH2_SCOPE", "openid profile email api"),
        )

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "HeySolConfig":
        """
        Create configuration from a JSON file.

        Args:
            file_path: Path to the configuration file

        Returns:
            HeySolConfig: Configuration instance loaded from file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            json.JSONDecodeError: If the configuration file is not valid JSON
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle nested default_spaces
        default_spaces = data.get("default_spaces", {})

        return cls(
            api_key=data.get("api_key"),
            base_url=data.get("base_url", "https://core.heysol.ai/api/v1/mcp"),
            source=data.get("source", "heysol-python-client"),
            timeout=data.get("timeout", 60),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            backoff_factor=data.get("backoff_factor", 2.0),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
            rate_limit_enabled=data.get("rate_limit_enabled", True),
            default_spaces=default_spaces,
            batch_size=data.get("batch_size", 10),
            max_concurrent_requests=data.get("max_concurrent_requests", 5),
            log_level=data.get("log_level", "INFO"),
            log_to_file=data.get("log_to_file", False),
            log_file_path=data.get("log_file_path"),
            async_enabled=data.get("async_enabled", False),
            max_async_workers=data.get("max_async_workers", 10),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeySolConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values

        Returns:
            HeySolConfig: Configuration instance loaded from dictionary
        """
        # Handle nested default_spaces
        default_spaces = data.get("default_spaces", {})

        return cls(
            api_key=data.get("api_key"),
            base_url=data.get("base_url", "https://core.heysol.ai/api/v1/mcp"),
            source=data.get("source", "heysol-python-client"),
            timeout=data.get("timeout", 60),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 1.0),
            backoff_factor=data.get("backoff_factor", 2.0),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
            rate_limit_enabled=data.get("rate_limit_enabled", True),
            default_spaces=default_spaces,
            batch_size=data.get("batch_size", 10),
            max_concurrent_requests=data.get("max_concurrent_requests", 5),
            log_level=data.get("log_level", "INFO"),
            log_to_file=data.get("log_to_file", False),
            log_file_path=data.get("log_file_path"),
            async_enabled=data.get("async_enabled", False),
            max_async_workers=data.get("max_async_workers", 10),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        return asdict(self)

    def to_json(self) -> str:
        """
        Convert configuration to JSON string.

        Returns:
            str: JSON representation of the configuration
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.

        Args:
            file_path: Path where to save the configuration file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def get_api_key(self) -> Optional[str]:
        """Get the API key."""
        return self.api_key

    def get_base_url(self) -> str:
        """Get the base URL."""
        return self.base_url

    def get_source(self) -> str:
        """Get the source identifier."""
        return self.source

    def get_timeout(self) -> int:
        """Get the request timeout."""
        return self.timeout

    def get_max_retries(self) -> int:
        """Get the maximum number of retries."""
        return self.max_retries

    def get_retry_delay(self) -> float:
        """Get the retry delay."""
        return self.retry_delay

    def get_backoff_factor(self) -> float:
        """Get the backoff factor for exponential backoff."""
        return self.backoff_factor

    def get_rate_limit_per_minute(self) -> int:
        """Get the rate limit per minute."""
        return self.rate_limit_per_minute

    def is_rate_limit_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.rate_limit_enabled

    def get_default_spaces(self) -> Dict[str, str]:
        """Get the default spaces configuration."""
        return self.default_spaces

    def get_batch_size(self) -> int:
        """Get the batch size for bulk operations."""
        return self.batch_size

    def get_max_concurrent_requests(self) -> int:
        """Get the maximum number of concurrent requests."""
        return self.max_concurrent_requests

    def get_log_level(self) -> str:
        """Get the log level."""
        return self.log_level

    def should_log_to_file(self) -> bool:
        """Check if logging to file is enabled."""
        return self.log_to_file

    def get_log_file_path(self) -> Optional[str]:
        """Get the log file path."""
        return self.log_file_path

    def is_async_enabled(self) -> bool:
        """Check if async support is enabled."""
        return self.async_enabled

    def get_max_async_workers(self) -> int:
        """Get the maximum number of async workers."""
        return self.max_async_workers