"""
Tests for configuration management in the HeySol API client.
"""

import os
import tempfile
import json
from pathlib import Path
import pytest

from heysol.config import HeySolConfig


class TestHeySolConfig:
    """Test HeySolConfig class functionality."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = HeySolConfig()

        assert config.api_key is None
        assert config.base_url == "https://core.heysol.ai/api/v1/mcp"
        assert config.source == "heysol-python-client"
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.rate_limit_per_minute == 60
        assert config.rate_limit_enabled is True
        assert config.log_level == "INFO"
        assert config.async_enabled is False

    def test_config_from_env(self, monkeypatch):
        """Test loading config from environment variables."""
        monkeypatch.setenv("COREAI_API_KEY", "test-api-key")
        monkeypatch.setenv("COREAI_BASE_URL", "https://test.coreai.ai/api/v1/mcp")
        monkeypatch.setenv("COREAI_SOURCE", "test-client")
        monkeypatch.setenv("COREAI_TIMEOUT", "30")
        monkeypatch.setenv("COREAI_MAX_RETRIES", "5")
        monkeypatch.setenv("COREAI_RATE_LIMIT_PER_MINUTE", "30")
        monkeypatch.setenv("COREAI_RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("COREAI_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("COREAI_ASYNC_ENABLED", "true")

        config = HeySolConfig.from_env()

        assert config.api_key == "test-api-key"
        assert config.base_url == "https://test.coreai.ai/api/v1/mcp"
        assert config.source == "test-client"
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.rate_limit_per_minute == 30
        assert config.rate_limit_enabled is False
        assert config.log_level == "DEBUG"
        assert config.async_enabled is True

    def test_config_from_file(self):
        """Test loading config from a JSON file."""
        config_data = {
            "api_key": "file-api-key",
            "base_url": "https://file.heysol.ai/api/v1/mcp",
            "source": "file-client",
            "timeout": 45,
            "max_retries": 2,
            "rate_limit_per_minute": 45,
            "rate_limit_enabled": False,
            "log_level": "WARNING",
            "async_enabled": True,
            "max_async_workers": 15,
            "default_spaces": {
                "test_space": "Test Space"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = HeySolConfig.from_file(config_file)

            assert config.api_key == "file-api-key"
            assert config.base_url == "https://file.heysol.ai/api/v1/mcp"
            assert config.source == "file-client"
            assert config.timeout == 45
            assert config.max_retries == 2
            assert config.rate_limit_per_minute == 45
            assert config.rate_limit_enabled is False
            assert config.log_level == "WARNING"
            assert config.async_enabled is True
            assert config.max_async_workers == 15
            assert config.default_spaces["test_space"] == "Test Space"

        finally:
            Path(config_file).unlink()

    def test_config_from_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config file."""
        with pytest.raises(FileNotFoundError):
            HeySolConfig.from_file("nonexistent_file.json")

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "api_key": "dict-api-key",
            "base_url": "https://dict.heysol.ai/api/v1/mcp",
            "timeout": 25,
            "default_spaces": {
                "dict_space": "Dict Space"
            }
        }

        config = HeySolConfig.from_dict(config_dict)

        assert config.api_key == "dict-api-key"
        assert config.base_url == "https://dict.heysol.ai/api/v1/mcp"
        assert config.timeout == 25
        assert config.default_spaces["dict_space"] == "Dict Space"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = HeySolConfig(
            api_key="test-key",
            base_url="https://test.heysol.ai/api/v1/mcp",
            timeout=30
        )

        config_dict = config.to_dict()

        assert config_dict["api_key"] == "test-key"
        assert config_dict["base_url"] == "https://test.heysol.ai/api/v1/mcp"
        assert config_dict["timeout"] == 30

    def test_config_to_json(self):
        """Test converting config to JSON string."""
        config = HeySolConfig(api_key="test-key", timeout=30)
        json_str = config.to_json()

        # Should be valid JSON
        config_dict = json.loads(json_str)
        assert config_dict["api_key"] == "test-key"
        assert config_dict["timeout"] == 30

    def test_config_save_to_file(self):
        """Test saving config to file."""
        config = HeySolConfig(api_key="test-key", timeout=30)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            config.save_to_file(config_file)

            # Verify file was created and contains correct data
            with open(config_file, "r") as f:
                saved_data = json.load(f)

            assert saved_data["api_key"] == "test-key"
            assert saved_data["timeout"] == 30

        finally:
            Path(config_file).unlink()

    def test_config_from_env_with_coreai_key(self, monkeypatch):
        """Test config loading from environment with CORE AI API key."""
        # Set up environment variables for CORE AI API
        monkeypatch.setenv("COREAI_API_KEY", "test-coreai-key")
        monkeypatch.setenv("COREAI_BASE_URL", "https://test.coreai.ai/api/v1/mcp")
        monkeypatch.setenv("COREAI_SOURCE", "test-client")
        monkeypatch.setenv("COREAI_TIMEOUT", "30")
        monkeypatch.setenv("COREAI_MAX_RETRIES", "5")
        monkeypatch.setenv("COREAI_RATE_LIMIT_PER_MINUTE", "30")
        monkeypatch.setenv("COREAI_RATE_LIMIT_ENABLED", "false")
        monkeypatch.setenv("COREAI_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("COREAI_ASYNC_ENABLED", "true")

        config = HeySolConfig.from_env()

        assert config.api_key == "test-coreai-key"
        assert config.base_url == "https://test.coreai.ai/api/v1/mcp"
        assert config.source == "test-client"
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.rate_limit_per_minute == 30
        assert config.rate_limit_enabled is False
        assert config.log_level == "DEBUG"
        assert config.async_enabled is True

    def test_config_integration_with_client(self, monkeypatch):
        """Test that config works with actual client instantiation."""
        # Mock the session initialization to avoid actual API calls
        def mock_initialize_session(self):
            self.session_id = "mock-session-id"
            self.logger.info("MCP session initialized (mock)")

        # Test with sync client
        from heysol.client import HeySolClient

        config = HeySolConfig(api_key="test-key", timeout=30)
        # Use a mock base URL to avoid actual API calls
        config.base_url = "https://mock-api.example.com/api/v1/mcp"

        # Mock the initialization method
        monkeypatch.setattr(HeySolClient, "_initialize_session", mock_initialize_session)

        client = HeySolClient(config=config)

        assert client.config.api_key == "test-key"
        assert client.config.timeout == 30

        # Test with async client
        from heysol.async_client import AsyncHeySolClient

        # Mock the async initialization method
        def mock_initialize_session_async(self):
            self.session_id = "mock-session-id"
            self.logger.info("MCP session initialized (mock)")

        monkeypatch.setattr(AsyncHeySolClient, "_initialize_session_sync", mock_initialize_session_async)

        async_client = AsyncHeySolClient(config=config)
        assert async_client.config.api_key == "test-key"
        assert async_client.config.timeout == 30

    def test_config_getters(self):
        """Test configuration getter methods."""
        config = HeySolConfig(
            api_key="test-key",
            base_url="https://test.heysol.ai/api/v1/mcp",
            timeout=30,
            max_retries=5,
            rate_limit_per_minute=45,
            log_level="DEBUG",
            async_enabled=True,
            max_async_workers=20
        )

        assert config.get_api_key() == "test-key"
        assert config.get_base_url() == "https://test.heysol.ai/api/v1/mcp"
        assert config.get_timeout() == 30
        assert config.get_max_retries() == 5
        assert config.get_rate_limit_per_minute() == 45
        assert config.get_log_level() == "DEBUG"
        assert config.is_async_enabled() is True
        assert config.get_max_async_workers() == 20

    def test_config_default_spaces_post_init(self):
        """Test that default_spaces is initialized properly."""
        config = HeySolConfig()
        expected_defaults = {
            "clinical_trials": "Clinical Trials",
            "patients": "Patients"
        }

        assert config.default_spaces == expected_defaults

    def test_config_custom_default_spaces(self):
        """Test config with custom default spaces."""
        custom_spaces = {"custom_space": "Custom Space"}
        config = HeySolConfig(default_spaces=custom_spaces)

        assert config.default_spaces == custom_spaces

    def test_config_rate_limit_methods(self):
        """Test rate limiting related methods."""
        config = HeySolConfig(rate_limit_enabled=False, rate_limit_per_minute=100)

        assert config.is_rate_limit_enabled() is False
        assert config.get_rate_limit_per_minute() == 100

    def test_config_logging_methods(self):
        """Test logging related methods."""
        config = HeySolConfig(
            log_level="ERROR",
            log_to_file=True,
            log_file_path="/var/log/heysol.log"
        )

        assert config.get_log_level() == "ERROR"
        assert config.should_log_to_file() is True
        assert config.get_log_file_path() == "/var/log/heysol.log"

    def test_config_batch_methods(self):
        """Test batch operation methods."""
        config = HeySolConfig(batch_size=50, max_concurrent_requests=10)

        assert config.get_batch_size() == 50
        assert config.get_max_concurrent_requests() == 10

    def test_config_env_with_invalid_values(self, monkeypatch):
        """Test config from env with invalid values (should use defaults)."""
        monkeypatch.setenv("HEYSOL_TIMEOUT", "invalid")
        monkeypatch.setenv("HEYSOL_MAX_RETRIES", "not_a_number")

        config = HeySolConfig.from_env()

        # Should use default values for invalid env vars
        assert config.timeout == 60  # default
        assert config.max_retries == 3  # default

    def test_config_from_empty_file(self):
        """Test loading config from empty file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")
            config_file = f.name

        try:
            config = HeySolConfig.from_file(config_file)

            # Should use default values
            assert config.api_key is None
            assert config.base_url == "https://core.heysol.ai/api/v1/mcp"
            assert config.timeout == 60

        finally:
            Path(config_file).unlink()