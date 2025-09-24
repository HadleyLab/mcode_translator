"""
Unit tests for llm_loader module.
"""

import json
import os
from unittest.mock import Mock, patch, mock_open
import pytest
from src.utils.llm_loader import (
    LLMConfig,
    LLMLoader,
    load_llm,
    reload_llms_config,
    get_default_llm,
)


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_init(self):
        """Test LLMConfig initialization."""
        config = LLMConfig(
            name="test-llm",
            model_type="openai",
            model_identifier="gpt-4",
            api_key_env_var="TEST_API_KEY",
            base_url="https://api.test.com",
            default_parameters={"temperature": 0.7},
            timeout_seconds=30,
            rate_limit_per_minute=50,
            default=True,
        )

        assert config.name == "test-llm"
        assert config.model_type == "openai"
        assert config.model_identifier == "gpt-4"
        assert config.api_key_env_var == "TEST_API_KEY"
        assert config.base_url == "https://api.test.com"
        assert config.default_parameters == {"temperature": 0.7}
        assert config.timeout_seconds == 30
        assert config.rate_limit_per_minute == 50
        assert config.default is True

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key-123"})
    def test_api_key_property_with_env_var(self):
        """Test api_key property with environment variable set."""
        config = LLMConfig(api_key_env_var="TEST_API_KEY")
        assert config.api_key == "test-key-123"

    def test_api_key_property_without_env_var(self):
        """Test api_key property without environment variable."""
        config = LLMConfig(api_key_env_var="NONEXISTENT_VAR")
        assert config.api_key == ""

    def test_api_key_property_no_env_var_configured(self):
        """Test api_key property when no env var is configured."""
        config = LLMConfig(api_key_env_var="")
        assert config.api_key == ""

    def test_to_dict(self):
        """Test to_dict method."""
        config = LLMConfig(
            name="test-llm",
            model_type="openai",
            model_identifier="gpt-4",
            api_key_env_var="TEST_API_KEY",
            base_url="https://api.test.com",
            default_parameters={"temperature": 0.7},
            timeout_seconds=30,
            rate_limit_per_minute=50,
            default=True,
        )

        result = config.to_dict()

        expected = {
            "name": "test-llm",
            "model_type": "openai",
            "model_identifier": "gpt-4",
            "api_key_env_var": "TEST_API_KEY",
            "base_url": "https://api.test.com",
            "default_parameters": {"temperature": 0.7},
            "timeout_seconds": 30,
            "rate_limit_per_minute": 50,
            "default": True,
        }

        assert result == expected

    def test_from_dict_complete(self):
        """Test from_dict with complete data."""
        data = {
            "name": "test-llm",
            "model_type": "openai",
            "model_identifier": "gpt-4",
            "api_key_env_var": "TEST_API_KEY",
            "base_url": "https://api.test.com",
            "default_parameters": {"temperature": 0.7},
            "timeout_seconds": 30,
            "rate_limit_per_minute": 50,
            "default": True,
        }

        config = LLMConfig.from_dict(data)

        assert config.name == "test-llm"
        assert config.model_type == "openai"
        assert config.model_identifier == "gpt-4"
        assert config.api_key_env_var == "TEST_API_KEY"
        assert config.base_url == "https://api.test.com"
        assert config.default_parameters == {"temperature": 0.7}
        assert config.timeout_seconds == 30
        assert config.rate_limit_per_minute == 50
        assert config.default is True

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {}

        config = LLMConfig.from_dict(data)

        assert config.name == ""
        assert config.model_type == ""
        assert config.model_identifier == ""
        assert config.api_key_env_var == ""
        assert config.base_url == ""
        assert config.default_parameters == {}
        assert config.timeout_seconds == 60
        assert config.rate_limit_per_minute == 100
        assert config.default is False

    def test_from_dict_legacy_fields(self):
        """Test from_dict with legacy field names."""
        data = {
            "model_name": "legacy-llm",
            "provider": "legacy-provider",
            "api_base": "https://legacy.api.com",
        }

        config = LLMConfig.from_dict(data)

        assert config.name == "legacy-llm"
        assert config.model_type == "legacy-provider"
        assert config.model_identifier == "legacy-llm"
        assert config.base_url == "https://legacy.api.com"


class TestLLMLoader:
    """Test LLMLoader class."""

    def test_init_custom_path(self):
        """Test initialization with custom path."""
        custom_path = "/custom/path/llms.json"
        loader = LLMLoader(custom_path)

        assert str(loader.llms_config_path) == custom_path

    @patch("src.utils.llm_loader.Path")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils.llm_loader.json.load")
    def test_load_llms_config_nested_structure(
        self, mock_json_load, mock_file, mock_path
    ):
        """Test loading config with nested structure."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_json_load.return_value = {
            "models": {
                "available": {
                    "llm1": {"name": "LLM1"},
                    "llm2": {"name": "LLM2"},
                }
            }
        }

        loader = LLMLoader("/fake/path")
        config = loader._load_llms_config()

        expected = {
            "llm1": {"name": "LLM1"},
            "llm2": {"name": "LLM2"},
        }
        assert config == expected

    @patch("src.utils.llm_loader.Path")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils.llm_loader.json.load")
    def test_load_llms_config_flat_structure(
        self, mock_json_load, mock_file, mock_path
    ):
        """Test loading config with flat structure."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_json_load.return_value = {
            "llm1": {"name": "LLM1"},
            "llm2": {"name": "LLM2"},
        }

        loader = LLMLoader("/fake/path")
        config = loader._load_llms_config()

        expected = {
            "llm1": {"name": "LLM1"},
            "llm2": {"name": "LLM2"},
        }
        assert config == expected

    @patch("src.utils.llm_loader.Path")
    def test_load_llms_config_file_not_found(self, mock_path):
        """Test loading config when file doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        loader = LLMLoader("/fake/path")
        config = loader._load_llms_config()

        # Should return empty dict when file not found
        assert config == {}

    @patch("src.utils.llm_loader.Path")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils.llm_loader.json.load")
    def test_load_llms_config_json_error(self, mock_json_load, mock_file, mock_path):
        """Test loading config with JSON error."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        loader = LLMLoader("/fake/path")
        config = loader._load_llms_config()

        # Should return empty dict on error
        assert config == {}

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"})
    def test_get_llm_success(self):
        """Test successful LLM retrieval."""
        loader = LLMLoader.__new__(LLMLoader)  # Create without calling __init__
        loader.llms_config = {
            "test-llm": {
                "name": "Test LLM",
                "model_type": "openai",
                "model_identifier": "gpt-4",
                "api_key_env_var": "TEST_API_KEY",
                "base_url": "https://api.test.com",
                "default_parameters": {"temperature": 0.7},
                "timeout_seconds": 30,
                "rate_limit_per_minute": 50,
                "default": False,
            }
        }

        result = loader.get_llm("test-llm")

        assert isinstance(result, LLMConfig)
        assert result.name == "Test LLM"
        assert result.api_key == "test-key"

    def test_get_llm_not_found(self):
        """Test LLM retrieval when key doesn't exist."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {}

        with pytest.raises(ValueError, match="LLM key 'nonexistent' not found"):
            loader.get_llm("nonexistent")

    def test_get_llm_missing_api_key_env_var(self):
        """Test LLM retrieval with missing API key env var."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "test-llm": {
                "name": "Test LLM",
                "api_key_env_var": "",  # Missing
            }
        }

        with pytest.raises(ValueError, match="'api_key_env_var' not set"):
            loader.get_llm("test-llm")

    @patch.dict(os.environ, {}, clear=True)  # Clear all env vars
    def test_get_llm_missing_env_var(self):
        """Test LLM retrieval when environment variable is not set."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "test-llm": {
                "name": "Test LLM",
                "api_key_env_var": "MISSING_VAR",
            }
        }

        with pytest.raises(
            ValueError, match="Environment variable 'MISSING_VAR' is not set"
        ):
            loader.get_llm("test-llm")

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"})
    def test_get_all_llms_success(self):
        """Test getting all LLMs successfully."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "llm1": {
                "name": "LLM1",
                "api_key_env_var": "TEST_API_KEY",
            },
            "llm2": {
                "name": "LLM2",
                "api_key_env_var": "TEST_API_KEY",
            },
        }

        result = loader.get_all_llms()

        assert len(result) == 2
        assert "llm1" in result
        assert "llm2" in result
        assert all(isinstance(config, LLMConfig) for config in result.values())

    def test_get_all_llms_with_failure(self):
        """Test getting all LLMs when one fails."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "llm1": {
                "name": "LLM1",
                "api_key_env_var": "TEST_API_KEY",
            },
            "llm2": {
                "name": "LLM2",
                "api_key_env_var": "",  # Will fail
            },
        }

        with pytest.raises(ValueError):
            loader.get_all_llms()

    def test_reload_config(self):
        """Test config reloading."""
        loader = LLMLoader.__new__(LLMLoader)
        loader._load_llms_config = Mock(return_value={"new": "config"})

        loader.reload_config()

        assert loader.llms_config == {"new": "config"}

    def test_get_llm_metadata(self):
        """Test getting LLM metadata."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {"test-llm": {"name": "Test LLM", "meta": "data"}}

        result = loader.get_llm_metadata("test-llm")

        assert result == {"name": "Test LLM", "meta": "data"}

    def test_get_llm_metadata_not_found(self):
        """Test getting metadata for non-existent LLM."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {}

        result = loader.get_llm_metadata("nonexistent")

        assert result is None

    def test_list_available_llms(self):
        """Test listing available LLMs."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "llm1": {"name": "LLM1"},
            "llm2": {"name": "LLM2"},
        }

        result = loader.list_available_llms()

        expected = {
            "llm1": {"name": "LLM1"},
            "llm2": {"name": "LLM2"},
        }
        assert result == expected

    def test_list_available_llms_with_duplicates(self):
        """Test listing LLMs with duplicate keys."""
        loader = LLMLoader.__new__(LLMLoader)
        # Simulate duplicate keys (though dict would normally prevent this)
        loader.llms_config = {
            "llm1": {"name": "LLM1"},
            "llm1": {"name": "Duplicate LLM1"},  # This would overwrite in real dict
        }

        result = loader.list_available_llms()

        # Should work fine since dict handles duplicates by overwriting
        assert len(result) == 1

    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"})
    def test_get_default_llm_success(self):
        """Test getting default LLM successfully."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "deepseek-coder": {
                "name": "DeepSeek Coder",
                "api_key_env_var": "TEST_API_KEY",
            }
        }
        loader.get_llm = Mock(return_value=LLMConfig(name="DeepSeek Coder"))

        result = loader.get_default_llm()

        assert isinstance(result, LLMConfig)
        loader.get_llm.assert_called_with("deepseek-coder")

    def test_get_default_llm_fallback(self):
        """Test default LLM fallback when deepseek-coder fails."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {
            "other-llm": {
                "name": "Other LLM",
            }
        }
        loader.get_llm = Mock(
            side_effect=[Exception("deepseek failed"), LLMConfig(name="Other LLM")]
        )

        result = loader.get_default_llm()

        assert result.name == "Other LLM"
        assert loader.get_llm.call_count == 2

    def test_get_default_llm_no_llms_available(self):
        """Test default LLM when no LLMs are available."""
        loader = LLMLoader.__new__(LLMLoader)
        loader.llms_config = {}
        loader.get_llm = Mock(side_effect=Exception("No LLMs"))

        with pytest.raises(ValueError, match="No LLMs available"):
            loader.get_default_llm()


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch("src.utils.llm_loader.llm_loader")
    def test_load_llm(self, mock_global_loader):
        """Test load_llm function."""
        mock_config = LLMConfig(name="Test LLM")
        mock_global_loader.get_llm.return_value = mock_config

        result = load_llm("test-llm")

        assert result == mock_config
        mock_global_loader.get_llm.assert_called_with("test-llm")

    @patch("src.utils.llm_loader.llm_loader")
    def test_reload_llms_config(self, mock_global_loader):
        """Test reload_llms_config function."""
        reload_llms_config()

        mock_global_loader.reload_config.assert_called_once()

    @patch("src.utils.llm_loader.llm_loader")
    def test_get_default_llm(self, mock_global_loader):
        """Test get_default_llm function."""
        mock_config = LLMConfig(name="Default LLM")
        mock_global_loader.get_default_llm.return_value = mock_config

        result = get_default_llm()

        assert result == mock_config
        mock_global_loader.get_default_llm.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
