"""
LLM Loader Utility

This module provides functionality to load LLM configurations from the file-based LLM library
instead of using hardcoded configurations in the source code.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Import centralized logging configuration
from .logging_config import get_logger

# Use centralized logger
logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Dataclass representing an LLM configuration"""

    name: str = ""
    model_type: str = ""
    model_identifier: str = ""
    api_key_env_var: str = ""
    base_url: str = ""
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60
    rate_limit_per_minute: int = 100
    default: bool = False

    @property
    def api_key(self) -> str:
        """Get the API key from the environment variable"""
        if not self.api_key_env_var:
            return ""
        return os.getenv(self.api_key_env_var, "")

    def to_dict(self) -> Dict[str, Any]:
        """Convert LLMConfig to dictionary"""
        return {
            "name": self.name,
            "model_type": self.model_type,
            "model_identifier": self.model_identifier,
            "api_key_env_var": self.api_key_env_var,
            "base_url": self.base_url,
            "default_parameters": self.default_parameters,
            "timeout_seconds": self.timeout_seconds,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "default": self.default,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from dictionary"""
        return cls(
            name=data.get("name", data.get("model_name", "")),
            model_type=data.get("model_type", data.get("provider", "")),
            model_identifier=data.get("model_identifier", data.get("model_name", "")),
            api_key_env_var=data.get("api_key_env_var", ""),
            base_url=data.get("base_url", data.get("api_base", "")),
            default_parameters=data.get("default_parameters", {}),
            timeout_seconds=data.get("timeout_seconds", 60),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 100),
            default=data.get("default", False),
        )


class LLMLoader:
    """Utility class for loading LLM configurations from the file-based LLM library"""

    def __init__(self, llms_config_path: Optional[str] = None) -> None:
        if llms_config_path is None:
            # Use the correct path relative to this file
            config_path = Path(__file__).parent.parent / "config" / "llms_config.json"
        else:
            config_path = Path(llms_config_path)
        self.llms_config_path: Path = config_path
        self.llms_config: Dict[str, Any] = self._load_llms_config()

    def _load_llms_config(self) -> Dict[str, Any]:
        """Load the LLMs configuration JSON file"""
        try:
            if not self.llms_config_path.exists():
                raise FileNotFoundError(f"LLMs config file not found: {self.llms_config_path}")

            with open(self.llms_config_path) as f:
                config_data: Dict[str, Any] = json.load(f)

            # Handle nested structure: models.available contains the actual models
            if "models" in config_data and "available" in config_data["models"]:
                nested_data: Dict[str, Any] = config_data["models"]["available"]
                return nested_data

            # Fallback to flattened structure if nested structure not found
            return config_data

        except Exception as e:
            logger.error(f"Failed to load LLMs config: {str(e)}")
            return {}

    def get_llm(self, llm_key: str) -> LLMConfig:
        """
        Get an LLM configuration by key

        Args:
            llm_key: The key identifying the LLM in the config

        Returns:
            LLMConfig object with the LLM configuration

        Raises:
            ValueError: If LLM validation fails or LLM not found
        """
        try:
            # Get LLM config from loaded configuration
            llm_config = self.llms_config.get(llm_key)
            if not llm_config:
                raise ValueError(f"LLM key '{llm_key}' not found in config")

            # STRICT IMPLEMENTATION: Validate API keys for the specified provider
            api_key_env_var = llm_config.get("api_key_env_var")
            if not api_key_env_var:
                raise ValueError(
                    f"Configuration Error: 'api_key_env_var' not set for LLM '{llm_key}'."
                )

            if not os.getenv(api_key_env_var):
                raise ValueError(
                    f"Configuration Error: Environment variable '{api_key_env_var}' is not set for LLM '{llm_key}'. "
                    "Please set the environment variable to use this LLM."
                )

            # Convert to LLMConfig object
            return LLMConfig.from_dict(llm_config)

        except ValueError:
            # Re-raise validation errors to be handled by caller
            raise
        except Exception as e:
            logger.error(f"Error loading LLM '{llm_key}': {str(e)}")
            raise

    def get_all_llms(self) -> Dict[str, LLMConfig]:
        """Get all LLMs from the library - STRICT implementation"""
        all_llms: Dict[str, LLMConfig] = {}
        for llm_key in self.llms_config.keys():
            # STRICT: Load all LLMs - throw exception if any LLM fails validation
            all_llms[llm_key] = self.get_llm(llm_key)

        return all_llms

    def reload_config(self) -> None:
        """Reload the LLMs configuration from disk"""
        self.llms_config = self._load_llms_config()
        logger.info("LLM configuration reloaded from disk")

    def get_llm_metadata(self, llm_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific LLM"""
        return self.llms_config.get(llm_key)

    def list_available_llms(self) -> Dict[str, Dict[str, Any]]:
        """List all available LLMs with their metadata"""
        # Validate that all LLM keys are unique
        llm_keys = list(self.llms_config.keys())
        if len(llm_keys) != len(set(llm_keys)):
            duplicates = [key for key in llm_keys if llm_keys.count(key) > 1]
            raise ValueError(f"Duplicate LLM keys found: {set(duplicates)}")

        return self.llms_config.copy()

    def get_default_llm(self) -> LLMConfig:
        """Get the default LLM configuration (deepseek-coder)

        Returns:
            LLMConfig: Default LLM configuration for deepseek-coder
        """
        try:
            return self.get_llm("deepseek-coder")
        except Exception as e:
            logger.warning(f"Could not load default LLM 'deepseek-coder': {e}")
            # Fallback to first available LLM if deepseek-coder fails
            available_llms = list(self.llms_config.keys())
            if available_llms:
                logger.info(f"Falling back to first available LLM: {available_llms[0]}")
                return self.get_llm(available_llms[0])
            else:
                raise ValueError("No LLMs available in configuration")


# Global instance for easy access
llm_loader = LLMLoader()


def load_llm(llm_key: str) -> LLMConfig:
    """
    Convenience function to load an LLM using the global loader

    Args:
        llm_key: The key identifying the LLM in the config

    Returns:
        LLMConfig object with the LLM configuration
    """
    return llm_loader.get_llm(llm_key)


def reload_llms_config() -> None:
    """Reload the LLMs configuration using the global loader"""
    llm_loader.reload_config()


def get_default_llm() -> LLMConfig:
    """Get the default LLM configuration using the global loader"""
    return llm_loader.get_default_llm()


# Example usage
if __name__ == "__main__":
    # Test the LLM loader
    try:
        # Load a specific LLM
        coder_llm = load_llm("deepseek-coder")
        print(f"Loaded LLM: {coder_llm.name}")
        print(f"LLM identifier: {coder_llm.model_identifier}")
        print(f"Base URL: {coder_llm.base_url}")

        # List all available LLMs
        available_llms = llm_loader.list_available_llms()
        print(f"Available LLMs: {list(available_llms.keys())}")

    except Exception as e:
        print(f"Error testing LLM loader: {str(e)}")
