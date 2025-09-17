import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import LLM loader for file-based LLM library
from .llm_loader import LLMConfig, llm_loader


class ConfigurationError(Exception):
    """Exception raised for configuration issues in strict infrastructure"""

    pass


class Config:
    """
    MODULAR Configuration class for the mCODE Translator
    Loads configuration from separate modular config files for better organization
    """

    def __init__(self):
        # Load all modular configurations
        self.cache_config = self._load_cache_config()
        self.apis_config = self._load_apis_config()
        self.core_memory_config = self._load_core_memory_config()
        self.models_config = self._load_models_config()
        self.prompts_config = self._load_prompts_config()
        self.synthetic_data_config = self._load_synthetic_data_config()
        self.validation_config = self._load_validation_config()
        self.logging_config = self._load_logging_config()
        self.patterns_config = self._load_patterns_config()

        # No legacy support - forward compatibility only

    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.cache_config["cache"]["enabled"]

    def get_api_cache_directory(self) -> str:
        """Get API cache directory path for clinical trials and other API data"""
        return self.cache_config["cache"]["api_cache_directory"]

    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self.cache_config["cache"]["ttl_seconds"]

    def get_rate_limit_delay(self) -> float:
        """Get rate limiting delay in seconds"""
        return self.cache_config["rate_limiting"]["delay_seconds"]

    def get_request_timeout(self) -> int:
        """Get request timeout in seconds"""
        return self.cache_config["request"]["timeout_seconds"]

    def get_clinical_trials_base_url(self) -> str:
        """Get clinical trials API base URL"""
        return self.apis_config["apis"]["clinical_trials"]["base_url"]

    def get_api_key(self, model_name: str) -> str:
        """
        Get API key for specified LLM model from environment variables.

        Args:
            model_name: REQUIRED name of the LLM model.

        Returns:
            API key string.

        Raises:
            ConfigurationError: If API key is missing or invalid.
        """
        if not model_name:
            raise ConfigurationError(
                "Model name is required - no fallback to default model allowed in strict mode"
            )

        model_config = self.get_llm_config(model_name)
        api_key_env_var = model_config.api_key_env_var

        if not api_key_env_var:
            raise ConfigurationError(
                f"API key environment variable not configured for model '{model_name}'."
            )

        api_key = os.getenv(api_key_env_var)

        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 20:
            raise ConfigurationError(
                f"Invalid or missing API key in environment variable '{api_key_env_var}' for model '{model_name}'."
            )

        return api_key

    def get_base_url(self, model_name: str) -> str:
        """
        Get base URL for specified LLM model

        Args:
            model_name: REQUIRED name of the LLM model

        Returns:
            Base URL string

        Raises:
            ConfigurationError: If base URL is missing or invalid
        """
        if not model_name:
            raise ConfigurationError(
                "Model name is required - no fallback to default model allowed in strict mode"
            )

        model_config = self.get_llm_config(model_name)
        base_url = model_config.base_url

        if not base_url or not isinstance(base_url, str):
            raise ConfigurationError(
                f"Invalid or missing base URL for model '{model_name}'"
            )

        return base_url

    def get_model_name(self, model_name: str) -> str:
        """
        Get model name for specified LLM model

        Args:
            model_name: REQUIRED name of the LLM model

        Returns:
            Model name string

        Raises:
            ConfigurationError: If model name is missing or invalid
        """
        if not model_name:
            raise ConfigurationError(
                "Model name is required - no fallback to default model allowed in strict mode"
            )

        model_config = self.get_llm_config(model_name)
        validated_model_name = model_config.name

        if not validated_model_name or not isinstance(validated_model_name, str):
            raise ConfigurationError(
                f"Invalid or missing model name for model '{model_name}'"
            )

        return validated_model_name

    def get_temperature(self, model_name: str) -> float:
        """
        Get temperature for specified LLM model

        Args:
            model_name: REQUIRED name of the LLM model

        Returns:
            Temperature float

        Raises:
            ConfigurationError: If temperature is missing or invalid
        """
        if not model_name:
            raise ConfigurationError(
                "Model name is required - no fallback to default model allowed in strict mode"
            )

        model_config = self.get_llm_config(model_name)
        temperature = model_config.default_parameters.get("temperature")

        if temperature is None or not isinstance(temperature, (int, float)):
            raise ConfigurationError(
                f"Invalid or missing temperature for model '{model_name}'"
            )

        return float(temperature)

    def get_max_tokens(self, model_name: str) -> int:
        """
        Get max tokens for specified LLM model

        Args:
            model_name: REQUIRED name of the LLM model

        Returns:
            Max tokens integer

        Raises:
            ConfigurationError: If max tokens is missing or invalid
        """
        if not model_name:
            raise ConfigurationError(
                "Model name is required - no fallback to default model allowed in strict mode"
            )

        model_config = self.get_llm_config(model_name)
        max_tokens = model_config.default_parameters.get("max_tokens")

        if max_tokens is None or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError(
                f"Invalid or missing max tokens for model '{model_name}'"
            )

        return max_tokens

    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled"""
        return self.validation_config["validation"]["strict_mode"]

    def require_api_keys(self) -> bool:
        """Check if API keys are required"""
        return self.validation_config["validation"]["require_api_keys"]

    def get_llm_config(self, llm_key: str) -> LLMConfig:
        """
        Get LLM configuration from the file-based LLM library

        Args:
            llm_key: REQUIRED key identifying the LLM in the library

        Returns:
            LLMConfig object with the LLM configuration

        Raises:
            ConfigurationError: If LLM configuration is missing or invalid
        """
        if not llm_key:
            raise ConfigurationError(
                "LLM key is required - no fallback to default LLM allowed in strict mode"
            )

        # STRICT: Load LLM configuration from file-based LLM library - throw exception if not found
        return llm_loader.get_llm(llm_key)

    def get_all_llm_configs(self) -> Dict[str, LLMConfig]:
        """
        Get all model configurations from the file-based model library

        Returns:
            Dictionary mapping model keys to ModelConfig objects

        Raises:
            ConfigurationError: If model configurations cannot be loaded
        """
        # STRICT: Load all LLM configurations - throw exception if any fail
        return llm_loader.get_all_llms()

    def reload_llm_configs(self) -> None:
        """Reload LLM configurations from the file-based LLM library"""
        # STRICT: Reload LLM configurations - throw exception if reload fails
        llm_loader.reload_config()

    def get_core_memory_api_key(self) -> str:
        """
        Get CORE Memory API key from environment variables.

        Returns:
            CORE Memory API key string.

        Raises:
            ConfigurationError: If API key is missing or invalid.
        """
        api_key = os.getenv("COREAI_API_KEY")
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 20:
            raise ConfigurationError(
                "Invalid or missing API key in environment variable 'COREAI_API_KEY'."
            )
        return api_key

    def get_core_memory_config(self) -> Dict[str, Any]:
        """
        Get centralized CORE Memory configuration from core_memory_config.json

        Returns:
            Dictionary with CORE Memory configuration

        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        config_path = (
            Path(__file__).parent.parent / "config" / "core_memory_config.json"
        )

        if not config_path.exists():
            raise ConfigurationError(
                f"CORE Memory configuration file not found: {config_path}"
            )

        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            return config_data
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in CORE Memory configuration file: {str(e)}"
            )
        except IOError as e:
            raise ConfigurationError(
                f"Failed to read CORE Memory configuration file: {str(e)}"
            )

    def get_core_memory_api_base_url(self) -> str:
        """Get CORE Memory API base URL from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["api_base_url"]

    def get_core_memory_source(self) -> str:
        """Get CORE Memory source identifier from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["source"]

    def get_core_memory_timeout(self) -> int:
        """Get CORE Memory request timeout from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["timeout_seconds"]

    def get_core_memory_max_retries(self) -> int:
        """Get CORE Memory max retries from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["max_retries"]

    def get_core_memory_default_spaces(self) -> Dict[str, str]:
        """Get CORE Memory default spaces from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["default_spaces"]

    def get_core_memory_batch_size(self) -> int:
        """Get CORE Memory batch size from centralized config"""
        config = self.get_core_memory_config()
        return config["core_memory"]["storage_settings"]["batch_size"]

    def get_mcode_summary_format(self) -> str:
        """Get mCODE summary format from centralized config"""
        config = self.get_core_memory_config()
        return config["mcode_settings"]["summary_format"]

    def get_mcode_include_codes(self) -> bool:
        """Get whether to include codes in mCODE summaries"""
        config = self.get_core_memory_config()
        return config["mcode_settings"]["include_codes"]

    def get_mcode_max_summary_length(self) -> int:
        """Get maximum mCODE summary length"""
        config = self.get_core_memory_config()
        return config["mcode_settings"]["max_summary_length"]

    # Synthetic Data Configuration Methods
    def get_synthetic_data_base_directory(self) -> str:
        """Get synthetic data base directory"""
        return self.synthetic_data_config["synthetic_data"]["base_directory"]

    def get_synthetic_data_default_archive(self) -> str:
        """Get default synthetic data archive"""
        return self.synthetic_data_config["synthetic_data"]["default_archive"]

    def get_synthetic_data_archives(self) -> Dict[str, Any]:
        """Get all synthetic data archives configuration"""
        return self.synthetic_data_config["synthetic_data"]["archives"]

    def is_synthetic_data_auto_download_enabled(self) -> bool:
        """Check if auto download is enabled for synthetic data"""
        return self.synthetic_data_config["synthetic_data"]["auto_download"]

    # Logging Configuration Methods
    def get_logging_config(self) -> Dict[str, Any]:
        """Get complete logging configuration"""
        return self.logging_config

    def get_default_log_level(self) -> str:
        """Get default logging level"""
        return self.logging_config["logging"]["default_level"]

    def get_log_format(self) -> str:
        """Get logging format string"""
        return self.logging_config["logging"]["format"]

    def is_colored_logging_enabled(self) -> bool:
        """Check if colored logging is enabled"""
        return self.logging_config["logging"]["colored_output"]

    # Patterns Configuration Methods
    def get_patterns_config(self) -> Dict[str, Any]:
        """Get complete patterns configuration"""
        return self.patterns_config

    def get_biomarker_patterns(self) -> Dict[str, Any]:
        """Get biomarker regex patterns"""
        return self.patterns_config["patterns"]["biomarker_patterns"]

    def get_genomic_patterns(self) -> Dict[str, Any]:
        """Get genomic variant patterns"""
        return self.patterns_config["patterns"]["genomic_patterns"]

    def get_condition_patterns(self) -> Dict[str, Any]:
        """Get condition patterns"""
        return self.patterns_config["patterns"]["condition_patterns"]

    def get_demographic_patterns(self) -> Dict[str, Any]:
        """Get demographic patterns"""
        return self.patterns_config["patterns"]["demographic_patterns"]

    def _load_cache_config(self) -> Dict[str, Any]:
        """Load cache configuration from modular config file"""
        return self._load_modular_config("cache_config.json")

    def _load_apis_config(self) -> Dict[str, Any]:
        """Load APIs configuration from modular config file"""
        return self._load_modular_config("apis_config.json")

    def _load_synthetic_data_config(self) -> Dict[str, Any]:
        """Load synthetic data configuration from modular config file"""
        return self._load_modular_config("synthetic_data_config.json")

    def _load_validation_config(self) -> Dict[str, Any]:
        """Load validation configuration from modular config file"""
        return self._load_modular_config("validation_config.json")

    def _load_logging_config(self) -> Dict[str, Any]:
        """Load logging configuration from modular config file"""
        return self._load_modular_config("logging_config.json")

    def _load_patterns_config(self) -> Dict[str, Any]:
        """Load patterns configuration from modular config file"""
        return self._load_modular_config("patterns_config.json")

    def _load_core_memory_config(self) -> Dict[str, Any]:
        """Load Core Memory configuration from modular config file"""
        return self._load_modular_config("core_memory_config.json")

    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from modular config file"""
        return self._load_modular_config("llms_config.json")

    def _load_prompts_config(self) -> Dict[str, Any]:
        """Load prompts configuration from modular config file"""
        return self._load_modular_config("prompts_config.json")

    def _load_modular_config(self, filename: str) -> Dict[str, Any]:
        """Load a modular configuration file"""
        config_path = Path(__file__).parent.parent / "config" / filename
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Modular configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in modular config file {filename}: {str(e)}"
            )
        except IOError as e:
            raise ConfigurationError(
                f"Failed to read modular config file {filename}: {str(e)}"
            )
