import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import model loader for file-based model library
from .model_loader import model_loader, ModelConfig


class ConfigurationError(Exception):
    """Exception raised for configuration issues in strict infrastructure"""
    pass


class Config:
    """
    STRICT Configuration class for the Mcode Translator
    Single source of truth with no fallbacks - throws exceptions for missing configuration
    """
    
    def __init__(self):
        # Load configuration from unified config.json
        self.config_data = self._load_config_file()
        
        # Validate configuration structure
        self._validate_config_structure()
        
    
    def _load_config_file(self) -> dict:
        """
        Load configuration from config.json file with strict validation
        
        Returns:
            Dictionary with configuration values
            
        Raises:
            ConfigurationError: If config file is missing or invalid
        """
        config_path = Path(__file__).parent.parent.parent / 'config.json'
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate version
            if config_data.get('version') != '1.0.0':
                raise ConfigurationError("Unsupported configuration version")
                
            return config_data
            
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {str(e)}")
        except IOError as e:
            raise ConfigurationError(f"Failed to read configuration file: {str(e)}")
    
    def _validate_config_structure(self) -> None:
        """Validate that required configuration sections exist"""
        required_sections = ['cache', 'rate_limiting', 'request', 'apis', 'validation']
        for section in required_sections:
            if section not in self.config_data:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Note: llm_providers validation removed - models are now managed through
        # centralized model_loader.py that reads from models/models_config.json
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.config_data['cache']['enabled']
    
    def get_api_cache_directory(self) -> str:
        """Get API cache directory path for clinical trials and other API data"""
        if 'api_cache_directory' in self.config_data['cache']:
            return self.config_data['cache']['api_cache_directory']
        elif 'directory' in self.config_data['cache']:
            # Fallback to legacy directory field for backward compatibility
            return self.config_data['cache']['directory']
        else:
            raise ConfigurationError("Missing API cache directory configuration")
    
    def get_cache_directory(self) -> str:
        """DEPRECATED: Get cache directory path - returns API cache directory as default
        Use get_api_cache_directory() instead"""
        return self.get_api_cache_directory()
    
    def get_cache_ttl(self) -> int:
        """Get cache TTL in seconds"""
        return self.config_data['cache']['ttl_seconds']
    
    def get_rate_limit_delay(self) -> float:
        """Get rate limiting delay in seconds"""
        return self.config_data['rate_limiting']['delay_seconds']
    
    def get_request_timeout(self) -> int:
        """Get request timeout in seconds"""
        return self.config_data['request']['timeout_seconds']
    
    def get_clinical_trials_base_url(self) -> str:
        """Get clinical trials API base URL"""
        return self.config_data['apis']['clinical_trials']['base_url']
    
    def get_api_key(self, model_name: Optional[str] = None) -> str:
        """
        Get API key for specified LLM model from environment variables.
        
        Args:
            model_name: Optional name of the LLM model.
        
        Returns:
            API key string.
            
        Raises:
            ConfigurationError: If API key is missing or invalid.
        """
        model_config = self.get_model_config(model_name)
        api_key_env_var = model_config.api_key_env_var

        if not api_key_env_var:
            raise ConfigurationError(f"API key environment variable not configured for model '{model_name}'.")

        api_key = os.getenv(api_key_env_var)
        
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 20:
            raise ConfigurationError(f"Invalid or missing API key in environment variable '{api_key_env_var}' for model '{model_name}'.")
        
        return api_key
    
    def get_base_url(self, model_name: Optional[str] = None) -> str:
        """
        Get base URL for specified LLM model
        
        Args:
            model_name: Optional name of the LLM model
            
        Returns:
            Base URL string
            
        Raises:
            ConfigurationError: If base URL is missing or invalid
        """
        model_config = self.get_model_config(model_name)
        base_url = model_config.base_url
        
        if not base_url or not isinstance(base_url, str):
            raise ConfigurationError(f"Invalid or missing base URL for model '{model_name}'")
        
        return base_url
    
    def get_model_name(self, model_name: Optional[str] = None) -> str:
        """
        Get model name for specified LLM model
        
        Args:
            model_name: Optional name of the LLM model
            
        Returns:
            Model name string
            
        Raises:
            ConfigurationError: If model name is missing or invalid
        """
        model_config = self.get_model_config(model_name)
        model_name = model_config.name
        
        if not model_name or not isinstance(model_name, str):
            raise ConfigurationError(f"Invalid or missing model name for model '{model_name}'")
        
        return model_name
    
    def get_temperature(self, model_name: Optional[str] = None) -> float:
        """
        Get temperature for specified LLM model
        
        Args:
            model_name: Optional name of the LLM model
            
        Returns:
            Temperature float
            
        Raises:
            ConfigurationError: If temperature is missing or invalid
        """
        model_config = self.get_model_config(model_name)
        temperature = model_config.default_parameters.get('temperature')
        
        if temperature is None or not isinstance(temperature, (int, float)):
            raise ConfigurationError(f"Invalid or missing temperature for model '{model_name}'")
        
        return float(temperature)
    
    def get_max_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Get max tokens for specified LLM model
        
        Args:
            model_name: Optional name of the LLM model
            
        Returns:
            Max tokens integer
            
        Raises:
            ConfigurationError: If max tokens is missing or invalid
        """
        model_config = self.get_model_config(model_name)
        max_tokens = model_config.default_parameters.get('max_tokens')
        
        if max_tokens is None or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError(f"Invalid or missing max tokens for model '{model_name}'")
        
        return max_tokens
    
    def is_strict_mode(self) -> bool:
        """Check if strict validation mode is enabled"""
        return self.config_data['validation']['strict_mode']
    
    def require_api_keys(self) -> bool:
        """Check if API keys are required"""
        return self.config_data['validation']['require_api_keys']
    
    def get_model_config(self, model_key: Optional[str]) -> ModelConfig:
        """
        Get model configuration from the file-based model library
        
        Args:
            model_key: The key identifying the model in the model library
            
        Returns:
            ModelConfig object with the model configuration
            
        Raises:
            ConfigurationError: If model configuration is missing or invalid
        """
        # If no model_key is provided, use the default model from the model library
        if model_key is None:
            model_key = model_loader.get_default_model()
            if not model_key:
                raise ConfigurationError("No default model set and no model key provided.")

        # STRICT: Load model configuration from file-based model library - throw exception if not found
        return model_loader.get_model(model_key)
    
    def get_all_model_configs(self) -> Dict[str, ModelConfig]:
        """
        Get all model configurations from the file-based model library
        
        Returns:
            Dictionary mapping model keys to ModelConfig objects
            
        Raises:
            ConfigurationError: If model configurations cannot be loaded
        """
        # STRICT: Load all model configurations - throw exception if any fail
        return model_loader.get_all_models()
    
    def reload_model_configs(self) -> None:
        """Reload model configurations from the file-based model library"""
        # STRICT: Reload model configurations - throw exception if reload fails
        model_loader.reload_config()