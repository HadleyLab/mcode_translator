"""
Model Loader Utility

This module provides functionality to load model configurations from the file-based model library
instead of using hardcoded configurations in the source code.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Import centralized logging configuration
from src.utils.logging_config import get_logger

# Use centralized logger
logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Dataclass representing a model configuration"""
    name: str = ""
    model_type: str = ""
    model_identifier: str = ""
    api_key_env_var: str = ""
    base_url: str = ""
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    default: bool = False
    
    @property
    def api_key(self) -> str:
        """Get the API key from the environment variable"""
        if not self.api_key_env_var:
            return ""
        return os.getenv(self.api_key_env_var, "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'model_identifier': self.model_identifier,
            'api_key_env_var': self.api_key_env_var,
            'base_url': self.base_url,
            'default_parameters': self.default_parameters,
            'default': self.default
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        return cls(
            name=data.get('name', ''),
            model_type=data.get('model_type', ''),
            model_identifier=data.get('model_identifier', ''),
            api_key_env_var=data.get('api_key_env_var', ''),
            base_url=data.get('base_url', ''),
            default_parameters=data.get('default_parameters', {}),
            default=data.get('default', False)
        )


class ModelLoader:
    """Utility class for loading model configurations from the file-based model library"""
    
    def __init__(self, models_config_path: str = "models/models_config.json"):
        self.models_config_path = Path(models_config_path)
        self.models_config = self._load_models_config()
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load the models configuration JSON file"""
        try:
            if not self.models_config_path.exists():
                raise FileNotFoundError(f"Models config file not found: {self.models_config_path}")
            
            with open(self.models_config_path, 'r') as f:
                config_data = json.load(f)
            
            # With the flattened structure, we can use the config data directly
            return config_data
                
        except Exception as e:
            logger.error(f"Failed to load models config: {str(e)}")
            return {}
    
    def get_model(self, model_key: str) -> ModelConfig:
        """
        Get a model configuration by key
        
        Args:
            model_key: The key identifying the model in the config
            
        Returns:
            ModelConfig object with the model configuration
            
        Raises:
            ValueError: If model validation fails or model not found
        """
        try:
            # Get model config from loaded configuration
            model_config = self.models_config.get(model_key)
            if not model_config:
                raise ValueError(f"Model key '{model_key}' not found in config")

            # STRICT IMPLEMENTATION: Validate API keys for the specified provider
            api_key_env_var = model_config.get("api_key_env_var")
            if not api_key_env_var:
                raise ValueError(f"Configuration Error: 'api_key_env_var' not set for model '{model_key}'.")

            if not os.getenv(api_key_env_var):
                raise ValueError(
                    f"Configuration Error: Environment variable '{api_key_env_var}' is not set for model '{model_key}'. "
                    "Please set the environment variable to use this model."
                )

            # Convert to ModelConfig object
            return ModelConfig.from_dict(model_config)
                
        except ValueError as e:
            # Re-raise validation errors to be handled by caller
            raise
        except Exception as e:
            logger.error(f"Error loading model '{model_key}': {str(e)}")
            raise
    
    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all models from the library - STRICT implementation"""
        all_models = {}
        for model_key in self.models_config.keys():
            # STRICT: Load all models - throw exception if any model fails validation
            all_models[model_key] = self.get_model(model_key)
        
        return all_models
    
    def reload_config(self) -> None:
        """Reload the models configuration from disk"""
        self.models_config = self._load_models_config()
        logger.info("Model configuration reloaded from disk")
    
    def get_model_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model"""
        return self.models_config.get(model_key)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their metadata"""
        # Validate that all model keys are unique
        model_keys = list(self.models_config.keys())
        if len(model_keys) != len(set(model_keys)):
            duplicates = [key for key in model_keys if model_keys.count(key) > 1]
            raise ValueError(f"Duplicate model keys found: {set(duplicates)}")
        
        return self.models_config.copy()

    def get_default_model(self) -> Optional[str]:
        """Get the default model from the configuration"""
        for model_key, model_config in self.models_config.items():
            if model_config.get("default"):
                return model_key
        return None


# Global instance for easy access
model_loader = ModelLoader()


def load_model(model_key: str) -> ModelConfig:
    """
    Convenience function to load a model using the global loader
    
    Args:
        model_key: The key identifying the model in the config
        
    Returns:
        ModelConfig object with the model configuration
    """
    return model_loader.get_model(model_key)


def reload_models_config() -> None:
    """Reload the models configuration using the global loader"""
    model_loader.reload_config()


# Example usage
if __name__ == "__main__":
    # Test the model loader
    try:
        # Load a specific model
        coder_model = load_model("deepseek-coder")
        print(f"Loaded model: {coder_model.name}")
        print(f"Model identifier: {coder_model.model_identifier}")
        print(f"Base URL: {coder_model.base_url}")
        
        # List all available models
        available_models = model_loader.list_available_models()
        print(f"Available models: {list(available_models.keys())}")
        
    except Exception as e:
        print(f"Error testing model loader: {str(e)}")