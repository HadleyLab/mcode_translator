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
    base_url: str = ""
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    status: str = "experimental"
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary"""
        return {
            'name': self.name,
            'model_type': self.model_type,
            'model_identifier': self.model_identifier,
            'base_url': self.base_url,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'created_date': self.created_date.isoformat() if isinstance(self.created_date, datetime) else str(self.created_date),
            'status': self.status,
            'default_parameters': self.default_parameters,
            'capabilities': self.capabilities,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create ModelConfig from dictionary"""
        # Handle datetime conversion
        created_date = data.get('created_date')
        if isinstance(created_date, str):
            try:
                created_date = datetime.fromisoformat(created_date)
            except ValueError:
                created_date = datetime.now()
        elif not isinstance(created_date, datetime):
            created_date = datetime.now()
        
        return cls(
            name=data.get('name', ''),
            model_type=data.get('model_type', ''),
            model_identifier=data.get('model_identifier', ''),
            base_url=data.get('base_url', ''),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            created_date=created_date,
            status=data.get('status', 'experimental'),
            default_parameters=data.get('default_parameters', {}),
            capabilities=data.get('capabilities', []),
            tags=data.get('tags', [])
        )


class ModelLoader:
    """Utility class for loading model configurations from the file-based model library"""
    
    def __init__(self, models_config_path: str = "models/models_config.json"):
        self.models_config_path = Path(models_config_path)
        self.models_config = self._load_models_config()
    
    def _load_models_config(self) -> Dict[str, Any]:
        """Load the models configuration JSON file and flatten the structure"""
        try:
            if not self.models_config_path.exists():
                raise FileNotFoundError(f"Models config file not found: {self.models_config_path}")
            
            with open(self.models_config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract the nested model library structure
            if "model_library" in config_data and "models" in config_data["model_library"]:
                models_config = config_data["model_library"]["models"]
                flattened_config = {}
                
                # Flatten the nested structure
                for category, subcategories in models_config.items():
                    for subcategory, model_list in subcategories.items():
                        for model_info in model_list:
                            model_name = model_info["name"]
                            flattened_config[model_name] = model_info
                
                return flattened_config
            else:
                logger.warning("Model library structure not found in config, using raw config")
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

            # STRICT IMPLEMENTATION: Validate API keys for specific model types
            model_type = model_config.get("model_type", "").lower()
            if "openai" in model_type or "gpt" in model_config.get("name", "").lower():
                if not os.getenv("OPENAI_API_KEY"):
                    raise ValueError(
                        "Configuration Error: OpenAI model selected, but OPENAI_API_KEY is not set. "
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