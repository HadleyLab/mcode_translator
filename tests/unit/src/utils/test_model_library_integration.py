"""
Test suite for model library integration
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils import ModelLoader, load_model, model_loader
from src.utils.config import Config


def test_model_loader_initialization():
    """Test that ModelLoader initializes correctly"""
    loader = ModelLoader()
    assert loader is not None
    assert isinstance(loader, ModelLoader)


def test_load_specific_model():
    """Test loading a specific model from the library"""
    # Test loading the deepseek-coder model
    model_config = load_model("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"
    assert model_config.model_identifier == "deepseek-coder"
    assert model_config.base_url == "https://api.deepseek.com/v1"
    assert model_config.model_type == "CODE_GENERATION"
    
    # Check default parameters
    assert "temperature" in model_config.default_parameters
    assert "max_tokens" in model_config.default_parameters
    assert model_config.default_parameters["temperature"] == 0.1
    assert model_config.default_parameters["max_tokens"] == 4000


def test_load_all_models():
    """Test loading all models from the library - STRICT implementation"""
    loader = ModelLoader()
    all_models = loader.get_all_models()
    
    # Should have exactly 5 models from the library
    assert len(all_models) == 5
    
    # Check that all expected models are present
    model_names = list(all_models.keys())
    assert "deepseek-coder" in model_names
    assert "deepseek-chat" in model_names
    assert "deepseek-reasoner" in model_names
    assert "gpt-4" in model_names
    assert "gpt-3.5-turbo" in model_names


def test_model_config_to_dict():
    """Test converting ModelConfig to dictionary"""
    model_config = load_model("deepseek-coder")
    model_dict = model_config.to_dict()
    
    assert isinstance(model_dict, dict)
    assert model_dict["name"] == "deepseek-coder"
    assert model_dict["model_identifier"] == "deepseek-coder"
    assert model_dict["base_url"] == "https://api.deepseek.com/v1"


def test_config_model_integration():
    """Test integration between Config class and model library - STRICT implementation"""
    config = Config()
    
    # Test getting model configuration through Config class
    model_config = config.get_model_config("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"
    assert model_config.model_identifier == "deepseek-coder"
    
    # Test getting all model configurations
    all_models = config.get_all_model_configs()
    assert isinstance(all_models, dict)
    assert len(all_models) >= 5  # Should have all 5 models from the library


def test_api_config_model_integration():
    """Test integration between APIConfig and model library"""
    from src.optimization.prompt_optimization_framework import APIConfig
    
    # Test creating APIConfig with model from library
    api_config = APIConfig(name="deepseek-coder")
    
    assert api_config.model == "deepseek-coder"
    assert api_config.base_url == "https://api.deepseek.com/v1"
    assert api_config.temperature == 0.1
    assert api_config.max_tokens == 4000


def test_reload_models_config():
    """Test reloading model configurations"""
    loader = ModelLoader()
    
    # Get initial model count
    initial_models = loader.get_all_models()
    initial_count = len(initial_models)
    
    # Reload configurations
    loader.reload_config()
    
    # Get models after reload
    reloaded_models = loader.get_all_models()
    reloaded_count = len(reloaded_models)
    
    # Counts should be the same
    assert initial_count == reloaded_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])