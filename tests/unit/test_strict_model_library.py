"""
Test suite for strict model library implementation
"""

import pytest
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.utils.model_loader import ModelLoader, load_model, model_loader
from src.utils.config import Config
from src.optimization.strict_prompt_optimization_framework import APIConfig


def test_strict_model_loading():
    """Test that strict model loading throws exceptions for missing models"""
    # Test loading an existing model - should succeed
    model_config = load_model("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"
    
    # Test loading a non-existent model - should throw exception
    with pytest.raises(ValueError, match=r"Model key 'non-existent-model' not found in config"):
        load_model("non-existent-model")


def test_strict_api_config_creation():
    """Test that APIConfig creation throws exceptions for missing models"""
    # Test creating APIConfig with existing model - should succeed
    api_config = APIConfig(name="deepseek-coder")
    assert api_config.model == "deepseek-coder"
    assert api_config.base_url == "https://api.deepseek.com/v1"
    
    # Test creating APIConfig with non-existent model - should throw exception
    with pytest.raises(ValueError, match=r"Model key 'non-existent-model' not found in config"):
        APIConfig(name="non-existent-model")


def test_strict_config_model_access():
    """Test that Config class throws exceptions for missing models"""
    config = Config()
    
    # Test getting existing model configuration - should succeed
    model_config = config.get_model_config("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"
    
    # Test getting non-existent model configuration - should throw exception
    with pytest.raises(ValueError, match=r"Model key 'non-existent-model' not found in config"):
        config.get_model_config("non-existent-model")


def test_strict_all_models_loading():
    """Test that loading all models works in strict mode"""
    loader = ModelLoader()
    
    # Test getting all models - should succeed with all 5 models
    all_models = loader.get_all_models()
    assert isinstance(all_models, dict)
    assert len(all_models) == 5  # Should have exactly 5 models
    
    # Check that all expected models are present
    model_names = list(all_models.keys())
    expected_models = ["deepseek-coder", "deepseek-chat", "deepseek-reasoner", "gpt-4", "gpt-3.5-turbo"]
    for expected_model in expected_models:
        assert expected_model in model_names


def test_strict_model_library_integration():
    """Test full integration of strict model library"""
    # Test that all components work together in strict mode
    config = Config()
    
    # Load all models through different interfaces
    model1 = config.get_model_config("deepseek-coder")
    model2 = load_model("deepseek-chat")
    model3 = model_loader.get_model("deepseek-reasoner")
    
    # Verify all models loaded successfully
    assert model1.name == "deepseek-coder"
    assert model2.name == "deepseek-chat"
    assert model3.name == "deepseek-reasoner"
    
    # Test APIConfig creation with all models
    api_config1 = APIConfig(name="deepseek-coder")
    api_config2 = APIConfig(name="deepseek-chat")
    api_config3 = APIConfig(name="deepseek-reasoner")
    
    # Verify APIConfigs created successfully
    assert api_config1.model == "deepseek-coder"
    assert api_config2.model == "deepseek-chat"
    assert api_config3.model == "deepseek-reasoner"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])