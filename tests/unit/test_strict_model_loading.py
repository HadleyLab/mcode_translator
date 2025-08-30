"""
Test suite for strict model loading behavior
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


def test_strict_model_loading_success():
    """Test that loading existing models succeeds in strict mode"""
    # Test loading an existing model - should succeed
    model_config = load_model("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"
    assert model_config.model_identifier == "deepseek-coder"
    assert model_config.base_url == "https://api.deepseek.com/v1"


def test_strict_model_loading_failure():
    """Test that loading non-existent models throws exceptions in strict mode"""
    # Test loading a non-existent model - should throw exception
    with pytest.raises(ValueError, match=r"Model key 'non-existent-model' not found in config"):
        load_model("non-existent-model")


def test_strict_api_config_creation_success():
    """Test that APIConfig creation with existing models succeeds in strict mode"""
    # Test creating APIConfig with existing model - should succeed
    api_config = APIConfig(name="deepseek-coder")
    assert api_config.model == "deepseek-coder"
    assert api_config.base_url == "https://api.deepseek.com/v1"


def test_strict_api_config_creation_failure():
    """Test that APIConfig creation with non-existent models throws exceptions in strict mode"""
    # Test creating APIConfig with non-existent model - should throw exception
    with pytest.raises(ValueError, match=r"Model key 'non-existent-model' not found in config"):
        APIConfig(name="non-existent-model")


def test_strict_config_model_access_success():
    """Test that Config class access to existing models succeeds in strict mode"""
    config = Config()
    
    # Test getting existing model configuration - should succeed
    model_config = config.get_model_config("deepseek-coder")
    assert model_config is not None
    assert model_config.name == "deepseek-coder"


def test_strict_config_model_access_failure():
    """Test that Config class access to non-existent models throws exceptions in strict mode"""
    config = Config()
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])