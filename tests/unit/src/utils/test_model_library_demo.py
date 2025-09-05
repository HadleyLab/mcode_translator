#!/usr/bin/env python3
"""
Demo script for the model library functionality
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.utils.model_loader import load_model, model_loader
from src.utils.config import Config
from src.optimization.prompt_optimization_framework import APIConfig


def demo_model_library():
    """Demonstrate model library functionality"""
    print("=" * 60)
    print("Model Library Demo")
    print("=" * 60)
    
    # Load all models and display information
    all_models = model_loader.get_all_models()
    print(f"\n📚 Loaded {len(all_models)} models from the library:")
    
    for model_name, model_config in all_models.items():
        print(f"\n  Model: {model_config.name}")
        print(f"    Type: {model_config.model_type}")
        print(f"    Identifier: {model_config.model_identifier}")
        print(f"    Base URL: {model_config.base_url}")
        print(f"    Default Temperature: {model_config.default_parameters.get('temperature', 'N/A')}")
        print(f"    Default Max Tokens: {model_config.default_parameters.get('max_tokens', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("API Config Integration Demo")
    print("=" * 60)
    
    # Demonstrate APIConfig integration
    print("\n🔧 Creating API configs from model library:")
    
    # Create API configs for different models
    model_configs = [
        ("deepseek-coder", "DeepSeek Coder"),
        ("deepseek-chat", "DeepSeek Chat"), 
        ("deepseek-reasoner", "DeepSeek Reasoner")
    ]
    
    for model_key, display_name in model_configs:
        try:
            api_config = APIConfig(name=model_key)
            print(f"\n  {display_name}:")
            print(f"    Model: {api_config.model}")
            print(f"    Base URL: {api_config.base_url}")
            print(f"    Temperature: {api_config.temperature}")
            print(f"    Max Tokens: {api_config.max_tokens}")
        except Exception as e:
            print(f"\n  ❌ Failed to create API config for {display_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Configuration Integration Demo")
    print("=" * 60)
    
    # Demonstrate Config class integration
    print("\n⚙️  Getting model configurations through Config class:")
    
    config = Config()
    
    for model_key, display_name in model_configs:
        try:
            model_config = config.get_model_config(model_key)
            print(f"\n  {display_name}:")
            print(f"    Model Identifier: {model_config.model_identifier}")
            print(f"    Base URL: {model_config.base_url}")
            print(f"    Default Temperature: {model_config.default_parameters.get('temperature', 'N/A')}")
            print(f"    Default Max Tokens: {model_config.default_parameters.get('max_tokens', 'N/A')}")
        except Exception as e:
            print(f"\n  ❌ Failed to get model config for {display_name}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Model Library Demo Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo_model_library()