# Model Library

This directory contains the file-based model library that centralizes all LLM model configurations for the mCODE Translator framework.

## Overview

The model library provides a unified, maintainable solution for model configuration management that eliminates technical debt from hardcoded strings while enabling better experimentation, version control, and collaboration across the development team.

## File Structure

```
models/
├── models_config.json          # Master configuration file
└── README.md                  # This documentation
```

## Configuration Schema

The [`models_config.json`](models_config.json) file follows this flattened structure:

```json
{
  "deepseek-coder": {
    "name": "deepseek-coder",
    "default": true,
    "model_type": "CODE_GENERATION",
    "model_identifier": "deepseek-coder",
    "api_key_env_var": "DEEPSEEK_API_KEY",
    "base_url": "https://api.deepseek.com/v1",
    "default_parameters": {
      "temperature": 0.1,
      "max_tokens": 4000
    }
  }
}
```

## Usage

To use models from the library in your code:

```python
from src.utils.model_loader import load_model

# Load a specific model configuration
model_config = load_model("deepseek-coder")
print(f"Model identifier: {model_config.model_identifier}")
print(f"Base URL: {model_config.base_url}")

# Access default parameters
temperature = model_config.default_parameters.get('temperature', 0.1)
max_tokens = model_config.default_parameters.get('max_tokens', 4000)
```

## Adding New Models

To add a new model to the library:

1. Add a new entry to [`models_config.json`](models_config.json)
2. Include all required fields:
   - `name`: Unique identifier for the model
   - `model_type`: Category of the model (e.g., CODE_GENERATION, GENERAL_CONVERSATION)
   - `model_identifier`: Actual model identifier used by the provider
   - `api_key_env_var`: Environment variable name for the API key
   - `base_url`: API endpoint for the model
   - `default_parameters`: Default parameters for the model
   - `default`: Boolean indicating if this is the default model

3. Validate the configuration by running the test suite

## Best Practices

1. **Version Control**: Treat model configurations as code and track changes through git
2. **Documentation**: Include comprehensive descriptions and metadata for each model
3. **Consistent Naming**: Use consistent naming conventions across all models
4. **Parameter Defaults**: Provide sensible default parameters for each model
5. **Status Tracking**: Maintain accurate status information for each model

## Related Documentation

- [Model Library Implementation Guide](../docs/MODEL_LIBRARY_GUIDE.md) - Comprehensive guide to the model library implementation
- [Configuration Management](../src/utils/config.py) - Centralized configuration management
- [Model Loader Utility](../src/utils/model_loader.py) - Utility for loading model configurations