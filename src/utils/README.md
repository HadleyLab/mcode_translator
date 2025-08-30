# Utility Modules Documentation

This directory contains utility modules that provide shared functionality across the mCODE Translator framework.

## Token Tracking System

The token tracking system provides standardized token usage reporting across all LLM calls in the framework. It consists of three main components:

### TokenUsage Dataclass
A standardized data structure for representing token usage information:
- `prompt_tokens`: Number of tokens in the prompt
- `completion_tokens`: Number of tokens in the completion/response
- `total_tokens`: Total tokens consumed
- `model_name`: Name of the model used
- `provider_name`: Name of the provider used

### TokenTracker Singleton
A thread-safe singleton class that aggregates token usage across all LLM calls:
- Maintains running totals of token usage
- Tracks usage by component (NLP extraction, mCODE mapping, etc.)
- Provides methods for retrieving and resetting usage data
- Thread-safe implementation using locks

### Global Token Tracker
A singleton instance of TokenTracker that is used throughout the framework:
- Automatically imported and used by all LLM components
- Resets at the beginning of each processing operation
- Captures aggregate usage for reporting in results

### Integration with LLM Components
The token tracking system is integrated with:
- `StrictLLMBase`: Base class that extracts token usage from LLM responses
- `StrictNlpExtractor`: NLP engine that tracks extraction token usage
- `StrictMcodeMapper`: mCODE mapper that tracks mapping token usage
- `StrictDynamicExtractionPipeline`: Pipeline that aggregates and reports token usage

### Usage Example
```python
from src.utils.token_tracker import global_token_tracker

# Get current aggregate token usage
current_usage = global_token_tracker.get_total_usage()

# Reset tracking for a new operation
global_token_tracker.reset()

# Add usage from an LLM call
global_token_tracker.add_usage(token_usage, "nlp_extraction")
```

## Model Library

The model library provides a file-based system for managing LLM model configurations:

### ModelLoader Class
A utility class for loading model configurations from the file-based model library:
- Loads model configurations from JSON files
- Caches configurations for performance
- Provides validation and error handling
- Supports nested configuration structures

### ModelConfig Dataclass
A standardized data structure for representing model configurations:
- `name`: Unique identifier for the model
- `model_type`: Category of the model (e.g., CODE_GENERATION, GENERAL_CONVERSATION)
- `model_identifier`: Actual model identifier used by the provider
- `base_url`: API endpoint for the model
- `description`: Brief description of the model's purpose
- `default_parameters`: Default parameters for the model
- `capabilities`: List of capabilities the model provides
- And other metadata fields

### Integration with Configuration Management
The model library integrates with the unified configuration system:
- `Config.get_model_config()`: Get model configuration from the library
- `Config.get_all_model_configs()`: Get all model configurations
- `Config.reload_model_configs()`: Reload model configurations from disk

### Usage Example
```python
from src.utils.model_loader import load_model
from src.utils.config import Config

# Load a specific model configuration
model_config = load_model("deepseek-coder")
print(f"Model: {model_config.model_identifier}")
print(f"Base URL: {model_config.base_url}")

# Access through Config class
config = Config()
model_config = config.get_model_config("deepseek-coder")

# Get default parameters
temperature = model_config.default_parameters.get('temperature', 0.1)
max_tokens = model_config.default_parameters.get('max_tokens', 4000)
```

## Other Utilities

### Logging Configuration
Standardized logging setup with colored output and consistent formatting.

### Configuration Management
Unified configuration management that loads settings from config.json and validates them.

### Cache Manager
Thread-safe caching system for storing and retrieving LLM responses to reduce API calls and improve performance.

### Prompt Loader
File-based prompt library system that centralizes all LLM prompts and eliminates hardcoded strings.