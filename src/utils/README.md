# mCODE-Translator Utility Modules

This directory contains a centralized library of utility modules that provide shared, standardized, and strictly implemented functionality across the mCODE-Translator framework.

## Centralized `__init__.py`

The `src/utils/__init__.py` file makes this directory a Python package and exposes all key utility classes and functions for easy and consistent importing across the application. This promotes a clear, centralized structure and avoids scattered or redundant utility implementations.

## Key Modules and Functionality

### API Manager (`api_manager.py`)
- **`UnifiedAPIManager`**: A centralized class for managing API response caching. It provides namespace-based separation for different types of cached data (e.g., LLM responses, clinical trial data) to ensure that different parts of the system do not interfere with each other's caches.
- **`APICache`**: A generic, disk-based cache class that supports time-to-live (TTL) for cache entries.

### Configuration (`config.py`)
- **`Config`**: A strict, centralized configuration management class that loads settings from `config.json`. It acts as a single source of truth for all configuration values and raises a `ConfigurationError` if any required settings are missing or invalid.

### Logging (`logging_config.py`)
- **`get_logger`**: A factory function for creating standardized, colored logger instances.
- **`setup_logging`**: A centralized function to configure the root logger for the application.
- **`Loggable`**: A base class that provides a logger instance to subclasses, ensuring consistent logging behavior.

### Model Loader (`model_loader.py`)
- **`model_loader`**: A global instance of the `ModelLoader` class, which manages loading LLM model configurations from the file-based model library.
- **`ModelConfig`**: A dataclass that represents a standardized model configuration.

### Prompt Loader (`prompt_loader.py`)
- **`prompt_loader`**: A global instance of the `PromptLoader` class, which centralizes all LLM prompts, improving maintainability and consistency.

### Token Tracker (`token_tracker.py`)
- **`global_token_tracker`**: A thread-safe singleton instance for tracking and aggregating token usage across all LLM calls.
- **`TokenUsage`**: A dataclass for a standardized token usage data structure.

### Feature Utilities (`feature_utils.py`)
- Provides functions for standardizing the structure of NLP feature extraction results, ensuring consistency across different NLP engines and models.

### Metrics (`metrics.py`)
- **`MatchingMetrics`**: A class for tracking and reporting metrics related to patient-trial matching, providing insights into the performance of the matching algorithm.

### Pattern Configuration (`pattern_config.py`)
- A centralized module that contains all regular expression patterns used for pattern matching in clinical text, organized by category for better maintainability.