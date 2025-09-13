"""
mCODE-Translator Utilities
==========================

This package provides a centralized library of utility modules that support various
components of the mCODE-Translator framework. The utilities are designed to be
reusable, well-documented, and strictly implemented to ensure consistency and
reliability across the system.

Key Modules:
------------
- **api_manager**: Unified API manager for handling caching of API responses from
  both the clinical trial fetcher and the LLM benchmark system.
- **config**: Centralized configuration management that loads settings from
  `config.json` and validates them, providing a single source of truth for all
  configuration values.
- **logging_config**: Standardized logging setup with colored output and
  consistent formatting for clear and readable logs.
- **model_loader**: File-based system for managing LLM model configurations,
  eliminating hardcoded model settings.
- **prompt_loader**: File-based prompt library that centralizes all LLM prompts,
  improving maintainability and consistency.
- **token_tracker**: Thread-safe singleton for tracking and aggregating token
  usage across all LLM calls, providing standardized reporting.
- **feature_utils**: Functions for standardizing the structure of NLP feature
  extraction results.
- **metrics**: Classes for tracking and reporting metrics related to
  patient-trial matching.
- **pattern_config**: Centralized configuration of regex patterns used for
  clinical text processing.

For detailed information on each module, please refer to the `README.md` file in this
directory.
"""

from .api_manager import APIManager
from .config import Config, ConfigurationError
from .feature_utils import (standardize_biomarkers, standardize_features,
                            standardize_variants)
from .logging_config import Loggable, get_logger, setup_logging
from .metrics import MatchingMetrics
from .model_loader import (ModelConfig, ModelLoader, load_model, model_loader,
                           reload_models_config)
from .pattern_config import (AGE_PATTERN, BIOMARKER_PATTERNS,
                             CANCER_TYPE_PATTERN, COMPLEX_VARIANT_PATTERN,
                             CONDITION_PATTERN, ECOG_PATTERN, GENDER_PATTERN,
                             GENE_PATTERN, STAGE_PATTERN, VARIANT_PATTERN)
from .prompt_loader import (PromptLoader, load_prompt, prompt_loader,
                            reload_prompts_config)
from .token_tracker import (TokenUsage, extract_token_usage_from_response,
                            global_token_tracker)

__all__ = [
    # API Manager
    "APIManager",
    # Config
    "Config",
    "ConfigurationError",
    # Logging
    "get_logger",
    "setup_logging",
    "Loggable",
    # Model Loader
    "ModelLoader",
    "model_loader",
    "ModelConfig",
    "load_model",
    "reload_models_config",
    # Prompt Loader
    "PromptLoader",
    "prompt_loader",
    "load_prompt",
    "reload_prompts_config",
    # Token Tracker
    "global_token_tracker",
    "TokenUsage",
    "extract_token_usage_from_response",
    # Feature Utils
    "standardize_features",
    "standardize_biomarkers",
    "standardize_variants",
    # Metrics
    "MatchingMetrics",
    # Pattern Config
    "BIOMARKER_PATTERNS",
    "GENE_PATTERN",
    "VARIANT_PATTERN",
    "COMPLEX_VARIANT_PATTERN",
    "STAGE_PATTERN",
    "CANCER_TYPE_PATTERN",
    "CONDITION_PATTERN",
    "ECOG_PATTERN",
    "GENDER_PATTERN",
    "AGE_PATTERN",
]
