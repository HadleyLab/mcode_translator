# Modular Configuration System

The mCODE Translator uses a **modular configuration system** where each component has its own dedicated configuration file. This provides better organization, maintainability, and flexibility.

## 📁 Configuration Structure

```
src/config/
├── cache_config.json          # Caching, rate limiting, requests
├── apis_config.json           # API endpoints and settings
├── core_memory_config.json    # Core Memory integration
├── synthetic_data_config.json # Synthetic patient data settings
├── validation_config.json     # Validation rules and settings
├── logging_config.json        # Logging configuration
├── patterns_config.json       # Regex patterns for text processing
└── README.md                  # This file
```

## 🔧 Configuration Files

### `cache_config.json`
Controls caching behavior, rate limiting, and request settings:
- Cache directories and TTL settings
- Rate limiting delays and burst limits
- Request timeouts and retry policies

### `apis_config.json`
API endpoint configurations:
- ClinicalTrials.gov API settings
- PubMed, CrossRef, and other external APIs
- Rate limits and authentication settings

### `core_memory_config.json`
Core Memory integration settings:
- API base URL and authentication
- Default spaces and storage settings
- mCODE summary formatting options

### `synthetic_data_config.json`
Synthetic patient data management:
- Archive locations and download settings
- Patient filtering criteria
- Processing batch sizes and limits

### `validation_config.json`
Data validation rules:
- Required fields and data types
- Validation constraints and patterns
- Quality checks and error handling

### `logging_config.json`
Logging system configuration:
- Log levels and formats
- Handler configurations (console, file)
- Logger-specific settings

### `patterns_config.json`
Regex patterns for clinical text processing:
- Biomarker detection patterns
- Genomic variant patterns
- Condition and demographic patterns

## 🚀 Usage

### Loading Configuration

```python
from src.utils.config import Config

config = Config()

# Access specific configurations
cache_settings = config.cache_config
api_settings = config.apis_config
core_memory_settings = config.core_memory_config
```

### Configuration Methods

The `Config` class provides convenient methods for accessing configuration:

```python
# Cache settings
cache_enabled = config.is_cache_enabled()
cache_ttl = config.get_cache_ttl()

# API settings
clinical_trials_url = config.get_clinical_trials_base_url()

# Core Memory settings
core_memory_key = config.get_core_memory_api_key()
memory_spaces = config.get_core_memory_default_spaces()

# Synthetic data settings
data_base_dir = config.get_synthetic_data_base_directory()
default_archive = config.get_synthetic_data_default_archive()

# Validation settings
strict_mode = config.is_strict_mode()

# Logging settings
default_level = config.get_default_log_level()
log_format = config.get_log_format()

# Pattern access
biomarker_patterns = config.get_biomarker_patterns()
genomic_patterns = config.get_genomic_patterns()
```

## 🔄 Legacy Support

The system maintains backward compatibility with the old `config.json` file. If the legacy file exists, it's loaded as `legacy_config` for migration purposes.

## 🛠️ Modifying Configuration

### Adding New Settings

1. **Choose the appropriate config file** based on the setting's purpose
2. **Add the setting** to the JSON file
3. **Add a method** to the `Config` class if needed
4. **Update documentation**

### Example: Adding a New API

```json
// In apis_config.json
{
  "apis": {
    "new_service": {
      "base_url": "https://api.newservice.com",
      "rate_limit_per_minute": 100,
      "timeout_seconds": 30
    }
  }
}
```

```python
# In config.py
def get_new_service_base_url(self) -> str:
    """Get new service API base URL"""
    return self.apis_config['apis']['new_service']['base_url']
```

## 📊 Benefits

### ✅ **Modular Organization**
- Each component has its own configuration file
- Clear separation of concerns
- Easy to find and modify specific settings

### ✅ **Maintainability**
- Smaller, focused configuration files
- Easier to review changes
- Reduced merge conflicts

### ✅ **Flexibility**
- Environment-specific configurations
- Easy to add new components
- Pluggable configuration sources

### ✅ **Type Safety**
- Structured JSON schemas
- Validation at load time
- Clear error messages for invalid configurations

## 🔍 Configuration Validation

All configuration files are validated when loaded:
- JSON syntax validation
- Required field checks
- Type validation where applicable
- Clear error messages for missing or invalid settings

## 🚨 Error Handling

Configuration errors are handled gracefully:
- Missing files result in clear error messages
- Invalid JSON is caught with helpful context
- Missing required settings fail fast with explanations

## 📝 Best Practices

1. **Use descriptive keys** that clearly indicate their purpose
2. **Group related settings** together in logical sections
3. **Document complex settings** with comments in the JSON
4. **Use consistent naming conventions** across all config files
5. **Version control** all configuration changes
6. **Test configuration changes** before deployment

## 🔗 Related Documentation

- [Main README](../README.md) - Overall project documentation
- [Configuration Class](../utils/config.py) - Implementation details
- [Environment Setup](../../README.md#prerequisites) - Environment variable setup