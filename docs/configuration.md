# Configuration Guide

## Environment Variables

### Required Variables

#### HEYSOL_API_KEY
API key for HeySol CORE Memory integration.

```bash
export HEYSOL_API_KEY="your-heysol-api-key-here"
```

Get your API key from: https://core.heysol.ai/settings/api

### Optional Variables

#### LLM API Keys
API keys for different LLM providers:

```bash
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

#### Logging Configuration
```bash
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
```

#### Testing Configuration
```bash
export ENABLE_LIVE_TESTS="false"  # Set to true to enable integration tests
```

#### Base URLs
Override default API base URLs:

```bash
export HEYSOL_BASE_URL="https://core.heysol.ai/api/v1"
export CLINICALTRIALS_BASE_URL="https://clinicaltrials.gov/api/v2"
```

## Configuration Files

### LLM Configuration (src/config/llms_config.json)

```json
{
  "models": {
    "gpt-4": {
      "model_identifier": "gpt-4",
      "api_key_env": "OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "temperature": 0.3,
      "max_tokens": 2000,
      "timeout": 30
    },
    "deepseek-coder": {
      "model_identifier": "deepseek-coder",
      "api_key_env": "DEEPSEEK_API_KEY",
      "base_url": "https://api.deepseek.com",
      "temperature": 0.2,
      "max_tokens": 1500,
      "timeout": 25
    }
  }
}
```

### API Configuration (src/config/apis_config.json)

```json
{
  "clinicaltrials": {
    "base_url": "https://clinicaltrials.gov/api/v2",
    "timeout": 30,
    "retry_attempts": 3,
    "rate_limit": 100
  },
  "heysol": {
    "base_url": "https://core.heysol.ai/api/v1",
    "timeout": 15,
    "retry_attempts": 2
  }
}
```

### Cache Configuration (src/config/cache_config.json)

```json
{
  "llm_cache": {
    "ttl_seconds": 3600,
    "max_size_mb": 500,
    "compression": true
  },
  "api_cache": {
    "ttl_seconds": 1800,
    "max_size_mb": 200,
    "compression": false
  }
}
```

### Memory Configuration (src/config/core_memory_config.json)

```json
{
  "spaces": {
    "patients": {
      "name": "OncoCore_Patients",
      "description": "Patient data and mCODE mappings"
    },
    "trials": {
      "name": "OncoCore_Trials",
      "description": "Clinical trial data and mCODE mappings"
    },
    "mcode": {
      "name": "OncoCore_mCODE",
      "description": "mCODE elements and mappings"
    }
  },
  "auto_create_spaces": true,
  "default_ttl_days": 365
}
```

### Logging Configuration (src/config/logging_config.json)

```json
{
  "version": 1,
  "disable_existing_loggers": false,
  "formatters": {
    "standard": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    },
    "detailed": {
      "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "INFO",
      "formatter": "standard"
    },
    "file": {
      "class": "logging.FileHandler",
      "level": "DEBUG",
      "formatter": "detailed",
      "filename": "logs/mcode_translator.log"
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console", "file"]
  }
}
```

### Prompt Configuration (src/config/prompts_config.json)

```json
{
  "prompts": {
    "direct_mcode_evidence_based_concise": {
      "template": "Extract mCODE elements from the following clinical trial text...",
      "description": "Concise evidence-based mCODE extraction",
      "version": "1.0"
    },
    "direct_mcode_comprehensive": {
      "template": "Perform comprehensive mCODE mapping from clinical trial data...",
      "description": "Detailed mCODE extraction with full context",
      "version": "1.0"
    }
  }
}
```

### Validation Configuration (src/config/validation_config.json)

```json
{
  "mcode_validation": {
    "required_elements": [
      "Patient",
      "CancerCondition",
      "CancerTreatment"
    ],
    "element_codes_required": true,
    "confidence_threshold": 0.7,
    "strict_mode": false
  },
  "data_validation": {
    "require_nct_id": true,
    "require_eligibility_criteria": true,
    "validate_fhir_format": true
  }
}
```

## Programmatic Configuration

### Using Config Class

```python
from src.utils.config import Config

# Initialize configuration
config = Config()

# Override API keys programmatically
config.set_api_key("gpt-4", "your-openai-key")
config.set_api_key("deepseek-coder", "your-deepseek-key")

# Configure model parameters
config.set_temperature("gpt-4", 0.3)
config.set_max_tokens("gpt-4", 2000)
config.set_timeout("gpt-4", 30)

# Get configuration values
api_key = config.get_api_key("gpt-4")
temperature = config.get_temperature("gpt-4")
```

### Custom HeySol Configuration

```python
from src.config.heysol_config import McodeHeySolConfig

# Create configuration with explicit authentication
config = McodeHeySolConfig.with_authentication(
    api_key="your-api-key",
    user="registry-user",
    base_url="https://custom.heysol.ai/api/v1"
)

# Use environment variables
config = McodeHeySolConfig.from_env()

# Get authentication details
api_key = config.get_api_key()
user = config.user
base_url = config.get_base_url()
```

## CLI Configuration

### Authentication Setup

```bash
# Using API key
python mcode_cli.py --api-key "your-heysol-key" trials process --input data.ndjson

# Using registry user
python mcode_cli.py --user "your-registry-user" trials process --input data.ndjson

# Using environment variable (recommended)
export HEYSOL_API_KEY="your-heysol-key"
python mcode_cli.py trials process --input data.ndjson
```

### Configuration Validation

```bash
# Check configuration
python mcode_cli.py config check

# Validate all settings
python mcode_cli.py config validate

# Show current configuration
python mcode_cli.py config show
```

## Advanced Configuration

### Custom Pipeline Configuration

```python
from src.core.dependency_container import DependencyContainer
from src.pipeline import McodePipeline

# Create custom container
container = DependencyContainer()

# Register custom components
container.register_component(
    "custom_pipeline",
    McodePipeline(model_name="gpt-4", prompt_name="custom_prompt")
)

# Get configured pipeline
pipeline = container.get_component("custom_pipeline")
```

### Memory Storage Configuration

```python
from src.storage.mcode_memory_storage import OncoCoreMemory

# Configure memory storage
memory = OncoCoreMemory()

# Custom space configuration
memory.create_space("custom_trials", "Custom trial data")
memory.create_space("custom_patients", "Custom patient data")

# Configure TTL
memory.set_default_ttl(30)  # 30 days
```

### Cache Configuration

```python
from src.utils.api_manager import APIManager

# Configure caching
api_manager = APIManager()

# Get cache instances
llm_cache = api_manager.get_cache("llm")
api_cache = api_manager.get_cache("api")

# Configure cache settings
llm_cache.set_ttl(3600)  # 1 hour
llm_cache.set_max_size(500 * 1024 * 1024)  # 500MB

# Clear caches
api_manager.clear_cache("llm")
api_manager.clear_cache("api")
```

## Troubleshooting Configuration

### Common Issues

#### Missing API Keys

**Error:** `ValueError: No API key available for gpt-4`

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
# Or pass via CLI
python mcode_cli.py --api-key "your-heysol-key" ...
```

#### Invalid Configuration Files

**Error:** `JSONDecodeError` or validation errors

**Solution:**
- Check JSON syntax in configuration files
- Validate against schema
- Use `python mcode_cli.py config validate`

#### Memory Connection Issues

**Error:** `ConnectionError: Unable to connect to HeySol API`

**Solution:**
- Verify `HEYSOL_API_KEY` is set
- Check network connectivity
- Validate API key validity
- Use `python mcode_cli.py status` to diagnose

### Configuration Validation

```python
from src.utils.config import Config

# Validate configuration
config = Config()
is_valid, errors = config.validate()

if not is_valid:
    for error in errors:
        print(f"Configuration error: {error}")
```

### Logging Configuration Issues

```python
import logging
from src.utils.logging_config import get_logger

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Get configured logger
logger = get_logger(__name__)
logger.debug("Debug message")
```

## Best Practices

### Security
- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Rotate API keys regularly
- Use least-privilege access

### Performance
- Configure appropriate cache TTL values
- Set reasonable timeout values
- Use concurrent processing for large datasets
- Monitor memory usage

### Maintainability
- Document custom configuration changes
- Use version control for configuration files
- Validate configuration on deployment
- Keep configuration files in sync across environments

### Environment Management
- Use different configurations for development/staging/production
- Document environment-specific settings
- Automate configuration validation in CI/CD
- Backup configuration before changes