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

## Integration Points

### CORE Memory Integration

The Expert Multi-LLM Curator integrates with HeySol CORE Memory for persistent storage and retrieval of clinical data, trial information, and processing results.

#### CORE Memory Configuration (src/config/core_memory_config.json)

```json
{
  "core_memory": {
    "api_base_url": "https://core.heysol.ai/api/v1/mcp",
    "source": "mcode_translator",
    "timeout_seconds": 60,
    "max_retries": 3,
    "retry_delay_seconds": 1,
    "default_spaces": {
      "clinical_trials": "Clinical Trials",
      "patients": "Patients"
    },
    "storage_settings": {
      "batch_size": 10,
      "max_concurrent_requests": 5,
      "rate_limit_per_minute": 60
    }
  },
  "mcode_settings": {
    "summary_format": "natural_language",
    "include_codes": true,
    "max_summary_length": 2000,
    "embedding_model": "text-embedding-ada-002"
  }
}
```

#### Configuration Parameters

- **api_base_url**: HeySol CORE Memory API endpoint
- **source**: Application identifier for memory operations
- **timeout_seconds**: Request timeout in seconds
- **max_retries**: Maximum retry attempts for failed requests
- **default_spaces**: Predefined memory spaces for clinical trials and patients
- **batch_size**: Number of records to process in each batch
- **rate_limit_per_minute**: Maximum API requests per minute
- **summary_format**: Output format for mCODE summaries
- **include_codes**: Whether to include SNOMED codes in summaries
- **max_summary_length**: Maximum length of generated summaries

#### Memory Spaces

The system automatically creates and manages the following memory spaces:

- **Clinical Trials**: Stores processed clinical trial data and mCODE mappings
- **Patients**: Stores patient data and eligibility assessments
- **OncoCore_mCODE**: Stores mCODE elements and standardized mappings

### Ensemble Configuration Parameters

The Expert Multi-LLM Curator uses an ensemble decision engine that combines multiple LLM experts with rule-based matching for optimal patient-trial matching accuracy.

#### Ensemble Configuration (src/config/ensemble_config.json)

```json
{
  "consensus_methods": {
    "default_method": "dynamic_weighting",
    "available_methods": [
      "weighted_majority_vote",
      "confidence_weighted",
      "bayesian_ensemble",
      "dynamic_weighting"
    ]
  },
  "expert_configuration": {
    "expert_types": [
      {
        "type": "clinical_reasoning",
        "description": "Specializes in detailed clinical rationale and safety considerations",
        "base_weight": 1.0,
        "reliability_score": 0.85,
        "expertise_areas": [
          "clinical_safety",
          "treatment_history",
          "comorbidity_assessment",
          "risk_benefit_analysis"
        ]
      },
      {
        "type": "pattern_recognition",
        "description": "Expert in identifying complex patterns in clinical data",
        "base_weight": 0.9,
        "expertise_areas": [
          "pattern_matching",
          "edge_case_detection",
          "complexity_assessment",
          "nuanced_criteria"
        ]
      },
      {
        "type": "comprehensive_analyst",
        "description": "Provides holistic assessment and risk-benefit analysis",
        "base_weight": 1.1,
        "expertise_areas": [
          "holistic_evaluation",
          "multi_criteria_analysis",
          "comprehensive_review",
          "integrated_assessment"
        ]
      }
    ]
  }
}
```

#### Key Configuration Parameters

##### Consensus Methods
- **default_method**: Primary consensus algorithm (dynamic_weighting)
- **available_methods**: Supported consensus algorithms
  - `weighted_majority_vote`: Simple weighted voting
  - `confidence_weighted`: Weights by expert confidence scores
  - `bayesian_ensemble`: Uses Bayesian inference
  - `dynamic_weighting`: Adjusts weights based on case complexity

##### Expert Configuration
- **expert_types**: Array of expert configurations with weights and specialties
- **min_experts**: Minimum number of experts to consult (default: 2)
- **max_experts**: Maximum number of experts to consult (default: 3)
- **diversity_threshold**: Minimum diversity score for expert selection (default: 0.7)

##### Decision Rules
- **confidence_thresholds**: Confidence levels for decision making
  - `high_confidence`: 0.8
  - `moderate_confidence`: 0.6
  - `minimum_acceptable`: 0.2
- **require_majority**: Whether majority agreement is required
- **fallback_to_rule_based**: Whether to fall back to rule-based matching on failure

##### Performance Optimization
- **caching**: Expert panel response caching configuration
- **batch_processing**: Batch processing settings for efficiency
- **resource_management**: Memory and timeout limits

### Expert Panel Prompts

The ensemble system uses specialized prompts for each expert type, stored in the `prompts/expert_panel/` directory.

#### Clinical Reasoning Specialist (clinical_reasoning_specialist.txt)

```
You are a Clinical Reasoning Specialist with deep expertise in oncology clinical trials and patient eligibility assessment.

Your role is to analyze patient-trial matches with rigorous clinical reasoning, focusing on:
1. Detailed clinical rationale for inclusion/exclusion decisions
2. Assessment of clinical appropriateness and safety considerations
3. Identification of nuanced clinical factors that might affect eligibility
4. Evaluation of disease stage, treatment history, and comorbidities
```

**Key Features:**
- Focuses on clinical safety and treatment history
- Provides detailed rationale for decisions
- Evaluates comorbidities and risk factors
- Assesses clinical appropriateness

#### Pattern Recognition Expert (pattern_recognition_expert.txt)

```
You are a Pattern Recognition Expert specializing in identifying complex patterns in clinical data and trial eligibility criteria.

Your expertise includes:
1. Recognition of subtle patterns in patient characteristics and trial requirements
2. Identification of eligibility patterns across multiple clinical dimensions
3. Detection of edge cases and unusual clinical presentations
4. Pattern matching for complex inclusion/exclusion scenarios
```

**Key Features:**
- Identifies complex patterns in clinical data
- Detects edge cases and unusual presentations
- Performs pattern matching across multiple dimensions
- Handles nuanced eligibility criteria

#### Comprehensive Analyst (comprehensive_analyst.txt)

```
You are a Comprehensive Clinical Analyst with expertise in holistic patient-trial matching assessment.

Your role encompasses:
1. Comprehensive evaluation of all clinical dimensions simultaneously
2. Integration of multiple data sources and clinical parameters
3. Holistic risk-benefit analysis for trial participation
4. Synthesis of complex clinical information into actionable insights
```

**Key Features:**
- Holistic evaluation of all clinical dimensions
- Risk-benefit analysis for trial participation
- Integration of multiple data sources
- Comprehensive clinical assessment

#### Response Format

All expert prompts use a standardized JSON response format:

```json
{
  "is_match": true,
  "confidence_score": 0.95,
  "reasoning": "Detailed clinical reasoning...",
  "matched_criteria": ["criterion1", "criterion2"],
  "unmatched_criteria": ["criterion3"],
  "clinical_notes": "Additional clinical observations...",
  "risk_assessment": "Risk level and considerations..."
}
```

### Caching Integration

The system implements multi-level caching to optimize performance and reduce API costs.

#### Cache Configuration (src/config/cache_config.json)

```json
{
  "cache": {
    "enabled": true,
    "api_cache_directory": ".api_cache",
    "llm_cache_directory": ".api_cache/llm",
    "clinical_trials_cache_directory": ".api_cache/clinical_trials",
    "ttl_seconds": 0,
    "max_cache_size_mb": 1000,
    "compression_enabled": true,
    "cleanup_interval_hours": 24
  },
  "rate_limiting": {
    "delay_seconds": 1.0,
    "max_requests_per_minute": 60,
    "burst_limit": 10,
    "backoff_multiplier": 2.0
  }
}
```

#### Cache Types

##### LLM Cache
- **Purpose**: Caches LLM API responses to reduce costs and latency
- **TTL**: Configurable time-to-live (default: 3600 seconds)
- **Directory**: `.api_cache/llm`
- **Compression**: Enabled for storage efficiency
- **Performance Impact**: 33%+ cost reduction achieved

##### API Cache
- **Purpose**: Caches ClinicalTrials.gov API responses
- **TTL**: Configurable (default: 1800 seconds)
- **Directory**: `.api_cache/clinical_trials`
- **Rate Limiting**: Built-in rate limiting to respect API limits

##### Expert Panel Cache
- **Purpose**: Caches expert panel responses for repeated queries
- **TTL**: 7200 seconds (2 hours)
- **Key Components**: Expert type, patient hash, trial hash, prompt hash
- **Performance Monitoring**: Tracks hit rates and response time savings

#### Cache Management

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

### HeySol API Integration

The system integrates with HeySol API for advanced memory operations and multi-instance management.

#### HeySol Configuration (src/config/heysol_config.py)

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

#### Authentication Methods

##### API Key Authentication
- Direct API key authentication
- Environment variable: `HEYSOL_API_KEY`
- Most secure for single-user scenarios

##### Registry Authentication
- User-based authentication with email identifiers
- Supports multi-instance management
- Automatic API key resolution from registry

#### Multi-Instance Operations

The system supports cross-instance operations for managing logs between different HeySol instances:

```python
# Move logs between instances
heysol_client.move_logs(
    source_user="user1@example.com",
    target_user="user2@example.com",
    log_ids=["log1", "log2"]
)

# Copy logs with full fidelity
heysol_client.copy_logs(
    source_user="user1@example.com",
    target_user="user2@example.com",
    log_ids=["log1", "log2"],
    preserve_metadata=True
)
```

#### Configuration Parameters

- **api_key**: HeySol API key for authentication
- **user**: Registry user identifier (email)
- **base_url**: HeySol API base URL
- **timeout**: Request timeout in seconds
- **mcode_cache_enabled**: Enable mCODE-specific caching
- **mcode_batch_size**: Batch size for operations
- **mcode_workers**: Number of worker threads

#### Registry System

The registry system manages multiple HeySol instances:

- **Email-based identification**: Users identified by email addresses
- **Automatic resolution**: Email identifiers resolved to API keys and base URLs
- **Secure storage**: Credentials stored securely with environment variable loading
- **Cross-instance operations**: Move/copy logs between different instances while preserving metadata

#### Integration Features

- **Memory ingestion**: Store conversation data for future reference
- **Memory search**: Query past conversations and decisions
- **Space management**: Organize memories by project/topic
- **Batch operations**: Efficient processing of multiple records
- **Rate limiting**: Respect API limits with automatic backoff
- **Error handling**: Comprehensive error handling with retries