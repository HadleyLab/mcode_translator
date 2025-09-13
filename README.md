# mCODE Translator

A high-performance clinical trial data processing pipeline that extracts and maps eligibility criteria to standardized mCODE elements using evidence-based LLM processing.

## 🏗️ Architecture Overview

The mCODE Translator follows a clean separation of concerns with distinct workflow types:

### 📊 Workflow Types

| Type | Purpose | Core Memory Storage | Examples |
|------|---------|-------------------|----------|
| **Fetchers** | Get raw data from APIs/archives | ❌ No storage | `trials_fetcher`, `patients_fetcher` |
| **Processors** | Apply mCODE processing & store summaries | ✅ Stores mCODE summaries | `trials_processor`, `patients_processor` |
| **Optimizers** | Test parameter combinations | ❌ No storage | `trials_optimizer` |

### 📁 Package Structure

```
src/
├── cli/                          # CLI entry points
│   ├── trials_fetcher.py         # Fetch raw trials JSON
│   ├── trials_processor.py       # Process trials + mCODE → Core Memory
│   ├── patients_fetcher.py       # Fetch synthetic patients
│   ├── patients_processor.py     # Process patients + mCODE → Core Memory
│   └── trials_optimizer.py       # Test optimization combinations
├── workflows/                    # Business logic workflows
│   ├── base_workflow.py          # Common workflow functionality
│   ├── trials_fetcher_workflow.py    # TrialsFetcherWorkflow
│   ├── trials_processor_workflow.py  # TrialsProcessorWorkflow
│   ├── patients_fetcher_workflow.py  # PatientsFetcherWorkflow
│   ├── patients_processor_workflow.py # PatientsProcessorWorkflow
│   └── trials_optimizer_workflow.py  # TrialsOptimizerWorkflow
├── storage/                      # Data persistence
│   └── mcode_memory_storage.py   # Unified Core Memory interface
└── shared/                       # Shared utilities
    ├── cli_utils.py             # Common CLI patterns
    └── types.py                 # Type definitions
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export CORE_MEMORY_API_KEY="your_api_key_here"
```

### Basic Usage

#### 1. Fetch Clinical Trials
```bash
# Fetch trials by condition (no core memory storage)
python -m src.cli.trials_fetcher --condition "breast cancer" -o trials.json
```

#### 2. Process Trials with mCODE
```bash
# Process trials and store mCODE summaries in Core Memory
python -m src.cli.trials_processor trials.json --store-in-core-memory
```

#### 3. Fetch Patient Data
```bash
# Fetch synthetic patients (no core memory storage)
python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.json
```

#### 4. Process Patients with mCODE
```bash
# Process patients with trial filtering and store summaries
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory
```

#### 5. Optimize mCODE Parameters
```bash
# Test different prompt×model combinations (no core memory storage)
python -m src.cli.trials_optimizer --save-config optimal_config.json
```

## 📖 CLI Reference

### Common Arguments

All CLI commands support:
- `-v, --verbose`: Enable debug logging
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--config`: Custom configuration file path

### Core Memory Arguments

Processor commands support:
- `--store-in-core-memory`: Store results in CORE Memory
- `--memory-source`: Source identifier for storage
- `--dry-run`: Preview without storing

### trials_fetcher

Fetch raw clinical trial data from ClinicalTrials.gov.

```bash
# Search by condition
python -m src.cli.trials_fetcher --condition "breast cancer" -o trials.json

# Fetch specific trial
python -m src.cli.trials_fetcher --nct-id NCT12345678 -o trial.json

# Fetch multiple trials
python -m src.cli.trials_fetcher --nct-ids NCT001,NCT002,NCT003 -o trials.json
```

### trials_processor

Process clinical trials with mCODE mapping and store summaries.

```bash
# Basic processing with storage
python -m src.cli.trials_processor trials.json --store-in-core-memory

# Custom model and prompt
python -m src.cli.trials_processor trials.json --model gpt-4 --prompt direct_mcode_evidence_based

# Dry run to preview
python -m src.cli.trials_processor trials.json --dry-run --verbose
```

### patients_fetcher

Fetch synthetic patient data from archives.

```bash
# List available archives
python -m src.cli.patients_fetcher --list-archives

# Fetch from specific archive
python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.json

# Fetch specific patient
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --patient-id patient_123 -o patient.json
```

### patients_processor

Process patient data with mCODE mapping and store summaries.

```bash
# Process with trial filtering
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory

# Process without filtering
python -m src.cli.patients_processor --patients patients.json --store-in-core-memory
```

### trials_optimizer

Test different combinations to find optimal mCODE settings.

```bash
# Basic optimization
python -m src.cli.trials_optimizer

# Test specific combinations
python -m src.cli.trials_optimizer --prompts direct_mcode_evidence_based,evidence_based_minimal --models gpt-4,claude-3

# Save optimal settings
python -m src.cli.trials_optimizer --save-config optimal_config.json

# List available options
python -m src.cli.trials_optimizer --list-prompts
python -m src.cli.trials_optimizer --list-models
```

## 🧠 Core Memory Integration

The system uses **centralized configuration** for all Core Memory settings and stores **only processed mCODE summaries**, not raw data.

### Configuration

Core Memory settings are centralized in `src/config/core_memory_config.json`:

```json
{
  "core_memory": {
    "api_base_url": "https://core.heysol.ai/api/v1/mcp",
    "source": "mcode_translator",
    "timeout_seconds": 60,
    "max_retries": 3,
    "default_spaces": {
      "clinical_trials": "Clinical Trials",
      "patients": "Patients",
      "research": "Research"
    }
  },
  "mcode_settings": {
    "summary_format": "natural_language",
    "include_codes": true,
    "max_summary_length": 2000
  }
}
```

### What Gets Stored

**✅ Trials Processor** → Clinical trial mCODE mappings
```
"Clinical Trial NCT123456: 'Study Title' sponsored by Sponsor.
mCODE Analysis:
Cancer Characteristics:
  - CancerCondition: Breast Cancer [SNOMED:254837009]
  - TNMStage: T2N1M0 [SNOMED:258215001]
Treatments:
  - CancerTreatment: chemotherapy [SNOMED:367336001]
mCODE Compliance Score: 0.950"
```

**✅ Patients Processor** → Patient mCODE profiles
```
"Patient Jane Doe (ID: patient_123), 45 years old, Female.
mCODE Profile:
Cancer Characteristics:
  - CancerCondition: Breast Cancer [SNOMED:254837009]
Biomarkers:
  - ERReceptorStatus: Positive [SNOMED:108283007]
Treatments:
  - CancerTreatment: tamoxifen [SNOMED:386897000]"
```

### What Doesn't Get Stored

**❌ Fetchers** → Only raw JSON data
**❌ Optimizer** → Only configuration recommendations

## 🔧 Configuration

The system uses a **strict modular configuration system** with separate config files for each component:

### 📁 Configuration Structure

```
src/config/
├── cache_config.json          # Caching, rate limiting, requests
├── apis_config.json           # API endpoints and settings
├── core_memory_config.json    # Core Memory integration
├── synthetic_data_config.json # Synthetic patient data settings
├── validation_config.json     # Validation rules and settings
├── logging_config.json        # Logging configuration
├── patterns_config.json       # Regex patterns for text processing
├── models_config.json         # LLM model configurations
├── prompts_config.json        # Prompt template configurations
└── README.md                  # Configuration documentation
```

### 🔧 Configuration Loading

All configurations are loaded strictly - **missing files throw exceptions** with clear error messages:

```python
from src.utils.config import Config

config = Config()  # Throws ConfigurationError if any config file is missing

# Access modular configurations
cache_settings = config.cache_config
api_settings = config.apis_config
core_memory_settings = config.core_memory_config
models_settings = config.models_config
prompts_settings = config.prompts_config
```

### 🚨 Strict Implementation

- **No fallbacks** - Missing configuration files throw exceptions
- **No defaults** - All settings must be explicitly configured
- **Clear errors** - Missing assets provide helpful error messages
- **Fail fast** - Configuration issues prevent application startup

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Check formatting
black --check src/

# Lint code
ruff check src/

# Type checking
mypy --strict src/
```

### Adding New Workflows

1. Create workflow class in `src/workflows/`
2. Create CLI entry point in `src/cli/`
3. Add to shared CLI utilities if needed
4. Update documentation

## 📊 Architecture Benefits

### ✅ Clear Separation of Concerns
- **Fetchers**: Data acquisition only
- **Processors**: mCODE processing + storage
- **Optimizers**: Parameter optimization

### ✅ Consistent Interfaces
- Unified workflow base class
- Standardized CLI argument patterns
- Common error handling

### ✅ Core Memory Efficiency
- Only stores processed mCODE summaries
- Natural language format with embedded codes
- Optimized for later analysis and retrieval

### ✅ Extensibility
- Easy to add new workflow types
- Pluggable storage backends
- Modular CLI architecture

## 🤝 Contributing

1. Follow the established patterns
2. Add tests for new functionality
3. Update documentation
4. Ensure code quality checks pass

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Related Documentation

- [mCODE Specification](https://mcodeinitiative.org/)
- [ClinicalTrials.gov API](https://clinicaltrials.gov/api/)
- [Synthea Patient Generator](https://github.com/synthetichealth/synthea)
