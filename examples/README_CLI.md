# mCODE Translator CLI Usage Guide

This guide provides comprehensive examples for using the mCODE Translator command-line interface (CLI) tools.

## Table of Contents

- [Quick Start](#quick-start)
- [CLI Tools Overview](#cli-tools-overview)
- [Common Usage Patterns](#common-usage-patterns)
- [Complete Workflows](#complete-workflows)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

1. **Activate the conda environment:**
   ```bash
   conda activate mcode_translator
   ```

2. **Download synthetic patient data (for patient workflows):**
   ```bash
   # Option 1: Complete setup and demo (recommended)
   python examples/setup_and_demo.py

   # Option 2: Manual download
   python -c "from src.utils.data_downloader import download_synthetic_patient_archives; download_synthetic_patient_archives()"

   # Option 3: Check existing archives
   python -m src.cli.patients_fetcher --list-archives
   ```

3. **Set environment variables (required for processing):**
   ```bash
   export COREAI_API_KEY="your-api-key-here"
   ```

4. **Navigate to project root:**
   ```bash
   cd /path/to/mcode_translator
   ```

### Basic Example

```bash
# 1. Fetch clinical trials
python -m src.cli.trials_fetcher --condition "breast cancer" --limit 3 -o trials.json

# 2. Process with mCODE mapping
python -m src.cli.trials_processor trials.json --store-in-core-memory

# 3. Fetch patient data
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 5 -o patients.json

# 4. Process patients
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory
```

## CLI Tools Overview

### 1. Trials Fetcher (`trials_fetcher.py`)

Fetch clinical trial data from ClinicalTrials.gov.

**Basic Usage:**
```bash
python -m src.cli.trials_fetcher --condition "breast cancer" -o trials.json
```

**Key Options:**
- `--condition "disease name"`: Search by medical condition
- `--nct-id NCT12345678`: Fetch specific trial by NCT ID
- `--nct-ids NCT1,NCT2,NCT3`: Fetch multiple trials
- `--limit N`: Maximum number of trials to fetch (default: 10)
- `-o, --output FILE`: Output file path
- `--verbose`: Enable detailed logging

**Examples:**
```bash
# Search by condition
python -m src.cli.trials_fetcher --condition "lung cancer" --limit 5 -o lung_trials.json --verbose

# Fetch specific trials
python -m src.cli.trials_fetcher --nct-ids NCT03170960,NCT03805399 -o specific_trials.json

# Single trial
python -m src.cli.trials_fetcher --nct-id NCT03170960 -o single_trial.json
```

### 2. Trials Processor (`trials_processor.py`)

Process clinical trial data with mCODE mapping and store results.

**Basic Usage:**
```bash
python -m src.cli.trials_processor trials.json --store-in-core-memory
```

**Key Options:**
- `input_file`: Path to JSON file containing trial data
- `--model MODEL`: LLM model (default: configured default)
- `--prompt PROMPT`: Prompt template (default: direct_mcode_evidence_based_concise)
- `--store-in-core-memory`: Store results in CORE Memory
- `--dry-run`: Preview without storing
- `--batch-size N`: Processing batch size (default: 10)

**Examples:**
```bash
# Process with specific model
python -m src.cli.trials_processor trials.json --model gpt-4 --store-in-core-memory

# Dry run to preview
python -m src.cli.trials_processor trials.json --dry-run --verbose

# Custom prompt
python -m src.cli.trials_processor trials.json --prompt direct_mcode_minimal --store-in-core-memory
```

### 3. Patients Fetcher (`patients_fetcher.py`)

Fetch synthetic patient data from archives.

**Data Download Required:**
Before using the patients fetcher, you need to download the synthetic patient data archives:
```bash
# Download all available archives
python -c "from src.utils.data_downloader import download_synthetic_patient_archives; download_synthetic_patient_archives()"

# Or use the complete setup script
python examples/setup_and_demo.py
```

**Basic Usage:**
```bash
python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.json
```

**Key Options:**
- `--archive ARCHIVE`: Patient archive identifier
- `--patient-id ID`: Specific patient ID to fetch
- `--limit N`: Maximum patients to fetch (default: 10)
- `--list-archives`: List available archives
- `-o, --output FILE`: Output file path

**Examples:**
```bash
# List available archives
python -m src.cli.patients_fetcher --list-archives

# Fetch from specific archive
python -m src.cli.patients_fetcher --archive mixed_cancer_lifetime --limit 20 -o patients.json

# Fetch specific patient
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --patient-id patient_001 -o patient.json
```

### 4. Patients Processor (`patients_processor.py`)

Process patient data with mCODE mapping and eligibility filtering.

**Basic Usage:**
```bash
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory
```

**Key Options:**
- `--patients FILE`: Patient data JSON file (required)
- `--trials FILE`: Trial data for eligibility filtering (optional)
- `--store-in-core-memory`: Store results in CORE Memory
- `--dry-run`: Preview without storing

**Examples:**
```bash
# Process patients with trial filtering
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory

# Process patients only
python -m src.cli.patients_processor --patients patients.json --store-in-core-memory

# Preview processing
python -m src.cli.patients_processor --patients patients.json --dry-run --verbose
```

### 5. Trials Optimizer (`trials_optimizer.py`)

Optimize mCODE translation parameters by testing different combinations.

**Basic Usage:**
```bash
python -m src.cli.trials_optimizer --trials-file trials.json
```

**Key Options:**
- `--trials-file FILE`: Trial data for optimization testing
- `--prompts P1,P2,P3`: Comma-separated prompt templates to test
- `--models M1,M2,M3`: Comma-separated models to test
- `--max-combinations N`: Maximum combinations to test
- `--save-config FILE`: Save optimal settings to file
- `--list-prompts`: List available prompts
- `--list-models`: List available models

**Examples:**
```bash
# List available options
python -m src.cli.trials_optimizer --list-prompts
python -m src.cli.trials_optimizer --list-models

# Optimize specific combinations
python -m src.cli.trials_optimizer --trials-file trials.json --prompts direct_mcode_evidence_based_concise,direct_mcode_minimal --models deepseek-coder,gpt-4 --max-combinations 4

# Save optimal configuration
python -m src.cli.trials_optimizer --trials-file trials.json --save-config optimal_config.json --verbose
```

## Common Usage Patterns

### Pattern 1: Basic Trial Processing

```bash
# Fetch trials
python -m src.cli.trials_fetcher --condition "breast cancer" --limit 5 -o trials.json

# Process trials
python -m src.cli.trials_processor trials.json --store-in-core-memory
```

### Pattern 2: Patient-Centric Workflow

```bash
# Fetch patients
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 10 -o patients.json

# Fetch trial criteria
python -m src.cli.trials_fetcher --condition "breast cancer" --limit 3 -o trial_criteria.json

# Process patients with eligibility filtering
python -m src.cli.patients_processor --patients patients.json --trials trial_criteria.json --store-in-core-memory
```

### Pattern 3: Optimization Workflow

```bash
# Fetch test data
python -m src.cli.trials_fetcher --condition "cancer" --limit 5 -o test_trials.json

# Optimize parameters
python -m src.cli.trials_optimizer --trials-file test_trials.json --save-config optimal.json

# Use optimized settings
python -m src.cli.trials_processor trials.json --config optimal.json --store-in-core-memory
```

## Complete Workflows

### Workflow 1: Full Research Pipeline

```bash
#!/bin/bash
# Complete research pipeline script

echo "🚀 Starting mCODE Research Pipeline"

# 1. Fetch clinical trials
echo "📋 Fetching clinical trials..."
python -m src.cli.trials_fetcher --condition "breast cancer" --limit 10 -o research_trials.json --verbose

# 2. Process trials with mCODE
echo "🔬 Processing trials..."
python -m src.cli.trials_processor research_trials.json --store-in-core-memory --verbose

# 3. Fetch patient cohorts
echo "👥 Fetching patient data..."
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 20 -o research_patients.json --verbose

# 4. Process patients with trial matching
echo "🎯 Processing patients with trial eligibility..."
python -m src.cli.patients_processor --patients research_patients.json --trials research_trials.json --store-in-core-memory --verbose

# 5. Optimize for future use
echo "⚡ Optimizing parameters..."
python -m src.cli.trials_optimizer --trials-file research_trials.json --max-combinations 6 --save-config research_config.json

echo "✅ Research pipeline completed!"
```

### Workflow 2: Development Testing

```bash
#!/bin/bash
# Development testing workflow

echo "🧪 Development Testing Workflow"

# Dry run everything first
echo "🔍 Dry run - trials"
python -m src.cli.trials_fetcher --condition "test cancer" --limit 2 -o test_trials.json
python -m src.cli.trials_processor test_trials.json --dry-run --verbose

echo "🔍 Dry run - patients"
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 3 -o test_patients.json
python -m src.cli.patients_processor --patients test_patients.json --dry-run --verbose

# Run optimization
echo "🎯 Testing optimization"
python -m src.cli.trials_optimizer --trials-file test_trials.json --max-combinations 2 --verbose

echo "✅ Testing completed!"
```

## Configuration

### Environment Variables

```bash
# Required for CORE Memory storage
export COREAI_API_KEY="your-core-ai-api-key"

# Optional: Custom logging level
export LOG_LEVEL="DEBUG"

# Optional: Custom configuration file
export MCODE_CONFIG="/path/to/custom/config.json"
```

### Configuration Files

The CLI tools use several configuration files:

- `src/config/core_memory_config.json`: CORE Memory settings
- `src/config/llms_config.json`: LLM model configurations
- `src/config/prompts_config.json`: Available prompt templates
- `src/config/apis_config.json`: API endpoints and settings

### Custom Configuration

```bash
# Use custom config file
python -m src.cli.trials_processor trials.json --config /path/to/custom_config.json --store-in-core-memory
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   ```bash
   # Ensure conda environment is activated
   conda activate mcode_translator

   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **CORE Memory connection failed:**
   ```bash
   # Check API key
   echo $COREAI_API_KEY

   # Test connection
   python -c "from src.utils.core_memory_client import CoreMemoryClient; print('Connection OK')"
   ```

3. **File not found errors:**
   ```bash
   # Use absolute paths
   python -m src.cli.trials_processor /full/path/to/trials.json

   # Or run from project root
   cd /path/to/mcode_translator
   python -m src.cli.trials_processor trials.json
   ```

4. **API rate limiting:**
   ```bash
   # Add delays between requests
   python -m src.cli.trials_fetcher --condition "cancer" --limit 5 --delay 2
   ```

### Debug Mode

Enable verbose logging for all commands:

```bash
# Add --verbose to any command
python -m src.cli.trials_fetcher --condition "breast cancer" --verbose
python -m src.cli.trials_processor trials.json --verbose --store-in-core-memory
```

### Getting Help

```bash
# Get help for any CLI tool
python -m src.cli.trials_fetcher --help
python -m src.cli.trials_processor --help
python -m src.cli.patients_fetcher --help
python -m src.cli.patients_processor --help
python -m src.cli.trials_optimizer --help
```

## Performance Tips

1. **Use batch processing** for large datasets
2. **Enable caching** for repeated operations
3. **Use dry-run** to test before production
4. **Monitor memory usage** with `--verbose`
5. **Optimize batch sizes** based on your system

## Next Steps

- Check the main [README.md](../README.md) for project overview
- Review the [configuration files](../src/config/) for customization options
- Explore the [examples directory](./) for more advanced usage patterns
- Check the [tests](../tests/) for additional usage examples

---

For questions or issues, please check the troubleshooting section above or create an issue in the project repository.