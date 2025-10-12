# 🚀 mCODE Translator Quickstart Guide

Get up and running with mCODE Translator in under 5 minutes! This guide covers installation, basic usage, and key features for new users.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Internet connection**: For API access to ClinicalTrials.gov
- **Optional**: CORE Memory API key for persistent storage

## 🛠️ Installation

### Option 1: Direct Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/HadleyLab/mcode-translator.git
cd mcode-translator

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/HadleyLab/mcode-translator.git
cd mcode-translator
pip install -e .
```

### Option 3: Using pyproject.toml

```bash
# Install with optional development dependencies
pip install -e ".[dev]"
```

## ⚙️ Configuration (Optional)

Create a `.env` file in the project root for CORE Memory integration:

```bash
# Optional: For persistent storage
HEYSOL_API_KEY=your_heysol_api_key_here

# Optional: Logging configuration
LOG_LEVEL=INFO
```

## 🎯 Your First mCODE Translation

### Basic Trial Processing

```bash
# Process a single clinical trial
python mcode-cli.py trials pipeline --fetch --trial-id NCT04348955 --process --engine regex
```

**Expected Output:**
```
🚀 Starting mCODE translation pipeline...
📥 Fetching trial NCT04348955...
🧪 Processing with regex engine...
📊 Extracted 12 mCODE elements
✅ Validation passed (95% confidence)
🧠 Stored in Core Memory
✨ Translation complete!
```

### Search by Condition

```bash
# Fetch trials for a specific condition
python mcode-cli.py trials pipeline --fetch --condition "breast cancer" --limit 3 --process --engine regex
```

### Process with Different Engines

```bash
# Fast processing with regex engine
python mcode-cli.py trials pipeline --fetch --condition "lung cancer" --limit 3 --process --engine regex

# Intelligent processing with LLM
python mcode-cli.py trials pipeline --fetch --condition "lung cancer" --limit 3 --process --engine llm --model deepseek-coder
```

## 🔑 Key Features

### 🤖 Multi-Engine Processing
- **Regex Engine**: Fast, deterministic processing (~0.1s/trial)
- **LLM Engine**: AI-powered extraction with higher accuracy (~2.5s/trial)
- **Automatic Selection**: Choose the best engine for your use case

### 📊 mCODE Standardization
- Converts free-text eligibility criteria to structured mCODE elements
- Supports 95%+ of common oncology trial criteria
- Full validation and confidence scoring

### 🔄 End-to-End Pipeline
- **Fetch**: Retrieve clinical trial data from ClinicalTrials.gov
- **Process**: Extract and standardize mCODE elements
- **Validate**: Ensure data quality and compliance
- **Store**: Persist results in CORE Memory (optional)

### ⚡ Performance Optimized
- Concurrent processing for multiple trials
- Memory-efficient batch operations
- Configurable processing parameters

### 🧠 Smart Storage
- CORE Memory integration for persistent, searchable results
- Automatic deduplication and versioning
- Query capabilities for historical data

## 📖 Basic Usage Examples

### Python API

```python
from src.core.data_flow_coordinator import process_clinical_trials_flow

# Process multiple trials
result = process_clinical_trials_flow(
    trial_ids=["NCT123456", "NCT789012"],
    config={"validate_data": True, "store_results": True}
)

print(f"Processed {len(result.data)} trials successfully")
```

### CLI Commands

```bash
# Fetch and process trials
python mcode-cli.py trials pipeline --fetch --condition "breast cancer" --limit 5 --process --engine regex

# View processing results
python mcode-cli.py trials list

# Optimize processing parameters
python mcode-cli.py optimize trials --trials-file data.ndjson --cv-folds 3
```

## 🧪 Testing Your Installation

```bash
# Run basic functionality test
python examples/basic_usage/basic_trial_processing.py

# Run full test suite
python run_tests.py unit
```

## 📚 Next Steps

### Explore Examples
- **[Basic Usage](examples/basic_usage/)**: Fundamental concepts and simple workflows
- **[Advanced Workflows](examples/advanced_workflow/)**: Multi-trial processing and engine comparison
- **[CLI Commands](examples/cli_commands/)**: Command-line automation patterns
- **[API Integration](examples/api_integration/)**: External system connectivity

### Learn More
- **[Full Documentation](docs/)**: Comprehensive guides and API reference
- **[Configuration Guide](docs/configuration.md)**: Advanced setup and customization
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

### Advanced Features
- **Cross-Validation**: Evaluate processing accuracy across different methods
- **Performance Analysis**: Identify bottlenecks and optimization opportunities
- **Batch Processing**: Handle large datasets efficiently
- **Webhook Integration**: Real-time processing notifications

## 🆘 Need Help?

- **Quick Issues**: Check the [troubleshooting guide](docs/troubleshooting.md)
- **Community Support**: [GitHub Discussions](https://github.com/HadleyLab/mcode-translator/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/HadleyLab/mcode-translator/issues)

---

**🎉 You're ready to start translating clinical trials to mCODE format!**

For detailed documentation, visit [docs.heysol.ai/mcode-translator](https://docs.heysol.ai/mcode-translator)