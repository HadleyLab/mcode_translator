# mCODE Translator v2.0

> **High-performance clinical trial data processing pipeline that extracts and maps eligibility criteria to standardized mCODE elements using evidence-based LLM processing.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/HadleyLab/mcode_translator.git
cd mcode_translator
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or DEEPSEEK_API_KEY

# Process clinical trial data
python mcode_translator.py data/selected_breast_cancer_trials.json -o results.json
```

## 💻 Usage

### mCODE Translator - Process Trial Data

```bash
# Basic processing
python mcode_translator.py --input-file data/selected_breast_cancer_trials.json -o results.json

# With specific model and prompt
python mcode_translator.py --input-file data/selected_breast_cancer_trials.json -m deepseek-coder -p direct_mcode_evidence_based_concise -o results.json

# Verbose processing with custom config
python mcode_translator.py --input-file data/selected_breast_cancer_trials.json --verbose --config custom_config.json

# Batch processing
python mcode_translator.py --input-file trials_batch.json --batch -o batch_results.json
```

### mCODE Fetcher - Search and Process Trials

```bash
# Search and fetch trials
python mcode_fetcher.py --condition "breast cancer" --limit 10 -o results.json

# Search with concurrent mCODE processing
python mcode_fetcher.py --condition "breast cancer" --concurrent --process \
  --workers 8 -m deepseek-coder -p direct_mcode_evidence_based_concise -o results.json

# Fetch specific trials with processing
python mcode_fetcher.py --nct-ids "NCT00109785,NCT00616135" --process -m deepseek-coder -p direct_mcode_evidence_based_concise -o trials.json

# Count available studies
python mcode_fetcher.py --condition "cancer" --count-only

# Store processed trials in CORE Memory with automatic duplicate detection
python mcode_fetcher.py --condition "breast cancer" --process --store-in-core-memory -m deepseek-coder

# Store a specific trial in CORE Memory
python mcode_fetcher.py --nct-id NCT00616135 --process --store-in-core-memory -m deepseek-coder
```

### mCODE Patients - Filter Patient Data

```bash
# Single file processing
python mcode_patients.py --input-file data/fetcher_output/deepseek-coder.results.json \
  --patient-file patient_data.json -o patient_filtered.json

# Batch processing with directory recursion
python mcode_patients.py --input-file data/fetcher_output/deepseek-coder.results.json \
  --input-dir data/mcode_downloads --output-dir data/mcode_filtered --workers 4

# With CORE Memory storage
python mcode_patients.py --input-file data/fetcher_output/deepseek-coder.results.json \
  --patient-file patient_data.json --store-in-core-memory --verbose
```

### mCODE Optimize - Cross-Validation Testing

```bash
# Full optimization across all prompt×model combinations
python mcode_optimize.py --trials-file data/selected_breast_cancer_trials.json --concurrent --detailed-report

# Quick optimization with limited combinations
python mcode_optimize.py --trials-file data/selected_breast_cancer_trials.json --max-combinations 10 --concurrent

# Test specific models and prompts
python mcode_optimize.py --trials-file data/selected_breast_cancer_trials.json --models deepseek-coder,gpt-4o \
  --prompts direct_mcode_evidence_based_concise,direct_mcode_simple \
  --detailed-report

# Set logging level (verbose)
python mcode_optimize.py --trials-file data/selected_breast_cancer_trials.json --verbose
```

### Python API

```python
from src.pipeline import McodePipeline

# Initialize pipeline with evidence-based processing
pipeline = McodePipeline(
    prompt_name="direct_mcode_evidence_based_concise"
)

# Process trial data
result = pipeline.process_clinical_trial(trial_data)
print(f"Generated {len(result.mcode_mappings)} mCODE mappings")
print(f"Quality Score: {result.validation_results['compliance_score']:.3f}")
```

## 🎯 Key Features

- **99.1% Quality Score** - Evidence-based processing with strict textual fidelity
- **Single-Step Pipeline** - Direct clinical text to mCODE mapping
- **Conservative Mapping** - Quality over quantity approach
- **Source Provenance** - Complete audit trail for all mappings
- **Production Ready** - Clean architecture with comprehensive error handling

## 📁 Project Structure

```
mcode_translator/
├── mcode_fetcher.py                # Fetcher CLI - Search and fetch trials
├── mcode_optimize.py               # Optimize CLI - Cross-Validation Testing
├── mcode_patients.py               # Patients CLI - Filter patient mCODE data
├── mcode_translator.py             # Main CLI - Process trial data with mCODE
├── src/pipeline/                   # Core processing components
│   ├── mcode_pipeline.py          # Main mCODE processing pipeline
│   ├── llm_base.py            # LLM base
│   ├── concurrent_fetcher.py      # Concurrent processing engine
│   └── fetcher.py                 # Core trial fetching functions
├── prompts/                        # Evidence-based prompt templates
├── models/                         # LLM model configurations
├── data/                           # Configuration and reference data
└── tests/                          # Test suite
```

## 📈 Quality Metrics

| Metric | Score | Improvement |
|--------|-------|-------------|
| Overall Quality | 98.9% | +8.5 points |
| Source Text Fidelity | 99.2% | +8.5 points |
| Average Confidence | 98.2% | +8.6 points |
| Mapping Efficiency | 44.9% reduction in over-mapping | Quality over quantity |

*Validated across 5 comprehensive breast cancer clinical trials*

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Required: Choose one LLM provider
OPENAI_API_KEY=your_openai_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
# Optional: CORE Memory API key for storage
COREAI_API_KEY=your_coreai_api_key
```

### Model Settings (models/models_config.json)

```json
{
  "models": {
    "deepseek-coder": {
      "provider": "deepseek",
      "max_tokens": 8000,
      "temperature": 0.1
    }
  }
}
```

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## ⚙️ End-to-End Pipeline Guide

### Synthetic Patient Workflow

1. **Download Synthetic Data**:
```bash
python mcode_fetcher.py --download-synthetic-patients --cancer-type breast_cancer
```

2. **Process Patients Against Trials**:
```bash
# Filter synthetic patients based on breast cancer trial criteria
python mcode_patients.py --input-file breast_cancer_trials.json \
  --archive-path "breast_cancer/10_years" --limit 50 \
  --output-dir data/filtered_synthetic_patients --store-in-core-memory
```

3. **Validate Results**:
```bash
# Test the integration end-to-end
pytest tests/unit/test_mcode_patients.py::test_synthetic_patient_filtering -v
```

This workflow enables matching synthetic patients to clinical trial eligibility criteria using mCODE mappings, supporting patient-trial matching research and validation.

### Prerequisites

1. **Python 3.8+**: Ensure you have Python 3.8 or a later version installed.
2. **Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **API Keys**: Obtain API keys for your chosen LLM provider (e.g., OpenAI, DeepSeek) and set them as environment variables in a `.env` file.
4. **Configuration**: Configure the pipeline settings in `config.json` and `models/models_config.json`, including API keys, model names, and other parameters.

### Prerequisites

1. **Python 3.8+**: Ensure you have Python 3.8 or a later version installed.
2. **Dependencies**: Install the required dependencies using `pip install -r requirements.txt`.
3. **API Keys**: Obtain API keys for your chosen LLM provider (e.g., OpenAI, DeepSeek) and set them as environment variables in a `.env` file.
4. **Configuration**: Configure the pipeline settings in `config.json` and `models/models_config.json`, including API keys, model names, and other parameters.

### Step-by-Step Instructions

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/HadleyLab/mcode_translator.git
    cd mcode_translator
    ```

2. **Set Up Environment**:

    ```bash
    # Create conda environment
    conda create -n mcode_translator python=3.11
    conda activate mcode_translator
    # Install dependencies
    pip install -r requirements.txt
    ```

3. **Configure API Keys**:

    - Copy the `.env.example` file to `.env` and edit it to include your API keys:

        ```bash
        cp .env.example .env
        nano .env  # Or your favorite text editor
        ```

    - Ensure the `.env` file contains the necessary API keys for your chosen LLM provider:

        ```
        OPENAI_API_KEY=your_openai_api_key
        # Or
        DEEPSEEK_API_KEY=your_deepseek_api_key
        # Optional: CORE Memory API key for storage
        COREAI_API_KEY=your_coreai_api_key
        ```

4. **Fetch and Process Clinical Trials (mcode_fetcher.py)**:

    - Search for clinical trials and process them with mCODE mapping, storing results in CORE Memory:

        ```bash
        python mcode_fetcher.py --condition "breast cancer" --limit 10 --process \
          --store-in-core-memory -m deepseek-coder -p direct_mcode_evidence_based_concise -o fetcher_results.json
        ```

    - This command:
      - Searches ClinicalTrials.gov for "breast cancer" trials (limit 10).
      - Processes each trial with the mCODE pipeline using the specified model and prompt.
      - Stores processed trials in CORE Memory for later use.
      - Saves results to `fetcher_results.json`.

5. **Process Patient Data Against Trials (mcode_patients.py)**:

    - Filter patient mCODE data based on the clinical trial mappings stored in CORE Memory, and store filtered patient summaries in CORE Memory:

        ```bash
        python mcode_patients.py --input-file fetcher_results.json \
          --input-dir data/mcode_downloads --output-dir data/mcode_filtered \
          --store-in-core-memory --verbose
        ```

    - This command:
      - Uses the trial data from `fetcher_results.json` (which contains mCODE elements from step 4).
      - Recursively processes patient JSON files from `data/mcode_downloads`.
      - Filters patient mCODE elements to match those in the clinical trials.
      - Saves filtered patient data to `data/mcode_filtered/` (mirroring the input directory structure).
      - Stores filtered patient summaries in CORE Memory.
      - Verbose output shows extraction, filtering, and storage progress.

6. **View and Analyze Results**:

    - Examine the processed trial data in `fetcher_results.json`.
    - Review filtered patient data in `data/mcode_filtered/`.
    - Use the stored data in CORE Memory for querying or further analysis (via API or future tools).

### Workflows

1. **Basic mCODE Mapping**:

    - Use the `mcode_translator.py` script to map clinical trial data to mCODE elements:

        ```bash
        python mcode_translator.py --input-file data/selected_breast_cancer_trials.json -m deepseek-coder -o results.json
        ```

2. **Fetching and Processing Trials**:

    - Use the `mcode_fetcher.py` script to search for clinical trials and process them:

        ```bash
        python mcode_fetcher.py --condition "breast cancer" --limit 10 --process \
          --store-in-core-memory -m deepseek-coder -o fetcher_results.json
        ```

3. **Filtering Patient Data**:

    - Use the `mcode_patients.py` script to filter patient data based on trial mCODE elements:

        ```bash
        python mcode_patients.py --input-file fetcher_results.json \
          --input-dir data/mcode_downloads --output-dir data/mcode_filtered \
          --store-in-core-memory --verbose
        ```

4. **Optimization and Validation**:

    - Use the `mcode_optimize.py` script to test and optimize prompt/model combinations:

        ```bash
        python mcode_optimize.py --trials-file data/selected_breast_cancer_trials.json \
          --concurrent --detailed-report
        ```

### Error Handling

1. **Missing API Keys**:

    - If you encounter an error related to missing API keys, ensure you have set the `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` environment variables in your `.env` file.

2. **Configuration Errors**:

    - If you encounter an error related to invalid configuration settings, check the `config.json` and `models/models_config.json` files for any inconsistencies or missing values.

3. **Rate Limiting**:

    - If you encounter rate limiting errors, adjust the `rate_limiting` settings in the `config.json` file.

### Validation

1. **Compliance Scores**:

    - The pipeline generates compliance scores to indicate the quality and completeness of the mCODE mappings. A higher compliance score indicates better mapping quality.

2. **Error Reporting**:

    - The pipeline provides detailed error messages and warnings to help you identify and resolve any issues with the mCODE mappings.

### CORE Memory Integration

1. **Storing Data**:

    - To store processed trials and patient data in CORE Memory, use the `--store-in-core-memory` flag with `mcode_fetcher.py` and `mcode_patients.py`.

2. **Automatic Duplicate Detection**:

    - The pipeline automatically detects and prevents duplicate entries in CORE Memory based on trial and patient identifiers.

### Retrieving Data from CORE Memory

1. **Using CORE Memory Data**:

    - To retrieve data from CORE Memory, you can use the CORE Memory API or command-line tools (coming soon).

    - Once you have retrieved the data, you can use it as input for the `mcode_translator.py` script. For example:

        ```bash
        # Assuming you have retrieved data from CORE Memory and saved it to core_memory_data.json
        python mcode_translator.py --input-file core_memory_data.json -o results.json
        ```

    - The `mcode_translator.py` script will process the data from CORE Memory and generate mCODE mappings.

---

**mCODE Translator v2.0** - Clinical trial data to standardized mCODE elements with 99.1% quality score.
