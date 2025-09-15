# mCODE Translator - End-to-End Examples

This directory contains streamlined examples demonstrating the complete mCODE workflow from data fetching to CORE memory storage.

## 🚀 Quick Start

### Single Command End-to-End Processing

The easiest way to get started is using the new end-to-end processor CLI:

```bash
# Activate the conda environment
source activate mcode_translator

# Process breast cancer data (trials + patients)
python -m src.cli.end_to_end_processor --condition "breast cancer" --trials-limit 5 --patients-limit 5

# Process with custom model and limits
python -m src.cli.end_to_end_processor \
  --condition "lung cancer" \
  --model deepseek-coder \
  --trials-limit 10 \
  --patients-limit 10 \
  --workers 4 \
  --verbose
```

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--condition` | Medical condition to search for (required) | - |
| `--model` | LLM model for mCODE processing | `deepseek-coder` |
| `--trials-limit` | Number of clinical trials to fetch | `5` |
| `--patients-limit` | Number of patients to fetch | `5` |
| `--workers` | Number of concurrent workers | `2` |
| `--dry-run` | Preview mode - process but don't store | `false` |
| `--store-in-core-memory` | Explicitly enable CORE memory storage | `true` |
| `--verbose` | Enable verbose logging | `false` |
| `--quiet` | Disable most logging output | `false` |

## 🔄 Complete Workflow

The end-to-end processor performs these steps automatically:

1. **🔬 Fetch Clinical Trials** - Searches ClinicalTrials.gov for the specified condition
2. **🧪 Process Trials** - Applies mCODE mapping with detailed codes (SNOMED, MeSH, etc.)
3. **👥 Fetch Patients** - Retrieves synthetic patients from appropriate archives
4. **🩺 Process Patients** - Applies mCODE mapping with detailed codes
5. **💾 Store in CORE Memory** - Saves all processed data for future retrieval

## 📊 Example Output

```
🚀 STARTING END-TO-END PROCESSING FOR CONDITION: breast cancer
📊 Configuration: trials_limit=1, patients_limit=1, workers=2

🔬 STEP 1: Fetching clinical trials...
✅ Fetched 1 trials

🧪 STEP 2: Processing trials with mCODE mapping...
✅ Processed 1 trials

👥 STEP 3: Fetching synthetic patients...
✅ Fetched 1 patients

🩺 STEP 4: Processing patients with mCODE mapping...
✅ Processed 1 patients

📊 END-TO-END PROCESSING SUMMARY
   Condition: breast cancer
   Trials: 1 processed, 1 stored
   Patients: 1 processed, 1 stored

🎉 END-TO-END PROCESSING COMPLETE
✅ All processing completed successfully!
```

## 🔧 Memory & Storage Options

The CLI provides flexible memory storage options:

### Default Behavior (Store in CORE Memory)
```bash
# Default: stores results in CORE memory
python -m src.cli.end_to_end_processor --condition "breast cancer"
```

### Preview Mode (Dry Run)
```bash
# Process data but don't store in CORE memory
python -m src.cli.end_to_end_processor --condition "breast cancer" --dry-run
```

### Explicit Storage Control
```bash
# Explicitly enable storage (same as default)
python -m src.cli.end_to_end_processor --condition "breast cancer" --store-in-core-memory
```

### Logging Control
```bash
# Verbose logging for debugging
python -m src.cli.end_to_end_processor --condition "breast cancer" --verbose

# Quiet mode (minimal logging)
python -m src.cli.end_to_end_processor --condition "breast cancer" --quiet
```

## 🎯 Key Features

### Active Sentence Structure
- Clinical features serve as subjects in sentences
- Better NLP processing and entity extraction
- Example: `"Trial cancer conditions include Estrogen Receptor-positive Breast Cancer (SNOMED:417742003)"`

### Detailed Code Integration
- **SNOMED codes**: For conditions, procedures, and outcomes
- **MeSH codes**: For medications and interventions
- **RxNorm codes**: For medications
- **LOINC codes**: For laboratory observations
- **ICD codes**: For diagnoses

### CORE Memory Integration
- All processed data automatically stored in CORE memory
- Enables cross-session continuity and knowledge persistence
- Supports future retrieval and analysis

## 📁 Available Patient Archives

The system includes these synthetic patient archives:

- `breast_cancer_10_years` - Breast cancer patients over 10-year period
- `breast_cancer_lifetime` - Breast cancer patients lifetime data
- `mixed_cancer_10_years` - Mixed cancer types over 10-year period
- `mixed_cancer_lifetime` - Mixed cancer types lifetime data

## 🔧 Advanced Usage

### Custom Model Selection
```bash
# Use different LLM models
python -m src.cli.end_to_end_processor --condition "prostate cancer" --model gpt-4
```

### High-Volume Processing
```bash
# Process large datasets with more workers
python -m src.cli.end_to_end_processor \
  --condition "diabetes" \
  --trials-limit 50 \
  --patients-limit 100 \
  --workers 8 \
  --verbose
```

### Preview Mode (Dry Run)
```bash
# Process data but skip CORE memory storage for testing
python -m src.cli.end_to_end_processor \
  --condition "hypertension" \
  --dry-run
```

## 🏗️ Architecture

The end-to-end processor integrates multiple components:

- **Workflows**: Modular processing pipelines
- **Services**: Core business logic (summarization, mCODE mapping)
- **Storage**: CORE memory persistence
- **Utils**: Configuration, logging, API management

## 📈 Performance Notes

- **Caching**: API responses cached to reduce external calls
- **Concurrency**: Multiple workers for parallel processing
- **Memory Management**: Efficient processing of large datasets
- **Error Handling**: Comprehensive error reporting and recovery

## 🐛 Troubleshooting

### Common Issues

**"Model name is required"**
- Ensure `--model` parameter is provided or use default

**"Archive path is required"**
- Patient archives must exist in `data/synthetic_patients/`
- Use available archives listed above

**"Failed to fetch trials"**
- Check internet connection
- Verify ClinicalTrials.gov API availability
- Try reducing `--trials-limit`

### Debug Mode
```bash
# Enable verbose logging for troubleshooting
python -m src.cli.end_to_end_processor --condition "test" --verbose
```

## 🎉 Success Metrics

After successful processing, you should see:
- ✅ Clinical trials fetched and processed
- ✅ Synthetic patients retrieved and mapped
- ✅ mCODE elements with detailed codes
- ✅ Data stored in CORE memory
- ✅ Comprehensive logging output

## 📚 Next Steps

1. **Explore CORE Memory**: Query stored data across sessions
2. **Custom Processing**: Modify workflows for specific use cases
3. **Batch Processing**: Process multiple conditions programmatically
4. **Integration**: Connect with downstream analysis tools

---

**Need Help?** Check the logs for detailed error messages and refer to the main project documentation.