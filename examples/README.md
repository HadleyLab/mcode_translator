# mCODE Translator - Complete Pipeline Examples

This directory contains streamlined examples demonstrating the complete mCODE workflow from data fetching to CORE memory storage.

## 📓 Current Jupyter Notebook Example

The primary example is the **complete mCODE translator workflow**:

- [`mcode_translator_complete_workflow.ipynb`](mcode_translator_complete_workflow.ipynb) - A self-contained, end-to-end demonstration with:
  - **Self-Contained Execution**: Downloads patient archives automatically.
  - **Concurrent Processing**: Uses 5 workers for fetching and processing.
  - **Optimized for Breast Cancer**: Determines and uses the best model and prompt combination.
  - **Native IPython Commands**: Uses `!` for streamlined shell command execution.
  - **Comprehensive Explanations**: Aligns with project documentation to explain each step.

### Obsolete Notebooks

The following notebooks have been marked as obsolete and are no longer maintained:
- `obsolete_core_ingest_from_uploaded_files.ipynb`
- `obsolete_enhanced_mcode_demo.ipynb`
- `obsolete_mcode_pipeline_demo.ipynb`
- `obsolete_comprehensive_mcode_demo.ipynb`

Please use `mcode_translator_complete_workflow.ipynb` for the most current and complete example.

## � Quick Start

### Single Command Complete Pipeline

The easiest way to get started is using the `mcode_translate.py` convenience script:

```bash
# Activate the conda environment
source activate mcode_translator

# Process breast cancer data (trials + patients)
python mcode_translate.py --condition "breast cancer" --limit 5

# Process with custom model and limits
python mcode_translate.py \
  --condition "lung cancer" \
  --limit 10 \
  --ingest \
  --verbose
```

## 📋 CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--condition` | Medical condition to search for (required) | - |
| `--nct-ids` | Specific NCT IDs to process | - |
| `--limit` | Number of trials/patients to process | `3` |
| `--optimize` | Run parameter optimization first | `false` |
| `--ingest` | Store results in CORE Memory | `false` |
| `--verbose` | Enable verbose logging | `false` |

## 🔄 Complete Workflow

The `mcode_translate.py` script performs these steps automatically:

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
python mcode_translate.py --condition "breast cancer" --ingest
```

### File Output Mode
```bash
# Process and save results to files (no CORE memory storage)
python mcode_translate.py --condition "breast cancer"
```

### Logging Control
```bash
# Verbose logging for debugging
python mcode_translate.py --condition "breast cancer" --verbose
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
python mcode_translate.py --condition "prostate cancer" --optimize
```

### High-Volume Processing
```bash
# Process large datasets with more workers
python mcode_translate.py \
  --condition "diabetes" \
  --limit 50 \
  --ingest \
  --verbose
```

## 🏗️ Architecture

The `mcode_translate.py` script integrates multiple components:

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
python mcode_translate.py --condition "test" --verbose
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