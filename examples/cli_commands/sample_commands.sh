#!/bin/bash
# Sample CLI Commands for mCODE Translator
# Copy and modify these commands for your use cases

# Set environment variables (if using CORE Memory)
export HEYSOL_API_KEY="your-api-key-here"

# Basic trial processing with RegexEngine (fast)
python mcode-cli.py data ingest-trials \
  --cancer-type "breast" \
  --limit 5 \
  --engine "regex" \
  --batch-size 3

# Intelligent processing with LLMEngine
python mcode-cli.py data ingest-trials \
  --cancer-type "lung" \
  --limit 5 \
  --engine "llm" \
  --model "deepseek-coder" \
  --prompt "direct_mcode_evidence_based_concise"

# Compare both engines on a specific trial
python mcode-cli.py mcode summarize NCT02364999 --compare-engines

# Process patient data
python mcode-cli.py patients pipeline \
  --input-file "patient_data.ndjson" \
  --output-file "processed_patients.ndjson"

# Optimize processing parameters
python mcode-cli.py mcode optimize-trials \
  --trials-file "raw_trials.ndjson" \
  --cv-folds 3 \
  --output-dir "optimization_results"

# Check system status
python mcode-cli.py status

# View configuration
python mcode-cli.py config show

# Run comprehensive diagnostics
python mcode-cli.py doctor

# Memory operations (requires API key)
python mcode-cli.py memory stats
python mcode-cli.py memory search "breast cancer trials"
python mcode-cli.py memory ingest --data-file "processed_trials.ndjson"

# Batch processing with error recovery
python mcode-cli.py data ingest-trials \
  --trial-ids NCT123456 NCT789012 NCT345678 \
  --engine "regex" \
  --batch-size 2 \
  --max-retries 3

# Export results in different formats
python mcode-cli.py mcode summarize NCT02364999 \
  --output "trial_summary.json" \
  --format "json"

# Verbose logging for debugging
python mcode-cli.py data ingest-trials \
  --cancer-type "breast" \
  --limit 2 \
  --verbose \
  --log-level "DEBUG"