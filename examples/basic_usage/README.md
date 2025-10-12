# Basic Trial Processing Example

This example demonstrates the fundamental usage of the mCODE Translator for processing a single clinical trial and extracting mCODE elements.

## What You'll Learn

- Basic trial data fetching from ClinicalTrials.gov
- Simple mCODE extraction using the regex engine
- Result validation and output formatting
- Understanding processing metrics

## Quick Start

```bash
cd examples/basic_usage
python basic_trial_processing.py
```

## Expected Output

```
🚀 mCODE Translator - Basic Trial Processing
============================================================

📋 Configuration:
   • Engine: regex
   • Validation: True
   • Storage: False

🎯 Processing Trial: NCT02364999
────────────────────────────────────────

✅ Processing completed successfully!

📊 Trial Information:
   • NCT ID: NCT02364999
   • Title: PALOMA-2: Phase III Trial of Palbociclib (PD-03329901) Plus Letrozole Versus Placebo Plus Letrozole for ER+...
   • Phase: Phase 3
   • Condition: Breast Cancer

🧬 mCODE Elements Extracted:
   1. CancerCondition (95.0% confidence)
   2. CancerTreatment (92.0% confidence)
   3. TrialPhase (100.0% confidence)
   ... and 2 more

⚡ Processing Metrics:
   • Processing Time: 0.15s
   • Elements Extracted: 5

✅ Validation Results:
   • Compliance Score: 95.0%
   • Errors: 0

🎉 Basic trial processing example completed!
```

## Configuration Options

The example uses these basic settings:

- **Engine**: `regex` (fast, deterministic processing)
- **Validation**: `True` (validate extracted elements)
- **Storage**: `False` (don't store in CORE Memory for demo)

## Files in This Example

- `basic_trial_processing.py` - Main example script
- `README.md` - This documentation

## Next Steps

After running this example, try:

1. **Different Trial**: Change the NCT ID to process other trials
2. **LLM Engine**: Modify `processing_engine` to `"llm"` for AI-powered extraction
3. **Multiple Trials**: Process several trials at once
4. **Storage**: Enable `store_results` to save to CORE Memory

## Troubleshooting

- **Import Errors**: Ensure you're running from the project root directory
- **Network Issues**: Check your internet connection for ClinicalTrials.gov API
- **Trial Not Found**: Verify the NCT ID exists and is accessible