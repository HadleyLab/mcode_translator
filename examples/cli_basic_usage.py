#!/usr/bin/env python3
"""
mCODE Translator CLI - Basic Usage Examples

This script provides simple, copy-paste ready examples for common CLI operations.

Usage:
    python examples/cli_basic_usage.py

Or copy individual commands to run them manually.
"""

import os
import sys

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print('='*60)


def print_data_download_info():
    """Print information about data downloads."""
    print("""
📥 DATA DOWNLOAD REQUIRED for patient workflows:

Before using patient-related commands, download the synthetic data:

1. Complete setup (recommended):
   python examples/setup_and_demo.py

2. Manual download:
   python -c "from src.utils.data_downloader import download_synthetic_patient_archives; download_synthetic_patient_archives()"

3. Check available archives:
   python -m src.cli.patients_fetcher --list-archives

The download includes:
• breast_cancer_10_years (~320MB)
• breast_cancer_lifetime (~620MB)
• mixed_cancer_10_years (~450MB)
• mixed_cancer_lifetime (~850MB)
    """)

def print_command(description: str, command: str):
    """Print a formatted command with description."""
    print(f"\n🔧 {description}")
    print(f"   {command}")

def main():
    """Display basic CLI usage examples."""
    print("🧪 mCODE Translator CLI - Basic Usage Examples")
    print("="*60)
    print("Copy and paste these commands to get started!")
    print("Make sure you're in the mcode_translator conda environment.")

    # Environment check
    if "CONDA_DEFAULT_ENV" not in os.environ or "mcode_translator" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("\n⚠️  WARNING: Not in mcode_translator environment!")
        print("   Run: conda activate mcode_translator")
        print("   Then re-run this script.")

    print_section("1. LIST AVAILABLE OPTIONS")

    print_command(
        "List available patient data archives",
        "python -m src.cli.patients_fetcher --list-archives"
    )

    print_command(
        "List available prompt templates",
        "python -m src.cli.trials_optimizer --list-prompts"
    )

    print_command(
        "List available LLM models",
        "python -m src.cli.trials_optimizer --list-models"
    )

    print_section("2. FETCH CLINICAL TRIALS")

    print_command(
        "Fetch trials by condition (basic)",
        "python -m src.cli.trials_fetcher --condition 'breast cancer' -o trials.json"
    )

    print_command(
        "Fetch specific trial by NCT ID",
        "python -m src.cli.trials_fetcher --nct-id NCT03170960 -o trial.json"
    )

    print_command(
        "Fetch multiple trials with verbose output",
        "python -m src.cli.trials_fetcher --nct-ids NCT03170960,NCT03805399 -o trials.json --verbose"
    )

    print_section("3. FETCH PATIENT DATA")

    # Add data download info
    print_data_download_info()

    print_command(
        "Fetch patients from breast cancer archive",
        "python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.json"
    )

    print_command(
        "Fetch specific patient by ID",
        "python -m src.cli.patients_fetcher --archive breast_cancer_10_years --patient-id patient_001 -o patient.json"
    )

    print_command(
        "Fetch limited patients with verbose logging",
        "python -m src.cli.patients_fetcher --archive mixed_cancer_lifetime --limit 10 -o patients.json --verbose"
    )

    print_section("4. PROCESS TRIALS WITH MCODE")

    print_command(
        "Process trials with default settings",
        "python -m src.cli.trials_processor trials.json --store-in-core-memory"
    )

    print_command(
        "Process with specific model and prompt",
        "python -m src.cli.trials_processor trials.json --model deepseek-coder --prompt direct_mcode_evidence_based_concise --store-in-core-memory"
    )

    print_command(
        "Dry run to preview processing",
        "python -m src.cli.trials_processor trials.json --dry-run --verbose"
    )

    print_section("5. PROCESS PATIENTS WITH MCODE")

    print_command(
        "Process patients with trial eligibility filtering",
        "python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory"
    )

    print_command(
        "Process patients only (no trial filtering)",
        "python -m src.cli.patients_processor --patients patients.json --store-in-core-memory"
    )

    print_command(
        "Preview patient processing (dry run)",
        "python -m src.cli.patients_processor --patients patients.json --dry-run --verbose"
    )

    print_section("6. OPTIMIZE PARAMETERS")

    print_command(
        "Optimize with default settings",
        "python -m src.cli.trials_optimizer --trials-file trials.json"
    )

    print_command(
        "Test specific prompt and model combinations",
        "python -m src.cli.trials_optimizer --trials-file trials.json --prompts direct_mcode_evidence_based_concise,direct_mcode_minimal --models deepseek-coder,gpt-4 --max-combinations 4"
    )

    print_command(
        "Save optimal configuration",
        "python -m src.cli.trials_optimizer --trials-file trials.json --save-config optimal_config.json --verbose"
    )

    print_section("7. COMPLETE WORKFLOWS")

    print("""
🔄 Complete Workflow Example:
# 1. Fetch trials
python -m src.cli.trials_fetcher --condition "breast cancer" --limit 5 -o trials.json

# 2. Process trials
python -m src.cli.trials_processor trials.json --store-in-core-memory

# 3. Fetch patients
python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 10 -o patients.json

# 4. Process patients with trial criteria
python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory

# 5. Optimize parameters (optional)
python -m src.cli.trials_optimizer --trials-file trials.json --save-config config.json
    """)

    print_section("8. TROUBLESHOOTING")

    print("""
🔧 Common Issues & Solutions:

1. "Module not found" errors:
   - Ensure you're in the mcode_translator conda environment
   - Run: conda activate mcode_translator

2. "CORE Memory connection failed":
   - Set your COREAI_API_KEY environment variable
   - Check core_memory_config.json settings

3. "File not found" errors:
   - Use absolute paths or run from project root
   - Check that input files exist: ls -la <filename>

4. Verbose logging for debugging:
   - Add --verbose to any command for detailed output

5. Dry run to test without side effects:
   - Add --dry-run to processing commands

📚 For more help:
- Run any command with --help for detailed options
- Check the README.md for setup instructions
- Review logs in the terminal output
    """)

    print("\n" + "="*60)
    print("🎉 Ready to start using mCODE Translator CLI!")
    print("Copy any command above and run it in your terminal.")
    print("="*60)

if __name__ == "__main__":
    main()