#!/usr/bin/env python3
"""
mCODE Translator - Complete Setup and Demo

This script demonstrates the complete setup process including:
1. Downloading synthetic patient data archives
2. Setting up the environment
3. Running end-to-end CLI workflows

Usage:
    python examples/setup_and_demo.py

Requirements:
    - conda environment: mcode_translator
    - internet connection for data downloads
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.utils.data_downloader import download_synthetic_patient_archives, get_archive_paths


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print('='*60)


def print_step(step_num: int, description: str):
    """Print a formatted step."""
    print(f"\n🔧 Step {step_num}: {description}")
    print("-" * 40)


def check_environment():
    """Check if we're in the correct environment."""
    print_section("Environment Check")

    # Check conda environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if "mcode_translator" in conda_env:
        print("✅ Conda environment: mcode_translator")
    else:
        print("⚠️  Warning: Not in mcode_translator conda environment")
        print("   Run: conda activate mcode_translator")
        return False

    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        print(f"✅ Python version: {python_version.major}.{python_version.minor}")
    else:
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor} (need >= 3.10)")
        return False

    # Check required modules
    try:
        import requests
        print("✅ requests library available")
    except ImportError:
        print("❌ requests library not found")
        return False

    return True


def download_data(force_download=False):
    """Download synthetic patient data archives."""
    print_section("Data Download")

    if force_download:
        print("📥 Force downloading synthetic patient data archives...")
        print("   This will re-download all archives even if they exist.")
    else:
        print("📥 Downloading synthetic patient data archives...")
        print("   This may take several minutes depending on your connection.")
        print("   Note: Existing archives will be skipped. Use --force to re-download.")

    try:
        downloaded = download_synthetic_patient_archives(force_download=force_download)

        if downloaded:
            print("✅ Successfully downloaded archives:")
            for name, path in downloaded.items():
                file_size = os.path.getsize(path) / (1024 * 1024)  # MB
                print(f"   • {name}: {file_size:.1f} MB")
        else:
            print("ℹ️  No new archives to download")

        # Show all available archives
        all_archives = get_archive_paths()
        if all_archives:
            print("\n📚 Available archives:")
            for name, path in all_archives.items():
                print(f"   • {name}")

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def run_cli_demo():
    """Run CLI demonstration commands."""
    print_section("CLI Demo")

    # Check if we have data
    archives = get_archive_paths()
    if not archives:
        print("❌ No data archives found. Please run data download first.")
        return False

    print("🚀 Running CLI demonstrations...")

    # Demo 1: List archives
    print_step(1, "List Available Archives")
    os.system("python -m src.cli.patients_fetcher --list-archives")

    # Demo 2: Fetch trials
    print_step(2, "Fetch Clinical Trials")
    os.system("python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 2 -o demo_trials.json --verbose")

    # Demo 3: Fetch patients (if data available)
    if any('breast_cancer_10_years' in name for name in archives.keys()):
        print_step(3, "Fetch Patient Data")
        os.system("python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 3 -o demo_patients.json --verbose")
    else:
        print_step(3, "Patient Data Fetch (Data Not Available)")
        print("   Skipping patient fetch - breast_cancer_10_years archive not found")

    # Demo 4: Show help
    print_step(4, "Show CLI Help")
    print("Available CLI commands:")
    print("• python -m src.cli.trials_fetcher --help")
    print("• python -m src.cli.trials_processor --help")
    print("• python -m src.cli.patients_fetcher --help")
    print("• python -m src.cli.patients_processor --help")
    print("• python -m src.cli.trials_optimizer --help")

    return True


def show_next_steps():
    """Show next steps for full functionality."""
    print_section("Next Steps for Full Functionality")

    print("""
🔑 To enable full functionality:

1. Set up API keys for LLM services:
   export COREAI_API_KEY="your-api-key-here"

2. Configure LLM models in src/config/llms_config.json

3. For processing trials:
   python -m src.cli.trials_processor demo_trials.json --store-in-core-memory

4. For processing patients (if data downloaded):
   python -m src.cli.patients_processor --patients demo_patients.json --trials demo_trials.json --store-in-core-memory

5. For optimization:
   python -m src.cli.trials_optimizer --trials-file demo_trials.json --save-config optimal.json

📚 Full documentation: examples/README_CLI.md
🛠️  Troubleshooting: Check examples/cli_basic_usage.py
    """)


def main():
    """Main setup and demo function."""
    import sys

    # Parse command line arguments
    force_download = "--force" in sys.argv

    print("🧪 mCODE Translator - Complete Setup and Demo")
    print("=" * 60)

    if force_download:
        print("🔄 Force download mode enabled - will re-download all archives")
        print()

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix issues above and try again.")
        return 1

    # Download data
    if not download_data(force_download=force_download):
        print("\n⚠️  Data download had issues, but continuing with demo...")

    # Run CLI demo
    run_cli_demo()

    # Show next steps
    show_next_steps()

    print("\n" + "=" * 60)
    print("🎉 Setup and demo completed!")
    print("Check the generated files and examples for more details.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())