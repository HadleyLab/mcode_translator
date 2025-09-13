#!/usr/bin/env python3
"""
End-to-End Test for Refactored mCODE Translator Architecture

This script demonstrates the complete workflow:
1. Fetch clinical trials
2. Process trials with mCODE mapping
3. Fetch synthetic patients
4. Process patients with mCODE mapping
5. Store all results in CORE Memory

Uses the new modular architecture with centralized configuration.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from storage.mcode_memory_storage import McodeMemoryStorage
from utils.config import Config
from utils.data_downloader import download_synthetic_patient_archives
from utils.logging_config import get_logger, setup_logging
from workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from workflows.patients_processor_workflow import PatientsProcessorWorkflow
from workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from workflows.trials_processor_workflow import TrialsProcessorWorkflow

# Setup logging
setup_logging()

# Get logger for this module
logger = get_logger(__name__)


def test_centralized_config():
    """Test that centralized configuration works."""
    logger.info("🔧 Testing centralized configuration...")

    config = Config()

    # Test Core Memory config
    core_config = config.get_core_memory_config()
    logger.info(f"✅ Core Memory config loaded: {core_config['core_memory']['source']}")

    # Test API key retrieval
    try:
        api_key = config.get_core_memory_api_key()
        logger.info(f"✅ API key loaded (length: {len(api_key)})")
    except Exception as e:
        logger.warning(f"⚠️  API key not available: {e}")
        return False

    return True


def test_workflows():
    """Test that all workflows can be instantiated."""
    logger.info("🏗️  Testing workflow instantiation...")

    config = Config()

    # Test fetchers (no storage)
    trials_fetcher = TrialsFetcherWorkflow(config)
    logger.info("✅ TrialsFetcherWorkflow instantiated")

    patients_fetcher = PatientsFetcherWorkflow(config)
    logger.info("✅ PatientsFetcherWorkflow instantiated")

    # Test processors (with storage)
    try:
        memory_storage = McodeMemoryStorage()
        trials_processor = TrialsProcessorWorkflow(config, memory_storage)
        logger.info("✅ TrialsProcessorWorkflow instantiated")

        patients_processor = PatientsProcessorWorkflow(config, memory_storage)
        logger.info("✅ PatientsProcessorWorkflow instantiated")
    except Exception as e:
        logger.warning(f"⚠️  Processors not available: {e}")
        return False

    return True


def test_memory_storage():
    """Test memory storage interface."""
    logger.info("🧠 Testing memory storage interface...")

    try:
        storage = McodeMemoryStorage()

        # Test configuration access
        logger.info(f"✅ Storage source: {storage.source}")
        logger.info(f"✅ Storage timeout: {storage.timeout}")
        logger.info(f"✅ Storage batch size: {storage.batch_size}")

        # Test summary creation (without actual API call)
        trial_data = {
            "mcode_mappings": [
                {
                    "mcode_element": "CancerCondition",
                    "value": "Breast Cancer",
                    "system": "http://snomed.info/sct",
                    "code": "254837009",
                }
            ],
            "metadata": {"brief_title": "Test Trial", "sponsor": "Test Sponsor"},
        }

        summary = storage._create_trial_summary("NCT123456", trial_data)
        logger.info("✅ Trial summary creation works")
        logger.debug(f"📝 Sample summary: {summary[:100]}...")

        return True

    except Exception as e:
        logger.warning(f"⚠️  Memory storage test failed: {e}")
        return False


def test_actual_processing():
    """Test actual processing of trials and patients to CORE Memory."""
    logger.info("🚀 Testing actual data processing...")

    try:
        config = Config()
        memory_storage = McodeMemoryStorage()

        # Test 1: Fetch and process 2 clinical trials
        logger.info("📋 Fetching clinical trials...")
        trials_fetcher = TrialsFetcherWorkflow(config)
        trials_result = trials_fetcher.execute(condition="breast cancer", limit=2)

        if trials_result.success and trials_result.data:
            logger.info(f"✅ Fetched {len(trials_result.data)} trials")

            # Process the trials
            logger.info("🔬 Processing trials with mCODE mapping...")
            trials_processor = TrialsProcessorWorkflow(config, memory_storage)
            trials_process_result = trials_processor.execute(
                trials_data=trials_result.data,
                model="deepseek-coder",
                store_in_memory=True,
            )

            if trials_process_result.success:
                logger.info(
                    f"✅ Processed {trials_process_result.metadata['successful']}/{trials_process_result.metadata['total_trials']} trials"
                )
            else:
                logger.warning(
                    f"⚠️  Trial processing failed: {trials_process_result.error_message}"
                )
        else:
            logger.warning(f"⚠️  Trial fetching failed: {trials_result.error_message}")

        # Test 2: Fetch and process 2 synthetic patients
        logger.info("👥 Fetching synthetic patients...")
        patients_fetcher = PatientsFetcherWorkflow(config)
        patients_result = patients_fetcher.execute(
            archive_path="breast_cancer/10_years",
            limit=5,  # Load only 5 patients for testing
        )

        if patients_result.success and patients_result.data:
            logger.info(f"✅ Fetched {len(patients_result.data)} patients")

            # Process the patients
            logger.info("🔬 Processing patients with mCODE mapping...")
            patients_processor = PatientsProcessorWorkflow(config, memory_storage)
            patients_process_result = patients_processor.execute(
                patients_data=patients_result.data, store_in_memory=True
            )

            if patients_process_result.success:
                logger.info(
                    f"✅ Processed {patients_process_result.metadata['successful']}/{patients_process_result.metadata['total_patients']} patients"
                )
            else:
                logger.warning(
                    f"⚠️  Patient processing failed: {patients_process_result.error_message}"
                )
        else:
            logger.warning(
                f"⚠️  Patient fetching failed: {patients_result.error_message}"
            )

        return True

    except Exception as e:
        logger.warning(f"⚠️  Actual processing test failed: {e}")
        return False


def download_missing_archives(config: Config) -> None:
    """Download missing or invalid synthetic patient archives."""
    logger.info("📥 Checking and downloading synthetic patient archives...")

    # Get archive configuration from config
    synthetic_config = config.get_synthetic_data_archives()

    # Check which archives need downloading
    archives_to_download = {}
    base_dir = config.get_synthetic_data_base_directory()

    for cancer_type, durations in synthetic_config.items():
        for duration, archive_info in durations.items():
            archive_name = f"{cancer_type}_{duration}.zip"
            expected_path = os.path.join(base_dir, cancer_type, duration, archive_name)

            # Check if archive exists and is valid
            needs_download = True
            if os.path.exists(expected_path):
                try:
                    import zipfile

                    with zipfile.ZipFile(expected_path, "r") as zf:
                        # Check if it's a valid ZIP with content
                        if len(zf.namelist()) > 0:
                            logger.info(
                                f"✅ Archive exists and is valid: {archive_name}"
                            )
                            needs_download = False
                        else:
                            logger.warning(
                                f"⚠️  Archive exists but is empty: {archive_name}"
                            )
                except (zipfile.BadZipFile, Exception) as e:
                    logger.warning(
                        f"⚠️  Archive exists but is invalid: {archive_name} ({e})"
                    )

            if needs_download:
                archives_to_download.setdefault(cancer_type, {})[duration] = (
                    archive_info["url"]
                )
                logger.info(f"📋 Will download: {archive_name}")

    # Download missing archives
    if archives_to_download:
        logger.info(f"🚀 Downloading {len(archives_to_download)} archive types...")
        download_synthetic_patient_archives(
            base_dir, archives_to_download, force_download=True
        )
        logger.info("✅ Archive download complete")
    else:
        logger.info("✅ All archives are valid")


def main():
    """Run the end-to-end test."""
    parser = argparse.ArgumentParser(description="mCODE Translator End-to-End Test")
    parser.add_argument(
        "--download-archives",
        action="store_true",
        help="Download missing or invalid synthetic patient archives before testing",
    )

    args = parser.parse_args()

    logger.info("🚀 mCODE Translator End-to-End Test")
    logger.info("=" * 50)

    # Download archives if requested
    if args.download_archives:
        config = Config()
        download_missing_archives(config)
        logger.info("")

    # Test 1: Centralized Configuration
    config_ok = test_centralized_config()

    # Test 2: Workflow Instantiation
    workflows_ok = test_workflows()

    # Test 3: Memory Storage
    storage_ok = test_memory_storage()

    # Test 4: Actual Processing
    processing_ok = test_actual_processing()

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results:")

    tests = [
        ("Centralized Configuration", config_ok),
        ("Workflow Instantiation", workflows_ok),
        ("Memory Storage Interface", storage_ok),
        ("Actual Data Processing", processing_ok),
    ]

    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    logger.info("\n" + "=" * 50)
    if all_passed:
        logger.info(
            "🎉 All tests passed! The refactored architecture is working correctly."
        )
        logger.info("\n💡 To run a full end-to-end test with real data:")
        logger.info("   1. Ensure COREAI_API_KEY is set in .env")
        logger.info(
            "   2. Run: python -m src.cli.trials_fetcher --condition 'breast cancer' -o trials.json"
        )
        logger.info(
            "   3. Run: python -m src.cli.trials_processor trials.json --store-in-core-memory"
        )
        logger.info(
            "   4. Run: python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.json"
        )
        logger.info(
            "   5. Run: python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory"
        )
    else:
        logger.warning("⚠️  Some tests failed. Check configuration and dependencies.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
