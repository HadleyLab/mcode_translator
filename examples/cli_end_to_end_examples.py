#!/usr/bin/env python3
"""
mCODE Translator CLI - End-to-End Usage Examples

This script demonstrates complete end-to-end workflows using the mCODE Translator CLI.
It shows how to fetch, process, and store clinical trial and patient data.

Usage:
    python examples/cli_end_to_end_examples.py

Requirements:
    - Set COREAI_API_KEY environment variable for memory storage
    - Ensure you're in the mcode_translator conda environment
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Project root for relative paths
PROJECT_ROOT = Path(__file__).parent.parent

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Use centralized logging
from src.utils.logging_config import setup_logging, get_logger

# Setup centralized logging
setup_logging()
logger = get_logger("cli_end_to_end_examples")


def run_command(cmd: str, cwd: Optional[Path] = None, env: Optional[dict] = None) -> bool:
    """Run a CLI command and return success status."""
    try:
        working_dir = cwd or PROJECT_ROOT
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        logger.info(f"\n🔧 Running: {cmd}")
        logger.info(f"   📁 Working directory: {working_dir}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            env=full_env,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )

        logger.info("✅ Command completed successfully!")
        if result.stdout.strip():
            logger.info(f"📄 Output:\n{result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            logger.info(f"📄 stdout:\n{e.stdout}")
        if e.stderr:
            logger.info(f"📄 stderr:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False


def example_1_basic_trial_workflow():
    """Example 1: Basic clinical trial fetching and processing."""
    logger.info("\n" + "="*80)
    logger.info("📋 EXAMPLE 1: Basic Clinical Trial Workflow")
    logger.info("="*80)

    # Step 1: Fetch trials
    logger.info("\n🔍 Step 1: Fetch clinical trials")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 3 -o examples/data/fetched_trials.json --verbose"
    )

    if not success:
        logger.error("❌ Trial fetching failed, skipping processing step")
        return False

    # Step 2: Process trials
    logger.info("\n🔬 Step 2: Process trials with mCODE mapping")
    success = run_command(
        "python -m src.cli.trials_processor examples/data/fetched_trials.json --model deepseek-coder --prompt direct_mcode_evidence_based_concise --store-in-core-memory --verbose"
    )

    return success


def example_2_patient_workflow():
    """Example 2: Patient data fetching and processing."""
    logger.info("\n" + "="*80)
    logger.info("📋 EXAMPLE 2: Patient Data Workflow")
    logger.info("="*80)

    # Step 1: Fetch patients
    logger.info("\n🔍 Step 1: Fetch synthetic patient data")
    success = run_command(
        "python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 5 -o examples/data/fetched_patients.json --verbose"
    )

    if not success:
        logger.error("❌ Patient fetching failed, skipping processing step")
        return False

    # Step 2: Fetch trials for criteria
    logger.info("\n🔍 Step 2: Fetch trial criteria")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 2 -o examples/data/trial_criteria.json --verbose"
    )

    if not success:
        logger.error("❌ Trial criteria fetching failed, skipping patient processing")
        return False

    # Step 3: Process patients with trial criteria
    logger.info("\n🔬 Step 3: Process patients with trial eligibility filtering")
    success = run_command(
        "python -m src.cli.patients_processor --patients examples/data/fetched_patients.json --trials examples/data/trial_criteria.json --store-in-core-memory --verbose"
    )

    return success


def example_3_optimization_workflow():
    """Example 3: Parameter optimization workflow."""
    logger.info("\n" + "="*80)
    logger.info("📋 EXAMPLE 3: Parameter Optimization Workflow")
    logger.info("="*80)

    # Step 1: Fetch sample trials for optimization
    logger.info("\n🔍 Step 1: Fetch sample trials for optimization")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition cancer --limit 5 -o examples/data/optimization_trials.json --verbose"
    )

    if not success:
        logger.error("❌ Sample data fetching failed, skipping optimization")
        return False

    # Step 2: Run optimization
    logger.info("\n🎯 Step 2: Optimize mCODE translation parameters")
    success = run_command(
        "python -m src.cli.trials_optimizer --trials-file examples/data/optimization_trials.json --prompts direct_mcode_evidence_based_concise,direct_mcode_minimal --models deepseek-coder,gpt-4 --max-combinations 4 --save-config examples/data/optimized_config.json --verbose"
    )

    return success


def example_4_complete_workflow():
    """Example 4: Complete end-to-end workflow."""
    logger.info("\n" + "="*80)
    logger.info("📋 EXAMPLE 4: Complete End-to-End Workflow")
    logger.info("="*80)

    # Step 1: Fetch and process trials
    logger.info("\n🔬 Step 1: Fetch and process clinical trials")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 3 -o examples/data/complete_trials.json"
    )

    if success:
        success = run_command(
            "python -m src.cli.trials_processor examples/data/complete_trials.json --store-in-core-memory"
        )

    # Step 2: Fetch and process patients
    if success:
        logger.info("\n👥 Step 2: Fetch and process patient data")
        success = run_command(
            "python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 3 -o examples/data/complete_patients.json"
        )

        if success:
            success = run_command(
                "python -m src.cli.patients_processor --patients examples/data/complete_patients.json --trials examples/data/complete_trials.json --store-in-core-memory"
            )

    return success


def example_5_dry_run_and_validation():
    """Example 5: Dry run and validation examples."""
    logger.info("\n" + "="*80)
    logger.info("📋 EXAMPLE 5: Dry Run and Validation")
    logger.info("="*80)

    # Dry run trials processing
    logger.info("\n🔍 Step 1: Dry run - preview what would be processed")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'lung cancer' --limit 2 -o examples/data/dry_run_trials.json"
    )

    if success:
        success = run_command(
            "python -m src.cli.trials_processor examples/data/dry_run_trials.json --dry-run --verbose"
        )

    # List available archives
    logger.info("\n📚 Step 2: List available patient archives")
    success = run_command(
        "python -m src.cli.patients_fetcher --list-archives"
    )

    # List available prompts and models
    logger.info("\n📝 Step 3: List available optimization options")
    run_command(
        "python -m src.cli.trials_optimizer --list-prompts"
    )

    run_command(
        "python -m src.cli.trials_optimizer --list-models"
    )

    return success


def main():
    """Run all CLI examples."""
    print("🧪 STARTING: mCODE Translator CLI - End-to-End Examples")
    logger.info("🧪 mCODE Translator CLI - End-to-End Examples")
    logger.info("="*80)
    logger.info("This script demonstrates complete workflows using the CLI tools.")
    logger.info("Make sure you're in the mcode_translator conda environment!")
    logger.info("")

    # Check environment
    if "CONDA_DEFAULT_ENV" not in os.environ or "mcode_translator" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        logger.warning("⚠️  Warning: Not in mcode_translator conda environment")
        logger.warning("   Run: conda activate mcode_translator")
        logger.warning("")

    # Create examples data directory
    examples_data_dir = PROJECT_ROOT / "examples" / "data"
    examples_data_dir.mkdir(exist_ok=True)

    # Run examples
    examples = [
        ("Basic Trial Workflow", example_1_basic_trial_workflow),
        ("Patient Data Workflow", example_2_patient_workflow),
        ("Optimization Workflow", example_3_optimization_workflow),
        ("Complete End-to-End", example_4_complete_workflow),
        ("Dry Run & Validation", example_5_dry_run_and_validation),
    ]

    results = []
    for name, func in examples:
        logger.info(f"\n🚀 Running {name}...")
        try:
            success = func()
            results.append((name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"\n{status}: {name}")
        except Exception as e:
            logger.error(f"\n💥 ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅" if success else "❌"
        logger.info(f"{status} {name}")

    logger.info(f"\n📈 Results: {passed}/{total} examples passed")

    if passed == total:
        logger.info("🎉 All examples completed successfully!")
    else:
        logger.warning("⚠️  Some examples failed. Check the output above for details.")

    logger.info("\n💡 Tips:")
    logger.info("• Set COREAI_API_KEY for memory storage functionality")
    logger.info("• Use --verbose for detailed logging")
    logger.info("• Use --dry-run to preview operations")
    logger.info("• Check examples/data/ for generated files")


if __name__ == "__main__":
    main()