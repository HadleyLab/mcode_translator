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


def run_command(cmd: str, cwd: Optional[Path] = None, env: Optional[dict] = None) -> bool:
    """Run a CLI command and return success status."""
    try:
        working_dir = cwd or PROJECT_ROOT
        full_env = os.environ.copy()
        if env:
            full_env.update(env)

        print(f"\n🔧 Running: {cmd}")
        print(f"   📁 Working directory: {working_dir}")

        result = subprocess.run(
            cmd,
            cwd=working_dir,
            env=full_env,
            capture_output=True,
            text=True,
            check=True,
            shell=True
        )

        print("✅ Command completed successfully!")
        if result.stdout.strip():
            print(f"📄 Output:\n{result.stdout}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"📄 stdout:\n{e.stdout}")
        if e.stderr:
            print(f"📄 stderr:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def example_1_basic_trial_workflow():
    """Example 1: Basic clinical trial fetching and processing."""
    print("\n" + "="*80)
    print("📋 EXAMPLE 1: Basic Clinical Trial Workflow")
    print("="*80)

    # Step 1: Fetch trials
    print("\n🔍 Step 1: Fetch clinical trials")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 3 -o examples/data/fetched_trials.json --verbose"
    )

    if not success:
        print("❌ Trial fetching failed, skipping processing step")
        return False

    # Step 2: Process trials
    print("\n🔬 Step 2: Process trials with mCODE mapping")
    success = run_command(
        "python -m src.cli.trials_processor examples/data/fetched_trials.json --model deepseek-coder --prompt direct_mcode_evidence_based_concise --store-in-core-memory --verbose"
    )

    return success


def example_2_patient_workflow():
    """Example 2: Patient data fetching and processing."""
    print("\n" + "="*80)
    print("📋 EXAMPLE 2: Patient Data Workflow")
    print("="*80)

    # Step 1: Fetch patients
    print("\n🔍 Step 1: Fetch synthetic patient data")
    success = run_command(
        "python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 5 -o examples/data/fetched_patients.json --verbose"
    )

    if not success:
        print("❌ Patient fetching failed, skipping processing step")
        return False

    # Step 2: Fetch trials for criteria
    print("\n🔍 Step 2: Fetch trial criteria")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 2 -o examples/data/trial_criteria.json --verbose"
    )

    if not success:
        print("❌ Trial criteria fetching failed, skipping patient processing")
        return False

    # Step 3: Process patients with trial criteria
    print("\n🔬 Step 3: Process patients with trial eligibility filtering")
    success = run_command(
        "python -m src.cli.patients_processor --patients examples/data/fetched_patients.json --trials examples/data/trial_criteria.json --store-in-core-memory --verbose"
    )

    return success


def example_3_optimization_workflow():
    """Example 3: Parameter optimization workflow."""
    print("\n" + "="*80)
    print("📋 EXAMPLE 3: Parameter Optimization Workflow")
    print("="*80)

    # Step 1: Fetch sample trials for optimization
    print("\n🔍 Step 1: Fetch sample trials for optimization")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition cancer --limit 5 -o examples/data/optimization_trials.json --verbose"
    )

    if not success:
        print("❌ Sample data fetching failed, skipping optimization")
        return False

    # Step 2: Run optimization
    print("\n🎯 Step 2: Optimize mCODE translation parameters")
    success = run_command(
        "python -m src.cli.trials_optimizer --trials-file examples/data/optimization_trials.json --prompts direct_mcode_evidence_based_concise,direct_mcode_minimal --models deepseek-coder,gpt-4 --max-combinations 4 --save-config examples/data/optimized_config.json --verbose"
    )

    return success


def example_4_complete_workflow():
    """Example 4: Complete end-to-end workflow."""
    print("\n" + "="*80)
    print("📋 EXAMPLE 4: Complete End-to-End Workflow")
    print("="*80)

    # Step 1: Fetch and process trials
    print("\n🔬 Step 1: Fetch and process clinical trials")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 3 -o examples/data/complete_trials.json"
    )

    if success:
        success = run_command(
            "python -m src.cli.trials_processor examples/data/complete_trials.json --store-in-core-memory"
        )

    # Step 2: Fetch and process patients
    if success:
        print("\n👥 Step 2: Fetch and process patient data")
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
    print("\n" + "="*80)
    print("📋 EXAMPLE 5: Dry Run and Validation")
    print("="*80)

    # Dry run trials processing
    print("\n🔍 Step 1: Dry run - preview what would be processed")
    success = run_command(
        "python -m src.cli.trials_fetcher --condition 'lung cancer' --limit 2 -o examples/data/dry_run_trials.json"
    )

    if success:
        success = run_command(
            "python -m src.cli.trials_processor examples/data/dry_run_trials.json --dry-run --verbose"
        )

    # List available archives
    print("\n📚 Step 2: List available patient archives")
    success = run_command(
        "python -m src.cli.patients_fetcher --list-archives"
    )

    # List available prompts and models
    print("\n📝 Step 3: List available optimization options")
    run_command(
        "python -m src.cli.trials_optimizer --list-prompts"
    )

    run_command(
        "python -m src.cli.trials_optimizer --list-models"
    )

    return success


def main():
    """Run all CLI examples."""
    print("🧪 mCODE Translator CLI - End-to-End Examples")
    print("="*80)
    print("This script demonstrates complete workflows using the CLI tools.")
    print("Make sure you're in the mcode_translator conda environment!")
    print()

    # Check environment
    if "CONDA_DEFAULT_ENV" not in os.environ or "mcode_translator" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("⚠️  Warning: Not in mcode_translator conda environment")
        print("   Run: conda activate mcode_translator")
        print()

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
        print(f"\n🚀 Running {name}...")
        try:
            success = func()
            results.append((name, success))
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {name}")
        except Exception as e:
            print(f"\n💥 ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    print(f"\n📈 Results: {passed}/{total} examples passed")

    if passed == total:
        print("🎉 All examples completed successfully!")
    else:
        print("⚠️  Some examples failed. Check the output above for details.")

    print("\n💡 Tips:")
    print("• Set COREAI_API_KEY for memory storage functionality")
    print("• Use --verbose for detailed logging")
    print("• Use --dry-run to preview operations")
    print("• Check examples/data/ for generated files")


if __name__ == "__main__":
    main()