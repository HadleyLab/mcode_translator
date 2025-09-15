#!/usr/bin/env python3
"""
Interactive Breast Cancer End-to-End Demo - Live Color Logging

This script demonstrates the complete breast cancer mCODE workflow with:
- Real-time colorful logging
- Interactive progress updates
- Concurrent processing with 5+ workers
- Complete CORE memory integration
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.utils.logging_config import setup_logging, get_logger
from src.utils.config import Config

setup_logging()
logger = get_logger("breast_cancer_interactive")
config = Config()


def run_interactive_command(cmd: str, description: str = "", cwd: Optional[Path] = None) -> bool:
    """Run command with real-time colorful output."""
    try:
        working_dir = cwd or PROJECT_ROOT

        if description:
            print(f"\n🎯 {description}")
        print(f"🔧 Executing: {cmd}")
        print(f"📁 Working directory: {working_dir}")
        print("⏳ Processing... (you'll see live output below)")
        print("-" * 80)

        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            shell=True,
            check=True
        )
        end_time = time.time()

        print("-" * 80)
        duration = end_time - start_time
        print(f"✅ Command completed successfully in {duration:.1f}s!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def show_workflow_header():
    """Display the workflow header with colorful formatting."""
    print("\n" + "🎗️" * 25)
    print("🧬 BREAST CANCER END-TO-END mCODE DEMO 🧬")
    print("🎗️" * 25)
    print("🎯 Complete Clinical Trial & Patient Processing Workflow")
    print("⚡ Real-time colorful logging with concurrent processing")
    print("💾 CORE Memory integration with new version storage")
    print()


def check_environment():
    """Check environment and configuration."""
    print("🔍 Checking environment...")

    # Check conda environment
    if "CONDA_DEFAULT_ENV" not in os.environ or "mcode_translator" not in os.environ.get("CONDA_DEFAULT_ENV", ""):
        print("⚠️  Warning: Not in mcode_translator conda environment")
        print("   Run: conda activate mcode_translator")
    else:
        print("✅ Conda environment: mcode_translator")

    # Check CORE Memory API key
    try:
        config.get_core_memory_api_key()
        print("✅ CORE Memory API key configured")
    except Exception as e:
        print(f"⚠️  Warning: {str(e)}")
        print("   Set the COREAI_API_KEY environment variable")

    print()


def step_1_fetch_trials():
    """Step 1: Fetch 5 breast cancer trials."""
    print("\n" + "🔬" * 15 + " STEP 1/5 " + "🔬" * 15)
    print("📥 FETCHING 5 BREAST CANCER CLINICAL TRIALS")
    print("🔬" * 40)

    success = run_interactive_command(
        "python -m src.cli.trials_fetcher --condition 'breast cancer' --limit 5 -o breast_cancer_trials_demo.json --workers 5 --verbose",
        "Searching ClinicalTrials.gov for 5 recent breast cancer trials using 5 concurrent workers"
    )

    if success:
        print("✅ Successfully fetched 5 breast cancer trials!")
        print("📁 Saved to: breast_cancer_trials_demo.json")
    else:
        print("❌ Failed to fetch trials")
        return False

    return success


def step_2_process_trials():
    """Step 2: Process trials with mCODE mapping."""
    print("\n" + "🧪" * 15 + " STEP 2/5 " + "🧪" * 15)
    print("🔬 PROCESSING TRIALS WITH mCODE MAPPING")
    print("🧪" * 40)

    success = run_interactive_command(
        "python -m src.cli.trials_processor breast_cancer_trials_demo.json --model deepseek-coder --prompt direct_mcode_evidence_based_concise --store-in-core-memory --workers 5 --verbose",
        "Processing 5 trials with LLM-based mCODE extraction and storing comprehensive summaries to CORE memory"
    )

    if success:
        print("✅ Successfully processed trials with mCODE mapping!")
        print("💾 Trial summaries stored in CORE memory with new versions")
    else:
        print("❌ Failed to process trials")
        return False

    return success


def step_3_fetch_patients():
    """Step 3: Fetch 5 breast cancer patients."""
    print("\n" + "👥" * 15 + " STEP 3/5 " + "👥" * 15)
    print("📥 FETCHING 5 BREAST CANCER PATIENTS")
    print("👥" * 40)

    success = run_interactive_command(
        "python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit 5 -o breast_cancer_patients_demo.json --verbose",
        "Generating 5 synthetic breast cancer patients from 10-year survival archive"
    )

    if success:
        print("✅ Successfully fetched 5 breast cancer patients!")
        print("📁 Saved to: breast_cancer_patients_demo.json")
    else:
        print("❌ Failed to fetch patients")
        return False

    return success


def step_4_process_patients():
    """Step 4: Process patients with trial matching."""
    print("\n" + "🩺" * 15 + " STEP 4/5 " + "🩺" * 15)
    print("👤 PROCESSING PATIENTS WITH TRIAL MATCHING")
    print("🩺" * 40)

    success = run_interactive_command(
        "python -m src.cli.patients_processor --patients breast_cancer_patients_demo.json --trials breast_cancer_trials_demo.json --store-in-core-memory --verbose",
        "Processing 5 patients with clinical trial matching and storing patient summaries to CORE memory"
    )

    if success:
        print("✅ Successfully processed patients with trial matching!")
        print("💾 Patient summaries stored in CORE memory with new versions")
    else:
        print("❌ Failed to process patients")
        return False

    return success


def step_5_final_summary():
    """Step 5: Show final workflow summary."""
    print("\n" + "🎉" * 15 + " STEP 5/5 " + "🎉" * 15)
    print("📊 WORKFLOW COMPLETION SUMMARY")
    print("🎉" * 40)

    print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    print()
    print("📋 What was accomplished:")
    print("   🔬 5 breast cancer clinical trials fetched and processed")
    print("   🧪 Trials analyzed with comprehensive mCODE mapping")
    print("   👥 5 breast cancer patients generated and processed")
    print("   🩺 Patients matched with eligible clinical trials")
    print("   💾 All data stored in CORE memory with new versions")
    print("   ⚡ Concurrent processing used 5+ workers for speed")
    print()
    print("🧬 CORE Memory now contains:")
    print("   • Rich clinical trial summaries with mCODE mappings")
    print("   • Patient profiles with trial eligibility matching")
    print("   • New versions added for continuous learning")
    print()
    print("🎯 Ready for clinical decision support and patient-trial matching!")
    print()
    print("📁 Generated files:")
    print("   📄 breast_cancer_trials_demo.json")
    print("   👤 breast_cancer_patients_demo.json")
    print("   🧠 CORE Memory (enhanced with new mCODE data)")

    return True


def main():
    """Run the interactive breast cancer demo."""
    show_workflow_header()
    check_environment()

    # Define workflow steps
    steps = [
        ("Fetch Trials", step_1_fetch_trials),
        ("Process Trials", step_2_process_trials),
        ("Fetch Patients", step_3_fetch_patients),
        ("Process Patients", step_4_process_patients),
        ("Final Summary", step_5_final_summary),
    ]

    results = []
    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n🚀 STARTING STEP {i}/5: {step_name}")
        print("=" * 60)

        try:
            start_time = time.time()
            success = step_func()
            end_time = time.time()
            duration = end_time - start_time

            results.append((step_name, success, duration))

            if success:
                print(f"✅ STEP {i}/5 COMPLETED: {step_name} ({duration:.1f}s)")
            else:
                print(f"❌ STEP {i}/5 FAILED: {step_name} ({duration:.1f}s)")
                print("🛑 Stopping workflow due to failure")
                break

        except KeyboardInterrupt:
            print("\n⏹️  Workflow interrupted by user")
            break
        except Exception as e:
            print(f"\n💥 ERROR in {step_name}: {e}")
            results.append((step_name, False, 0))
            break

    # Show final status
    print("\n" + "=" * 80)
    print("🏁 WORKFLOW FINAL STATUS")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    total_time = sum(duration for _, _, duration in results)

    print(f"📊 Completed: {passed}/{total} steps")
    print(f"⏱️  Total time: {total_time:.1f}s")

    if passed == total:
        print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print("🧬 Breast cancer mCODE workflow is fully operational")
    else:
        print("⚠️  WORKFLOW PARTIALLY COMPLETED")
        print("🔄 Failed steps can be retried individually")


if __name__ == "__main__":
    main()