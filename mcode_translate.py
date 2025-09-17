#!/usr/bin/env python3
"""
mCODE Translator - Complete Pipeline CLI

A convenience CLI that runs the complete mCODE translation pipeline:
1. Fetch clinical trials and patients
2. Process with mCODE mapping
3. Generate summaries and store in CORE Memory

Usage:
    python mcode_translate.py --nct-ids NCT04348955,NCT03247478
    python mcode_translate.py --condition "breast cancer" --limit 3
    python mcode_translate.py --nct-ids NCT04348955 --optimize
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def run_command(cmd: str, capture_output: bool = False) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=capture_output,
            text=True,
            cwd=Path(__file__).parent
        )
        return result.stdout if capture_output else ""
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        sys.exit(1)


def optimize_parameters(nct_ids: List[str]) -> dict:
    """Run optimization to find best parameters."""
    print("🔧 Optimizing parameters...")

    # Create temporary file for optimization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
        # Fetch a sample trial for optimization
        sample_nct = nct_ids[0] if nct_ids else "NCT04348955"
        run_command(f"source activate mcode_translator && python -m src.cli.trials_fetcher --nct-id {sample_nct} --out {f.name}")

        # Run optimization
        config_file = f.name.replace('.ndjson', '_optimal.json')
        run_command(f"source activate mcode_translator && python -m src.cli.trials_optimizer --trials-file {f.name} --max-combinations 2 --save-config {config_file}")

        # Load optimal config
        try:
            with open(config_file, 'r') as cf:
                optimal_config = json.load(cf)
            print(f"✅ Optimal config: {optimal_config.get('model')} + {optimal_config.get('prompt')}")
            return optimal_config
        except Exception as e:
            print(f"⚠️  Optimization failed, using defaults: {e}")
            return {"model": "deepseek-coder", "prompt": "direct_mcode_evidence_based_concise"}


def run_complete_pipeline(
    nct_ids: Optional[List[str]] = None,
    condition: Optional[str] = None,
    limit: int = 3,
    optimize: bool = False,
    ingest: bool = False,
    verbose: bool = False
) -> None:
    """Run the complete mCODE translation pipeline."""

    print("🚀 mCODE Translator - Complete Pipeline")
    print("=" * 50)

    # Determine optimal parameters if requested
    optimal_config = None
    if optimize:
        sample_ids = nct_ids if nct_ids else [f"NCT043489{i}" for i in range(min(limit, 3))]
        optimal_config = optimize_parameters(sample_ids)

    model = optimal_config.get("model", "deepseek-coder") if optimal_config else "deepseek-coder"
    prompt = optimal_config.get("prompt", "direct_mcode_evidence_based_concise") if optimal_config else "direct_mcode_evidence_based_concise"

    print(f"🤖 Using model: {model}")
    print(f"📝 Using prompt: {prompt}")
    print()

    # Step 1: Fetch clinical trials
    print("🔬 STEP 1: Fetching clinical trials...")
    if nct_ids:
        nct_str = ",".join(nct_ids)
        cmd = f"source activate mcode_translator && python -m src.cli.trials_fetcher --nct-ids {nct_str} --out raw_trials.ndjson"
    else:
        cmd = f"source activate mcode_translator && python -m src.cli.trials_fetcher --condition \"{condition}\" --limit {limit} --out raw_trials.ndjson"

    if verbose:
        cmd += " --verbose"

    run_command(cmd)
    print("✅ Trials fetched successfully")
    print()

    # Step 2: Fetch patients
    print("👥 STEP 2: Fetching synthetic patients...")
    cmd = f"source activate mcode_translator && python -m src.cli.patients_fetcher --archive breast_cancer_10_years --limit {limit} --out raw_patients.ndjson"
    if verbose:
        cmd += " --verbose"
    run_command(cmd)
    print("✅ Patients fetched successfully")
    print()

    # Step 3: Process trials
    print("🧪 STEP 3: Processing trials with mCODE mapping...")
    cmd = f"source activate mcode_translator && python -m src.cli.trials_processor raw_trials.ndjson --out mcode_trials.ndjson --model {model} --prompt {prompt} --workers 2"
    if verbose:
        cmd += " --verbose"
    run_command(cmd)
    print("✅ Trials processed successfully")
    print()

    # Step 4: Process patients
    print("🩺 STEP 4: Processing patients with mCODE mapping...")
    cmd = f"source activate mcode_translator && python -m src.cli.patients_processor --in raw_patients.ndjson --out mcode_patients.ndjson --workers 2"
    if verbose:
        cmd += " --verbose"
    run_command(cmd)
    print("✅ Patients processed successfully")
    print()

    # Step 5: Generate summaries
    if ingest:
        print("📝 STEP 5: Generating summaries and storing in CORE Memory...")
    else:
        print("📝 STEP 5: Generating summaries...")

    # Trial summaries
    cmd = f"source activate mcode_translator && python -m src.cli.trials_summarizer --in mcode_trials.ndjson --model {model}"
    if ingest:
        cmd += " --ingest"
    if verbose:
        cmd += " --verbose"
    run_command(cmd)

    # Patient summaries
    cmd = f"source activate mcode_translator && python -m src.cli.patients_summarizer --in mcode_patients.ndjson"
    if ingest:
        cmd += " --ingest"
    if verbose:
        cmd += " --verbose"
    run_command(cmd)

    if ingest:
        print("✅ Summaries generated and stored in CORE Memory")
    else:
        print("✅ Summaries generated (CORE Memory storage disabled)")

    print()
    print("🎉 Pipeline completed successfully!")
    print("=" * 50)

    # Show summary
    print("📊 Summary:")
    if nct_ids:
        print(f"   Trials processed: {len(nct_ids)}")
    else:
        print(f"   Condition: {condition}")
        print(f"   Trials limit: {limit}")
    print(f"   Patients processed: {limit}")
    print(f"   Model used: {model}")
    print(f"   Prompt used: {prompt}")
    print(f"   CORE Memory storage: {'Enabled' if ingest else 'Disabled'}")

    # Show generated files
    print("\n📁 Generated files:")
    files = ["raw_trials.ndjson", "raw_patients.ndjson", "mcode_trials.ndjson", "mcode_patients.ndjson"]
    for file in files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} (not found)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="mCODE Translator - Complete Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete mCODE translation pipeline in one command!

Examples:
  # Process specific trials
  python mcode_translate.py --nct-ids NCT04348955,NCT03247478

  # Search by condition
  python mcode_translate.py --condition "breast cancer" --limit 5

  # Optimize parameters first
  python mcode_translate.py --nct-ids NCT04348955 --optimize

  # Store results in CORE Memory
  python mcode_translate.py --condition "lung cancer" --ingest

  # Verbose output
  python mcode_translate.py --nct-ids NCT04348955 --verbose

The pipeline will:
1. Fetch clinical trials and patients
2. Process with mCODE mapping
3. Generate summaries (use --ingest to enable storage)
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--nct-ids",
        nargs="+",
        help="Specific NCT IDs to process (e.g., NCT04348955 NCT03247478)"
    )
    input_group.add_argument(
        "--condition",
        help="Medical condition to search for (e.g., 'breast cancer')"
    )

    # Configuration options
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of trials/patients to process (default: 3)"
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run optimization to find best model/prompt combination"
    )


    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Store results in CORE Memory (disabled by default)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from all pipeline steps"
    )

    args = parser.parse_args()

    # Convert single NCT ID to list
    nct_ids = args.nct_ids if args.nct_ids else None

    # Run the complete pipeline
    try:
        run_complete_pipeline(
            nct_ids=nct_ids,
            condition=args.condition,
            limit=args.limit,
            optimize=args.optimize,
            ingest=args.ingest,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()