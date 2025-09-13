#!/usr/bin/env python3
"""
Patients Processor - Process patient data with mCODE mapping.

A command-line interface for processing patient data with mCODE mapping
and storing the resulting summaries to CORE Memory.
"""

import argparse
import os
import sys
from pathlib import Path

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for patients processor."""
    parser = argparse.ArgumentParser(
        description="Process patient data with mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process patients and store in core memory
  python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory

  # Process patients with trial filtering
  python -m src.cli.patients_processor --patients patients.json --trials trials.json --store-in-core-memory

  # Dry run to preview what would be stored
  python -m src.cli.patients_processor --patients patients.json --dry-run --verbose

  # Custom core memory settings
  python -m src.cli.patients_processor --patients patients.json --store-in-core-memory --memory-source custom_source
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)

    # Input arguments
    parser.add_argument(
        "--patients", required=True, help="Path to JSON file containing patient data"
    )

    parser.add_argument(
        "--trials",
        help="Path to JSON file containing trial data for eligibility filtering",
    )

    return parser


def main() -> None:
    """Main entry point for patients processor CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Validate input files
    patients_path = Path(args.patients)
    if not patients_path.exists():
        print(f"❌ Patients file not found: {patients_path}")
        sys.exit(1)

    trials_criteria = None
    if args.trials:
        trials_path = Path(args.trials)
        if not trials_path.exists():
            print(f"❌ Trials file not found: {trials_path}")
            sys.exit(1)

        # Load trials data for criteria extraction
        try:
            with open(trials_path, "r", encoding="utf-8") as f:
                trials_data = f.read().strip()

            if not trials_data:
                print(f"❌ Trials file is empty: {trials_path}")
                sys.exit(1)

            import json

            trials_json = json.loads(trials_data)

            # Extract mCODE criteria from trials
            trials_criteria = extract_mcode_criteria_from_trials(trials_json)
            print(
                f"📋 Extracted mCODE criteria from {len(trials_criteria)} trial elements"
            )

        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in trials file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to read trials file: {e}")
            sys.exit(1)

    # Load patients data
    try:
        with open(patients_path, "r", encoding="utf-8") as f:
            patients_data = f.read().strip()

        if not patients_data:
            print(f"❌ Patients file is empty: {patients_path}")
            sys.exit(1)

        import json

        patients_json = json.loads(patients_data)

        # Handle different patient data formats
        if isinstance(patients_json, dict) and "entry" in patients_json:
            # Single FHIR Bundle
            patients_list = [patients_json]
        elif isinstance(patients_json, list):
            # List of FHIR Bundles
            patients_list = patients_json
        else:
            print("❌ Invalid patients data format. Expected FHIR Bundle(s).")
            sys.exit(1)

        print(f"🔬 Processing {len(patients_list)} patient records...")

        # Debug: Check patient data integrity
        for i, patient in enumerate(patients_list):
            entries = patient.get("entry", [])
            print(f"Patient {i+1}: {len(entries)} entries")
            for j, entry in enumerate(entries[:3]):  # Check first 3 entries
                print(f"  Entry {j}: type={type(entry)}")
                if isinstance(entry, dict):
                    resource = entry.get("resource", {})
                    print(f"    Resource: type={type(resource)}")
                    if isinstance(resource, dict):
                        resource_type = resource.get("resourceType", "Unknown")
                        print(f"    ResourceType: {resource_type}")
                        if resource_type == "Patient":
                            name = resource.get("name", [])
                            print(f"    Patient {i+1} name: {name}")
                    else:
                        print(f"    Resource content: {resource}")
                else:
                    print(f"    Entry content: {entry}")
                if j >= 2:  # Only check first 3
                    break

    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in patients file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to read patients file: {e}")
        sys.exit(1)

    # Prepare workflow parameters
    workflow_kwargs = {
        "patients_data": patients_list,
        "trials_criteria": trials_criteria,
        "store_in_memory": args.store_in_core_memory and not args.dry_run,
    }

    # Initialize core memory storage if needed
    memory_storage = None
    if args.store_in_core_memory:
        try:
            # Use centralized configuration
            memory_storage = McodeMemoryStorage(source=args.memory_source)
            print(f"🧠 Initialized CORE Memory storage (source: {args.memory_source})")
        except Exception as e:
            print(f"❌ Failed to initialize CORE Memory: {e}")
            print(
                "💡 Check your COREAI_API_KEY environment variable and core_memory_config.json"
            )
            sys.exit(1)

    # Initialize and execute workflow
    try:
        workflow = PatientsProcessorWorkflow(config, memory_storage)
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            print("✅ Patients processing completed successfully!")

            # Print summary
            metadata = result.metadata
            if metadata:
                total_patients = metadata.get("total_patients", 0)
                successful = metadata.get("successful", 0)
                failed = metadata.get("failed", 0)
                success_rate = metadata.get("success_rate", 0)

                print(f"📊 Total patients: {total_patients}")
                print(f"✅ Successful: {successful}")
                print(f"❌ Failed: {failed}")
                print(f"📈 Success rate: {success_rate:.1f}%")

                if trials_criteria:
                    filtered = metadata.get("trial_criteria_applied", False)
                    if filtered:
                        print("🎯 Applied trial eligibility filtering")

                if args.store_in_core_memory:
                    stored = metadata.get("stored_in_memory", False)
                    if stored:
                        print("🧠 mCODE summaries stored in CORE Memory")
                    else:
                        print("💾 mCODE summaries NOT stored (dry run or error)")

        else:
            print(f"❌ Patients processing failed: {result.error_message}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def extract_mcode_criteria_from_trials(trials_data) -> dict:
    """
    Extract mCODE eligibility criteria from trial data.

    This is a simplified version - in practice, this would use
    the full mCODE criteria extraction logic.
    """
    criteria = {}

    # Handle different trial data formats
    if isinstance(trials_data, list):
        trials_list = trials_data
    elif isinstance(trials_data, dict):
        if "studies" in trials_data:
            trials_list = trials_data["studies"]
        elif "successful_trials" in trials_data:
            trials_list = trials_data["successful_trials"]
        else:
            trials_list = [trials_data]
    else:
        return criteria

    # Extract mCODE elements from all trials
    for trial in trials_list:
        if "McodeResults" in trial:
            mcode_results = trial["McodeResults"]
            mappings = mcode_results.get("mcode_mappings", [])

            for mapping in mappings:
                element = mapping.get("mcode_element")
                value = mapping.get("value")

                if element and value and value != "N/A":
                    if element not in criteria:
                        criteria[element] = []
                    if value not in criteria[element]:
                        criteria[element].append(value)

    return criteria


if __name__ == "__main__":
    main()
