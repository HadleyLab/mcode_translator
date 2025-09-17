#!/usr/bin/env python3
"""
Patients Processor - Process patient data with mCODE mapping.

A command-line interface for processing patient data with mCODE mapping
and storing the resulting summaries to CORE Memory or saving as JSON/NDJSON files.

Features:
- Extract mCODE elements from FHIR patient bundles
- Save processed data in JSON array format or NDJSON format
- Optional CORE Memory storage (use --ingest to enable)
- Trial eligibility filtering capabilities
- Concurrent processing with worker threads
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
  # Process patients and save as NDJSON (recommended for large datasets)
  python -m src.cli.patients_processor --in patients.ndjson --output mcode_patients.ndjson

  # Process patients and store in core memory
  python -m src.cli.patients_processor --in patients.ndjson --trials trials.ndjson --ingest

  # Process patients with trial filtering
  python -m src.cli.patients_processor --in patients.ndjson --trials trials.ndjson --ingest

  # Custom core memory settings
  python -m src.cli.patients_processor --in patients.ndjson --ingest --memory-source custom_source

Output Formats:
  JSON:  Standard JSON array format - [{"patient_bundle": [...]}]
  NDJSON: Newline-delimited JSON - one JSON object per line (recommended for streaming)
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)
    McodeCLI.add_processor_args(parser)

    # Input arguments
    parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file with patient data (NDJSON format). If not specified, reads from stdin",
    )

    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for mCODE data (NDJSON format). If not specified, writes to stdout",
    )

    parser.add_argument(
        "--trials",
        help="Path to NDJSON file containing trial data for eligibility filtering",
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

    # Determine input file (positional or --in)
    input_file = getattr(args, 'input_file', None)
    if not input_file:
        print("❌ No input file specified. Use --in argument")
        sys.exit(1)

    patients_path = Path(input_file)
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

    # Load patients data from file or stdin
    try:
        import json

        if args.input_file:
            # Read NDJSON file (one JSON object per line)
            patients_list = []
            with open(patients_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        patient_data = json.loads(line)
                        # Handle different patient data formats
                        if isinstance(patient_data, dict) and "entry" in patient_data:
                            # Single FHIR Bundle
                            patients_list.append(patient_data)
                        else:
                            print(f"❌ Invalid patient data format in line. Expected FHIR Bundle.")
                            continue
        else:
            # Read from stdin as NDJSON
            patients_list = []
            for line in sys.stdin:
                line = line.strip()
                if line:  # Skip empty lines
                    patient_data = json.loads(line)
                    # Handle different patient data formats
                    if isinstance(patient_data, dict) and "entry" in patient_data:
                        # Single FHIR Bundle
                        patients_list.append(patient_data)
                    else:
                        print(f"❌ Invalid patient data format in line. Expected FHIR Bundle.")
                        continue

        if not patients_list:
            print("❌ No patient data provided")
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
        "store_in_memory": args.ingest,
    }

    # Initialize core memory storage if requested
    memory_storage = None
    if args.ingest:
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

            # Save processed mCODE data to file or stdout
            if result.data:
                try:
                    import json

                    # Extract mCODE elements for output
                    mcode_data = []
                    for patient_bundle in result.data:
                        if (
                            isinstance(patient_bundle, dict)
                            and "entry" in patient_bundle
                        ):
                            # Extract patient ID
                            patient_id = "unknown"
                            for entry in patient_bundle["entry"]:
                                if (isinstance(entry, dict) and
                                    "resource" in entry and
                                    isinstance(entry["resource"], dict) and
                                    entry["resource"].get("resourceType") == "Patient"):
                                    patient_id = entry["resource"].get("id", "unknown")
                                    break

                            # Process each entry in the FHIR bundle
                            mcode_entries = []
                            for entry in patient_bundle["entry"]:
                                if (
                                    isinstance(entry, dict)
                                    and "resource" in entry
                                    and isinstance(entry["resource"], dict)
                                ):
                                    resource = entry["resource"]
                                    resource_type = resource.get("resourceType")

                                    # Extract mCODE-relevant information
                                    if resource_type == "Patient":
                                        name = (
                                            resource.get("name", [{}])[0]
                                            if resource.get("name")
                                            else {}
                                        )
                                        mcode_entries.append(
                                            {
                                                "resource_type": "Patient",
                                                "id": resource.get("id"),
                                                "name": name,
                                            }
                                        )
                                    elif resource_type in [
                                        "Condition",
                                        "Observation",
                                        "MedicationStatement",
                                        "Procedure",
                                    ]:
                                        # These contain clinical data that would be mCODE-mapped
                                        mcode_entries.append(
                                            {
                                                "resource_type": resource_type,
                                                "id": resource.get("id"),
                                                "clinical_data": resource,
                                            }
                                        )

                            # Create output structure with mCODE data and original patient
                            output_item = {
                                "patient_id": patient_id,
                                "mcode_elements": mcode_entries,
                                "original_patient_data": patient_bundle,  # Keep original for summarizer
                            }
                            mcode_data.append(output_item)

                    # Output as NDJSON to file or stdout
                    if args.output_file:
                        with open(args.output_file, "w", encoding="utf-8") as f:
                            for item in mcode_data:
                                json.dump(item, f, ensure_ascii=False, default=str)
                                f.write("\n")
                        print(f"💾 mCODE data saved as NDJSON to: {args.output_file}")
                    else:
                        # Write to stdout
                        for item in mcode_data:
                            json.dump(item, sys.stdout, ensure_ascii=False, default=str)
                            sys.stdout.write("\n")
                        sys.stdout.flush()
                        print("📤 mCODE data written to stdout")

                except Exception as e:
                    print(f"❌ Failed to save processed data: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

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

                if args.ingest:
                    stored = metadata.get("stored_in_memory", False)
                    if stored:
                        print("🧠 mCODE summaries stored in CORE Memory")
                    else:
                        print("💾 mCODE summaries NOT stored (error)")
                else:
                    print("💾 mCODE summaries NOT stored (storage disabled)")

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
