#!/usr/bin/env python3
"""
Patients Processor - Process patient data with mCODE mapping.

A streamlined command-line interface for processing patient data with mCODE mapping.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.utils.data_loader import load_ndjson_data
from src.utils.logging_config import get_logger
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create streamlined argument parser for patients processor."""
    parser = argparse.ArgumentParser(
        description="Process patient data with mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.patients_processor --in patients.ndjson --out mcode_patients.ndjson
  python -m src.cli.patients_processor --in patients.ndjson --trials trials.ndjson --ingest
        """,
    )

    # Core arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)
    McodeCLI.add_processor_args(parser)

    # I/O arguments
    parser.add_argument(
        "--in", dest="input_file", help="Input file with patient data (NDJSON format)"
    )
    parser.add_argument(
        "--out", dest="output_file", help="Output file for mCODE data (NDJSON format)"
    )
    parser.add_argument(
        "--trials",
        help="Path to NDJSON file containing trial data for eligibility filtering",
    )

    return parser


# Data loading function removed - now using shared src.utils.data_loader.load_ndjson_data


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for patients processor CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging and configuration
    McodeCLI.setup_logging(args)
    logger = get_logger(__name__)
    config = McodeCLI.create_config(args)

    # Validate and load input file
    if not args.input_file:
        logger.error("No input file specified")
        sys.exit(1)

    patients_path = Path(args.input_file)
    if not patients_path.exists():
        logger.error(f"Patients file not found: {patients_path}")
        sys.exit(1)

    # Load trials criteria if provided
    trials_criteria = None
    if args.trials:
        trials_path = Path(args.trials)
        if not trials_path.exists():
            logger.error(f"Trials file not found: {trials_path}")
            sys.exit(1)

        try:
            trials_list = load_ndjson_data(trials_path, "trials")
            trials_criteria = extract_mcode_criteria_from_trials(trials_list)
            logger.info(
                f"📋 Extracted mCODE criteria from {len(trials_criteria)} trial elements"
            )
        except Exception as e:
            logger.error(f"Failed to load trials file: {e}")
            sys.exit(1)

    # Load patients data
    try:
        patients_list = load_ndjson_data(patients_path, "patients")
        if not patients_list:
            logger.error("No patient data found in input file")
            sys.exit(1)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to load patients file: {e}")
        sys.exit(1)

    # Initialize memory storage if requested
    memory_storage = None
    if args.ingest:
        try:
            memory_storage = OncoCoreMemory(source=args.memory_source)
            logger.info(
                f"🧠 Initialized CORE Memory storage (source: {args.memory_source})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CORE Memory: {e}")
            sys.exit(1)

    # Execute workflow
    try:
        logger.info(f"🔬 Processing {len(patients_list)} patients...")
        workflow = PatientsProcessorWorkflow(config, memory_storage)
        result = workflow.execute(
            patients_data=patients_list,
            trials_criteria=trials_criteria,
            store_in_memory=args.ingest,
            workers=args.workers,
        )

        if not result.success:
            logger.error(f"Patients processing failed: {result.error_message}")
            sys.exit(1)

        logger.info("✅ Patients processing completed successfully!")

        # Save results
        if result.data:
            data_list = result.data if isinstance(result.data, list) else [result.data]
            save_processed_data(data_list, args.output_file, logger)

        # Print summary
        print_processing_summary(result.metadata, args.ingest, trials_criteria, logger)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def save_processed_data(
    data: List[Any], output_file: Optional[str], logger: Any
) -> None:
    """Save processed mCODE data to file or stdout."""
    mcode_data = []
    for patient_bundle in data:
        if not (isinstance(patient_bundle, dict) and "entry" in patient_bundle):
            continue

        # Extract patient ID
        patient_id = "unknown"
        for entry in patient_bundle["entry"]:
            if (
                isinstance(entry, dict)
                and entry.get("resource", {}).get("resourceType") == "Patient"
            ):
                patient_id = entry["resource"].get("id", "unknown")
                break

        # Process entries
        mcode_entries = []
        for entry in patient_bundle["entry"]:
            if not (isinstance(entry, dict) and "resource" in entry):
                continue

            resource = entry["resource"]
            resource_type = resource.get("resourceType")

            if resource_type == "Patient":
                name = resource.get("name", [{}])[0] if resource.get("name") else {}
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
                mcode_entries.append(
                    {
                        "resource_type": resource_type,
                        "id": resource.get("id"),
                        "clinical_data": resource,
                    }
                )

        mcode_data.append(
            {
                "patient_id": patient_id,
                "mcode_elements": mcode_entries,
                "original_patient_data": patient_bundle,
            }
        )

    # Output as NDJSON
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in mcode_data:
                json.dump(item, f, ensure_ascii=False, default=str)
                f.write("\n")
        logger.info(f"💾 mCODE data saved to: {output_file}")
    else:
        for item in mcode_data:
            json.dump(item, sys.stdout, ensure_ascii=False, default=str)
            sys.stdout.write("\n")
        sys.stdout.flush()
        logger.info("📤 mCODE data written to stdout")


def print_processing_summary(
    metadata: Any,
    ingested: bool,
    trials_criteria: Optional[Dict[str, Any]],
    logger: Any,
) -> None:
    """Print processing summary."""
    if not metadata:
        return

    total = metadata.get("total_patients", 0)
    successful = metadata.get("successful", 0)
    failed = metadata.get("failed", 0)
    success_rate = metadata.get("success_rate", 0)

    logger.info(f"📊 Total patients: {total}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success rate: {success_rate:.1f}%")

    if trials_criteria:
        logger.info("🎯 Applied trial eligibility filtering")

    if ingested:
        stored = metadata.get("stored_in_memory", False)
        status = "🧠 Stored in CORE Memory" if stored else "💾 Storage failed"
        logger.info(status)
    else:
        logger.info("💾 Storage disabled")


def extract_mcode_criteria_from_trials(trials_data: Any) -> Dict[str, List[str]]:
    """
    Extract mCODE eligibility criteria from trial data.

    This is a simplified version - in practice, this would use
    the full mCODE criteria extraction logic.
    """
    criteria: Dict[str, List[str]] = {}

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
