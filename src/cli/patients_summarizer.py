#!/usr/bin/env python3
"""
Patients Summarizer - Generate natural language summaries from mCODE patient data.

A command-line interface for generating comprehensive natural language summaries
from processed mCODE patient data. Takes mCODE-mapped patient data as input and
produces human-readable summaries for CORE Memory storage.

Features:
- Generate comprehensive patient summaries from mCODE data
- Support for stdin/stdout I/O streams
- Concurrent processing with worker threads
- CORE Memory storage integration
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.workflows.patients_summarizer_workflow import PatientsSummarizerWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for patients summarizer."""
    parser = argparse.ArgumentParser(
        description="Generate natural language summaries from mCODE patient data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize from file and store in CORE Memory
  python -m src.cli.patients_summarizer --in mcode_patients.ndjson --ingest

  # Summarize from stdin, output to stdout
  cat mcode_patients.ndjson | python -m src.cli.patients_summarizer

  # Summarize from file, save summaries to file
  python -m src.cli.patients_summarizer --in mcode_patients.ndjson --out patient_summaries.ndjson

  # Preview summaries without storing
  python -m src.cli.patients_summarizer --in mcode_patients.ndjson --verbose

Input/Output:
  Input: NDJSON format with mCODE patient data (one patient per line)
  Output: NDJSON format with natural language summaries (one summary per line)
  If --in not specified: reads from stdin
  If --out not specified: writes to stdout
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)
    McodeCLI.add_processor_args(parser)

    # I/O arguments
    parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file with mCODE patient data (NDJSON format). If not specified, reads from stdin",
    )

    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for summaries (NDJSON format). If not specified, writes to stdout",
    )

    return parser


def load_mcode_patients(input_file: Optional[str]) -> List[Dict]:
    """Load mCODE patient data from file or stdin."""
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    patients = []
    for line in content.strip().split("\n"):
        if line.strip():
            try:
                patient = json.loads(line)
                patients.append(patient)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    return patients


def save_summaries(summaries: List[Dict], output_file: Optional[str]) -> None:
    """Save summaries to file or stdout."""
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            for summary in summaries:
                json.dump(summary, f, ensure_ascii=False, default=str)
                f.write("\n")
    else:
        for summary in summaries:
            json.dump(summary, sys.stdout, ensure_ascii=False, default=str)
            sys.stdout.write("\n")
        sys.stdout.flush()


def main() -> None:
    """Main entry point for patients summarizer CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)
    global logger
    logger = get_logger(__name__)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Load mCODE patient data
    try:
        logger.info("Loading mCODE patient data...")
        mcode_patients = load_mcode_patients(args.input_file)
        logger.info(f"Loaded {len(mcode_patients)} mCODE patient records")
    except Exception as e:
        logger.error(f"Failed to load mCODE patient data: {e}")
        sys.exit(1)

    if not mcode_patients:
        logger.error("No valid mCODE patient data found")
        sys.exit(1)

    # Initialize core memory storage if needed
    memory_storage = None
    if args.ingest:
        try:
            memory_storage = McodeMemoryStorage(source=args.memory_source)
            logger.info(f"🧠 Initialized CORE Memory storage (source: {args.memory_source})")
        except Exception as e:
            logger.error(f"Failed to initialize CORE Memory: {e}")
            logger.info("Check your COREAI_API_KEY environment variable and core_memory_config.json")
            sys.exit(1)

    # Prepare workflow parameters
    workflow_kwargs = {
        "patients_data": mcode_patients,
        "store_in_memory": args.ingest and not args.dry_run,
        "workers": args.workers,
    }

    # Initialize and execute workflow
    try:
        logger.info("Initializing patients summarizer workflow...")
        workflow = PatientsSummarizerWorkflow(config, memory_storage)

        logger.info("Generating natural language summaries...")
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            logger.info("✅ Patient summarization completed successfully!")

            # Extract and save summaries
            summaries = []
            if result.data:
                for patient_result in result.data:
                    if isinstance(patient_result, dict) and "McodeResults" in patient_result:
                        mcode_results = patient_result["McodeResults"]
                        if "natural_language_summary" in mcode_results:
                            # Extract patient ID
                            patient_id = "unknown"
                            if "entry" in patient_result:
                                for entry in patient_result["entry"]:
                                    if (isinstance(entry, dict) and
                                        "resource" in entry and
                                        isinstance(entry["resource"], dict) and
                                        entry["resource"].get("resourceType") == "Patient"):
                                        patient_id = entry["resource"].get("id", "unknown")
                                        break

                            summary = {
                                "patient_id": patient_id,
                                "summary": mcode_results["natural_language_summary"],
                                "mcode_elements": mcode_results.get("mcode_mappings", []),
                            }
                            summaries.append(summary)

            # Save summaries
            if summaries:
                save_summaries(summaries, args.output_file)
                logger.info(f"💾 Saved {len(summaries)} patient summaries")

                if args.output_file:
                    logger.info(f"📁 Output saved to: {args.output_file}")
                else:
                    logger.info("📤 Output written to stdout")

            # Print summary
            metadata = result.metadata
            if metadata:
                total_patients = metadata.get("total_patients", 0)
                successful = metadata.get("successful", 0)
                failed = metadata.get("failed", 0)
                success_rate = metadata.get("success_rate", 0)

                logger.info(f"📊 Total patients: {total_patients}")
                logger.info(f"✅ Successful: {successful}")
                logger.info(f"❌ Failed: {failed}")
                logger.info(f"📈 Success rate: {success_rate:.1f}%")

                if args.ingest:
                    stored = metadata.get("stored_in_memory", False)
                    if stored:
                        logger.info("🧠 Summaries stored in CORE Memory")
                    else:
                        logger.info("💾 Summaries NOT stored (dry run or error)")

        else:
            logger.error(f"Patient summarization failed: {result.error_message}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()