#!/usr/bin/env python3
"""
Patients Fetcher - Fetch synthetic patient data from archives.

A command-line interface for fetching raw synthetic patient data from
archives without any processing or core memory storage.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.shared.cli_utils import McodeCLI
from src.utils.config import Config
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for patients fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch synthetic patient data from archives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch patients from breast cancer archive
  python -m src.cli.patients_fetcher --archive breast_cancer_10_years -o patients.ndjson

  # Fetch specific patient by ID
  python -m src.cli.patients_fetcher --archive breast_cancer_10_years --patient-id patient_123 -o patient.ndjson

  # List available archives
  python -m src.cli.patients_fetcher --list-archives

  # Verbose output with custom limit
  python -m src.cli.patients_fetcher --archive mixed_cancer_lifetime --limit 50 --verbose -o patients.ndjson
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)

    # Fetcher-specific arguments
    fetcher_group = parser.add_argument_group("fetch options")
    fetcher_group.add_argument(
        "--archive", help="Patient archive identifier (e.g., breast_cancer_10_years)"
    )

    fetcher_group.add_argument("--patient-id", help="Specific patient ID to fetch")

    fetcher_group.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of patients to fetch (default: 10)",
    )

    fetcher_group.add_argument(
        "--list-archives",
        action="store_true",
        help="List available patient archives and exit",
    )

    # I/O arguments
    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for patient data (NDJSON format). If not specified, writes to stdout",
    )

    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for patients fetcher CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Handle list archives command
    if args.list_archives:
        workflow = PatientsFetcherWorkflow(config)
        archives = workflow.list_available_archives()
        print("📚 Available patient archives:")
        for archive in archives:
            print(f"  • {archive}")
        return

    # Validate arguments
    if not args.archive:
        parser.error("Must specify --archive (or use --list-archives)")

    # Prepare workflow parameters
    workflow_kwargs = {"archive_path": args.archive, "limit": args.limit}

    if args.patient_id:
        workflow_kwargs["patient_id"] = args.patient_id

    if args.output_file:
        workflow_kwargs["output_path"] = args.output_file

    # Initialize and execute workflow
    try:
        # Disable CORE memory for fetching operations unless explicitly requested
        workflow = PatientsFetcherWorkflow(config, memory_storage=False)
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            print("✅ Patients fetch completed successfully!")

            # Print summary
            metadata = result.metadata
            if metadata:
                total_fetched = metadata.get("total_fetched", 0)
                print(f"📊 Total patients fetched: {total_fetched}")

                if args.output_file:
                    print(f"💾 Results saved to: {args.output_file}")
                else:
                    print(f"📤 Results written to stdout: {total_fetched} records (NDJSON format)")

                # Print additional details
                fetch_type = metadata.get("fetch_type")
                if fetch_type:
                    print(f"🔍 Fetch type: {fetch_type}")

                # Print archive info
                archive_path = metadata.get("archive_path")
                if archive_path:
                    print(f"📁 Archive: {archive_path}")

        else:
            print(f"❌ Patients fetch failed: {result.error_message}")
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


if __name__ == "__main__":
    main()
