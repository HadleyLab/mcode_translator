#!/usr/bin/env python3
"""
Trials Fetcher - Fetch clinical trials from ClinicalTrials.gov.

A command-line interface for fetching raw clinical trial data from
ClinicalTrials.gov without any processing or core memory storage.
"""

import argparse
import sys
from pathlib import Path

from src.shared.cli_utils import McodeCLI
from src.utils.config import Config
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch clinical trials from ClinicalTrials.gov",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch trials by condition
  python -m src.cli.trials_fetcher --condition "breast cancer" -o trials.ndjson

  # Fetch specific trial by NCT ID
  python -m src.cli.trials_fetcher --nct-id NCT12345678 -o trial.ndjson

  # Fetch multiple trials
  python -m src.cli.trials_fetcher --nct-ids NCT12345678,NCT87654321 -o trials.ndjson

  # Verbose output with custom config
  python -m src.cli.trials_fetcher --condition "lung cancer" --verbose --config custom.json
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)

    # Fetcher-specific arguments
    fetcher_group = parser.add_argument_group("fetch options")
    fetcher_group.add_argument(
        "--condition", help="Medical condition to search for (e.g., 'breast cancer')"
    )

    fetcher_group.add_argument(
        "--nct-id", help="Specific NCT ID to fetch (e.g., NCT12345678)"
    )

    fetcher_group.add_argument(
        "--nct-ids", help="Comma-separated list of NCT IDs to fetch"
    )

    fetcher_group.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of trials to fetch (default: 10)",
    )

    # I/O arguments
    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for trial data (NDJSON format). If not specified, writes to stdout",
    )

    # Processing arguments (for concurrency)
    McodeCLI.add_processor_args(parser)

    return parser


def main() -> None:
    """Main entry point for trials fetcher CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Validate arguments
    if not any([args.condition, args.nct_id, args.nct_ids]):
        parser.error("Must specify one of: --condition, --nct-id, or --nct-ids")

    # Prepare workflow parameters
    workflow_kwargs = {}

    if args.condition:
        workflow_kwargs["condition"] = args.condition
        workflow_kwargs["limit"] = args.limit
    elif args.nct_id:
        workflow_kwargs["nct_id"] = args.nct_id
    elif args.nct_ids:
        workflow_kwargs["nct_ids"] = [nct.strip() for nct in args.nct_ids.split(",")]

    if args.output_file:
        workflow_kwargs["output_path"] = args.output_file

    # Handle concurrent execution
    try:
        if args.workers > 0:
            # Use concurrent fetching
            import asyncio

            from src.pipeline.concurrent_fetcher import \
                concurrent_search_and_process

            async def run_concurrent():
                if args.condition:
                    return await concurrent_search_and_process(
                        condition=args.condition,
                        limit=args.limit,
                        max_workers=args.workers,
                        export_path=args.output if args.output else None,
                        progress_updates=args.verbose,
                    )
                elif args.nct_ids:
                    return await concurrent_search_and_process(
                        nct_ids=[nct.strip() for nct in args.nct_ids.split(",")],
                        max_workers=args.workers,
                        export_path=args.output if args.output else None,
                        progress_updates=args.verbose,
                    )
                else:
                    raise ValueError(
                        "Concurrent fetching requires --condition or --nct-ids"
                    )

            result_data = asyncio.run(run_concurrent())

            # Convert concurrent result to workflow result format
            class MockResult:
                def __init__(self, concurrent_result):
                    self.success = concurrent_result.successful_trials > 0
                    self.error_message = ""
                    self.metadata = {
                        "total_fetched": concurrent_result.total_trials,
                        "fetch_type": "concurrent",
                        "successful": concurrent_result.successful_trials,
                        "failed": concurrent_result.failed_trials,
                        "duration_seconds": concurrent_result.duration_seconds,
                    }

            result = MockResult(result_data)

        else:
            # Use sequential workflow
            workflow = TrialsFetcherWorkflow(config)
            result = workflow.execute(**workflow_kwargs)

        # Print results (common for both concurrent and sequential)
        if result.success:
            print("✅ Trials fetch completed successfully!")

            # Print summary
            metadata = result.metadata
            if metadata:
                total_fetched = metadata.get("total_fetched", 0)
                print(f"📊 Total trials fetched: {total_fetched}")

                if args.output_file:
                    print(f"💾 Results saved to: {args.output_file}")
                else:
                    print("📤 Results written to stdout")

                # Print additional details
                fetch_type = metadata.get("fetch_type")
                if fetch_type:
                    print(f"🔍 Fetch type: {fetch_type}")

                # Print concurrency info if applicable
                if args.workers > 0:
                    successful = metadata.get("successful", 0)
                    failed = metadata.get("failed", 0)
                    duration = metadata.get("duration_seconds", 0)
                    print(f"⚡ Concurrent processing: {args.workers} workers")
                    print(f"✅ Successful: {successful}, ❌ Failed: {failed}")
                    print(f"⏱️  Duration: {duration:.2f}s")

        else:
            print(f"❌ Trials fetch failed: {result.error_message}")
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
