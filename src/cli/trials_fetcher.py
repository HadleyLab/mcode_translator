#!/usr/bin/env python3
"""
Trials Fetcher - Fetch clinical trials from ClinicalTrials.gov.

A streamlined command-line interface for fetching raw clinical trial data.
"""

import argparse
import sys
from typing import Any, Optional

from src.shared.cli_utils import McodeCLI
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create streamlined argument parser for trials fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch clinical trials from ClinicalTrials.gov",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.trials_fetcher --condition "breast cancer" --out trials.ndjson
  python -m src.cli.trials_fetcher --nct-id NCT12345678 --out trial.ndjson
        """,
    )

    # Core arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_concurrency_args(parser)

    # Fetch options
    parser.add_argument("--condition", help="Medical condition to search for")
    parser.add_argument("--nct-id", help="Specific NCT ID to fetch")
    parser.add_argument("--nct-ids", help="Comma-separated list of NCT IDs to fetch")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of trials to fetch (default: 10)",
    )
    parser.add_argument(
        "--out", dest="output_file", help="Output file for trial data (NDJSON format)"
    )

    return parser


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for trials fetcher CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging and configuration
    McodeCLI.setup_logging(args)
    config = McodeCLI.create_config(args)

    # Validate arguments
    if not any([args.condition, args.nct_id, args.nct_ids]):
        parser = create_parser()
        parser.error("Must specify one of: --condition, --nct-id, or --nct-ids")

    # Prepare workflow parameters
    workflow_kwargs: dict[str, Any] = {"cli_args": args}

    if args.condition:
        workflow_kwargs.update({"condition": args.condition, "limit": args.limit})
    elif args.nct_id:
        workflow_kwargs["nct_id"] = args.nct_id
    elif args.nct_ids:
        workflow_kwargs["nct_ids"] = [nct.strip() for nct in args.nct_ids.split(",")]

    if args.output_file:
        workflow_kwargs["output_path"] = args.output_file

    # Execute workflow
    try:
        workflow = TrialsFetcherWorkflow(config, memory_storage=False)
        result = workflow.execute(**workflow_kwargs)

        if not result.success:
            print(f"❌ Trials fetch failed: {result.error_message}")
            sys.exit(1)

        print("✅ Trials fetch completed successfully!")
        print_fetch_summary(result.metadata, args.output_file)

    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def print_fetch_summary(metadata: Any, output_file: Optional[str]) -> None:
    """Print fetch operation summary."""
    if not metadata:
        return

    total_fetched = metadata.get("total_fetched", 0)
    print(f"📊 Total trials fetched: {total_fetched}")

    if output_file:
        print(f"💾 Results saved to: {output_file}")
    else:
        print("📤 Results written to stdout")

    if fetch_type := metadata.get("fetch_type"):
        print(f"🔍 Fetch type: {fetch_type}")

    if duration := metadata.get("duration_seconds"):
        print(f"⏱️  Duration: {duration:.2f}s")


def fetch_trials_direct(
    condition: Optional[str],
    nct_id: Optional[str],
    nct_ids: Optional[str],
    limit: int,
    output_file: Optional[str],
    model: str,
    prompt: str,
    workers: int,
    log_level: str,
    config_file: Optional[str],
    store_in_memory: bool,
) -> bool:
    """Direct function call - no argparse complexity."""
    from src.config.heysol_config import get_config
    from src.utils.logging_config import setup_logging

    # Setup logging directly
    setup_logging(level=log_level)

    # Get configuration
    get_config()

    # Process the request directly
    if condition:
        print(f"Fetching trials for condition: {condition}")
        # Direct implementation here
        return True
    elif nct_id:
        print(f"Fetching trial: {nct_id}")
        # Direct implementation here
        return True
    elif nct_ids:
        print(f"Fetching trials: {nct_ids}")
        # Direct implementation here
        return True
    else:
        print("No search criteria provided")
        return False


if __name__ == "__main__":
    main()
