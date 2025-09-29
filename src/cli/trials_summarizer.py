#!/usr/bin/env python3
"""
Trials Summarizer - Generate natural language summaries from mCODE trial data.

A streamlined command-line interface for generating summaries from mCODE trial data.
"""

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import OncoCoreMemory
from src.utils.logging_config import get_logger
from src.workflows.trials_summarizer_workflow import TrialsSummarizerWorkflow

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create streamlined argument parser for trials summarizer."""
    parser = argparse.ArgumentParser(
        description="Generate natural language summaries from mCODE trial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.trials_summarizer --in mcode_trials.ndjson --ingest
  python -m src.cli.trials_summarizer --in mcode_trials.ndjson --out summaries.ndjson
        """,
    )

    # Core arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)
    McodeCLI.add_processor_args(parser)

    # I/O arguments
    parser.add_argument(
        "--in",
        dest="input_file",
        help="Input file with mCODE trial data (NDJSON format)",
    )
    parser.add_argument(
        "--out", dest="output_file", help="Output file for summaries (NDJSON format)"
    )

    return parser


def load_mcode_trials(input_file: Optional[str]) -> List[Dict[str, Any]]:
    """Load mCODE trial data from file or stdin."""
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sys.stdin.read()

    trials = []
    for line in content.strip().split("\n"):
        if line.strip():
            try:
                trial = json.loads(line)
                trials.append(trial)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    return trials


def save_summaries(summaries: List[Dict[str, Any]], output_file: Optional[str]) -> None:
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


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for trials summarizer CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging and configuration
    McodeCLI.setup_logging(args)
    logger = get_logger(__name__)
    config = McodeCLI.create_config(args)

    # Load mCODE trial data
    try:
        logger.info("Loading mCODE trial data...")
        mcode_trials = load_mcode_trials(args.input_file)
        if not mcode_trials:
            logger.error("No valid mCODE trial data found")
            sys.exit(1)
        logger.info(f"Loaded {len(mcode_trials)} mCODE trial records")
    except Exception as e:
        logger.error(f"Failed to load mCODE trial data: {e}")
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

    # Prepare trial data for summarization
    trial_data = []
    for mcode_trial in mcode_trials:
        trial_data.append(
            mcode_trial.get("original_trial_data")
            or mcode_trial.get("trial_data")
            or mcode_trial
        )

    # Execute workflow
    try:
        logger.info("Generating natural language summaries...")
        workflow = TrialsSummarizerWorkflow(config, memory_storage)
        result = workflow.execute(
            trials_data=trial_data,
            model=args.model,
            prompt=args.prompt,
            store_in_memory=args.ingest,
            workers=args.workers,
        )

        if not result.success:
            logger.error(f"Trial summarization failed: {result.error_message}")
            sys.exit(1)

        logger.info("✅ Trial summarization completed successfully!")

        # Extract and save summaries
        summaries = extract_summaries(result.data)
        if summaries:
            save_summaries(summaries, args.output_file)
            logger.info(f"💾 Saved {len(summaries)} trial summaries")

            if args.output_file:
                logger.info(f"📁 Output saved to: {args.output_file}")
            else:
                logger.info("📤 Output written to stdout")

        # Print summary
        print_processing_summary(result.metadata, args.ingest, logger)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def extract_summaries(data: Any) -> List[Dict[str, Any]]:
    """Extract summaries from workflow results."""
    summaries: List[Dict[str, Any]] = []
    if not data:
        return summaries

    for trial_result in data:
        if not (isinstance(trial_result, dict) and "McodeResults" in trial_result):
            continue

        mcode_results = trial_result["McodeResults"]
        if "natural_language_summary" not in mcode_results:
            continue

        # Extract trial ID
        trial_id = "unknown"
        if protocol := trial_result.get("protocolSection"):
            if ident := protocol.get("identificationModule"):
                trial_id = ident.get("nctId", "unknown")

        summaries.append(
            {
                "trial_id": trial_id,
                "summary": mcode_results["natural_language_summary"],
                "mcode_elements": mcode_results.get("mcode_mappings", []),
            }
        )

    return summaries


def print_processing_summary(metadata: Any, ingested: bool, logger: Any) -> None:
    """Print processing summary."""
    if not metadata:
        return

    total = metadata.get("total_trials", 0)
    successful = metadata.get("successful", 0)
    failed = metadata.get("failed", 0)
    success_rate = metadata.get("success_rate", 0)

    logger.info(f"📊 Total trials: {total}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📈 Success rate: {success_rate:.1f}%")

    if ingested:
        stored = metadata.get("stored_in_memory", False)
        status = "🧠 Stored in CORE Memory" if stored else "💾 Storage failed"
        logger.info(status)
    else:
        logger.info("💾 Storage disabled")


if __name__ == "__main__":
    main()
