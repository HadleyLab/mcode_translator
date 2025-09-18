#!/usr/bin/env python3
"""
Trials Summarizer - Generate natural language summaries from mCODE trial data.

A command-line interface for generating comprehensive natural language summaries
from processed mCODE trial data. Takes mCODE-mapped trial data as input and
produces human-readable summaries for CORE Memory storage.

Features:
- Generate comprehensive trial summaries from mCODE data
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
from src.workflows.trials_summarizer_workflow import TrialsSummarizerWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials summarizer."""
    parser = argparse.ArgumentParser(
        description="Generate natural language summaries from mCODE trial data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize from file and store in CORE Memory
  python -m src.cli.trials_summarizer --in mcode_trials.ndjson --ingest

  # Summarize from stdin, output to stdout
  cat mcode_trials.ndjson | python -m src.cli.trials_summarizer

  # Summarize from file, save summaries to file
  python -m src.cli.trials_summarizer --in mcode_trials.ndjson --out trial_summaries.ndjson

  # Preview summaries without storing
  python -m src.cli.trials_summarizer --in mcode_trials.ndjson --verbose

Input/Output:
  Input: NDJSON format with mCODE trial data (one trial per line)
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
        help="Input file with mCODE trial data (NDJSON format). If not specified, reads from stdin",
    )

    parser.add_argument(
        "--out",
        dest="output_file",
        help="Output file for summaries (NDJSON format). If not specified, writes to stdout",
    )

    return parser


def load_mcode_trials(input_file: Optional[str]) -> List[Dict]:
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
    """Main entry point for trials summarizer CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)
    global logger
    logger = get_logger(__name__)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Load mCODE trial data
    try:
        logger.info("Loading mCODE trial data...")
        mcode_trials = load_mcode_trials(args.input_file)
        logger.info(f"Loaded {len(mcode_trials)} mCODE trial records")
    except Exception as e:
        logger.error(f"Failed to load mCODE trial data: {e}")
        sys.exit(1)

    if not mcode_trials:
        logger.error("No valid mCODE trial data found")
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
        "model": args.model,
        "prompt": args.prompt,
        "store_in_memory": args.ingest and not args.dry_run,
        "workers": args.workers,
    }

    # Convert mCODE data back to trial format for summarization
    trial_data = []
    for mcode_trial in mcode_trials:
        # Extract the original trial data from mCODE format
        if "original_trial_data" in mcode_trial:
            trial_data.append(mcode_trial["original_trial_data"])
        elif "trial_data" in mcode_trial:
            trial_data.append(mcode_trial["trial_data"])
        else:
            logger.warning("Trial missing original data, using mCODE data as fallback")
            trial_data.append(mcode_trial)

    workflow_kwargs["trials_data"] = trial_data

    # Initialize and execute workflow
    try:
        logger.info("Initializing trials summarizer workflow...")
        workflow = TrialsSummarizerWorkflow(config, memory_storage)

        logger.info("Generating natural language summaries...")
        result = workflow.execute(**workflow_kwargs)

        if result.success:
            logger.info("✅ Trial summarization completed successfully!")

            # Extract and save summaries
            summaries = []
            if result.data:
                for trial_result in result.data:
                    if isinstance(trial_result, dict) and "McodeResults" in trial_result:
                        mcode_results = trial_result["McodeResults"]
                        if "natural_language_summary" in mcode_results:
                            # Extract trial ID
                            trial_id = "unknown"
                            if "protocolSection" in trial_result:
                                protocol = trial_result["protocolSection"]
                                if "identificationModule" in protocol:
                                    trial_id = protocol["identificationModule"].get("nctId", "unknown")

                            summary = {
                                "trial_id": trial_id,
                                "summary": mcode_results["natural_language_summary"],
                                "mcode_elements": mcode_results.get("mcode_mappings", []),
                            }
                            summaries.append(summary)

            # Save summaries
            if summaries:
                save_summaries(summaries, args.output_file)
                logger.info(f"💾 Saved {len(summaries)} trial summaries")

                if args.output_file:
                    logger.info(f"📁 Output saved to: {args.output_file}")
                else:
                    logger.info("📤 Output written to stdout")

            # Print summary
            metadata = result.metadata
            if metadata:
                total_trials = metadata.get("total_trials", 0)
                successful = metadata.get("successful", 0)
                failed = metadata.get("failed", 0)
                success_rate = metadata.get("success_rate", 0)

                logger.info(f"📊 Total trials: {total_trials}")
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
            logger.error(f"Trial summarization failed: {result.error_message}")
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