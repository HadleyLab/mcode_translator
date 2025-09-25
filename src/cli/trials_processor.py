#!/usr/bin/env python3
"""
Trials Processor - Process clinical trials with mCODE mapping.

A streamlined command-line interface for processing clinical trial data with mCODE mapping
and storing results to CORE Memory or NDJSON files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.data_loader import load_ndjson_data, extract_trial_id
from src.utils.error_handler import (
    handle_cli_error,
    log_operation_start,
    log_operation_success,
)
from src.utils.logging_config import get_logger
from src.workflows.trials_processor_workflow import ClinicalTrialsProcessorWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create streamlined argument parser for trials processor."""
    parser = argparse.ArgumentParser(
        description="Process clinical trials with mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli.trials_processor trials.ndjson --out mcode_trials.ndjson
  python -m src.cli.trials_processor trials.ndjson --ingest --model gpt-4
        """,
    )

    # Core arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)
    McodeCLI.add_processor_args(parser)

    # Essential input/output arguments
    parser.add_argument("input_file", help="Path to NDJSON file containing trial data")
    parser.add_argument(
        "--out",
        dest="output_file",
        help="Path to save processed mCODE data (NDJSON format)",
    )

    return parser


def save_processed_data(data: List[Any], output_file: Optional[str], logger) -> None:
    """Save processed mCODE data to file or stdout."""
    mcode_data = []
    for item in data:
        item_dict = item if isinstance(item, dict) else item.__dict__
        trial_id = extract_trial_id(item_dict)

        mcode_results = item_dict.get("McodeResults", {})
        if isinstance(mcode_results, dict) and "mcode_mappings" in mcode_results:
            # Convert McodeElement objects to dicts
            mappings = []
            for mapping in mcode_results["mcode_mappings"]:
                if hasattr(mapping, "__dict__"):
                    mappings.append(mapping.__dict__)
                else:
                    mappings.append(mapping)
            mcode_results["mcode_mappings"] = mappings

        output_item = {
            "trial_id": trial_id,
            "mcode_elements": mcode_results,
            "original_trial_data": item_dict,
        }
        mcode_data.append(output_item)

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
    metadata: Optional[Dict[str, Any]], ingested: bool, logger
) -> None:
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

    # Print model/prompt info
    if model := metadata.get("model_used"):
        logger.info(f"🤖 Model: {model}")
    if prompt := metadata.get("prompt_used"):
        logger.info(f"📝 Prompt: {prompt}")


def main(args: Optional[argparse.Namespace] = None) -> None:
    """Main entry point for trials processor CLI."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    # Setup logging and configuration
    McodeCLI.setup_logging(args)
    logger = get_logger(__name__)
    config = McodeCLI.create_config(args)

    # Validate and load input file
    if not args.input_file:
        handle_cli_error(ValueError("Input file is required"))

    try:
        input_path = Path(args.input_file)
    except TypeError:
        handle_cli_error(ValueError("Input file path is invalid"))

    if not input_path.exists():
        handle_cli_error(FileNotFoundError(f"Input file not found: {input_path}"))

    try:
        trial_data = load_ndjson_data(input_path, "trials")
        if not trial_data:
            handle_cli_error(ValueError("No trial data found in input file"))
    except (json.JSONDecodeError, FileNotFoundError) as e:
        handle_cli_error(e, "Failed to load input file")

    # Initialize memory storage if requested
    memory_storage = None
    if args.ingest:
        try:
            memory_storage = McodeMemoryStorage(source=args.memory_source)
            logger.info(
                f"🧠 Initialized CORE Memory storage (source: {args.memory_source})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CORE Memory: {e}")
            sys.exit(1)

    # Execute workflow
    try:
        log_operation_start(f"Processing {len(trial_data)} trials")
        workflow = ClinicalTrialsProcessorWorkflow(config, memory_storage)
        result = workflow.execute(
            trials_data=trial_data,
            model=args.model,
            prompt=args.prompt,
            store_in_memory=args.ingest,
            workers=args.workers,
            cli_args=args,
        )

        if not result.success:
            handle_cli_error(
                ValueError(f"Trials processing failed: {result.error_message}")
            )

        log_operation_success("Trials processing completed successfully")

        # Save results
        if result.data:
            data_list = result.data if isinstance(result.data, list) else [result.data]
            save_processed_data(data_list, args.output_file, logger)

        # Print summary
        print_processing_summary(result.metadata, args.ingest, logger)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        handle_cli_error(
            e, "Unexpected error during trials processing", verbose=args.verbose
        )


if __name__ == "__main__":
    main()
