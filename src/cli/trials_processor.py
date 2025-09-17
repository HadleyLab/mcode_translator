#!/usr/bin/env python3
"""
Trials Processor - Process clinical trials with mCODE mapping.

A command-line interface for processing clinical trial data with mCODE mapping
and storing the resulting summaries to CORE Memory or saving as JSON/NDJSON files.

Features:
- Extract mCODE elements from clinical trial data
- Save processed data in JSON array format or NDJSON format
- Optional CORE Memory storage (use --ingest to enable)
- Configurable LLM models and prompts
- Concurrent processing with worker threads
"""

import argparse
import os
import sys
from pathlib import Path

from src.shared.cli_utils import McodeCLI
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config
from src.utils.logging_config import get_logger
from src.workflows.trials_processor_workflow import ClinicalTrialsProcessorWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials processor."""
    parser = argparse.ArgumentParser(
        description="Process clinical trials with mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Process trials and save as NDJSON (recommended for large datasets)
   python -m src.cli.trials_processor trials.json --out mcode_trials.ndjson

   # Process trials from file and store in core memory
   python -m src.cli.trials_processor trials.json --ingest

   # Process with specific model and prompt
   python -m src.cli.trials_processor trials.json --model gpt-4 --prompt direct_mcode_evidence_based

   # Process single trial
   python -m src.cli.trials_processor single_trial.json --ingest

   # Custom core memory settings
   python -m src.cli.trials_processor trials.json --ingest --memory-source custom_source

 Input Formats:
   JSON:  Standard JSON array format - [{"protocolSection": {...}, ...}]
   NDJSON: Newline-delimited JSON - one JSON object per line

 Output Formats:
   NDJSON: Newline-delimited JSON - one JSON object per line (recommended for streaming)
         """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)

    # Processor-specific arguments
    McodeCLI.add_processor_args(parser)

    # Input arguments
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to NDJSON file containing trial data (or use --in)"
    )

    parser.add_argument(
        "--in",
        dest="input_file",
        help="Path to NDJSON file containing trial data (alternative to positional argument)"
    )

    parser.add_argument(
        "--out",
        dest="output_file",
        help="Path to save processed mCODE data (NDJSON format). If not specified, writes to stdout",
    )

    return parser


def main() -> None:
    """Main entry point for trials processor CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)
    logger = get_logger(__name__)

    # Determine input file (positional argument takes precedence)
    input_file = args.input_file
    if not input_file:
        logger.error("No input file specified. Use positional argument or --in")
        sys.exit(1)

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Load input data from file or stdin
    try:
        import json

        if args.input_file:
            # Try to read as JSON array first, then fall back to NDJSON
            trial_data = []
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                logger.error("Input file is empty")
                sys.exit(1)

            # Try to parse as JSON array first
            try:
                json_data = json.loads(content)
                if isinstance(json_data, list):
                    # JSON array format
                    trial_data = json_data
                    logger.info(f"📄 Read {len(trial_data)} trials from JSON array format")
                elif isinstance(json_data, dict):
                    # Single JSON object
                    trial_data = [json_data]
                    logger.info("📄 Read single trial from JSON format")
                else:
                    logger.error("Invalid JSON format. Expected array or object.")
                    sys.exit(1)
            except json.JSONDecodeError:
                # Fall back to NDJSON format
                logger.info("📄 JSON parsing failed, trying NDJSON format...")
                for line in content.split('\n'):
                    line = line.strip()
                    if line:  # Skip empty lines
                        trial_data.append(json.loads(line))
                logger.info(f"📄 Read {len(trial_data)} trials from NDJSON format")
        else:
            # Read from stdin as NDJSON
            trial_data = []
            for line in sys.stdin:
                line = line.strip()
                if line:  # Skip empty lines
                    trial_data.append(json.loads(line))

        if not trial_data:
            logger.error("No input data provided")
            sys.exit(1)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        sys.exit(1)

    # Prepare workflow parameters
    workflow_kwargs = {
        "model": args.model,
        "prompt": args.prompt,
        "store_in_memory": args.ingest,
        "workers": args.workers,
    }

    # Handle different input formats
    if isinstance(trial_data, list):
        # Multiple trials
        workflow_kwargs["trials_data"] = trial_data
        logger.info(f"🔬 Processing {len(trial_data)} trials...")
    elif isinstance(trial_data, dict):
        # Single trial or batch format
        if "studies" in trial_data:
            # ClinicalTrials.gov API format
            workflow_kwargs["trials_data"] = trial_data["studies"]
            logger.info(
                f"🔬 Processing {len(trial_data['studies'])} trials from API format..."
            )
        elif "successful_trials" in trial_data:
            # Batch processing format
            workflow_kwargs["trials_data"] = trial_data["successful_trials"]
            logger.info(
                f"🔬 Processing {len(trial_data['successful_trials'])} trials from batch format..."
            )
        else:
            # Single trial
            workflow_kwargs["trials_data"] = [trial_data]
            logger.info("🔬 Processing single trial...")
    else:
        logger.error("Invalid input format. Expected JSON array or object.")
        sys.exit(1)

    # Initialize core memory storage if requested
    memory_storage = False  # Default to disabled
    if args.ingest:
        try:
            # Use centralized configuration
            memory_storage = McodeMemoryStorage(source=args.memory_source)
            logger.info(
                f"🧠 Initialized CORE Memory storage (source: {args.memory_source})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize CORE Memory: {e}")
            logger.info(
                "Check your COREAI_API_KEY environment variable and core_memory_config.json"
            )
            sys.exit(1)

    # Initialize and execute workflow
    try:
        logger.info("Initializing trials processor workflow...")
        workflow = ClinicalTrialsProcessorWorkflow(config, memory_storage)
        logger.info("Executing trials processor workflow...")
        result = workflow.execute(**workflow_kwargs)
        logger.info("Trials processor workflow execution completed")

        if result.success:
            logger.info("✅ Trials processing completed successfully!")

            # Save processed mCODE data to file or stdout
            if result.data:
                try:
                    import json

                    # Extract mCODE elements for output
                    mcode_data = []
                    for item in result.data:
                        # Handle both dict objects and objects with __dict__
                        if isinstance(item, dict):
                            item_dict = item
                        elif hasattr(item, "__dict__"):
                            item_dict = item.__dict__.copy()
                        else:
                            continue

                        # Extract trial ID from nested structure
                        trial_id = None
                        if (
                            "protocolSection" in item_dict
                            and "identificationModule" in item_dict["protocolSection"]
                        ):
                            trial_id = item_dict["protocolSection"][
                                "identificationModule"
                            ].get("nctId")

                        # Extract mCODE results and original trial data
                        if "McodeResults" in item_dict and item_dict["McodeResults"]:
                            mcode_results = item_dict["McodeResults"]

                            # Convert McodeElement objects to dictionaries
                            if (
                                "mcode_mappings" in mcode_results
                                and mcode_results["mcode_mappings"]
                            ):
                                mappings = []
                                for mapping in mcode_results["mcode_mappings"]:
                                    if hasattr(mapping, "__dict__"):
                                        mappings.append(mapping.__dict__)
                                    else:
                                        mappings.append(mapping)
                                mcode_results["mcode_mappings"] = mappings

                            # Create output structure with mCODE data and original trial
                            output_item = {
                                "trial_id": trial_id,
                                "mcode_elements": mcode_results,
                                "original_trial_data": item_dict,  # Keep original for summarizer
                            }
                            mcode_data.append(output_item)
                        else:
                            # If no McodeResults, include original data anyway
                            logger.warning(
                                f"No McodeResults found for trial {trial_id}"
                            )
                            output_item = {
                                "trial_id": trial_id,
                                "mcode_elements": {
                                    "note": "No mCODE mappings generated"
                                },
                                "original_trial_data": item_dict,
                            }
                            mcode_data.append(output_item)

                    # Output as NDJSON to file or stdout
                    if args.output_file:
                        with open(args.output_file, "w", encoding="utf-8") as f:
                            for item in mcode_data:
                                json.dump(item, f, ensure_ascii=False, default=str)
                                f.write("\n")
                        logger.info(f"💾 mCODE data saved as NDJSON to: {args.output_file}")
                    else:
                        # Write to stdout
                        for item in mcode_data:
                            json.dump(item, sys.stdout, ensure_ascii=False, default=str)
                            sys.stdout.write("\n")
                        sys.stdout.flush()
                        logger.info("📤 mCODE data written to stdout")

                except Exception as e:
                    logger.error(f"Failed to save processed data: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

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
                        logger.info("🧠 mCODE summaries stored in CORE Memory")
                    else:
                        logger.info("💾 mCODE summaries NOT stored (error)")
                else:
                    logger.info("💾 mCODE summaries NOT stored (storage disabled)")

                # Print model/prompt info
                model_used = metadata.get("model_used")
                prompt_used = metadata.get("prompt_used")
                if model_used:
                    logger.info(f"🤖 Model: {model_used}")
                if prompt_used:
                    logger.info(f"📝 Prompt: {prompt_used}")

        else:
            logger.error(f"Trials processing failed: {result.error_message}")
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
