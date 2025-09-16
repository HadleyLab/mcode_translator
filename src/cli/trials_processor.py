#!/usr/bin/env python3
"""
Trials Processor - Process clinical trials with mCODE mapping.

A command-line interface for processing clinical trial data with mCODE mapping
and storing the resulting summaries to CORE Memory or saving as JSON/NDJSON files.

Features:
- Extract mCODE elements from clinical trial data
- Save processed data in JSON array format or NDJSON format
- Support for dry-run mode (no CORE Memory storage)
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
from src.workflows.trials_processor_workflow import TrialsProcessorWorkflow


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for trials processor."""
    parser = argparse.ArgumentParser(
        description="Process clinical trials with mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process trials and save as JSON array
  python -m src.cli.trials_processor trials.json --output mcode_trials.json --dry-run

  # Process trials and save as NDJSON (recommended for large datasets)
  python -m src.cli.trials_processor trials.json --output mcode_trials.ndjson --dry-run

  # Process trials from file and store in core memory
  python -m src.cli.trials_processor trials.json --store-in-core-memory

  # Process with specific model and prompt
  python -m src.cli.trials_processor trials.json --model gpt-4 --prompt direct_mcode_evidence_based

  # Process single trial
  python -m src.cli.trials_processor single_trial.json --store-in-core-memory

  # Dry run to preview what would be stored
  python -m src.cli.trials_processor trials.json --dry-run --verbose

  # Custom core memory settings
  python -m src.cli.trials_processor trials.json --store-in-core-memory --memory-source custom_source

Output Formats:
  JSON:  Standard JSON array format - [{"trial_id": "...", "mcode_elements": {...}}]
  NDJSON: Newline-delimited JSON - one JSON object per line (recommended for streaming)
        """,
    )

    # Add shared arguments
    McodeCLI.add_core_args(parser)
    McodeCLI.add_memory_args(parser)

    # Processor-specific arguments
    McodeCLI.add_processor_args(parser)

    # Input arguments
    parser.add_argument("input_file", help="Path to JSON file containing trial data")

    parser.add_argument(
        "--output",
        help="Path to save processed mCODE data (JSON array or NDJSON format). Use .ndjson extension for newline-delimited format"
    )

    return parser


def main() -> None:
    """Main entry point for trials processor CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    McodeCLI.setup_logging(args)
    logger = get_logger(__name__)

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create configuration
    config = McodeCLI.create_config(args)

    # Load input data
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = f.read().strip()

        if not input_data:
            logger.error(f"Input file is empty: {input_path}")
            sys.exit(1)

        # Parse JSON
        import json

        trial_data = json.loads(input_data)

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
        "store_in_memory": args.store_in_core_memory and not args.dry_run,
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

    # Initialize core memory storage if needed
    memory_storage = None
    if args.store_in_core_memory:
        try:
            # Use centralized configuration
            memory_storage = McodeMemoryStorage(source=args.memory_source)
            logger.info(f"🧠 Initialized CORE Memory storage (source: {args.memory_source})")
        except Exception as e:
            logger.error(f"Failed to initialize CORE Memory: {e}")
            logger.info(
                "Check your COREAI_API_KEY environment variable and core_memory_config.json"
            )
            sys.exit(1)

    # Initialize and execute workflow
    try:
        logger.info("Initializing trials processor workflow...")
        workflow = TrialsProcessorWorkflow(config, memory_storage)
        logger.info("Executing trials processor workflow...")
        result = workflow.execute(**workflow_kwargs)
        logger.info("Trials processor workflow execution completed")

        if result.success:
            logger.info("✅ Trials processing completed successfully!")

            # Save processed data to JSON file if requested
            if hasattr(args, 'output') and args.output and result.data:
                try:
                    import json
                    # Debug: Check what we actually have
                    logger.info(f"Debug: result.data type: {type(result.data)}")
                    if result.data:
                        logger.info(f"Debug: result.data length: {len(result.data)}")
                        if len(result.data) > 0:
                            logger.info(f"Debug: first item type: {type(result.data[0])}")
                            if hasattr(result.data[0], '__dict__'):
                                logger.info(f"Debug: first item keys: {list(result.data[0].__dict__.keys())}")
                                # Check if McodeResults exists
                                if 'McodeResults' in result.data[0].__dict__:
                                    logger.info(f"Debug: McodeResults found: {type(result.data[0].__dict__['McodeResults'])}")
                                else:
                                    logger.info("Debug: No McodeResults in first item")
                                    # Show all keys to understand structure
                                    for key in result.data[0].__dict__.keys():
                                        logger.info(f"Debug: Key '{key}': {type(result.data[0].__dict__[key])}")
                            elif isinstance(result.data[0], dict):
                                logger.info(f"Debug: first item dict keys: {list(result.data[0].keys())}")
                                if 'McodeResults' in result.data[0]:
                                    logger.info(f"Debug: McodeResults found in dict: {type(result.data[0]['McodeResults'])}")
                                else:
                                    logger.info("Debug: No McodeResults in first dict item")

                    # Extract only mCODE elements for cleaner JSON structure
                    mcode_only_data = []
                    for item in result.data:
                        # Handle both dict objects and objects with __dict__
                        if isinstance(item, dict):
                            item_dict = item
                        elif hasattr(item, '__dict__'):
                            item_dict = item.__dict__.copy()
                        else:
                            continue

                        # Extract trial ID from nested structure
                        trial_id = None
                        if ('protocolSection' in item_dict and
                            'identificationModule' in item_dict['protocolSection']):
                            trial_id = item_dict['protocolSection']['identificationModule'].get('nctId')

                        # Extract only the McodeResults
                        if 'McodeResults' in item_dict and item_dict['McodeResults']:
                            mcode_results = item_dict['McodeResults']

                            # Convert McodeElement objects to dictionaries
                            if 'mcode_mappings' in mcode_results and mcode_results['mcode_mappings']:
                                mappings = []
                                for mapping in mcode_results['mcode_mappings']:
                                    if hasattr(mapping, '__dict__'):
                                        mappings.append(mapping.__dict__)
                                    else:
                                        mappings.append(mapping)
                                mcode_results['mcode_mappings'] = mappings

                            # Create clean mCODE-only structure
                            clean_item = {
                                'trial_id': trial_id,
                                'mcode_elements': mcode_results
                            }
                            mcode_only_data.append(clean_item)
                        else:
                            # If no McodeResults, create a placeholder
                            logger.warning(f"No McodeResults found for trial {trial_id}")
                            clean_item = {
                                'trial_id': trial_id,
                                'mcode_elements': {'note': 'No mCODE mappings generated'}
                            }
                            mcode_only_data.append(clean_item)

                    logger.info(f"Debug: mcode_only_data length: {len(mcode_only_data)}")
                    # Save as NDJSON (Newline Delimited JSON) format
                    with open(args.output, 'w', encoding='utf-8') as f:
                        for item in mcode_only_data:
                            json.dump(item, f, ensure_ascii=False, default=str)
                            f.write('\n')
                    logger.info(f"💾 mCODE-only data saved as NDJSON to: {args.output}")
                except Exception as e:
                    logger.error(f"Failed to save processed data: {e}")
                    # Try alternative serialization
                    try:
                        with open(args.output, 'w', encoding='utf-8') as f:
                            json.dump(str(result.data), f, indent=2, ensure_ascii=False)
                        logger.info(f"💾 Processed data saved as string to: {args.output}")
                    except Exception as e2:
                        logger.error(f"Failed alternative serialization: {e2}")

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

                if args.store_in_core_memory:
                    stored = metadata.get("stored_in_memory", False)
                    if stored:
                        logger.info("🧠 mCODE summaries stored in CORE Memory")
                    else:
                        logger.info("💾 mCODE summaries NOT stored (dry run or error)")

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
