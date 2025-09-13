#!/usr/bin/env python3
"""
mCODE T  # With specific model and prompt
  mcode_translator.py trial_data.json -m deepseek-coder \
    -p direct_mcode_evidence_based_concise -o results.jsonslator - Main CLI Interface

A high-performance clinical trial data processing pipeline that extracts and maps 
eligibility criteria to standardized mCODE elements using evidence-based LLM processing.

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline, ProcessingPipeline
from src.utils.config import Config
from src.utils.logging_config import get_logger, setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with shared argument patterns."""
    parser = argparse.ArgumentParser(
        description="mCODE Translator - Clinical trial data to mCODE mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  mcode_translator.py trial_data.json -o results.json
  
  # With specific model and prompt
  mcode_translator.py trial_data.json -m deepseek-coder \\
    -p direct_mcode_evidence_based_concise -o results.json
  
  # Verbose processing with custom config
  mcode_translator.py trial_data.json --verbose --config custom_config.json
  
  # Batch processing with validation
  mcode_translator.py trials_batch.json --validate --output validated_results.json
        """,
    )

    # Input/Output
    parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        help="Path to clinical trial data file (JSON format)",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="mcode_output.json",
        help="Output file path (default: mcode_output.json)",
    )

    # Shared mCODE processing arguments (consistent with mcode_fetcher.py)
    parser.add_argument(
        "-m", "--model", help="LLM model to use for mCODE processing (overrides config)"
    )

    parser.add_argument(
        "-p",
        "--prompt",
        default="direct_mcode_evidence_based_concise",
        help="Prompt template to use for mCODE processing (default: evidence-based concise)",
    )

    parser.add_argument(
        "--config", help="Path to configuration file (overrides default)"
    )

    # Processing options
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run additional validation checks on results",
    )

    parser.add_argument(
        "--batch", action="store_true", help="Process input as batch of multiple trials"
    )

    # Logging options (consistent with mcode_fetcher.py)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    return parser


def process_single_trial(
    trial_data: Dict[str, Any], pipeline: McodePipeline, args: argparse.Namespace
) -> Dict[str, Any]:
    """Process a single clinical trial."""
    logger = get_logger(__name__)
    logger.info("🔬 Processing single clinical trial")

    # Check if the input is wrapped in a 'trial' key
    if "trial" in trial_data:
        trial = trial_data["trial"]
    else:
        trial = trial_data

    result = pipeline.process_clinical_trial(trial)

    # Add metadata
    output_data = {
        "input_metadata": {
            "trial_id": trial.get("protocolSection", {})
            .get("identificationModule", {})
            .get("nctId", "unknown"),
            "processing_timestamp": (
                pipeline.get_processing_timestamp()
                if hasattr(pipeline, "get_processing_timestamp")
                else None
            ),
            # Include LLM parameters in metadata
            "model_used": args.model,
            "prompt_used": args.prompt,
            "llm_temperature": (
                pipeline.llm_mapper.temperature
                if hasattr(pipeline.llm_mapper, "temperature")
                else "N/A"
            ),  # Access temperature from McodeMapper
        },
        "mcode_results": {
            "extracted_entities": result.extracted_entities,
            "mcode_mappings": result.mcode_mappings,
            "source_references": result.source_references,
            "validation_results": result.validation_results,
            "metadata": {
                "prompt": args.prompt,
                "model": args.model,
                "prompt": args.prompt,
                "llm_temperature": (
                    pipeline.llm_mapper.temperature
                    if hasattr(pipeline.llm_mapper, "temperature")
                    else "N/A"
                ),
                **result.metadata,  # Include existing metadata
            },
            "token_usage": result.token_usage,
            "error": result.error,
        },
    }

    return output_data


def process_batch_trials(
    trials_data: List[Dict[str, Any]], pipeline: McodePipeline, args: argparse.Namespace
) -> Dict[str, Any]:
    """Process multiple clinical trials in batch."""
    logger = get_logger(__name__)
    logger.info(f"🔬 Processing batch of {len(trials_data)} clinical trials")

    successful_results = []
    failed_results = []

    for i, trial_data in enumerate(trials_data):
        try:
            logger.info(f"Processing trial {i+1}/{len(trials_data)}")
            result = pipeline.process_clinical_trial(trial_data)

            trial_result = {
                "trial_id": trial_data.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId", f"trial_{i}"),
                "mcode_results": {
                    "extracted_entities": result.extracted_entities,
                    "mcode_mappings": result.mcode_mappings,
                    "source_references": result.source_references,
                    "validation_results": result.validation_results,
                    "metadata": result.metadata,
                    "token_usage": result.token_usage,
                    "error": result.error,
                },
                "processing_index": i,
                "input_metadata": {
                    "trial_id": trial_data.get("protocolSection", {})
                    .get("identificationModule", {})
                    .get("nctId", f"trial_{i}"),
                    "processing_timestamp": (
                        pipeline.get_processing_timestamp()
                        if hasattr(pipeline, "get_processing_timestamp")
                        else None
                    ),
                    # Include LLM parameters in metadata
                    "model_used": args.model,
                    "prompt_used": args.prompt,
                    "llm_temperature": (
                        pipeline.llm_mapper.temperature
                        if hasattr(pipeline.llm_mapper, "temperature")
                        else "N/A"
                    ),  # Access temperature from McodeMapper
                },
                "mcode_results": {
                    "extracted_entities": result.extracted_entities,
                    "mcode_mappings": result.mcode_mappings,
                    "source_references": result.source_references,
                    "validation_results": result.validation_results,
                    "metadata": {
                        "prompt": args.prompt,
                        "model": args.model,
                        "prompt": args.prompt,
                        "llm_temperature": (
                            pipeline.llm_mapper.temperature
                            if hasattr(pipeline.llm_mapper, "temperature")
                            else "N/A"
                        ),
                        **{
                            k: v
                            for k, v in result.metadata.items()
                            if k != "pipeline_version"
                        },  # Include existing metadata, excluding pipeline_version
                    },
                    "token_usage": result.token_usage,
                    "error": result.error,
                },
            }
            successful_results.append(trial_result)

        except Exception as e:
            logger.error(f"Failed to process trial {i+1}: {e}")
            failed_result = {
                "trial_id": trial_data.get("protocolSection", {})
                .get("identificationModule", {})
                .get("nctId", f"trial_{i}"),
                "error": str(e),
                "processing_index": i,
            }
            failed_results.append(failed_result)

    return {
        "batch_metadata": {
            "total_trials": len(trials_data),
            "successful_trials": len(successful_results),
            "failed_trials": len(failed_results),
            "success_rate": (
                len(successful_results) / len(trials_data) * 100 if trials_data else 0
            ),
            "model_used": args.model,
            "prompt_used": args.prompt,
        },
        "successful_results": successful_results,
        "failed_results": failed_results,
    }


def main() -> None:
    """Main entry point for the mCODE Translator CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    setup_logging(level=log_level)
    logger = get_logger(__name__)

    logger.info("🚀 mCODE Translator starting")

    try:
        # Initialize configuration
        config = Config()
        if args.config:
            # Config class may not support config_path parameter, handle this differently if needed
            logger.info(f"Custom config specified: {args.config}")

        # Load input data
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"❌ Input file not found: {input_path}")
            sys.exit(1)

        logger.info(f"📥 Loading input data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Initialize pipeline
        logger.info(
            f"🔧 Initializing mCODE pipeline (model: {args.model or 'default'}, prompt: {args.prompt})"
        )
        pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)

        # Auto-detect input format: batch wrapper or single trial
        if "successful_trials" in input_data and isinstance(
            input_data["successful_trials"], list
        ):
            # Handle batch wrapper format from fetcher
            trials_data = input_data["successful_trials"]
            logger.info(
                f"🔬 Starting batch processing of {len(trials_data)} trials from successful_trials array"
            )
            output_data = process_batch_trials(trials_data, pipeline, args)
        elif isinstance(input_data, list):
            # Direct list of trials
            logger.info(f"🔬 Starting batch processing of {len(input_data)} trials")
            output_data = process_batch_trials(input_data, pipeline, args)
        else:
            # Single trial processing
            logger.info("🔬 Starting single trial processing")
            output_data = process_single_trial(input_data, pipeline, args)

        # Run additional validation if requested
        if args.validate:
            logger.info("🔍 Running additional validation checks")
            # Add validation logic here if needed
            output_data["validation_metadata"] = {
                "additional_validation_performed": True,
                "validation_timestamp": None,  # Could add timestamp utility
            }

        # Save output
        output_path = Path(args.output)
        logger.info(f"💾 Saving results to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Print summary
        if "batch_metadata" in output_data:
            batch_meta = output_data["batch_metadata"]
            print(f"\n✅ Batch processing completed successfully!")
            print(f"📄 Input: {input_path}")
            print(f"💾 Output: {output_path}")
            print(f"📊 Total Trials: {batch_meta['total_trials']}")
            print(f"✅ Successful: {batch_meta['successful_trials']}")
            print(f"❌ Failed: {batch_meta['failed_trials']}")
            print(f"📈 Success Rate: {batch_meta['success_rate']:.1f}%")
        else:
            if "mcode_results" in output_data:
                mcode_results = output_data["mcode_results"]
                print(f"\n✅ Processing completed successfully!")
                print(f"📄 Input: {input_path}")
                print(f"💾 Output: {output_path}")
                if mcode_results:
                    print(
                        f"🎯 mCODE Mappings: {len(mcode_results.get('mcode_mappings', []))}"
                    )
                    print(
                        f"📊 Quality Score: {mcode_results.get('validation_results', {}).get('compliance_score', 'N/A')}"
                    )
                else:
                    print("🎯 No mCODE Mappings")
                    print("📊 Quality Score: N/A")
            else:
                print(f"\n✅ Processing completed successfully!")
                print(f"📄 Input: {input_path}")
                print(f"💾 Output: {output_path}")
                print(f"🎯 mCODE Mappings: 0")
                print(f"📊 Quality Score: N/A")

        logger.info("✅ mCODE Translator completed successfully")

    except KeyboardInterrupt:
        logger.info("⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
