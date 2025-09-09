#!/usr/bin/env python3
"""
mCODE Fetcher - Clinical T  # Concurrent processing with mCODE mapping
  mcode_fetcher.py --condition "breast cancer" --concurrent --process \
    --workers 8 -m deepseek-coder -p direct_mcode_evidence_based_concisel Data Fetcher with Concurrent Processing

A high-performance clinical trial data fetcher that searches ClinicalTrials.gov
and optionally processes results with mCODE mapping using concurrent processing.

Author: mCODE Translation Team
Version: 2.0.0
License: MIT
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.pipeline.concurrent_fetcher import (
    ConcurrentFetcher, 
    ProcessingConfig,
    concurrent_search_and_process,
    concurrent_process_trials
)
from src.pipeline.fetcher import (
    search_trials,
    get_full_study,
    calculate_total_studies,
    ClinicalTrialsAPIError
)
from src.utils.config import Config
from src.utils.logging_config import get_logger, setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with shared argument patterns."""
    parser = argparse.ArgumentParser(
        description="mCODE Fetcher - Clinical trial data fetcher with concurrent processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search and export
  mcode_fetcher.py --condition "breast cancer" --limit 10 --output results.json
  
  # Fetch specific trials
  mcode_fetcher.py --nct-id NCT12345678 --output trial.json
  mcode_fetcher.py --nct-ids "NCT001,NCT002,NCT003" --output trials.json
  
  # Search with mCODE processing (sequential)
  mcode_fetcher.py --condition "lung cancer" --process -m deepseek-coder
  
  # Concurrent processing with mCODE mapping
  mcode_fetcher.py --condition "breast cancer" --concurrent --process \\
    --workers 8 --model deepseek-coder --prompt direct_mcode_evidence_based_concise
  
  # Count total available studies
  mcode_fetcher.py --condition "cancer" --count-only
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--condition", "-c",
        help="Medical condition to search for (e.g., 'breast cancer')"
    )
    input_group.add_argument(
        "--nct-id", "-n", 
        help="Specific NCT ID to fetch (e.g., 'NCT12345678')"
    )
    input_group.add_argument(
        "--nct-ids",
        help="Comma-separated list of NCT IDs to process"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path (JSON format). If not specified, prints to stdout"
    )
    
    # Search parameters
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count total studies matching condition (no data fetch)"
    )
    
    # Processing options
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process results with mCODE mapping pipeline"
    )
    
    # Shared mCODE processing arguments (consistent with mcode_translator.py)
    parser.add_argument(
        "-m", "--model",
        help="LLM model to use for mCODE processing (overrides config)"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="direct_mcode_evidence_based_concise", 
        help="Prompt template to use for mCODE processing (default: evidence-based concise)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (overrides default)"
    )
    
    # Concurrent processing options
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Use concurrent processing for improved performance"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of concurrent workers (default: 5, only with --concurrent)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for concurrent processing (default: 10)"
    )
    
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress updates during concurrent processing"
    )
    
    # Logging options (consistent with mcode_translator.py)
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    return parser


async def fetch_and_process_concurrent(args: argparse.Namespace) -> Dict[str, Any]:
    """Handle concurrent fetching and processing."""
    get_logger(__name__).info("🚀 Starting concurrent processing")
    
    if args.condition:
        # Concurrent search and process
        result = await concurrent_search_and_process(
            condition=args.condition,
            limit=args.limit,
            max_workers=args.workers,
            batch_size=args.batch_size,
            process_criteria=args.process,
            process_trials=args.process,
            model_name=args.model,
            prompt_name=args.prompt,
            export_path=None,  # We'll handle export separately
            progress_updates=not args.no_progress
        )
        
        return {
            "search_condition": args.condition,
            "processing_type": "concurrent_search_and_process",
            "summary": {
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "failed_trials": result.failed_trials,
                "success_rate": (result.successful_trials / result.total_trials * 100) if result.total_trials > 0 else 0,
                "duration_seconds": result.duration_seconds,
                "processing_rate": result.total_trials / result.duration_seconds if result.duration_seconds > 0 else 0
            },
            "task_statistics": result.task_stats,
            "successful_trials": result.results,
            "failed_trials": result.errors
        }
    
    elif args.nct_ids:
        # Concurrent process specific trials
        nct_id_list = [nct_id.strip() for nct_id in args.nct_ids.split(',')]
        
        result = await concurrent_process_trials(
            nct_ids=nct_id_list,
            max_workers=args.workers,
            batch_size=args.batch_size,
            process_criteria=args.process,
            process_trials=args.process,
            model_name=args.model,
            prompt_name=args.prompt,
            export_path=None,  # We'll handle export separately
            progress_updates=not args.no_progress
        )
        
        return {
            "nct_ids": nct_id_list,
            "processing_type": "concurrent_process_trials",
            "summary": {
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "failed_trials": result.failed_trials,
                "success_rate": (result.successful_trials / result.total_trials * 100) if result.total_trials > 0 else 0,
                "duration_seconds": result.duration_seconds,
                "processing_rate": result.total_trials / result.duration_seconds if result.duration_seconds > 0 else 0
            },
            "task_statistics": result.task_stats,
            "successful_trials": result.results,
            "failed_trials": result.errors
        }
    
    else:
        raise ValueError("Concurrent processing requires either --condition or --nct-ids")


def fetch_and_process_sequential(args: argparse.Namespace) -> Dict[str, Any]:
    """Handle sequential fetching and processing."""
    logger = get_logger(__name__)
    
    if args.condition:
        # Search for trials
        logger.info(f"🔍 Searching for trials: '{args.condition}' (limit: {args.limit})")
        search_result = search_trials(args.condition, fields=None, max_results=args.limit)
        trials = search_result.get('studies', [])
        
        if not trials:
            logger.warning("No trials found")
            return {
                "search_condition": args.condition,
                "processing_type": "sequential_search",
                "total_found": 0,
                "trials": []
            }
        
        logger.info(f"📋 Found {len(trials)} trials")
        
        # Process with mCODE if requested
        if args.process:
            logger.info("🔬 Processing trials with mCODE pipeline")
            from src.pipeline import McodePipeline
            
            pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
            processed_trials = []
            
            for i, trial in enumerate(trials):
                try:
                    logger.info(f"Processing trial {i+1}/{len(trials)}")
                    result = pipeline.process_clinical_trial(trial)
                    
                    # Add mCODE results to trial
                    enhanced_trial = trial.copy()
                    enhanced_trial['McodeResults'] = {
                        'extracted_entities': result.extracted_entities,
                        'mcode_mappings': result.mcode_mappings,
                        'source_references': result.source_references,
                        'validation': result.validation_results,
                        'metadata': result.metadata,
                        'error': result.error
                    }
                    processed_trials.append(enhanced_trial)
                    
                except Exception as e:
                    logger.error(f"Failed to process trial {i+1}: {e}")
                    trial['McodeProcessingError'] = str(e)
                    processed_trials.append(trial)
            
            trials = processed_trials
        
        return {
            "search_condition": args.condition,
            "processing_type": "sequential_search",
            "total_found": len(trials),
            "trials": trials
        }
    
    elif args.nct_id:
        # Fetch single trial
        logger.info(f"📥 Fetching trial: {args.nct_id}")
        trial = get_full_study(args.nct_id)
        
        # Process with mCODE if requested
        if args.process:
            logger.info("🔬 Processing trial with mCODE pipeline")
            from src.pipeline import McodePipeline
            
            pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
            result = pipeline.process_clinical_trial(trial)
            
            # Add mCODE results to trial
            trial['McodeResults'] = {
                'extracted_entities': result.extracted_entities,
                'mcode_mappings': result.mcode_mappings,
                'source_references': result.source_references,
                'validation': result.validation_results,
                'metadata': result.metadata,
                'error': result.error
            }
        
        return {
            "nct_id": args.nct_id,
            "processing_type": "sequential_single",
            "trial": trial
        }
    
    elif args.nct_ids:
        # Fetch multiple trials sequentially
        nct_id_list = [nct_id.strip() for nct_id in args.nct_ids.split(',')]
        logger.info(f"📥 Fetching {len(nct_id_list)} trials sequentially")
        
        trials = []
        failed_trials = []
        
        for nct_id in nct_id_list:
            try:
                trial = get_full_study(nct_id)
                
                # Process with mCODE if requested
                if args.process:
                    from src.pipeline import McodePipeline
                    
                    pipeline = McodePipeline(prompt_name=args.prompt, model_name=args.model)
                    result = pipeline.process_clinical_trial(trial)
                    
                    # Add mCODE results to trial
                    trial['McodeResults'] = {
                        'extracted_entities': result.extracted_entities,
                        'mcode_mappings': result.mcode_mappings,
                        'source_references': result.source_references,
                        'validation': result.validation_results,
                        'metadata': result.metadata,
                        'error': result.error
                    }
                
                trials.append(trial)
                logger.info(f"✅ Successfully processed {nct_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {nct_id}: {e}")
                failed_trials.append({"nct_id": nct_id, "error": str(e)})
        
        return {
            "nct_ids": nct_id_list,
            "processing_type": "sequential_multiple",
            "successful_count": len(trials),
            "failed_count": len(failed_trials),
            "trials": trials,
            "failed_trials": failed_trials
        }


def count_studies(args: argparse.Namespace) -> Dict[str, Any]:
    """Count total studies for a condition."""
    logger = get_logger(__name__)
    logger.info(f"📊 Counting studies for condition: '{args.condition}'")
    
    stats = calculate_total_studies(args.condition)
    
    return {
        "condition": args.condition,
        "total_studies": stats['total_studies'],
        "total_pages": stats['total_pages'],
        "page_size": stats['page_size']
    }


def output_results(results: Dict[str, Any], output_path: Optional[str]) -> None:
    """Output results to file or stdout."""
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results exported to {output_path}")
    else:
        print(json.dumps(results, indent=2))


async def main() -> None:
    """Main entry point for the mCODE Fetcher CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else getattr(logging, args.log_level)
    setup_logging(level=log_level)
    
    logger = get_logger(__name__)
    logger.info("🚀 mCODE Fetcher starting")
    
    try:
        # Initialize configuration
        config = Config()
        if args.config:
            # Config class may not support config_path parameter, handle this differently if needed
            logger.info(f"Custom config specified: {args.config}")
        
        # Handle count-only requests
        if args.count_only:
            if not args.condition:
                parser.error("--count-only requires --condition")
            
            results = count_studies(args)
            output_results(results, args.output)
            return
        
        # Handle concurrent vs sequential processing
        if args.concurrent and args.process:
            logger.info("🔄 Using concurrent processing with mCODE mapping")
            results = await fetch_and_process_concurrent(args)
        else:
            if args.concurrent:
                logger.warning("⚠️  --concurrent specified but --process not enabled. Using sequential processing.")
            logger.info("📝 Using sequential processing")
            results = fetch_and_process_sequential(args)
        
        # Output results
        output_results(results, args.output)
        
        logger.info("✅ mCODE Fetcher completed successfully")
        
    except ClinicalTrialsAPIError as e:
        logger.error(f"❌ ClinicalTrials.gov API Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⏹️  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())