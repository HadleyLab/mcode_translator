#!/usr/bin/env python3
"""
End-to-End Processor CLI - Complete mCODE Workflow

This CLI tool provides a single command to fetch clinical trials and patients
for a given condition, process them with mCODE mapping, and store summaries
in CORE memory.

Usage:
    python -m src.cli.end_to_end_processor --condition "breast cancer"
    python -m src.cli.end_to_end_processor --condition "lung cancer" --dry-run --verbose
    python -m src.cli.end_to_end_processor --condition "diabetes" --trials-limit 10 --patients-limit 20
    python -m src.cli.end_to_end_processor --condition "prostate cancer" --store-in-core-memory --quiet
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import get_logger, setup_logging
from src.workflows.patients_fetcher_workflow import PatientsFetcherWorkflow
from src.workflows.patients_processor_workflow import PatientsProcessorWorkflow
from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
from src.workflows.trials_processor_workflow import TrialsProcessorWorkflow
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config

logger = get_logger("end_to_end_processor")


class EndToEndProcessor:
    """Complete end-to-end processor for clinical data."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the processor."""
        self.config = config or Config()
        self.memory_storage = McodeMemoryStorage()

    def process_condition(
        self,
        condition: str,
        model: str = "deepseek-coder",
        trials_limit: int = 5,
        patients_limit: int = 5,
        workers: int = 2,
        store_in_memory: bool = True
    ) -> Dict[str, any]:
        """
        Process a complete condition workflow.

        Args:
            condition: Medical condition to search for
            trials_limit: Number of trials to fetch
            patients_limit: Number of patients to fetch
            workers: Number of concurrent workers
            store_in_memory: Whether to store results in CORE memory

        Returns:
            Processing results summary
        """
        logger.info(f"🚀 Starting end-to-end processing for condition: {condition}")
        logger.info(f"📊 Configuration: trials_limit={trials_limit}, patients_limit={patients_limit}, workers={workers}")

        results = {
            "condition": condition,
            "trials_processed": 0,
            "patients_processed": 0,
            "trials_stored": 0,
            "patients_stored": 0,
            "errors": []
        }

        try:
            # Step 1: Fetch clinical trials
            logger.info("🔬 STEP 1: Fetching clinical trials...")
            trials_workflow = TrialsFetcherWorkflow(self.config)
            trials_result = trials_workflow.execute(
                condition=condition,
                limit=trials_limit,
                workers=workers
            )

            if trials_result.success and trials_result.data:
                trials_data = trials_result.data
                results["trials_fetched"] = len(trials_data)
                logger.info(f"✅ Fetched {len(trials_data)} trials")

                # Step 2: Process trials with mCODE mapping
                logger.info("🧪 STEP 2: Processing trials with mCODE mapping...")
                trials_processor = TrialsProcessorWorkflow(self.config, self.memory_storage)
                trials_process_result = trials_processor.execute(
                    trials_data=trials_data,
                    model=model,
                    store_in_memory=store_in_memory,
                    workers=workers
                )

                if trials_process_result.success:
                    results["trials_processed"] = len(trials_data)
                    if store_in_memory:
                        results["trials_stored"] = len(trials_data)
                    logger.info(f"✅ Processed {len(trials_data)} trials")
                else:
                    results["errors"].append(f"Failed to process trials: {trials_process_result.error_message}")
            else:
                results["errors"].append(f"Failed to fetch trials: {trials_result.error_message}")

            # Step 3: Fetch synthetic patients
            logger.info("👥 STEP 3: Fetching synthetic patients...")
            patients_workflow = PatientsFetcherWorkflow(self.config)
            patients_result = patients_workflow.execute(
                archive_path="breast_cancer_10_years",
                limit=patients_limit
            )

            if patients_result.success and patients_result.data:
                patients_data = patients_result.data
                results["patients_fetched"] = len(patients_data)
                logger.info(f"✅ Fetched {len(patients_data)} patients")

                # Step 4: Process patients with mCODE mapping
                logger.info("🩺 STEP 4: Processing patients with mCODE mapping...")
                patients_processor = PatientsProcessorWorkflow(self.config, self.memory_storage)
                patients_process_result = patients_processor.execute(
                    patients_data=patients_data,
                    store_in_memory=store_in_memory,
                    workers=workers
                )

                if patients_process_result.success:
                    results["patients_processed"] = len(patients_data)
                    if store_in_memory:
                        results["patients_stored"] = len(patients_data)
                    logger.info(f"✅ Processed {len(patients_data)} patients")
                else:
                    results["errors"].append(f"Failed to process patients: {patients_process_result.error_message}")
            else:
                results["errors"].append(f"Failed to fetch patients: {patients_result.error_message}")

        except Exception as e:
            results["errors"].append(f"Unexpected error: {str(e)}")
            logger.error(f"❌ End-to-end processing failed: {e}")

        # Summary
        logger.info("📊 END-TO-END PROCESSING SUMMARY")
        logger.info(f"   Condition: {condition}")
        logger.info(f"   Trials: {results.get('trials_processed', 0)} processed, {results.get('trials_stored', 0)} stored")
        logger.info(f"   Patients: {results.get('patients_processed', 0)} processed, {results.get('patients_stored', 0)} stored")

        if results["errors"]:
            logger.warning(f"   Errors: {len(results['errors'])}")
            for error in results["errors"]:
                logger.warning(f"     • {error}")

        return results


def main():
    """Main CLI entry point."""
    # Setup logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="End-to-End mCODE Processor - Fetch and process clinical data for a condition"
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Medical condition to search for (e.g., 'breast cancer')"
    )
    parser.add_argument(
        "--model",
        default="deepseek-coder",
        help="LLM model to use for mCODE processing (default: deepseek-coder)"
    )
    parser.add_argument(
        "--trials-limit",
        type=int,
        default=5,
        help="Number of clinical trials to fetch (default: 5)"
    )
    parser.add_argument(
        "--patients-limit",
        type=int,
        default=5,
        help="Number of patients to fetch (default: 5)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent workers (default: 2)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode - process data but skip storing results in CORE memory"
    )
    parser.add_argument(
        "--store-in-core-memory",
        action="store_true",
        help="Explicitly enable storing results in CORE memory (default behavior)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable most logging output"
    )

    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine memory storage behavior
    # Default: store in memory unless dry-run is specified
    store_in_memory = True
    if args.dry_run:
        store_in_memory = False
    elif args.store_in_core_memory:
        store_in_memory = True

    # Run the processor
    processor = EndToEndProcessor()

    # Run processing
    results = processor.process_condition(
        condition=args.condition,
        model=args.model,
        trials_limit=args.trials_limit,
        patients_limit=args.patients_limit,
        workers=args.workers,
        store_in_memory=store_in_memory
    )

    # Print final results
    print("\n" + "="*60)
    print("🎉 END-TO-END PROCESSING COMPLETE")
    print("="*60)
    print(f"Condition: {args.condition}")
    print(f"Trials processed: {results.get('trials_processed', 0)}")
    print(f"Patients processed: {results.get('patients_processed', 0)}")
    print(f"CORE Memory storage: {'Enabled' if store_in_memory else 'Disabled'}")

    if results.get("errors"):
        print(f"\n⚠️  Errors encountered: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"   • {error}")
        return 1
    else:
        print("\n✅ All processing completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())