"""
Batch Processor Component - Handles batch processing of clinical trial data.

This module provides specialized functionality for processing clinical trials
in batches with progress tracking and error handling.
"""

from typing import Any, Dict, List

from src.pipeline import McodePipeline
from src.shared.models import WorkflowResult
from src.utils.logging_config import get_logger


class BatchProcessor:
    """
    Specialized component for batch processing of clinical trial data.

    Handles batch processing, progress tracking, and error handling
    for clinical trial data processing operations.
    """

    def __init__(self, pipeline: McodePipeline):
        """
        Initialize the batch processor.

        Args:
            pipeline: The pipeline to use for processing
        """
        self.pipeline = pipeline
        self.logger = get_logger(__name__)

    def process_trials_in_batches(
        self,
        trials_data: List[Dict[str, Any]],
        validate_data: bool = True,
        store_results: bool = True,
        batch_size: int = 5,
    ) -> WorkflowResult:
        """
        Process trials in batches using the unified pipeline.

        Args:
            trials_data: List of trial data to process
            validate_data: Whether to validate data
            store_results: Whether to store results
            batch_size: Size of processing batches

        Returns:
            WorkflowResult with batch processing results
        """
        self.logger.info("🔬 Processing trials in batches")

        if not trials_data:
            return WorkflowResult(
                success=False, error_message="No trial data to process", data={}
            )

        # Process in batches to manage memory and provide progress updates
        all_results = []
        total_processed = 0
        total_successful = 0

        for i in range(0, len(trials_data), batch_size):
            batch = trials_data[i : i + batch_size]
            batch_number = (i // batch_size) + 1
            total_batches = (len(trials_data) + batch_size - 1) // batch_size

            self.logger.info(
                f"Processing batch {batch_number}/{total_batches} ({len(batch)} trials)"
            )

            # Process batch using simplified pipeline
            batch_results = []
            for trial_data in batch:
                try:
                    result = self.pipeline.process(trial_data)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process trial: {e}")
                    # Create a failed result
                    batch_results.append(
                        type(
                            "FailedResult",
                            (),
                            {
                                "success": False,
                                "error_message": str(e),
                                "mcode_mappings": [],
                                "validation_results": {},
                            },
                        )()
                    )

            # Count successful results in this batch
            successful_in_batch = sum(
                1
                for r in batch_results
                if r is not None
                and getattr(r, "error", None) is None
                and getattr(r, "success", True)
            )
            total_successful += successful_in_batch
            total_processed += len(batch)

            all_results.append(
                {
                    "batch_number": batch_number,
                    "batch_size": len(batch),
                    "results": batch_results,
                    "successful": successful_in_batch,
                    "failed": len(batch) - successful_in_batch,
                }
            )

            self.logger.info(
                f"✅ Batch {batch_number} completed: {successful_in_batch}/{len(batch)} successful"
            )

        return WorkflowResult(
            success=total_successful > 0,
            data=all_results,
            error_message=None,
            metadata={
                "total_trials": len(trials_data),
                "total_processed": total_processed,
                "total_successful": total_successful,
                "total_failed": total_processed - total_successful,
                "success_rate": (
                    total_successful / total_processed if total_processed > 0 else 0
                ),
                "batches_processed": len(all_results),
            },
        )
