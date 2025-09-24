"""
Data Fetcher Component - Handles clinical trial data fetching operations.

This module provides specialized functionality for fetching clinical trial data
from external APIs with batch processing and error handling.
"""

from typing import Dict, List

from src.shared.models import WorkflowResult
from src.utils.fetcher import get_full_studies_batch
from src.utils.logging_config import get_logger


class DataFetcher:
    """
    Specialized component for fetching clinical trial data.

    Handles batch processing, error handling, and progress tracking
    for clinical trial data retrieval operations.
    """

    def __init__(self):
        """Initialize the data fetcher with logging."""
        self.logger = get_logger(__name__)

    def fetch_trial_data(self, trial_ids: List[str]) -> WorkflowResult:
        """
        Fetch clinical trial data from ClinicalTrials.gov API.

        Args:
            trial_ids: List of NCT IDs to fetch

        Returns:
            WorkflowResult with fetched trial data
        """
        self.logger.info("📥 Fetching clinical trial data")

        try:
            # Use batch processing for better performance
            self.logger.info(f"🔄 Batch fetching {len(trial_ids)} trials")
            batch_results = get_full_studies_batch(trial_ids, max_workers=8)

            fetched_trials = []
            failed_fetches = []

            for trial_id, result in batch_results.items():
                if isinstance(result, dict) and "error" not in result:
                    fetched_trials.append(result)
                    self.logger.debug(f"✅ Fetched: {trial_id}")
                else:
                    error_msg = (
                        result.get("error", "Unknown error")
                        if isinstance(result, dict)
                        else str(result)
                    )
                    self.logger.warning(f"❌ Failed to fetch {trial_id}: {error_msg}")
                    failed_fetches.append({"trial_id": trial_id, "error": error_msg})

            if not fetched_trials:
                return WorkflowResult(
                    success=False,
                    error_message="No trials could be fetched",
                    data={},
                    metadata={"failed_fetches": failed_fetches},
                )

            success_rate = len(fetched_trials) / len(trial_ids) if trial_ids else 0
            self.logger.info(
                f"📊 Batch fetch complete: {len(fetched_trials)}/{len(trial_ids)} successful ({success_rate:.1%})"
            )

            return WorkflowResult(
                success=True,
                data=fetched_trials,
                metadata={
                    "total_requested": len(trial_ids),
                    "total_fetched": len(fetched_trials),
                    "total_failed": len(failed_fetches),
                    "success_rate": success_rate,
                    "failed_fetches": failed_fetches,
                    "processing_method": "batch_api_calls",
                },
            )

        except Exception as e:
            return WorkflowResult(
                success=False, error_message=f"Data fetching failed: {str(e)}", data={}
            )