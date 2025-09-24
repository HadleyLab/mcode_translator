"""
Flow Summary Generator Component - Generates comprehensive flow summaries.

This module provides specialized functionality for generating detailed summaries
of data processing flows with statistics and success rates.
"""

from typing import Any, Dict, List

from src.shared.models import WorkflowResult
from src.utils.logging_config import get_logger


class FlowSummaryGenerator:
    """
    Specialized component for generating comprehensive flow summaries.

    Handles summary generation, statistics calculation, and success rate
    analysis for data processing flows.
    """

    def __init__(self):
        """Initialize the summary generator with logging."""
        self.logger = get_logger(__name__)

    def generate_flow_summary(
        self,
        trial_ids: List[str],
        fetch_result: WorkflowResult,
        processing_result: WorkflowResult,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive flow summary.

        Args:
            trial_ids: Original list of requested trial IDs
            fetch_result: Results from data fetching phase
            processing_result: Results from processing phase

        Returns:
            Comprehensive summary dictionary
        """
        fetch_metadata = fetch_result.metadata
        processing_metadata = processing_result.metadata

        total_requested = len(trial_ids)
        total_fetched = fetch_metadata.get("total_fetched", 0)
        total_processed = processing_metadata.get("total_processed", 0)
        total_successful = processing_metadata.get("total_successful", 0)

        # Calculate overall success rate
        if total_requested > 0:
            fetch_success_rate = total_fetched / total_requested
            processing_success_rate = (
                total_successful / total_fetched if total_fetched > 0 else 0
            )
            overall_success_rate = total_successful / total_requested
        else:
            fetch_success_rate = 0.0
            processing_success_rate = 0.0
            overall_success_rate = 0.0

        return {
            "total_requested": total_requested,
            "total_fetched": total_fetched,
            "total_processed": total_processed,
            "total_successful": total_successful,
            "total_failed": total_processed - total_successful,
            "fetch_success_rate": fetch_success_rate,
            "processing_success_rate": processing_success_rate,
            "overall_success_rate": overall_success_rate,
            "failed_fetches": fetch_metadata.get("failed_fetches", []),
            "flow_phases": {
                "fetch": {
                    "completed": fetch_result.success,
                    "trials_fetched": total_fetched,
                    "success_rate": fetch_success_rate,
                },
                "process": {
                    "completed": processing_result.success,
                    "trials_processed": total_processed,
                    "success_rate": processing_success_rate,
                },
            },
        }