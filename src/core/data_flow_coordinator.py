"""
Data Flow Coordinator - Streamlined end-to-end data processing pipeline.

This module provides a complete data flow coordinator that orchestrates
the entire pipeline from data fetching through validation, processing,
and storage using the new unified architecture.
"""

from typing import Any, Dict, List, Optional

from src.core.batch_processor import BatchProcessor
from src.core.data_fetcher import DataFetcher
from src.core.dependency_container import create_trial_pipeline
from src.core.flow_summary_generator import FlowSummaryGenerator
from src.pipeline import McodePipeline
from src.shared.models import WorkflowResult
from src.utils.logging_config import get_logger


class DataFlowCoordinator:
    """
    Complete data flow coordinator implementing fetch → validate → process → store.

    This coordinator demonstrates the streamlined architecture by orchestrating
    all pipeline components in a clean, maintainable way.
    """

    def __init__(
        self,
        pipeline: Optional[McodePipeline] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data flow coordinator.

        Args:
            pipeline: Injected pipeline (created automatically if None)
            config: Configuration for the coordinator
        """
        self.logger = get_logger(__name__)
        self.config = config or {}

        # Dependency injection: pipeline is injected or created
        if pipeline is None:
            processor_config = self.config.get("processor", {})
            self.pipeline = create_trial_pipeline(
                processor_config=processor_config, include_storage=False
            )
        else:
            self.pipeline = pipeline

        # Initialize specialized components
        self.data_fetcher = DataFetcher()
        self.batch_processor = BatchProcessor(self.pipeline)
        self.summary_generator = FlowSummaryGenerator()

    def process_clinical_trials_complete_flow(
        self,
        trial_ids: List[str],
        validate_data: bool = True,
        store_results: bool = True,
        batch_size: int = 5,
    ) -> WorkflowResult:
        """
        Execute complete data flow: fetch → validate → process → store.

        Args:
            trial_ids: List of NCT IDs to process
            validate_data: Whether to validate fetched data
            store_results: Whether to store processing results
            batch_size: Number of trials to process in each batch

        Returns:
            WorkflowResult with complete flow outcome
        """
        self.logger.info("🚀 Starting complete clinical trials data flow")
        self.logger.info(f"   📊 Trials to process: {len(trial_ids)}")
        self.logger.info(
            f"   ✅ Validation: {'enabled' if validate_data else 'disabled'}"
        )
        self.logger.info(f"   💾 Storage: {'enabled' if store_results else 'disabled'}")
        self.logger.info(f"   📦 Batch size: {batch_size}")

        # Phase 1: Fetch trial data
        fetch_result = self.data_fetcher.fetch_trial_data(trial_ids)
        if not fetch_result.success:
            self.logger.error("❌ Data fetching failed")
            return fetch_result

        fetched_trials = fetch_result.data
        self.logger.info(f"✅ Successfully fetched {len(fetched_trials)} trials")

        # Phase 2: Validate and process in batches
        processing_result = self.batch_processor.process_trials_in_batches(
            fetched_trials,
            validate_data=validate_data,
            store_results=store_results,
            batch_size=batch_size,
        )

        # Phase 3: Generate comprehensive summary
        summary = self.summary_generator.generate_flow_summary(
            trial_ids=trial_ids,
            fetch_result=fetch_result,
            processing_result=processing_result,
        )

        if processing_result.success:
            self.logger.info("🎉 Complete data flow finished successfully!")
            self.logger.info(
                f"   📈 Success rate: {summary['overall_success_rate']:.1%}"
            )
            self.logger.info(
                f"   ✅ Processed: {summary['total_processed']}/{summary['total_requested']}"
            )
        else:
            self.logger.error("💥 Complete data flow failed!")
            self.logger.error(f"   Error: {processing_result.error_message}")

        return WorkflowResult(
            success=processing_result.success,
            data={
                "fetched_trials": fetched_trials,
                "processing_results": processing_result.data,
                "summary": summary,
            },
            error_message=None,
            metadata={
                "flow_type": "complete_data_flow",
                "phases_completed": ["fetch", "validate", "process", "store"],
                "total_trials_requested": len(trial_ids),
                "trials_fetched": len(fetched_trials),
                "trials_processed": summary.get("total_processed", 0),
                "success_rate": summary.get("overall_success_rate", 0.0),
            },
        )

    def _fetch_trial_data(self, trial_ids: List[str]) -> WorkflowResult:
        """
        Fetch trial data for given NCT IDs.

        Args:
            trial_ids: List of NCT IDs to fetch

        Returns:
            WorkflowResult with fetched trial data
        """
        return self.data_fetcher.fetch_trial_data(trial_ids)

    def _process_trials_in_batches(
        self,
        trial_data: List[Dict[str, Any]],
        batch_size: int = 5,
        validate_data: bool = True,
        store_results: bool = True,
    ) -> WorkflowResult:
        """
        Process trial data in batches.

        Args:
            trial_data: List of trial data to process
            batch_size: Size of each batch
            validate_data: Whether to validate data
            store_results: Whether to store results

        Returns:
            WorkflowResult with processing results
        """
        if not trial_data:
            return WorkflowResult(
                success=False,
                error_message="No trial data to process",
                data=[],
                metadata={
                    "total_processed": 0,
                    "total_successful": 0,
                    "total_failed": 0,
                },
            )

        return self.batch_processor.process_trials_in_batches(
            trial_data,
            validate_data=validate_data,
            store_results=store_results,
            batch_size=batch_size,
        )

    def _generate_flow_summary(
        self,
        trial_ids: List[str],
        fetch_result: WorkflowResult,
        processing_result: WorkflowResult,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive flow summary.

        Args:
            trial_ids: Original trial IDs requested
            fetch_result: Result from data fetching
            processing_result: Result from data processing

        Returns:
            Dictionary with flow summary statistics
        """
        return self.summary_generator.generate_flow_summary(
            trial_ids=trial_ids,
            fetch_result=fetch_result,
            processing_result=processing_result,
        )

    def get_flow_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the data flow coordinator.

        Returns:
            Dictionary with flow statistics
        """
        return {
            "coordinator_type": "data_flow_coordinator",
            "pipeline_type": "simplified",
            "has_validator": False,  # Simplified pipeline doesn't have separate validator
            "has_processor": True,
            "has_storage": False,  # Simplified pipeline doesn't include storage
            "config": self.config,
            "capabilities": {
                "batch_processing": True,
                "data_validation": False,  # Validation is handled internally
                "result_storage": False,
                "progress_tracking": True,
                "error_handling": True,
            },
        }


# Convenience functions for easy coordinator creation
def create_data_flow_coordinator(
    config: Optional[Dict[str, Any]] = None,
) -> DataFlowCoordinator:
    """
    Create a data flow coordinator with default configuration.

    Args:
        config: Coordinator configuration

    Returns:
        Configured DataFlowCoordinator
    """
    return DataFlowCoordinator(config=config)


def process_clinical_trials_flow(
    trial_ids: List[str], config: Optional[Dict[str, Any]] = None
) -> WorkflowResult:
    """
    Convenience function to process clinical trials using complete data flow.

    Args:
        trial_ids: List of NCT IDs to process
        config: Processing configuration

    Returns:
        WorkflowResult with complete flow outcome
    """
    coordinator = create_data_flow_coordinator(config=config)
    return coordinator.process_clinical_trials_complete_flow(trial_ids)
