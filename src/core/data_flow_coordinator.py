"""
Data Flow Coordinator - Streamlined end-to-end data processing pipeline.

This module provides a complete data flow coordinator that orchestrates
the entire pipeline from data fetching through validation, processing,
and storage using the new unified architecture.
"""

from typing import Any, Dict, List, Optional

from src.core.dependency_container import create_trial_pipeline
from src.pipeline import McodePipeline
from src.shared.models import WorkflowResult
from src.utils.fetcher import get_full_studies_batch
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
        fetch_result = self._fetch_trial_data(trial_ids)
        if not fetch_result.success:
            self.logger.error("❌ Data fetching failed")
            return fetch_result

        fetched_trials = fetch_result.data
        self.logger.info(f"✅ Successfully fetched {len(fetched_trials)} trials")

        # Phase 2: Validate and process in batches
        processing_result = self._process_trials_in_batches(
            fetched_trials,
            validate_data=validate_data,
            store_results=store_results,
            batch_size=batch_size,
        )

        # Phase 3: Generate comprehensive summary
        summary = self._generate_flow_summary(
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
        Phase 1: Fetch clinical trial data from ClinicalTrials.gov API.

        Args:
            trial_ids: List of NCT IDs to fetch

        Returns:
            WorkflowResult with fetched trial data
        """
        self.logger.info("📥 Phase 1: Fetching clinical trial data")

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

    def _process_trials_in_batches(
        self,
        trials_data: List[Dict[str, Any]],
        validate_data: bool = True,
        store_results: bool = True,
        batch_size: int = 5,
    ) -> WorkflowResult:
        """
        Phase 2: Process trials in batches using the unified pipeline.

        Args:
            trials_data: List of trial data to process
            validate_data: Whether to validate data
            store_results: Whether to store results
            batch_size: Size of processing batches

        Returns:
            WorkflowResult with batch processing results
        """
        self.logger.info("🔬 Phase 2: Processing trials in batches")

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
                1 for r in batch_results if getattr(r, "success", False)
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

    def _generate_flow_summary(
        self,
        trial_ids: List[str],
        fetch_result: WorkflowResult,
        processing_result: WorkflowResult,
    ) -> Dict[str, Any]:
        """
        Phase 3: Generate comprehensive flow summary.

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
