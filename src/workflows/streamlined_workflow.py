"""
Streamlined Workflow - Composition-based workflow using unified pipeline.

This module demonstrates the new architecture using composition over inheritance,
dependency injection, and clear separation of concerns.
"""

from typing import Any, Dict, List, Optional

from src.core.dependency_container import create_trial_pipeline
from src.pipeline.unified_pipeline import UnifiedPipeline
from src.shared.models import WorkflowResult
from src.utils.logging_config import get_logger


class StreamlinedTrialProcessor:
    """
    Streamlined clinical trial processor using composition and dependency injection.

    This class demonstrates the improved architecture:
    - Uses composition instead of inheritance
    - Injects dependencies through constructor
    - Clear separation of concerns
    - Simplified interface
    """

    def __init__(
        self,
        pipeline: Optional[UnifiedPipeline] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the streamlined processor.

        Args:
            pipeline: Injected pipeline instance (created automatically if None)
            config: Configuration dictionary
        """
        self.logger = get_logger(__name__)
        self.config = config or {}

        # Dependency injection: pipeline is injected or created
        if pipeline is None:
            processor_config = self.config.get("processor", {})
            include_storage = self.config.get("include_storage", True)
            self.pipeline = create_trial_pipeline(
                processor_config=processor_config, include_storage=include_storage
            )
        else:
            self.pipeline = pipeline

    def process_single_trial(
        self,
        trial_data: Dict[str, Any],
        validate: bool = True,
        store_results: bool = True,
    ) -> WorkflowResult:
        """
        Process a single clinical trial.

        Args:
            trial_data: Clinical trial data to process
            validate: Whether to validate data
            store_results: Whether to store results

        Returns:
            WorkflowResult with processing outcome
        """
        self.logger.info("🔬 Processing single clinical trial")

        result = self.pipeline.process_trial(
            trial_data=trial_data, validate=validate, store_results=store_results
        )

        if result.success:
            self.logger.info("✅ Trial processing completed successfully")
        else:
            self.logger.error(f"❌ Trial processing failed: {result.error_message}")

        return result

    def process_multiple_trials(
        self,
        trials_data: List[Dict[str, Any]],
        validate: bool = True,
        store_results: bool = True,
        batch_size: int = 10,
    ) -> WorkflowResult:
        """
        Process multiple clinical trials in batches.

        Args:
            trials_data: List of clinical trial data
            validate: Whether to validate data
            store_results: Whether to store results
            batch_size: Size of processing batches

        Returns:
            WorkflowResult with batch processing outcome
        """
        self.logger.info(f"🔬 Processing {len(trials_data)} clinical trials in batches")

        # For now, process all at once (could be enhanced with actual batching)
        result = self.pipeline.process_trials_batch(
            trials_data=trials_data, validate=validate, store_results=store_results
        )

        if result.success:
            metadata = result.metadata
            successful = metadata.get("successful", 0)
            total = metadata.get("total_trials", 0)
            self.logger.info(
                f"✅ Batch processing completed: {successful}/{total} successful"
            )
        else:
            self.logger.error("❌ Batch processing failed")

        return result

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processing capabilities.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "pipeline_type": "unified",
            "has_validator": hasattr(self.pipeline, "validator"),
            "has_processor": hasattr(self.pipeline, "processor"),
            "has_storage": self.pipeline.storage is not None,
            "config": self.config,
        }


class StreamlinedWorkflowCoordinator:
    """
    Coordinator for streamlined workflows.

    This class orchestrates multiple streamlined processors and provides
    a high-level interface for complex processing scenarios.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow coordinator.

        Args:
            config: Configuration for the coordinator
        """
        self.logger = get_logger(__name__)
        self.config = config or {}

        # Create processors for different data types
        self.trial_processor = StreamlinedTrialProcessor(
            config=self.config.get("trial_processor", {})
        )

        # Could add patient_processor, etc. in the future
        self.patient_processor = None

    def process_clinical_trials_workflow(
        self,
        trials_data: List[Dict[str, Any]],
        workflow_config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute a complete clinical trials processing workflow.

        Args:
            trials_data: List of clinical trial data
            workflow_config: Workflow-specific configuration

        Returns:
            WorkflowResult with complete workflow outcome
        """
        config = workflow_config or {}
        validate = config.get("validate", True)
        store_results = config.get("store_results", True)
        batch_size = config.get("batch_size", 10)

        self.logger.info("🚀 Starting clinical trials processing workflow")
        self.logger.info(f"   📊 Trials to process: {len(trials_data)}")
        self.logger.info(f"   ✅ Validation: {'enabled' if validate else 'disabled'}")
        self.logger.info(f"   💾 Storage: {'enabled' if store_results else 'disabled'}")

        # Execute the workflow
        result = self.trial_processor.process_multiple_trials(
            trials_data=trials_data,
            validate=validate,
            store_results=store_results,
            batch_size=batch_size,
        )

        # Log final results
        if result.success:
            metadata = result.metadata
            successful = metadata.get("successful", 0)
            total = metadata.get("total_trials", 0)
            success_rate = metadata.get("success_rate", 0)

            self.logger.info("🎉 Workflow completed successfully!")
            self.logger.info(f"   📈 Success rate: {success_rate:.1%}")
            self.logger.info(f"   ✅ Successful: {successful}/{total}")
        else:
            self.logger.error("💥 Workflow failed!")
            self.logger.error(f"   Error: {result.error_message}")

        return result

    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the workflow coordinator.

        Returns:
            Dictionary with workflow statistics
        """
        return {
            "coordinator_type": "streamlined",
            "trial_processor": self.trial_processor.get_processing_stats(),
            "patient_processor": (
                self.patient_processor.get_processing_stats()
                if self.patient_processor
                else "Not configured"
            ),
            "config": self.config,
        }


# Convenience functions for easy workflow creation
def create_trial_processor(
    config: Optional[Dict[str, Any]] = None,
) -> StreamlinedTrialProcessor:
    """
    Create a streamlined trial processor.

    Args:
        config: Processor configuration

    Returns:
        Configured StreamlinedTrialProcessor
    """
    return StreamlinedTrialProcessor(config=config)


def create_workflow_coordinator(
    config: Optional[Dict[str, Any]] = None,
) -> StreamlinedWorkflowCoordinator:
    """
    Create a streamlined workflow coordinator.

    Args:
        config: Coordinator configuration

    Returns:
        Configured StreamlinedWorkflowCoordinator
    """
    return StreamlinedWorkflowCoordinator(config=config)
