"""
Unified Pipeline Interface - Streamlined architecture for mCODE processing.

This module provides a unified interface for all pipeline operations,
implementing dependency injection and clear separation of concerns.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

from src.shared.models import (ClinicalTrialData, PatientData, PipelineResult,
                               WorkflowResult, clinical_trial_from_dict,
                               validate_clinical_trial_data,
                               validate_patient_data)


class DataValidator(Protocol):
    """Protocol for data validation components."""

    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate input data. Returns (is_valid, error_message)."""
        ...


class DataProcessor(Protocol):
    """Protocol for data processing components."""

    def process(self, data: Dict[str, Any], **kwargs) -> PipelineResult:
        """Process input data and return results."""
        ...


class DataStorage(Protocol):
    """Protocol for data storage components."""

    def store(self, key: str, data: Dict[str, Any]) -> bool:
        """Store data with given key. Returns success status."""
        ...

    def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        ...


class PipelineComponent:
    """Base class for pipeline components with dependency injection."""

    def __init__(self, **dependencies):
        """Initialize component with injected dependencies."""
        for name, dependency in dependencies.items():
            setattr(self, name, dependency)


class ClinicalTrialValidator(PipelineComponent):
    """Validator for clinical trial data."""

    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate clinical trial data."""
        return validate_clinical_trial_data(data)


class PatientDataValidator(PipelineComponent):
    """Validator for patient data."""

    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate patient data."""
        return validate_patient_data(data)


class UnifiedPipeline(PipelineComponent):
    """
    Unified pipeline interface that orchestrates the entire data processing flow.

    This class implements dependency injection and provides a clean interface
    for processing clinical trials and patient data through validation,
    processing, and storage phases.
    """

    def __init__(
        self,
        validator: DataValidator,
        processor: DataProcessor,
        storage: Optional[DataStorage] = None,
        **kwargs,
    ):
        """
        Initialize unified pipeline with injected dependencies.

        Args:
            validator: Component for data validation
            processor: Component for data processing
            storage: Optional component for data storage
            **kwargs: Additional dependencies
        """
        super().__init__(
            validator=validator, processor=processor, storage=storage, **kwargs
        )

    def process_trial(
        self,
        trial_data: Dict[str, Any],
        validate: bool = True,
        store_results: bool = True,
        **processing_kwargs,
    ) -> WorkflowResult:
        """
        Process a clinical trial through the complete pipeline.

        Args:
            trial_data: Raw clinical trial data
            validate: Whether to validate data before processing
            store_results: Whether to store results
            **processing_kwargs: Additional processing parameters

        Returns:
            WorkflowResult with processing outcome
        """
        try:
            # Phase 1: Validation
            if validate:
                is_valid, error_msg = self.validator.validate(trial_data)
                if not is_valid:
                    return WorkflowResult(
                        success=False,
                        error_message=f"Validation failed: {error_msg}",
                        data=trial_data,
                    )

            # Phase 2: Processing
            pipeline_result = self.processor.process(trial_data, **processing_kwargs)

            # Phase 3: Storage (optional)
            if store_results and self.storage and pipeline_result.error is None:
                # Extract trial ID for storage key
                trial_id = self._extract_trial_id(trial_data)
                storage_data = {
                    "trial_data": trial_data,
                    "pipeline_result": pipeline_result.model_dump(),
                    "processing_timestamp": datetime.utcnow(),
                }

                # Use the appropriate storage method for trials
                if hasattr(self.storage, 'store_trial_mcode_summary'):
                    storage_success = self.storage.store_trial_mcode_summary(trial_id, storage_data)
                else:
                    # Fallback to generic store if available
                    storage_success = self.storage.store(trial_id, storage_data)
                if not storage_success:
                    # Don't fail the entire operation for storage issues
                    pipeline_result.metadata.storage_success = False

            return WorkflowResult(
                success=True,
                data={"trial_data": trial_data, "pipeline_result": pipeline_result},
                metadata={
                    "validation_performed": validate,
                    "storage_attempted": store_results,
                    "processing_time": getattr(
                        pipeline_result.metadata, "processing_time_seconds", None
                    ),
                },
            )

        except Exception as e:
            return WorkflowResult(
                success=False,
                error_message=f"Pipeline processing failed: {str(e)}",
                data=trial_data,
            )

    def process_trials_batch(
        self,
        trials_data: List[Dict[str, Any]],
        validate: bool = True,
        store_results: bool = True,
        **processing_kwargs,
    ) -> WorkflowResult:
        """
        Process multiple clinical trials in batch.

        Args:
            trials_data: List of raw clinical trial data
            validate: Whether to validate data before processing
            store_results: Whether to store results
            **processing_kwargs: Additional processing parameters

        Returns:
            WorkflowResult with batch processing outcome
        """
        results = []
        successful_count = 0
        failed_count = 0

        for trial_data in trials_data:
            result = self.process_trial(
                trial_data,
                validate=validate,
                store_results=store_results,
                **processing_kwargs,
            )
            results.append(result)

            if result.success:
                successful_count += 1
            else:
                failed_count += 1

        return WorkflowResult(
            success=successful_count > 0,
            data=results,
            metadata={
                "total_trials": len(trials_data),
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": (
                    successful_count / len(trials_data) if trials_data else 0
                ),
            },
        )

    def _extract_trial_id(self, trial_data: Dict[str, Any]) -> str:
        """Extract trial ID from trial data for storage."""
        try:
            return trial_data["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_trial_{hash(str(trial_data)) % 10000}"


class PipelineFactory:
    """Factory for creating configured pipeline instances."""

    @staticmethod
    def create_clinical_trial_pipeline(
        processor: DataProcessor, storage: Optional[DataStorage] = None
    ) -> UnifiedPipeline:
        """
        Create a clinical trial processing pipeline.

        Args:
            processor: The processing component to use
            storage: Optional storage component

        Returns:
            Configured UnifiedPipeline instance
        """
        validator = ClinicalTrialValidator()
        return UnifiedPipeline(
            validator=validator, processor=processor, storage=storage
        )

    @staticmethod
    def create_patient_data_pipeline(
        processor: DataProcessor, storage: Optional[DataStorage] = None
    ) -> UnifiedPipeline:
        """
        Create a patient data processing pipeline.

        Args:
            processor: The processing component to use
            storage: Optional storage component

        Returns:
            Configured UnifiedPipeline instance
        """
        validator = PatientDataValidator()
        return UnifiedPipeline(
            validator=validator, processor=processor, storage=storage
        )


# Convenience functions for easy pipeline creation
def create_clinical_trial_pipeline(
    processor: DataProcessor, storage: Optional[DataStorage] = None
) -> UnifiedPipeline:
    """Create a clinical trial processing pipeline."""
    return PipelineFactory.create_clinical_trial_pipeline(processor, storage)


def create_patient_data_pipeline(
    processor: DataProcessor, storage: Optional[DataStorage] = None
) -> UnifiedPipeline:
    """Create a patient data processing pipeline."""
    return PipelineFactory.create_patient_data_pipeline(processor, storage)
