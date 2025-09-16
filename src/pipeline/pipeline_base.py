"""
Base class for processing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.shared.models import ClinicalTrialData, PipelineResult


class ProcessingPipeline(ABC):
    """
    Abstract base class for processing pipelines.
    Defines the common interface for all pipelines.
    """

    @abstractmethod
    def process_clinical_trial(self, trial_data: Dict[str, Any]) -> PipelineResult:
        """
        Process complete clinical trial data through the pipeline.

        Args:
            trial_data: Raw clinical trial data from API or source.

        Returns:
            PipelineResult with extracted entities, mCODE mappings, and source tracking.
        """
        pass

    def process_clinical_trial_validated(
        self, trial_data: ClinicalTrialData
    ) -> PipelineResult:
        """
        Process validated clinical trial data through the pipeline.

        Args:
            trial_data: Validated ClinicalTrialData instance.

        Returns:
            PipelineResult with extracted entities, mCODE mappings, and source tracking.
        """
        # Convert to dict for backward compatibility
        trial_dict = trial_data.dict()
        return self.process_clinical_trial(trial_dict)
