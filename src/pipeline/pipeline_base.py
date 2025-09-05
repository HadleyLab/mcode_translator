"""
Base class for processing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PipelineResult:
    """Comprehensive result from a processing pipeline"""
    extracted_entities: List[Dict[str, Any]]
    mcode_mappings: List[Dict[str, Any]]
    source_references: List[Any] 
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any]
    original_data: Dict[str, Any]
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None

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
            PipelineResult with extracted entities, Mcode mappings, and source tracking.
        """
        pass