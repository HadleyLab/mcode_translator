"""
Shared utilities and models for mCODE Translator.

This package contains shared components used across the application,
including data models, utility functions, and common interfaces.
"""

from .cli_utils import McodeCLI
from .extractors import DataExtractor
from .models import (
    ClinicalTrialData,
    McodeElement,
    PipelineResult,
    ProcessingMetadata,
    ValidationResult,
    WorkflowResult,
    create_mcode_results_structure,
    enhance_trial_with_mcode_results,
)
from .types import TaskStatus

__all__ = [
    "McodeCLI",
    "DataExtractor",
    "ClinicalTrialData",
    "McodeElement",
    "PipelineResult",
    "ProcessingMetadata",
    "ValidationResult",
    "WorkflowResult",
    "TaskStatus",
    "create_mcode_results_structure",
    "enhance_trial_with_mcode_results",
]
