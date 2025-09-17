"""
Base workflow classes for mCODE translator.

This module provides the foundation for all workflow implementations,
ensuring consistent interfaces, error handling, and CORE memory integration.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.shared.models import WorkflowResult
from src.storage.mcode_memory_storage import McodeMemoryStorage
from src.utils.config import Config
from src.utils.logging_config import get_logger

# WorkflowResult is now imported from shared.models


class WorkflowError(Exception):
    """Custom exception for workflow errors."""

    pass


class BaseWorkflow(ABC):
    """
    Base class for all mCODE translator workflows.

    Provides common functionality for configuration, logging, error handling,
    and CORE memory integration. All workflows can store to CORE memory.
    """

    def __init__(self, config: Optional[Config] = None, memory_storage: Optional[McodeMemoryStorage] = None):
        """
        Initialize the workflow with configuration and CORE memory.

        Args:
            config: Configuration instance. If None, creates default config.
            memory_storage: CORE memory storage instance. If None, creates default.
                          Pass False to disable CORE memory entirely.
        """
        self.config = config or Config()
        if memory_storage is False:
            self.memory_storage = None
        else:
            self.memory_storage = memory_storage or McodeMemoryStorage()
        self.logger = get_logger(self.__class__.__name__)

    @property
    @abstractmethod
    def memory_space(self) -> str:
        """
        Define the CORE memory space for this workflow type.

        Returns:
            str: Memory space name (e.g., 'trials', 'patients', 'summaries')
        """
        pass

    def store_to_core_memory(self, key: str, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data to CORE memory in this workflow's designated space.

        Args:
            key: Unique identifier for the data
            data: Data to store
            metadata: Optional metadata

        Returns:
            bool: True if storage was successful
        """
        if self.memory_storage is None:
            self.logger.debug(f"CORE memory disabled - skipping storage of {key}")
            return False

        try:
            # Create namespaced key
            namespaced_key = f"{self.memory_space}:{key}"

            # Prepare storage data
            storage_data = {
                "data": data,
                "metadata": metadata or {},
                "workflow_type": self.__class__.__name__,
                "memory_space": self.memory_space,
                "timestamp": self._get_timestamp()
            }

            success = self.memory_storage.store(namespaced_key, storage_data)
            if success:
                self.logger.info(f"✅ Stored {key} to CORE memory space '{self.memory_space}'")
            else:
                self.logger.warning(f"❌ Failed to store {key} to CORE memory space '{self.memory_space}'")

            return success

        except Exception as e:
            self.logger.error(f"❌ Error storing {key} to CORE memory: {e}")
            return False

    def retrieve_from_core_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data from CORE memory in this workflow's designated space.

        Args:
            key: Unique identifier for the data

        Returns:
            Optional[Dict[str, Any]]: Retrieved data or None if not found
        """
        if self.memory_storage is None:
            self.logger.debug(f"CORE memory disabled - cannot retrieve {key}")
            return None

        try:
            namespaced_key = f"{self.memory_space}:{key}"
            data = self.memory_storage.retrieve(namespaced_key)
            if data:
                self.logger.debug(f"✅ Retrieved {key} from CORE memory space '{self.memory_space}'")
            return data
        except Exception as e:
            self.logger.error(f"❌ Error retrieving {key} from CORE memory: {e}")
            return None

    def _get_timestamp(self) -> str:
        """Get current timestamp for storage metadata."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    @abstractmethod
    def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            **kwargs: Workflow-specific parameters

        Returns:
            WorkflowResult: Standardized result structure
        """
        pass

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate workflow inputs.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            bool: True if inputs are valid
        """
        # Default implementation - subclasses can override
        return True

    def _create_result(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Create a standardized workflow result.

        Args:
            success: Whether the workflow succeeded
            data: Result data
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            WorkflowResult: Standardized result
        """
        return WorkflowResult(
            success=success,
            data=data or {},
            error_message=error_message,
            metadata=metadata or {},
        )

    def _handle_error(self, error: Exception, context: str = "") -> WorkflowResult:
        """
        Handle workflow errors consistently.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            WorkflowResult: Error result
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        return self._create_result(
            success=False,
            error_message=error_msg,
            metadata={"error_type": type(error).__name__},
        )


class FetcherWorkflow(BaseWorkflow):
    """
    Base class for data fetching workflows.

    Fetchers retrieve raw data from external sources and can store metadata
    to CORE memory for tracking data provenance and processing status.
    """

    @property
    def memory_space(self) -> str:
        """Fetcher workflows use 'raw_data' space for metadata."""
        return "raw_data"


class ProcessorWorkflow(BaseWorkflow):
    """
    Base class for data processing workflows.

    Processors apply mCODE transformations and always store results to CORE memory
    in their designated space.
    """

    pass


class TrialsProcessorWorkflow(ProcessorWorkflow):
    """
    Base class for clinical trials processing workflows.

    All trials processing results are stored in the 'trials' CORE memory space.
    """

    @property
    def memory_space(self) -> str:
        """Trials processors use 'trials' space."""
        return "trials"


class PatientsProcessorWorkflow(ProcessorWorkflow):
    """
    Base class for patient data processing workflows.

    All patient processing results are stored in the 'patients' CORE memory space.
    """

    @property
    def memory_space(self) -> str:
        """Patients processors use 'patients' space."""
        return "patients"


class SummarizerWorkflow(BaseWorkflow):
    """
    Base class for data summarization workflows.

    Summarizers create natural language summaries and store them to CORE memory
    in their designated space.
    """

    pass


class TrialsSummarizerWorkflow(SummarizerWorkflow):
    """
    Base class for clinical trials summarization workflows.

    All trials summaries are stored in the 'trials_summaries' CORE memory space.
    """

    @property
    def memory_space(self) -> str:
        """Trials summarizers use 'trials_summaries' space."""
        return "trials_summaries"


class PatientsSummarizerWorkflow(SummarizerWorkflow):
    """
    Base class for patient data summarization workflows.

    All patient summaries are stored in the 'patients_summaries' CORE memory space.
    """

    @property
    def memory_space(self) -> str:
        """Patients summarizers use 'patients_summaries' space."""
        return "patients_summaries"
