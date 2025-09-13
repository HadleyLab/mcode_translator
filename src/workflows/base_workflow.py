"""
Base workflow classes for mCODE translator.

This module provides the foundation for all workflow implementations,
ensuring consistent interfaces and error handling across the application.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.config import Config
from src.utils.logging_config import get_logger


@dataclass
class WorkflowResult:
    """Standardized result structure for all workflows."""

    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowError(Exception):
    """Custom exception for workflow errors."""

    pass


class BaseWorkflow(ABC):
    """
    Base class for all mCODE translator workflows.

    Provides common functionality for configuration, logging, and error handling.
    Subclasses must implement the execute() method.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the workflow with configuration.

        Args:
            config: Configuration instance. If None, creates default config.
        """
        self.config = config or Config()
        self.logger = get_logger(self.__class__.__name__)

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

    Fetchers retrieve raw data from external sources but do not process it.
    They do not store results to core memory.
    """

    pass


class ProcessorWorkflow(BaseWorkflow):
    """
    Base class for data processing workflows.

    Processors apply mCODE transformations and can store results to core memory.
    """

    def __init__(
        self, config: Optional[Config] = None, memory_storage: Optional[Any] = None
    ):
        """
        Initialize processor workflow.

        Args:
            config: Configuration instance
            memory_storage: Optional core memory storage interface
        """
        super().__init__(config)
        self.memory_storage = memory_storage
