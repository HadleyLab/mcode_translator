"""
Error Handling Utilities - Consistent error handling patterns.

This module provides standardized error handling utilities to ensure
consistent error reporting and logging across the mCODE Translator application.
"""

import logging
import sys
from typing import Any, Callable, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class McodeError(Exception):
    """Base exception for mCODE Translator errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


class ConfigurationError(McodeError):
    """Configuration-related errors."""

    pass


class DataProcessingError(McodeError):
    """Data processing errors."""

    pass


class APIError(McodeError):
    """API-related errors."""

    pass


def handle_error(
    error: Exception,
    context: str = "",
    logger_instance: Optional[logging.Logger] = None,
    exit_code: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """
    Handle errors consistently across the application.

    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        logger_instance: Logger instance to use (defaults to module logger)
        exit_code: Exit code for sys.exit() (None means don't exit)
        verbose: Whether to show full traceback
    """
    log = logger_instance or logger

    error_msg = f"{context}: {str(error)}" if context else str(error)
    log.error(f"❌ {error_msg}")

    if verbose:
        import traceback

        log.debug(f"Full traceback:\n{traceback.format_exc()}")

    if exit_code is not None:
        sys.exit(exit_code)


def handle_cli_error(
    error: Exception, context: str = "", verbose: bool = False, exit_code: int = 1
) -> None:
    """
    Handle CLI command errors with user-friendly output.

    Args:
        error: The exception that occurred
        context: Additional context about the command
        verbose: Whether to show full traceback
        exit_code: Exit code for sys.exit()
    """
    error_msg = f"{context}: {str(error)}" if context else str(error)
    print(f"❌ {error_msg}", file=sys.stderr)

    if verbose:
        import traceback

        print(f"Full traceback:\n{traceback.format_exc()}", file=sys.stderr)

    sys.exit(exit_code)


def safe_execute(
    func: Callable[..., Any],
    *args: Any,
    error_msg: str = "Operation failed",
    logger_instance: Optional[logging.Logger] = None,
    **kwargs: Any,
) -> Any:
    """
    Execute a function safely with consistent error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        error_msg: Error message prefix
        logger_instance: Logger instance to use
        **kwargs: Keyword arguments for the function

    Returns:
        Function result

    Raises:
        Exception: Re-raises the original exception after logging
    """
    log = logger_instance or logger

    try:
        return func(*args, **kwargs)
    except Exception as e:
        log.error(f"❌ {error_msg}: {e}")
        raise


def validate_required(value: Any, name: str, error_msg: Optional[str] = None) -> None:
    """
    Validate that a required value is not None or empty.

    Args:
        value: Value to validate
        name: Name of the value for error messages
        error_msg: Custom error message

    Raises:
        ValueError: If validation fails
    """
    if value is None:
        msg = error_msg or f"Required value '{name}' is None"
        raise ValueError(msg)

    if isinstance(value, str) and not value.strip():
        msg = error_msg or f"Required value '{name}' is empty"
        raise ValueError(msg)

    # Handle list and dict separately to avoid mypy issues with len()
    # Use truthiness check instead of length for generic types
    if isinstance(value, (list, tuple)) and not value:
        msg = error_msg or f"Required value '{name}' is empty"
        raise ValueError(msg)
    if isinstance(value, dict) and not value:
        msg = error_msg or f"Required value '{name}' is empty"
        raise ValueError(msg)


def log_operation_start(operation: str, logger_instance: Optional[logging.Logger] = None) -> None:
    """Log the start of an operation."""
    log = logger_instance or logger
    log.info(f"🚀 Starting: {operation}")


def log_operation_success(operation: str, logger_instance: Optional[logging.Logger] = None) -> None:
    """Log the successful completion of an operation."""
    log = logger_instance or logger
    log.info(f"✅ Completed: {operation}")


def log_operation_failure(
    operation: str, error: Exception, logger_instance: Optional[logging.Logger] = None
) -> None:
    """Log the failure of an operation."""
    log = logger_instance or logger
    log.error(f"❌ Failed: {operation} - {error}")
