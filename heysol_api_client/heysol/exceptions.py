"""
Custom exceptions for the HeySol API client.

This module defines all custom exceptions used throughout the HeySol API client
library, providing clear error categorization and helpful error messages.
"""

from typing import Optional, Dict, Any


class HeySolError(Exception):
    """
    Base exception for all HeySol API client errors.

    This is the parent class for all custom exceptions in the HeySol API client.
    It provides a consistent interface for error handling and reporting.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a HeySol API error.

        Args:
            message: Human-readable error message
            status_code: HTTP status code if applicable
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}
        self.request_id = request_id

    def __str__(self) -> str:
        """Return a formatted error message."""
        parts = [self.message]
        if self.status_code:
            parts.append(f"(Status: {self.status_code})")
        if self.request_id:
            parts.append(f"(Request ID: {self.request_id})")
        return " ".join(parts)


class AuthenticationError(HeySolError):
    """
    Exception raised when authentication fails.

    This exception is raised when API key validation fails, tokens are invalid
    or expired, or when authentication credentials are missing.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        status_code: Optional[int] = 401,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize an authentication error.

        Args:
            message: Error message describing the authentication failure
            status_code: HTTP status code (defaults to 401)
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, status_code, response_data, request_id)


class RateLimitError(HeySolError):
    """
    Exception raised when API rate limits are exceeded.

    This exception is raised when the client exceeds the API rate limits.
    It includes information about when the rate limit will reset.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        status_code: Optional[int] = 429,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        """
        Initialize a rate limit error.

        Args:
            message: Error message describing the rate limit
            status_code: HTTP status code (defaults to 429)
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
            retry_after: Seconds to wait before retrying (if provided by API)
        """
        super().__init__(message, status_code, response_data, request_id)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """Return a formatted error message with retry information."""
        message = super().__str__()
        if self.retry_after:
            message += f" Retry after {self.retry_after} seconds."
        return message


class APIError(HeySolError):
    """
    Exception raised for general API errors.

    This exception is raised for various API-related errors that don't fall
    into more specific categories like authentication or rate limiting.
    """

    def __init__(
        self,
        message: str = "API error occurred",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a general API error.

        Args:
            message: Error message describing the API error
            status_code: HTTP status code
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, status_code, response_data, request_id)


class ValidationError(HeySolError):
    """
    Exception raised when request validation fails.

    This exception is raised when the client-side validation of request
    parameters or data fails before sending the request to the API.
    """

    def __init__(
        self,
        message: str = "Request validation failed",
        field_errors: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a validation error.

        Args:
            message: Error message describing the validation failure
            field_errors: Dictionary of field-specific validation errors
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, None, None, request_id)
        self.field_errors = field_errors or {}

    def __str__(self) -> str:
        """Return a formatted error message with field-specific errors."""
        message = super().__str__()
        if self.field_errors:
            field_details = ", ".join(
                f"{field}: {error}" for field, error in self.field_errors.items()
            )
            message += f" Field errors: {field_details}"
        return message


class ConnectionError(HeySolError):
    """
    Exception raised when network or connection issues occur.

    This exception is raised when there are network connectivity issues,
    timeouts, or other connection-related problems.
    """

    def __init__(
        self,
        message: str = "Connection error occurred",
        original_exception: Optional[Exception] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a connection error.

        Args:
            message: Error message describing the connection issue
            original_exception: The original exception that caused this error
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, None, None, request_id)
        self.original_exception = original_exception

    def __str__(self) -> str:
        """Return a formatted error message with original exception info."""
        message = super().__str__()
        if self.original_exception:
            message += f" Original error: {str(self.original_exception)}"
        return message


class ServerError(HeySolError):
    """
    Exception raised when the HeySol API returns a server error.

    This exception is raised when the API returns 5xx status codes,
    indicating server-side issues.
    """

    def __init__(
        self,
        message: str = "Server error occurred",
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a server error.

        Args:
            message: Error message describing the server error
            status_code: HTTP status code (5xx range)
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, status_code, response_data, request_id)


class NotFoundError(HeySolError):
    """
    Exception raised when a requested resource is not found.

    This exception is raised when the API returns a 404 status code,
    indicating that the requested resource does not exist.
    """

    def __init__(
        self,
        message: str = "Resource not found",
        resource_id: Optional[str] = None,
        status_code: Optional[int] = 404,
        response_data: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        """
        Initialize a not found error.

        Args:
            message: Error message describing the missing resource
            resource_id: ID of the resource that was not found
            status_code: HTTP status code (defaults to 404)
            response_data: Raw response data from the API
            request_id: Request ID for debugging purposes
        """
        super().__init__(message, status_code, response_data, request_id)
        self.resource_id = resource_id

    def __str__(self) -> str:
        """Return a formatted error message with resource ID."""
        message = super().__str__()
        if self.resource_id:
            message += f" Resource ID: {self.resource_id}"
        return message