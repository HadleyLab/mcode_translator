"""
Tests for custom exceptions in the HeySol API client.
"""

import pytest
from unittest.mock import Mock

from heysol.exceptions import (
    HeySolError,
    AuthenticationError,
    RateLimitError,
    APIError,
    ValidationError,
    ConnectionError,
    ServerError,
    NotFoundError,
)


class TestHeySolError:
    """Test the base HeySolError exception."""

    def test_base_error_creation(self):
        """Test creating a base HeySolError."""
        error = HeySolError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.status_code is None
        assert error.response_data == {}
        assert error.request_id is None

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = HeySolError("Test error", status_code=404)

        assert error.status_code == 404
        assert "Status: 404" in str(error)

    def test_error_with_request_id(self):
        """Test error with request ID."""
        error = HeySolError("Test error", request_id="req-123")

        assert error.request_id == "req-123"
        assert "Request ID: req-123" in str(error)

    def test_error_with_response_data(self):
        """Test error with response data."""
        response_data = {"error": "Invalid request"}
        error = HeySolError("Test error", response_data=response_data)

        assert error.response_data == response_data


class TestAuthenticationError:
    """Test AuthenticationError exception."""

    def test_authentication_error_creation(self):
        """Test creating an AuthenticationError."""
        error = AuthenticationError("Invalid API key")

        assert str(error) == "Invalid API key (Status: 401)"
        assert error.status_code == 401
        assert isinstance(error, HeySolError)

    def test_authentication_error_custom_message(self):
        """Test AuthenticationError with custom message."""
        error = AuthenticationError("Custom auth error", status_code=403)

        assert error.message == "Custom auth error"
        assert error.status_code == 403


class TestRateLimitError:
    """Test RateLimitError exception."""

    def test_rate_limit_error_creation(self):
        """Test creating a RateLimitError."""
        error = RateLimitError("Rate limit exceeded")

        assert str(error) == "Rate limit exceeded (Status: 429)"
        assert error.status_code == 429
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self):
        """Test RateLimitError with retry after value."""
        error = RateLimitError("Rate limit exceeded", retry_after=60)

        assert error.retry_after == 60
        assert "Retry after 60 seconds" in str(error)


class TestAPIError:
    """Test APIError exception."""

    def test_api_error_creation(self):
        """Test creating an APIError."""
        error = APIError("API call failed")

        assert str(error) == "API call failed"
        assert isinstance(error, HeySolError)


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating a ValidationError."""
        error = ValidationError("Validation failed")

        assert str(error) == "Validation failed"
        assert error.field_errors == {}
        assert isinstance(error, HeySolError)

    def test_validation_error_with_field_errors(self):
        """Test ValidationError with field-specific errors."""
        field_errors = {"email": "Invalid email format", "age": "Must be positive"}
        error = ValidationError("Validation failed", field_errors=field_errors)

        assert error.field_errors == field_errors
        assert "email: Invalid email format" in str(error)
        assert "age: Must be positive" in str(error)


class TestConnectionError:
    """Test ConnectionError exception."""

    def test_connection_error_creation(self):
        """Test creating a ConnectionError."""
        error = ConnectionError("Network timeout")

        assert str(error) == "Network timeout"
        assert error.original_exception is None
        assert isinstance(error, HeySolError)

    def test_connection_error_with_original_exception(self):
        """Test ConnectionError with original exception."""
        original_error = ValueError("Connection refused")
        error = ConnectionError("Network error", original_exception=original_error)

        assert error.original_exception == original_error
        assert "Original error: Connection refused" in str(error)


class TestServerError:
    """Test ServerError exception."""

    def test_server_error_creation(self):
        """Test creating a ServerError."""
        error = ServerError("Internal server error", status_code=500)

        assert str(error) == "Internal server error (Status: 500)"
        assert error.status_code == 500
        assert isinstance(error, HeySolError)


class TestNotFoundError:
    """Test NotFoundError exception."""

    def test_not_found_error_creation(self):
        """Test creating a NotFoundError."""
        error = NotFoundError("Resource not found")

        assert str(error) == "Resource not found (Status: 404)"
        assert error.status_code == 404
        assert error.resource_id is None
        assert isinstance(error, HeySolError)

    def test_not_found_error_with_resource_id(self):
        """Test NotFoundError with resource ID."""
        error = NotFoundError("User not found", resource_id="user-123")

        assert error.resource_id == "user-123"
        assert "Resource ID: user-123" in str(error)


class TestErrorInheritance:
    """Test that all errors properly inherit from HeySolError."""

    def test_all_errors_inherit_from_base(self):
        """Test that all custom errors inherit from HeySolError."""
        errors_to_test = [
            AuthenticationError("test"),
            RateLimitError("test"),
            APIError("test"),
            ValidationError("test"),
            ConnectionError("test"),
            ServerError("test", status_code=500),
            NotFoundError("test"),
        ]

        for error in errors_to_test:
            assert isinstance(error, HeySolError)