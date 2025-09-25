"""
Unit tests for error_handler module.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest

from src.utils.error_handler import (
    McodeError,
    ConfigurationError,
    DataProcessingError,
    APIError,
    handle_error,
    handle_cli_error,
    safe_execute,
    validate_required,
    log_operation_start,
    log_operation_success,
    log_operation_failure,
)


class TestMcodeError:
    """Test McodeError exception class."""

    def test_init_with_message_only(self):
        """Test initialization with message only."""
        error = McodeError("Test message")
        assert error.message == "Test message"
        assert error.error_code is None
        assert str(error) == "Test message"

    def test_init_with_message_and_code(self):
        """Test initialization with message and error code."""
        error = McodeError("Test message", "ERR001")
        assert error.message == "Test message"
        assert error.error_code == "ERR001"
        assert str(error) == "Test message"


class TestConfigurationError:
    """Test ConfigurationError exception class."""

    def test_inheritance(self):
        """Test that ConfigurationError inherits from McodeError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, McodeError)
        assert error.message == "Config error"


class TestDataProcessingError:
    """Test DataProcessingError exception class."""

    def test_inheritance(self):
        """Test that DataProcessingError inherits from McodeError."""
        error = DataProcessingError("Data error")
        assert isinstance(error, McodeError)
        assert error.message == "Data error"


class TestAPIError:
    """Test APIError exception class."""

    def test_inheritance(self):
        """Test that APIError inherits from McodeError."""
        error = APIError("API error")
        assert isinstance(error, McodeError)
        assert error.message == "API error"


class TestHandleError:
    """Test handle_error function."""

    @patch('src.utils.error_handler.logger')
    def test_handle_error_basic(self, mock_logger):
        """Test basic error handling."""
        error = ValueError("Test error")
        handle_error(error)

        mock_logger.error.assert_called_once_with("❌ Test error")

    @patch('src.utils.error_handler.logger')
    def test_handle_error_with_context(self, mock_logger):
        """Test error handling with context."""
        error = ValueError("Test error")
        handle_error(error, context="Operation failed")

        mock_logger.error.assert_called_once_with("❌ Operation failed: Test error")

    @patch('src.utils.error_handler.logger')
    def test_handle_error_with_custom_logger(self, mock_logger):
        """Test error handling with custom logger."""
        mock_custom_logger = MagicMock()
        error = ValueError("Test error")
        handle_error(error, logger_instance=mock_custom_logger)

        mock_custom_logger.error.assert_called_once_with("❌ Test error")
        mock_logger.error.assert_not_called()

    @patch('src.utils.error_handler.logger')
    @patch('sys.exit')
    def test_handle_error_with_exit_code(self, mock_exit, mock_logger):
        """Test error handling with exit code."""
        error = ValueError("Test error")
        handle_error(error, exit_code=1)

        mock_logger.error.assert_called_once_with("❌ Test error")
        mock_exit.assert_called_once_with(1)

    @patch('src.utils.error_handler.logger')
    @patch('traceback.format_exc')
    def test_handle_error_verbose(self, mock_format_exc, mock_logger):
        """Test error handling with verbose output."""
        mock_format_exc.return_value = "Traceback details"
        error = ValueError("Test error")
        handle_error(error, verbose=True)

        mock_logger.error.assert_called_once_with("❌ Test error")
        mock_logger.debug.assert_called_once_with("Full traceback:\nTraceback details")


class TestHandleCliError:
    """Test handle_cli_error function."""

    @patch('sys.stderr')
    @patch('sys.exit')
    def test_handle_cli_error_basic(self, mock_exit, mock_stderr):
        """Test basic CLI error handling."""
        error = ValueError("Test error")
        handle_cli_error(error)

        # print() calls write twice: once for message, once for newline
        mock_stderr.write.assert_any_call("❌ Test error")
        mock_stderr.write.assert_any_call("\n")
        mock_exit.assert_called_once_with(1)

    @patch('sys.stderr')
    @patch('sys.exit')
    def test_handle_cli_error_with_context(self, mock_exit, mock_stderr):
        """Test CLI error handling with context."""
        error = ValueError("Test error")
        handle_cli_error(error, context="Command failed")

        mock_stderr.write.assert_any_call("❌ Command failed: Test error")
        mock_stderr.write.assert_any_call("\n")
        mock_exit.assert_called_once_with(1)

    @patch('sys.stderr')
    @patch('sys.exit')
    @patch('traceback.format_exc')
    def test_handle_cli_error_verbose(self, mock_format_exc, mock_exit, mock_stderr):
        """Test CLI error handling with verbose output."""
        mock_format_exc.return_value = "Traceback details"
        error = ValueError("Test error")
        handle_cli_error(error, verbose=True)

        # Two print() calls: error message and traceback, each with newline
        assert mock_stderr.write.call_count == 4
        mock_stderr.write.assert_any_call("❌ Test error")
        mock_stderr.write.assert_any_call("\n")
        mock_stderr.write.assert_any_call("Full traceback:\nTraceback details")
        mock_exit.assert_called_once_with(1)

    @patch('sys.stderr')
    @patch('sys.exit')
    def test_handle_cli_error_custom_exit_code(self, mock_exit, mock_stderr):
        """Test CLI error handling with custom exit code."""
        error = ValueError("Test error")
        handle_cli_error(error, exit_code=42)

        mock_stderr.write.assert_any_call("❌ Test error")
        mock_stderr.write.assert_any_call("\n")
        mock_exit.assert_called_once_with(42)


class TestSafeExecute:
    """Test safe_execute function."""

    @patch('src.utils.error_handler.logger')
    def test_safe_execute_success(self, mock_logger):
        """Test successful function execution."""
        def test_func(x, y=10):
            return x + y

        result = safe_execute(test_func, 5, y=15)
        assert result == 20
        mock_logger.error.assert_not_called()

    @patch('src.utils.error_handler.logger')
    def test_safe_execute_failure(self, mock_logger):
        """Test function execution with exception."""
        def failing_func():
            raise ValueError("Function failed")

        with pytest.raises(ValueError, match="Function failed"):
            safe_execute(failing_func)

        mock_logger.error.assert_called_once_with("❌ Operation failed: Function failed")

    @patch('src.utils.error_handler.logger')
    def test_safe_execute_custom_error_msg(self, mock_logger):
        """Test function execution with custom error message."""
        def failing_func():
            raise ValueError("Function failed")

        with pytest.raises(ValueError, match="Function failed"):
            safe_execute(failing_func, error_msg="Custom error")

        mock_logger.error.assert_called_once_with("❌ Custom error: Function failed")

    @patch('src.utils.error_handler.logger')
    def test_safe_execute_custom_logger(self, mock_logger):
        """Test function execution with custom logger."""
        mock_custom_logger = MagicMock()

        def failing_func():
            raise ValueError("Function failed")

        with pytest.raises(ValueError, match="Function failed"):
            safe_execute(failing_func, logger_instance=mock_custom_logger)

        mock_custom_logger.error.assert_called_once_with("❌ Operation failed: Function failed")
        mock_logger.error.assert_not_called()


class TestValidateRequired:
    """Test validate_required function."""

    def test_validate_required_none_value(self):
        """Test validation of None value."""
        with pytest.raises(ValueError, match="Required value 'test' is None"):
            validate_required(None, "test")

    def test_validate_required_empty_string(self):
        """Test validation of empty string."""
        with pytest.raises(ValueError, match="Required value 'test' is empty"):
            validate_required("", "test")

    def test_validate_required_whitespace_string(self):
        """Test validation of whitespace-only string."""
        with pytest.raises(ValueError, match="Required value 'test' is empty"):
            validate_required("   ", "test")

    def test_validate_required_empty_list(self):
        """Test validation of empty list."""
        with pytest.raises(ValueError, match="Required value 'test' is empty"):
            validate_required([], "test")

    def test_validate_required_empty_dict(self):
        """Test validation of empty dict."""
        with pytest.raises(ValueError, match="Required value 'test' is empty"):
            validate_required({}, "test")

    def test_validate_required_valid_string(self):
        """Test validation of valid string."""
        # Should not raise
        validate_required("valid", "test")

    def test_validate_required_valid_list(self):
        """Test validation of valid list."""
        # Should not raise
        validate_required([1, 2, 3], "test")

    def test_validate_required_valid_dict(self):
        """Test validation of valid dict."""
        # Should not raise
        validate_required({"key": "value"}, "test")

    def test_validate_required_custom_error_msg(self):
        """Test validation with custom error message."""
        with pytest.raises(ValueError, match="Custom error message"):
            validate_required(None, "test", error_msg="Custom error message")


class TestLogOperationStart:
    """Test log_operation_start function."""

    @patch('src.utils.error_handler.logger')
    def test_log_operation_start_default_logger(self, mock_logger):
        """Test operation start logging with default logger."""
        log_operation_start("Test operation")
        mock_logger.info.assert_called_once_with("🚀 Starting: Test operation")

    @patch('src.utils.error_handler.logger')
    def test_log_operation_start_custom_logger(self, mock_logger):
        """Test operation start logging with custom logger."""
        mock_custom_logger = MagicMock()
        log_operation_start("Test operation", logger_instance=mock_custom_logger)

        mock_custom_logger.info.assert_called_once_with("🚀 Starting: Test operation")
        mock_logger.info.assert_not_called()


class TestLogOperationSuccess:
    """Test log_operation_success function."""

    @patch('src.utils.error_handler.logger')
    def test_log_operation_success_default_logger(self, mock_logger):
        """Test operation success logging with default logger."""
        log_operation_success("Test operation")
        mock_logger.info.assert_called_once_with("✅ Completed: Test operation")

    @patch('src.utils.error_handler.logger')
    def test_log_operation_success_custom_logger(self, mock_logger):
        """Test operation success logging with custom logger."""
        mock_custom_logger = MagicMock()
        log_operation_success("Test operation", logger_instance=mock_custom_logger)

        mock_custom_logger.info.assert_called_once_with("✅ Completed: Test operation")
        mock_logger.info.assert_not_called()


class TestLogOperationFailure:
    """Test log_operation_failure function."""

    @patch('src.utils.error_handler.logger')
    def test_log_operation_failure_default_logger(self, mock_logger):
        """Test operation failure logging with default logger."""
        error = ValueError("Test error")
        log_operation_failure("Test operation", error)

        mock_logger.error.assert_called_once_with("❌ Failed: Test operation - Test error")

    @patch('src.utils.error_handler.logger')
    def test_log_operation_failure_custom_logger(self, mock_logger):
        """Test operation failure logging with custom logger."""
        mock_custom_logger = MagicMock()
        error = ValueError("Test error")
        log_operation_failure("Test operation", error, logger_instance=mock_custom_logger)

        mock_custom_logger.error.assert_called_once_with("❌ Failed: Test operation - Test error")
        mock_logger.error.assert_not_called()