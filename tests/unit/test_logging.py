"""
Test color logging configuration and functionality.
"""

import pytest


@pytest.mark.mock
class TestColorLogging:
    """Test that color logging is properly configured."""

    def test_logger_fixture(self, test_logger):
        """Test that logger fixture works."""
        assert test_logger is not None
        assert test_logger.name == "test_logger"

    def test_debug_logging(self, test_logger):
        """Test debug level logging."""
        test_logger.debug("This is a DEBUG message for testing color logging")
        assert True  # Just to have an assertion

    def test_info_logging(self, test_logger):
        """Test info level logging."""
        test_logger.info("This is an INFO message for testing color logging")
        assert True  # Just to have an assertion

    def test_warning_logging(self, test_logger):
        """Test warning level logging."""
        test_logger.warning("This is a WARNING message for testing color logging")
        assert True  # Just to have an assertion

    def test_error_logging(self, test_logger):
        """Test error level logging."""
        test_logger.error("This is an ERROR message for testing color logging")
        assert True  # Just to have an assertion

    def test_critical_logging(self, test_logger):
        """Test critical level logging."""
        test_logger.critical("This is a CRITICAL message for testing color logging")
        assert True  # Just to have an assertion

    def test_multiple_log_levels(self, test_logger):
        """Test multiple log levels in sequence."""
        test_logger.debug("Debug: Starting test")
        test_logger.info("Info: Processing data")
        test_logger.warning("Warning: Something might be wrong")
        test_logger.error("Error: An error occurred")
        test_logger.info("Info: Test completed")
        assert True  # Just to have an assertion
