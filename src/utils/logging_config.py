import json
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import colorlog

# Export logging module for other modules to use
__all__ = ["setup_logging", "get_logger", "Loggable", "logging"]

# Global configuration state
_logging_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    """
    Setup centralized logging for the entire application using modular configuration.

    Args:
        level: Optional logging level override ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    global _logging_configured

    if _logging_configured:
        return

    # Load logging configuration directly to avoid circular imports
    config_path = Path(__file__).parent.parent / "config" / "logging_config.json"
    try:
        with open(config_path) as f:
            logging_config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(
            f"Logging configuration file not found or invalid: {config_path}. "
            f"Error: {e}. Please ensure the logging configuration file exists and is valid."
        ) from e

    # Use provided level or default from config
    log_level: str = level or logging_config["default_level"]

    # Convert string level to logging constant
    numeric_level: int = getattr(logging, log_level.upper(), logging.INFO)

    # Clear all existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set up colored formatter if enabled
    if logging_config.get("colored_output", True):
        formatter: Union[colorlog.ColoredFormatter, logging.Formatter] = colorlog.ColoredFormatter(
            logging_config["format"],
            log_colors=logging_config["handlers"]["console"]["colors"],
        )
    else:
        formatter = logging.Formatter(logging_config["format"])

    # Create console handler with immediate flushing
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, logging_config["handlers"]["console"]["level"]))
    # Force immediate output by setting stream to unbuffered
    console_handler.stream = sys.stdout
    # Set handler to flush after each log message
    console_handler.terminator = "\n"

    # Configure root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(numeric_level)

    # Configure specific loggers
    for logger_name, logger_config in logging_config["loggers"].items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, logger_config["level"]))
        logger.propagate = logger_config.get("propagate", True)

    # Ensure propagation is enabled for proper message flow
    logging.getLogger().propagate = True

    _logging_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. Call setup_logging() first.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    logger_name = name or "mcode_translator"
    logger = logging.getLogger(logger_name)

    # Don't add handlers here - they're handled by root logger
    # Just ensure propagation is enabled so messages reach root handler
    logger.propagate = True

    return logger


class Loggable:
    """
    Base class that provides a logger instance to subclasses.
    """

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
