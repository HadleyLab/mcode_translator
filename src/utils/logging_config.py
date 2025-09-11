import logging
import sys
import colorlog
from typing import Optional

# Global configuration state
_logging_configured = False

def setup_logging(level: str = "INFO") -> None:
    """
    Setup centralized logging for the entire application - call this ONCE.
    
    Args:
        level: Logging level as string ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Clear all existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set up colored formatter
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s:%(lineno)d%(reset)s %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green', 
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # Create single handler for all logging
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    # Prevent propagation to avoid duplicates
    logging.getLogger().propagate = False
    
    _logging_configured = True

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance. Call setup_logging() first.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    logger_name = name or 'mcode_translator'
    logger = logging.getLogger(logger_name)
    
    # Don't add handlers here - they're handled by root logger
    # Just ensure propagation is enabled so messages reach root handler
    logger.propagate = True
    
    return logger


class Loggable:
    """
    Base class that provides a logger instance to subclasses.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

class Loggable:
    """
    Base class that provides a logger instance to subclasses.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)