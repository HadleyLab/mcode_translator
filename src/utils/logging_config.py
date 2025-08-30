import logging
import sys
import colorlog
from typing import Optional

# Global logger configuration state
_loggers_configured = set()

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a centralized logger instance with color formatting.
    
    Args:
        name: Logger name (optional, defaults to 'mcode_translator')
        
    Returns:
        Configured logger instance
    """
    # Create logger with the specified name
    logger_name = name or 'mcode_translator'
    logger = colorlog.getLogger(logger_name)
    
    # Only configure this logger if it hasn't been configured yet
    if logger_name not in _loggers_configured:
        # Default to INFO level, can be overridden by setup_logging
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers = []
        
        # Create colored formatter
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
        
        # Create and configure handler
        handler = colorlog.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Mark this logger as configured
        _loggers_configured.add(logger_name)
    
    return logger

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Setup centralized logging for the application.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = get_logger()
    logger.setLevel(level)
    
    # Also configure the root logger to prevent duplicate messages
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers from root logger to avoid duplicate output
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(levelname)-8s %(name)s:%(lineno)d %(message)s')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    return logger

class Loggable:
    """
    Base class that provides a logger instance to subclasses.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)