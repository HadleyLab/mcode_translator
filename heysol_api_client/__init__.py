"""
HeySol API Client Package

This package provides a comprehensive Python client for the HeySol API.
"""

from .heysol import HeySolClient, HeySolError, ValidationError

__version__ = "1.0.0"
__all__ = [
    "HeySolClient",
    "HeySolError",
    "ValidationError",
]