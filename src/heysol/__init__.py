"""
Minimal HeySol API client for mCODE Translator CLI.

This module provides a minimal implementation of the HeySol API client
functionality needed for the mCODE Translator CLI to work.
"""

from .clients import HeySolClient
from .config import HeySolConfig
from .exceptions import HeySolError, ValidationError

__all__ = ["HeySolClient", "HeySolConfig", "HeySolError", "ValidationError"]