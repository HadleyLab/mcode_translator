"""
Minimal exceptions for the HeySol API client.

This module defines essential custom exceptions used in the HeySol API client.
"""


class HeySolError(Exception):
    """
    Base exception for all HeySol API client errors.
    """
    pass


class AuthenticationError(HeySolError):
    """
    Exception raised when authentication fails.
    """
    pass


class ValidationError(HeySolError):
    """
    Exception raised when request validation fails.
    """
    pass


class ConnectionError(HeySolError):
    """
    Exception raised when network or connection issues occur.
    """
    pass