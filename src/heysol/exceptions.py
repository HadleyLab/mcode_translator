"""
Minimal HeySol exceptions for mCODE Translator CLI.
"""


class HeySolError(Exception):
    """Base HeySol error."""
    pass


class ValidationError(HeySolError):
    """Validation error."""
    pass