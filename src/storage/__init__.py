"""
Storage package for mCODE Translator.

This package provides data persistence and memory storage functionality
for mCODE extraction results and related data.
"""

from .mcode_memory_storage import OncoCoreMemory

__all__ = [
    "OncoCoreMemory",
]
