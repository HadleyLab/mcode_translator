"""
Shared types and enums to avoid circular imports
"""

from enum import Enum


class TaskStatus(Enum):
    """Status of a benchmark task"""
    PENDING = "Pending"
    PROCESSING = "Processing"
    SUCCESS = "Success"
    FAILED = "Failed"
    CANCELLED = "Cancelled"