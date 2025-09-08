"""
Benchmark Task - Extended task class for benchmark validation tasks
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from src.shared.types import TaskStatus




class Task:
    """Simple task representation"""
    
    def __init__(self, name: str, description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.failed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.execution_time = 0.0
        self.estimated_time = 0.0
        self.tags: List[str] = []


# BenchmarkTask implementation moved to src/pipeline/task_queue.py
# This file now contains only the base Task class for reference