"""
Benchmark Task - Extended task class for benchmark validation tasks
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class TaskStatus(Enum):
    """Enumeration of possible task states"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    """Enumeration of task priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class Task:
    """Simple task representation"""
    
    def __init__(self, name: str, description: str = "", priority: TaskPriority = TaskPriority.NORMAL):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.priority = priority
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


class BenchmarkTask(Task):
    """Extended task class for benchmark validation tasks"""
    
    _tasks: Dict[str, 'BenchmarkTask'] = {}

    def __init__(self, name: str, description: str = "", priority: TaskPriority = TaskPriority.NORMAL,
                 prompt_key: str = "", model_key: str = "", trial_id: str = ""):
        super().__init__(name, description, priority)
        self.prompt_key = prompt_key
        self.model_key = model_key
        self.trial_id = trial_id
        self.expected_entities: List[Dict[str, Any]] = []
        self.expected_mappings: List[Dict[str, Any]] = []
        self.benchmark_result: Optional[Dict[str, Any]] = None
        self.token_usage = 0
        self.validation_passed = False
        self.compliance_score = 0.0
        self.f1_score = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.live_log: List[str] = []
        
        # Add task to the registry
        BenchmarkTask._tasks[self.id] = self

    def add_log(self, log_entry: str):
        """Add a log entry to the live log."""
        self.live_log.append(log_entry)

    @classmethod
    def get_task(cls, task_id: str) -> Optional['BenchmarkTask']:
        """Retrieve a task by its ID."""
        return cls._tasks.get(task_id)