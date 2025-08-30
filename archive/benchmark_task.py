"""
Benchmark Task - Extended task class for benchmark validation tasks
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from src.optimization.advanced_task_tracker import Task, TaskStatus, TaskPriority


class BenchmarkTask(Task):
    """Extended task class for benchmark validation tasks"""
    
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