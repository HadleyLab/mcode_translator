"""
Centralized task execution system for pipeline processing.
Provides queue-based concurrency with worker management.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

from src.pipeline.pipeline_base import PipelineResult
from src.pipeline.nlp_mcode_pipeline import NlpMcodePipeline
from src.pipeline.mcode_pipeline import McodePipeline
from src.shared.benchmark_result import BenchmarkResult


from src.shared.types import TaskStatus


@dataclass
class BenchmarkTask:
    """Data structure for benchmark task execution"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_name: str = ""
    model_name: str = ""
    trial_id: str = ""
    trial_data: Dict[str, Any] = field(default_factory=dict)
    prompt_type: str = "NLP_EXTRACTION"
    expected_entities: List[Dict[str, Any]] = field(default_factory=list)
    expected_mappings: List[Dict[str, Any]] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[PipelineResult] = None
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    compliance_score: float = 0.0
    live_log: List[str] = field(default_factory=list)


class PipelineTaskQueue:
    """Centralized task execution system for pipeline processing"""
    
    def __init__(self, max_workers: int = 5):
        self.task_queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.max_workers = max_workers
        self.is_running = False
        self.cancelled = False
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_tasks = 0
        
        # Task tracking
        self.tasks: Dict[str, BenchmarkTask] = {}
        self.task_callbacks: Dict[str, Callable] = {}
        
        # Pipeline instances cache
        self.pipeline_cache: Dict[str, Union[NlpMcodePipeline, McodePipeline]] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    async def start_workers(self) -> None:
        """Start worker tasks for processing"""
        if self.is_running:
            self.logger.warning("Workers are already running")
            return
        
        self.is_running = True
        self.cancelled = False
        
        # Create worker tasks
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._worker_loop(i + 1))
            self.workers.append(worker_task)
        
        self.logger.info(f"Started {self.max_workers} worker tasks")
    
    async def stop_workers(self) -> None:
        """Stop all worker tasks gracefully"""
        if not self.is_running:
            return
        
        self.cancelled = True
        
        # Send stop signals to workers
        for _ in range(self.max_workers):
            await self.task_queue.put(None)
        
        # Wait for workers to complete
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.workers.clear()
        
        self.is_running = False
        self.logger.info("All workers stopped")
    
    async def add_task(self, task: BenchmarkTask, callback: Optional[Callable] = None) -> str:
        """Add a benchmark task to the queue"""
        self.tasks[task.task_id] = task
        self.total_tasks += 1
        
        if callback:
            self.task_callbacks[task.task_id] = callback
        
        await self.task_queue.put(task)
        return task.task_id
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Worker task that processes benchmark tasks"""
        self.logger.info(f"Worker {worker_id} started")
        
        while not self.cancelled:
            try:
                # Get task from queue with timeout for cancellation check
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if task is None:  # Stop signal
                    break
                
                # Process the task
                await self._process_task(worker_id, task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
                if not self.task_queue.empty():
                    self.task_queue.task_done()
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, worker_id: int, task: BenchmarkTask) -> None:
        """Process a single benchmark task"""
        task.status = TaskStatus.PROCESSING
        task.start_time = time.time()

        # Add initial log entry
        timestamp = time.strftime("%H:%M:%S")
        task.live_log.append(f"[{timestamp}] 🔄 Processing {task.prompt_name} + {task.model_name} + {task.trial_id}")

        self.logger.info(f"Processing {task.prompt_name} + {task.model_name} + {task.trial_id}")

        try:
            # Get or create pipeline instance
            pipeline = await self._get_pipeline(task.prompt_name, task.prompt_type)

            # Process the trial with actual trial data - run in thread to avoid blocking
            result = await asyncio.to_thread(pipeline.process_clinical_trial, task.trial_data)

            # Update task with results
            task.result = result
            task.end_time = time.time()
            task.duration_ms = (task.end_time - task.start_time) * 1000
            task.token_usage = result.metadata.get('aggregate_token_usage', {}) if result.metadata else {}
            task.status = TaskStatus.SUCCESS

            # Calculate metrics using gold standard validation
            benchmark = BenchmarkResult()
            benchmark.success = True
            benchmark.extracted_entities = result.extracted_entities if result.extracted_entities else []
            benchmark.entities_extracted = len(benchmark.extracted_entities)
            benchmark.mcode_mappings = result.mcode_mappings if result.mcode_mappings else []
            benchmark.entities_mapped = len(benchmark.mcode_mappings)
            
            # Create a mock framework for logging
            class MockFramework:
                def __init__(self):
                    import logging
                    self.logger = logging.getLogger(__name__)
            
            mock_framework = MockFramework()
            
            # Calculate actual metrics using the shared BenchmarkResult class
            benchmark.calculate_metrics(
                expected_entities=task.expected_entities,
                expected_mappings=task.expected_mappings,
                framework=mock_framework
            )
            
            # Extract metrics from the benchmark result
            task.precision = benchmark.precision
            task.recall = benchmark.recall
            task.f1_score = benchmark.f1_score
            task.compliance_score = benchmark.compliance_score

            # Add completion log entry
            timestamp = time.strftime("%H:%M:%S")
            task.live_log.append(f"[{timestamp}] ✅ Completed {task.prompt_name} + {task.model_name} + {task.trial_id} - F1={task.f1_score:.3f}")

            self.logger.info(f"Completed {task.prompt_name} + {task.model_name} + {task.trial_id} - F1={task.f1_score:.3f}")

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.end_time = time.time()
            task.duration_ms = (task.end_time - task.start_time) * 1000

            # Add error log entry
            timestamp = time.strftime("%H:%M:%S")
            task.live_log.append(f"[{timestamp}] ❌ Failed {task.prompt_name} + {task.model_name} + {task.trial_id} - {str(e)}")

            self.logger.error(f"Failed {task.prompt_name} + {task.model_name} + {task.trial_id} - {str(e)}")

        finally:
            # Update completion count
            self.completed_tasks += 1
            if task.status == TaskStatus.FAILED:
                self.failed_tasks += 1

            # Call callback if registered
            if task.task_id in self.task_callbacks:
                try:
                    self.task_callbacks[task.task_id](task)
                except Exception as e:
                    self.logger.error(f"Callback error for task {task.task_id}: {str(e)}")
    
    async def _get_pipeline(self, prompt_name: str, prompt_type: str) -> Union[NlpMcodePipeline, McodePipeline]:
        """Get or create pipeline instance for the given prompt"""
        cache_key = f"{prompt_name}_{prompt_type}"
        
        if cache_key in self.pipeline_cache:
            return self.pipeline_cache[cache_key]
        
        # Create appropriate pipeline based on prompt type
        if prompt_type == "DIRECT_MCODE":
            pipeline = McodePipeline(prompt_name=prompt_name)
        else:
            # For NLP extraction or mapping, use NLP+mcode pipeline
            extraction_prompt = prompt_name if prompt_type == "NLP_EXTRACTION" else "generic_extraction"
            mapping_prompt = prompt_name if prompt_type == "MCODE_MAPPING" else "generic_mapping"
            pipeline = NlpMcodePipeline(
                extraction_prompt_name=extraction_prompt,
                mapping_prompt_name=mapping_prompt
            )
        
        self.pipeline_cache[cache_key] = pipeline
        return pipeline
    
    def get_task(self, task_id: str) -> Optional[BenchmarkTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[BenchmarkTask]:
        """Get all tasks"""
        return list(self.tasks.values())
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get statistics about task execution"""
        total = len(self.tasks)
        completed = sum(1 for task in self.tasks.values() if task.status in [TaskStatus.SUCCESS, TaskStatus.FAILED])
        success = sum(1 for task in self.tasks.values() if task.status == TaskStatus.SUCCESS)
        failed = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        
        return {
            "total_tasks": total,
            "completed_tasks": completed,
            "successful_tasks": success,
            "failed_tasks": failed,
            "completion_rate": completed / total if total > 0 else 0,
            "success_rate": success / total if total > 0 else 0,
            "workers_running": len(self.workers),
            "queue_size": self.task_queue.qsize(),
        }


# Global task queue instance
global_task_queue: Optional[PipelineTaskQueue] = None


def get_global_task_queue(max_workers: int = 5) -> PipelineTaskQueue:
    """Get or create the global task queue instance"""
    global global_task_queue
    if global_task_queue is None:
        global_task_queue = PipelineTaskQueue(max_workers=max_workers)
    return global_task_queue


async def initialize_task_queue(max_workers: int = 5) -> PipelineTaskQueue:
    """Initialize the global task queue and start workers"""
    task_queue = get_global_task_queue(max_workers)
    await task_queue.start_workers()
    return task_queue


async def shutdown_task_queue() -> None:
    """Shutdown the global task queue"""
    global global_task_queue
    if global_task_queue:
        await global_task_queue.stop_workers()
        global_task_queue = None