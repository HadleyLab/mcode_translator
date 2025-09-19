"""
Concurrency utilities for mCODE translator.

Provides a common worker pool and task queue system for concurrent processing
across all components (fetchers, processors, optimizers).
"""

import asyncio
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from queue import Queue
import time

from src.utils.logging_config import get_logger

T = TypeVar('T')
logger = get_logger(__name__)


@dataclass
class Task:
    """Represents a task to be executed by workers."""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: int = 0

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    success: bool
    result: Any = None
    error: Exception = None
    duration: float = 0.0


class WorkerPool:
    """
    Thread pool for concurrent task execution.

    Provides a common interface for running tasks concurrently across
    all mCODE translator components.
    """

    def __init__(self, max_workers: int = 4, name: str = "WorkerPool"):
        self.max_workers = max_workers
        self.name = name
        self.executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self.logger = get_logger(f"{name}")

    def start(self):
        """Start the worker pool."""
        if self._running:
            return

        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=f"{self.name}-worker"
        )
        self._running = True
        self.logger.info(f"🤖 Started {self.name} with {self.max_workers} workers")

    def stop(self):
        """Stop the worker pool."""
        if not self._running:
            return

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        self._running = False
        self.logger.info(f"🛑 Stopped {self.name}")

    def submit_task(self, task: Task) -> concurrent.futures.Future:
        """Submit a single task for execution."""
        if not self._running or not self.executor:
            raise RuntimeError(f"Worker pool {self.name} is not running")

        future = self.executor.submit(self._execute_task, task)
        return future

    def submit_tasks(self, tasks: List[Task]) -> List[concurrent.futures.Future]:
        """Submit multiple tasks for execution."""
        return [self.submit_task(task) for task in tasks]

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task and return result."""
        start_time = time.time()

        try:
            self.logger.info(f"⚡ Worker {threading.current_thread().name} executing task {task.id}")
            result = task.func(*task.args, **task.kwargs)
            duration = time.time() - start_time

            self.logger.info(f"✅ Worker {threading.current_thread().name} completed task {task.id} in {duration:.2f}s")
            return TaskResult(
                task_id=task.id,
                success=True,
                result=result,
                duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"❌ Worker {threading.current_thread().name} failed task {task.id} after {duration:.2f}s: {e}")
            return TaskResult(
                task_id=task.id,
                success=False,
                error=e,
                duration=duration
            )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class TaskQueue:
    """
    Priority queue for managing tasks with worker pool execution.

    Provides a high-level interface for submitting tasks and collecting results.
    """

    def __init__(self, max_workers: int = 4, name: str = "TaskQueue"):
        self.worker_pool = WorkerPool(max_workers, name)
        self.logger = get_logger(f"{name}")

    def execute_tasks(
        self,
        tasks: List[Task],
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """
        Execute tasks concurrently and return results.

        Args:
            tasks: List of tasks to execute
            timeout: Maximum time to wait for all tasks (None = no timeout)
            progress_callback: Optional callback for progress updates

        Returns:
            List of task results
        """
        if not tasks:
            return []

        with self.worker_pool:
            self.logger.info(f"📋 Submitting {len(tasks)} tasks for concurrent execution")
            self.logger.info(f"🔧 Worker pool status: {self.worker_pool.max_workers} workers available")

            # Submit all tasks
            futures = self.worker_pool.submit_tasks(tasks)
            self.logger.info(f"✅ Submitted {len(futures)} tasks to worker pool")

            # Collect results as they complete
            results = []
            completed_count = 0

            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                    completed_count += 1

                    if progress_callback:
                        progress_callback(completed_count, len(tasks), result)

                    if result.success:
                        self.logger.debug(f"📊 Task {result.task_id}: SUCCESS ({result.duration:.2f}s)")
                    else:
                        self.logger.warning(f"📊 Task {result.task_id}: FAILED ({result.duration:.2f}s)")

                except concurrent.futures.TimeoutError:
                    self.logger.error("⏰ Task execution timed out")
                    break
                except Exception as e:
                    self.logger.error(f"💥 Unexpected error collecting task result: {e}")

            # Sort results by original task order
            task_order = {task.id: i for i, task in enumerate(tasks)}
            results.sort(key=lambda r: task_order.get(r.task_id, 999))

            successful = sum(1 for r in results if r.success)
            self.logger.info(f"🎉 Completed {successful}/{len(tasks)} tasks successfully")

            return results

    def execute_task(self, task: Task, timeout: Optional[float] = None) -> TaskResult:
        """Execute a single task."""
        results = self.execute_tasks([task], timeout)
        return results[0] if results else TaskResult(
            task_id=task.id,
            success=False,
            error=Exception("No result returned")
        )


# Global worker pools for different component types
FETCHER_POOL = WorkerPool(max_workers=4, name="FetcherPool")
PROCESSOR_POOL = WorkerPool(max_workers=8, name="ProcessorPool")
OPTIMIZER_POOL = WorkerPool(max_workers=8, name="OptimizerPool")  # Increased from 2 to 8 for better parallelism


def get_fetcher_pool() -> WorkerPool:
    """Get the global fetcher worker pool."""
    return FETCHER_POOL


def get_processor_pool() -> WorkerPool:
    """Get the global processor worker pool."""
    return PROCESSOR_POOL


def get_optimizer_pool() -> WorkerPool:
    """Get the global optimizer worker pool."""
    return OPTIMIZER_POOL


def get_worker_pool_by_type(pool_type: str, custom_workers: int = None) -> WorkerPool:
    """
    Get a worker pool by type with optional custom worker count.

    Args:
        pool_type: Type of pool ("fetcher", "processor", "optimizer", "custom")
        custom_workers: Custom number of workers (only used for "custom" type)

    Returns:
        WorkerPool instance
    """
    if pool_type == "fetcher":
        return get_fetcher_pool()
    elif pool_type == "processor":
        return get_processor_pool()
    elif pool_type == "optimizer":
        return get_optimizer_pool()
    elif pool_type == "custom":
        if custom_workers is None:
            custom_workers = 4  # Default for custom
        return WorkerPool(max_workers=custom_workers, name="CustomPool")
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")


def create_task_queue_from_args(args, component_type: str = "custom") -> TaskQueue:
    """
    Create a TaskQueue from CLI arguments.

    Args:
        args: Parsed CLI arguments
        component_type: Default component type if not specified in args

    Returns:
        Configured TaskQueue
    """
    # Determine worker pool
    pool_type = getattr(args, 'worker_pool', component_type)
    custom_workers = getattr(args, 'workers', None)

    if custom_workers and custom_workers > 0:
        # Use custom worker count
        worker_pool = WorkerPool(max_workers=custom_workers, name=f"{pool_type.title()}Pool")
    else:
        # Use predefined pool
        worker_pool = get_worker_pool_by_type(pool_type, custom_workers)

    # Create task queue
    queue_name = f"{pool_type.title()}Queue"
    task_queue = TaskQueue(max_workers=worker_pool.max_workers, name=queue_name)

    return task_queue


def create_task(
    task_id: str,
    func: Callable,
    *args,
    priority: int = 0,
    **kwargs
) -> Task:
    """Create a task object."""
    return Task(
        id=task_id,
        func=func,
        args=args,
        kwargs=kwargs,
        priority=priority
    )


def run_concurrent(
    func: Callable,
    items: List[Any],
    max_workers: int = 4,
    task_prefix: str = "task",
    **kwargs
) -> List[TaskResult]:
    """
    Convenience function to run a function concurrently on a list of items.

    Args:
        func: Function to execute on each item
        items: List of items to process
        max_workers: Maximum number of concurrent workers
        task_prefix: Prefix for task IDs
        **kwargs: Additional arguments to pass to func

    Returns:
        List of task results
    """
    tasks = []
    for i, item in enumerate(items):
        # Create task with proper argument structure
        task_args = (item,)  # positional args
        task_kwargs = kwargs.copy()  # keyword args

        task = Task(
            id=f"{task_prefix}_{i}",
            func=func,
            args=task_args,
            kwargs=task_kwargs,
            priority=0
        )
        tasks.append(task)

    queue = TaskQueue(max_workers, f"Concurrent{task_prefix.title()}")
    return queue.execute_tasks(tasks)


# Initialize global pools on import
def init_global_pools():
    """Initialize global worker pools."""
    logger.info("🔧 Initializing global worker pools")
    # Pools are created lazily when first accessed
    pass


# Cleanup on module unload
def cleanup_global_pools():
    """Clean up global worker pools."""
    logger.info("🧹 Cleaning up global worker pools")
    FETCHER_POOL.stop()
    PROCESSOR_POOL.stop()
    OPTIMIZER_POOL.stop()


# Initialize on import
init_global_pools()