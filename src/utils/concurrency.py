"""
Pure async concurrency utilities for mCODE translator.

Provides a lean, high-performance async task queue with controlled concurrency.
"""

import asyncio
from typing import Any, Callable, List, Optional
from dataclasses import dataclass
import time

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """Async task representation."""
    id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class TaskResult:
    """Async task result."""
    task_id: str
    success: bool
    result: Any = None
    error: Exception = None
    duration: float = 0.0


class AsyncQueue:
    """
    Pure async task queue with controlled concurrency.

    Uses asyncio.Semaphore for concurrency control and thread pool for sync tasks.
    """

    def __init__(self, max_concurrent: int = 1, name: str = "AsyncQueue"):
        self.max_concurrent = max_concurrent
        self.name = name
        self.logger = get_logger(f"{name}")
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_tasks(
        self,
        tasks: List[Task],
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[TaskResult]:
        """Execute tasks with controlled concurrency."""
        if not tasks:
            return []

        self.logger.info(f"🚀 {self.name}: Processing {len(tasks)} tasks (max {self.max_concurrent} concurrent)")

        async def execute_task(task: Task) -> TaskResult:
            async with self.semaphore:
                start_time = time.time()
                try:
                    self.logger.debug(f"⚡ {self.name}: Executing {task.id}")

                    # Run sync function in thread pool - combine args and kwargs
                    loop = asyncio.get_event_loop()
                    # Create a wrapper function that handles both args and kwargs
                    def task_wrapper():
                        return task.func(*task.args, **task.kwargs)

                    result = await loop.run_in_executor(None, task_wrapper)

                    duration = time.time() - start_time
                    self.logger.debug(f"✅ {self.name}: Completed {task.id} ({duration:.2f}s)")

                    return TaskResult(
                        task_id=task.id,
                        success=True,
                        result=result,
                        duration=duration
                    )

                except Exception as e:
                    duration = time.time() - start_time
                    self.logger.error(f"❌ {self.name}: Failed {task.id} ({duration:.2f}s): {e}")
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error=e,
                        duration=duration
                    )

        # Execute with timeout if specified
        if timeout:
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*[execute_task(task) for task in tasks], return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ {self.name}: Timeout after {timeout}s")
                return []
        else:
            results = await asyncio.gather(*[execute_task(task) for task in tasks], return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id=tasks[i].id,
                    success=False,
                    error=result,
                    duration=0.0
                ))
            else:
                processed_results.append(result)

            if progress_callback:
                progress_callback(len(processed_results), len(tasks), processed_results[-1])

        successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)
        self.logger.info(f"🎉 {self.name}: Completed {successful}/{len(tasks)} tasks")

        return processed_results

    async def execute_task(self, task: Task, timeout: Optional[float] = None) -> TaskResult:
        """Execute single task."""
        results = await self.execute_tasks([task], timeout)
        return results[0] if results else TaskResult(
            task_id=task.id,
            success=False,
            error=Exception("No result")
        )


def create_task(
    task_id: str,
    func: Callable,
    *args,
    **kwargs
) -> Task:
    """Create async task."""
    return Task(id=task_id, func=func, args=args, kwargs=kwargs)


def create_async_queue_from_args(args, component_type: str = "custom") -> AsyncQueue:
    """
    Create AsyncQueue from CLI args.

    Args:
        args: CLI arguments
        component_type: Component type for naming

    Returns:
        Configured AsyncQueue
    """
    # Get worker count from args
    max_concurrent = getattr(args, 'workers', 1)
    if max_concurrent <= 0:
        max_concurrent = 1

    queue_name = f"{component_type.title()}AsyncQueue"
    return AsyncQueue(max_concurrent=max_concurrent, name=queue_name)