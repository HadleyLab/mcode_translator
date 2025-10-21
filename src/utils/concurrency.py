"""
Concurrency utilities for mCODE translator.

Provides pure async concurrency with controlled parallelism.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """Task representation for async execution."""

    id: str
    func: Callable[..., Awaitable[Any]]
    args: tuple[Any, ...] = ()
    kwargs: Optional[dict[str, Any]] = None
    priority: int = 0

    def __post_init__(self) -> None:
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class TaskResult:
    """Async task result."""

    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration: float = 0.0


class AsyncTaskQueue:
    """
    Pure async task queue with controlled concurrency.

    Uses asyncio.Queue and asyncio.Semaphore for controlled async execution.
    """

    def __init__(self, max_concurrent: int = 1, name: str = "AsyncTaskQueue"):
        self.max_concurrent = max_concurrent
        self.name = name
        self.logger = get_logger(f"{name}")
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def execute_tasks(
        self,
        tasks: List[Task],
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[int, int, TaskResult], None]] = None,
    ) -> List[TaskResult]:
        """Execute tasks with controlled concurrency using asyncio.Queue and Semaphore."""
        if not tasks:
            return []

        self.logger.info(
            f"🚀 {self.name}: Processing {len(tasks)} tasks (max {self.max_concurrent} concurrent)"
        )

        async def worker(worker_id: int, task_queue: asyncio.Queue[Optional[Task]]) -> None:
            """Worker function that processes tasks from the queue."""
            while True:
                try:
                    # Get task from queue
                    task = await task_queue.get()

                    # Sentinel value to stop worker
                    if task is None:
                        task_queue.task_done()
                        break

                    async with self.semaphore:
                        start_time = time.time()
                        self.logger.info(f"⚡ {self.name}: WORKER-{worker_id} STARTED {task.id}")

                        try:
                            # Await async function directly
                            result = await task.func(*task.args, **(task.kwargs or {}))

                            duration = time.time() - start_time
                            self.logger.info(
                                f"✅ {self.name}: WORKER-{worker_id} COMPLETED {task.id} ({duration:.2f}s)"
                            )

                            task_results.append(
                                TaskResult(
                                    task_id=task.id,
                                    success=True,
                                    result=result,
                                    duration=duration,
                                )
                            )

                        except Exception as e:
                            duration = time.time() - start_time
                            self.logger.error(
                                f"❌ {self.name}: WORKER-{worker_id} FAILED {task.id} ({duration:.2f}s): {e}"
                            )
                            task_results.append(
                                TaskResult(
                                    task_id=task.id,
                                    success=False,
                                    error=e,
                                    duration=duration,
                                )
                            )

                        finally:
                            task_queue.task_done()

                            if progress_callback:
                                progress_callback(len(task_results), len(tasks), task_results[-1])

                except Exception as e:
                    self.logger.error(f"❌ {self.name}: WORKER-{worker_id} error: {e}")

        # Create task queue and results list
        task_queue: asyncio.Queue[Optional[Task]] = asyncio.Queue()
        task_results: List[TaskResult] = []

        # Add all tasks to queue
        for task in tasks:
            await task_queue.put(task)

        # Add sentinel values to stop workers
        for _ in range(self.max_concurrent):
            await task_queue.put(None)

        # Start workers
        workers = []
        for i in range(self.max_concurrent):
            worker_task = asyncio.create_task(worker(i + 1, task_queue))
            workers.append(worker_task)

        # Wait for all workers to complete
        if timeout:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*workers, return_exceptions=True), timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.error(f"⏰ {self.name}: Timeout after {timeout}s")
                # Cancel remaining tasks
                for w in workers:
                    if not w.done():
                        w.cancel()
                return []
        else:
            await asyncio.gather(*workers, return_exceptions=True)

        successful = len([r for r in task_results if r.success])
        self.logger.info(f"🎉 {self.name}: Completed {successful}/{len(tasks)} tasks")

        return task_results

    async def execute_task(self, task: Task, timeout: Optional[float] = None) -> TaskResult:
        """Execute single task."""
        results = await self.execute_tasks([task], timeout)
        return (
            results[0]
            if results
            else TaskResult(task_id=task.id, success=False, error=Exception("No result"))
        )


def create_task(task_id: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Task:
    """Create async task."""
    return Task(id=task_id, func=func, args=args, kwargs=kwargs)


def create_async_task_queue_from_args(args: Any, component_type: str = "custom") -> AsyncTaskQueue:
    """
    Create AsyncTaskQueue from CLI args.

    Args:
        args: CLI arguments
        component_type: Component type for naming

    Returns:
        Configured AsyncTaskQueue
    """
    # Get worker count from args
    max_concurrent = getattr(args, "workers", 1)
    if max_concurrent <= 0:
        max_concurrent = 1

    queue_name = f"{component_type.title()}AsyncTaskQueue"
    return AsyncTaskQueue(max_concurrent=max_concurrent, name=queue_name)


async def run_concurrent_async(
    func: Callable[..., Awaitable[Any]],
    items: List[Any],
    max_workers: int = 4,
    task_prefix: str = "task",
) -> List[TaskResult]:
    """
    Run an async function concurrently on a list of items.

    Args:
        func: Async function to run on each item
        items: List of items to process
        max_workers: Maximum number of concurrent workers
        task_prefix: Prefix for task IDs

    Returns:
        List of TaskResult objects
    """
    tasks = [create_task(f"{task_prefix}_{i}", func, item) for i, item in enumerate(items)]

    queue = AsyncTaskQueue(max_concurrent=max_workers, name=f"{task_prefix.title()}AsyncTaskQueue")
    return await queue.execute_tasks(tasks)
