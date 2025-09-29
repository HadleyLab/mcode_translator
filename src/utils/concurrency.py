"""
Concurrency utilities for mCODE translator.

Provides both async and thread-based concurrency with controlled parallelism.
"""

import asyncio
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Task:
    """Task representation for both async and thread-based execution."""

    id: str
    func: Callable[..., Any]
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
        progress_callback: Optional[Callable[[int, int, TaskResult], None]] = None,
    ) -> List[TaskResult]:
        """Execute tasks with controlled concurrency using a worker pool approach."""
        if not tasks:
            return []

        self.logger.info(
            f"🚀 {self.name}: Processing {len(tasks)} tasks (max {self.max_concurrent} concurrent)"
        )

        async def worker(
            worker_id: int, task_queue: asyncio.Queue[Optional[Task]]
        ) -> None:
            """Worker function that processes tasks from the queue."""
            while True:
                try:
                    # Get task from queue
                    task = await task_queue.get()

                    # Sentinel value to stop worker
                    if task is None:
                        task_queue.task_done()
                        break

                    start_time = time.time()
                    self.logger.info(
                        f"⚡ {self.name}: WORKER-{worker_id} STARTED {task.id}"
                    )

                    try:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()

                        def task_wrapper() -> Any:
                            return task.func(*task.args, **(task.kwargs or {}))

                        result = await loop.run_in_executor(None, task_wrapper)

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
                            progress_callback(
                                len(task_results), len(tasks), task_results[-1]
                            )

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

    async def execute_task(
        self, task: Task, timeout: Optional[float] = None
    ) -> TaskResult:
        """Execute single task."""
        results = await self.execute_tasks([task], timeout)
        return (
            results[0]
            if results
            else TaskResult(
                task_id=task.id, success=False, error=Exception("No result")
            )
        )


def create_task(
    task_id: str, func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Task:
    """Create async task."""
    return Task(id=task_id, func=func, args=args, kwargs=kwargs)


def create_async_queue_from_args(
    args: Any, component_type: str = "custom"
) -> AsyncQueue:
    """
    Create AsyncQueue from CLI args.

    Args:
        args: CLI arguments
        component_type: Component type for naming

    Returns:
        Configured AsyncQueue
    """
    # Get worker count from args
    max_concurrent = getattr(args, "workers", 1)
    if max_concurrent <= 0:
        max_concurrent = 1

    queue_name = f"{component_type.title()}AsyncQueue"
    return AsyncQueue(max_concurrent=max_concurrent, name=queue_name)


# Thread-based concurrency classes for backward compatibility


class WorkerPool:
    """
    Thread-based worker pool for backward compatibility.

    Uses ThreadPoolExecutor for concurrent task execution.
    """

    def __init__(self, max_workers: int = 1, name: str = "WorkerPool"):
        self.max_workers = max_workers
        self.name = name
        self.logger = get_logger(f"{name}")
        self.executor: Optional[ThreadPoolExecutor] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the worker pool."""
        with self._lock:
            if not self._running:
                self.executor = ThreadPoolExecutor(
                    max_workers=self.max_workers, thread_name_prefix=self.name
                )
                self._running = True
                self.logger.info(
                    f"🚀 {self.name}: Started with {self.max_workers} workers"
                )

    def stop(self) -> None:
        """Stop the worker pool."""
        with self._lock:
            if self._running:
                if self.executor:
                    self.executor.shutdown(wait=True)
                    self.executor = None
                self._running = False
                self.logger.info(f"🛑 {self.name}: Stopped")

    def __enter__(self) -> "WorkerPool":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def submit_task(self, task: Task) -> Future[TaskResult]:
        """Submit a single task for execution."""
        if not self._running or not self.executor:
            raise RuntimeError(f"Worker pool {self.name} is not running")

        def task_wrapper() -> TaskResult:
            start_time = time.time()
            self.logger.info(f"⚡ {self.name}: STARTED {task.id}")

            try:
                result = task.func(*task.args, **(task.kwargs or {}))
                duration = time.time() - start_time
                self.logger.info(
                    f"✅ {self.name}: COMPLETED {task.id} ({duration:.2f}s)"
                )
                return TaskResult(
                    task_id=task.id, success=True, result=result, duration=duration
                )
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(
                    f"❌ {self.name}: FAILED {task.id} ({duration:.2f}s): {e}"
                )
                return TaskResult(
                    task_id=task.id, success=False, error=e, duration=duration
                )

        return self.executor.submit(task_wrapper)

    def submit_tasks(self, tasks: List[Task]) -> List[Future[TaskResult]]:
        """Submit multiple tasks for execution."""
        return [self.submit_task(task) for task in tasks]


class TaskQueue:
    """
    Thread-based task queue for backward compatibility.

    Wraps WorkerPool for queue-based task execution.
    """

    def __init__(self, max_workers: int = 1, name: str = "TaskQueue"):
        self.worker_pool = WorkerPool(max_workers=max_workers, name=name)
        self.logger = get_logger(f"{name}")

    def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        with self.worker_pool:
            future = self.worker_pool.submit_task(task)
            return future.result()

    def execute_tasks(
        self,
        tasks: List[Task],
        progress_callback: Optional[Callable[[int, int, TaskResult], None]] = None,
    ) -> List[TaskResult]:
        """Execute multiple tasks with optional progress callback."""
        if not tasks:
            return []

        self.logger.info(f"🚀 {self.worker_pool.name}: Processing {len(tasks)} tasks")

        with self.worker_pool:
            futures = self.worker_pool.submit_tasks(tasks)
            results = []

            for i, future in enumerate(futures):
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(tasks), result)

            successful = sum(1 for r in results if r.success)
            self.logger.info(
                f"🎉 {self.worker_pool.name}: Completed {successful}/{len(tasks)} tasks"
            )

            return results


def run_concurrent(
    func: Callable[..., Any],
    items: List[Any],
    max_workers: int = 4,
    task_prefix: str = "task",
) -> List[TaskResult]:
    """
    Run a function concurrently on a list of items.

    Args:
        func: Function to run on each item
        items: List of items to process
        max_workers: Maximum number of concurrent workers
        task_prefix: Prefix for task IDs

    Returns:
        List of TaskResult objects
    """
    tasks = [
        create_task(f"{task_prefix}_{i}", func, item) for i, item in enumerate(items)
    ]

    queue = TaskQueue(max_workers=max_workers, name=f"{task_prefix.title()}Queue")
    return queue.execute_tasks(tasks)


# Global worker pools for backward compatibility

_fetcher_pool = None
_processor_pool = None
_optimizer_pool = None


def get_fetcher_pool() -> WorkerPool:
    """Get global fetcher worker pool (singleton)."""
    global _fetcher_pool
    if _fetcher_pool is None:
        _fetcher_pool = WorkerPool(max_workers=4, name="FetcherPool")
    return _fetcher_pool


def get_processor_pool() -> WorkerPool:
    """Get global processor worker pool (singleton)."""
    global _processor_pool
    if _processor_pool is None:
        _processor_pool = WorkerPool(max_workers=8, name="ProcessorPool")
    return _processor_pool


def get_optimizer_pool() -> WorkerPool:
    """Get global optimizer worker pool (singleton)."""
    global _optimizer_pool
    if _optimizer_pool is None:
        _optimizer_pool = WorkerPool(max_workers=2, name="OptimizerPool")
    return _optimizer_pool
