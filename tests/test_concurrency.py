"""
Comprehensive tests for the concurrency layer.

Tests cover:
- WorkerPool lifecycle management
- TaskQueue execution
- Concurrent task processing
- Error handling in concurrent operations
- Progress tracking
- Resource cleanup
- Integration with existing workflows
"""

import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from src.utils.concurrency import (
    WorkerPool,
    TaskQueue,
    Task,
    create_task,
    run_concurrent,
    get_fetcher_pool,
    get_processor_pool,
    get_optimizer_pool
)


class TestWorkerPool:
    """Test WorkerPool functionality."""

    def test_worker_pool_initialization(self):
        """Test worker pool initialization."""
        pool = WorkerPool(max_workers=2, name="TestPool")
        assert pool.max_workers == 2
        assert pool.name == "TestPool"
        assert not pool._running
        assert pool.executor is None

    def test_worker_pool_lifecycle(self):
        """Test worker pool start/stop lifecycle."""
        pool = WorkerPool(max_workers=2, name="TestPool")

        # Start pool
        pool.start()
        assert pool._running
        assert pool.executor is not None
        assert isinstance(pool.executor, ThreadPoolExecutor)

        # Stop pool
        pool.stop()
        assert not pool._running
        assert pool.executor is None

    def test_worker_pool_context_manager(self):
        """Test worker pool as context manager."""
        with WorkerPool(max_workers=2, name="TestPool") as pool:
            assert pool._running
            assert pool.executor is not None

        assert not pool._running
        assert pool.executor is None

    def test_task_execution(self):
        """Test basic task execution."""
        def simple_task(x, y=10):
            return x + y

        pool = WorkerPool(max_workers=2, name="TestPool")
        with pool:
            future = pool.submit_task(create_task("test_task", simple_task, 5, y=15))
            result = future.result()
            assert result.success
            assert result.result == 20
            assert result.task_id == "test_task"
            assert result.duration > 0

    def test_task_execution_with_error(self):
        """Test task execution with error."""
        def failing_task():
            raise ValueError("Test error")

        pool = WorkerPool(max_workers=2, name="TestPool")
        with pool:
            future = pool.submit_task(create_task("failing_task", failing_task))
            result = future.result()
            assert not result.success
            assert isinstance(result.error, ValueError)
            assert str(result.error) == "Test error"
            assert result.task_id == "failing_task"
            assert result.duration > 0

    def test_multiple_tasks(self):
        """Test executing multiple tasks."""
        def multiply_task(x):
            time.sleep(0.01)  # Simulate work
            return x * 2

        tasks = [
            create_task(f"task_{i}", multiply_task, i)
            for i in range(5)
        ]

        pool = WorkerPool(max_workers=3, name="TestPool")
        with pool:
            futures = pool.submit_tasks(tasks)
            results = [future.result() for future in futures]

            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.success
                assert result.result == i * 2
                assert result.task_id == f"task_{i}"


class TestTaskQueue:
    """Test TaskQueue functionality."""

    def test_task_queue_initialization(self):
        """Test task queue initialization."""
        queue = TaskQueue(max_workers=2, name="TestQueue")
        assert queue.worker_pool.max_workers == 2
        assert queue.worker_pool.name == "TestQueue"

    def test_single_task_execution(self):
        """Test executing a single task."""
        def add_task(a, b):
            return a + b

        queue = TaskQueue(max_workers=2, name="TestQueue")
        task = create_task("add_task", add_task, 3, b=7)
        result = queue.execute_task(task)

        assert result.success
        assert result.result == 10
        assert result.task_id == "add_task"

    def test_multiple_tasks_execution(self):
        """Test executing multiple tasks."""
        def power_task(base, exp=2):
            time.sleep(0.01)  # Simulate work
            return base ** exp

        tasks = [
            create_task(f"power_{i}", power_task, i, exp=2)
            for i in range(1, 6)
        ]

        queue = TaskQueue(max_workers=3, name="TestQueue")
        results = queue.execute_tasks(tasks)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == (i + 1) ** 2
            assert result.task_id == f"power_{i + 1}"

    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_calls = []

        def progress_callback(completed, total, result):
            progress_calls.append((completed, total, result.success))

        def slow_task(delay):
            time.sleep(delay)
            return f"completed_{delay}"

        tasks = [
            create_task(f"slow_{i}", slow_task, 0.01 * i)
            for i in range(1, 4)
        ]

        queue = TaskQueue(max_workers=2, name="TestQueue")
        results = queue.execute_tasks(tasks, progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3, True)  # Final call

    def test_error_handling(self):
        """Test error handling in task execution."""
        def error_task():
            raise RuntimeError("Test runtime error")

        tasks = [
            create_task("good_task", lambda: "success"),
            create_task("error_task", error_task),
            create_task("another_good", lambda: "success2")
        ]

        queue = TaskQueue(max_workers=2, name="TestQueue")
        results = queue.execute_tasks(tasks)

        assert len(results) == 3

        # Check successful tasks
        success_results = [r for r in results if r.success]
        assert len(success_results) == 2

        # Check failed task
        failed_results = [r for r in results if not r.success]
        assert len(failed_results) == 1
        assert isinstance(failed_results[0].error, RuntimeError)


class TestConcurrencyUtilities:
    """Test concurrency utility functions."""

    def test_create_task(self):
        """Test task creation utility."""
        def sample_func(a, b=10):
            return a + b

        task = create_task("test", sample_func, 5, b=15)
        assert task.id == "test"
        assert task.func == sample_func
        assert task.args == (5,)
        assert task.kwargs == {"b": 15}
        assert task.priority == 0

    def test_run_concurrent(self):
        """Test run_concurrent utility function."""
        def double_func(x):
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = run_concurrent(double_func, items, max_workers=3, task_prefix="double")

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == (i + 1) * 2
            assert result.task_id == f"double_{i}"

    def test_run_concurrent_with_errors(self):
        """Test run_concurrent with some failing tasks."""
        def risky_func(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = run_concurrent(risky_func, items, max_workers=2, task_prefix="risky")

        assert len(results) == 5

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 4
        assert len(failed) == 1
        assert isinstance(failed[0].error, ValueError)


class TestGlobalPools:
    """Test global worker pool instances."""

    def test_global_pools_exist(self):
        """Test that global pools are properly initialized."""
        fetcher_pool = get_fetcher_pool()
        processor_pool = get_processor_pool()
        optimizer_pool = get_optimizer_pool()

        assert fetcher_pool.max_workers == 4
        assert processor_pool.max_workers == 8
        assert optimizer_pool.max_workers == 2

        assert fetcher_pool.name == "FetcherPool"
        assert processor_pool.name == "ProcessorPool"
        assert optimizer_pool.name == "OptimizerPool"

    def test_global_pools_are_singletons(self):
        """Test that global pools return the same instance."""
        pool1 = get_fetcher_pool()
        pool2 = get_fetcher_pool()
        assert pool1 is pool2

        pool3 = get_processor_pool()
        pool4 = get_processor_pool()
        assert pool3 is pool4


class TestIntegration:
    """Integration tests for concurrency with real workflows."""

    @patch('src.utils.fetcher.get_full_study')
    def test_fetcher_concurrency_integration(self, mock_get_full_study):
        """Test fetcher workflow with concurrent processing."""
        # Mock the API call
        mock_get_full_study.return_value = {"mock": "trial_data"}

        from src.workflows.trials_fetcher_workflow import TrialsFetcherWorkflow
        from src.utils.config import Config

        # Create workflow with test config
        config = Config()
        workflow = TrialsFetcherWorkflow(config)

        # Test concurrent fetching of multiple trials
        nct_ids = ["NCT001", "NCT002", "NCT003"]
        result = workflow.execute(nct_ids=nct_ids)

        assert result.success
        assert len(result.data) == 3
        assert mock_get_full_study.call_count == 3

    def test_optimizer_concurrency_integration(self):
        """Test optimizer workflow with concurrent processing."""
        from src.workflows.trials_optimizer_workflow import TrialsOptimizerWorkflow
        from src.utils.config import Config

        # Create test trial data
        trial_data = [
            {"nct_id": "NCT001", "title": "Trial 1"},
            {"nct_id": "NCT002", "title": "Trial 2"},
            {"nct_id": "NCT003", "title": "Trial 3"}
        ]

        # Create workflow with test config
        config = Config()
        workflow = TrialsOptimizerWorkflow(config, memory_storage=False)

        # Test concurrent optimization
        result = workflow.execute(
            trials_data=trial_data,
            cv_folds=2,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1
        )

        assert result.success
        assert result.metadata["total_combinations_tested"] == 1


class TestConcurrencyEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_task_list(self):
        """Test handling of empty task list."""
        queue = TaskQueue(max_workers=2, name="TestQueue")
        results = queue.execute_tasks([])
        assert results == []

    def test_task_timeout(self):
        """Test task execution with timeout."""
        def slow_task():
            time.sleep(2)
            return "completed"

        task = create_task("slow", slow_task)
        queue = TaskQueue(max_workers=1, name="TestQueue")

        # This should timeout
        with pytest.raises(Exception):  # Should raise TimeoutError
            queue.execute_tasks([task], timeout=0.1)

    def test_worker_pool_double_start_stop(self):
        """Test starting/stopping worker pool multiple times."""
        pool = WorkerPool(max_workers=2, name="TestPool")

        # Multiple starts should be safe
        pool.start()
        pool.start()
        assert pool._running

        # Multiple stops should be safe
        pool.stop()
        pool.stop()
        assert not pool._running

    def test_task_with_no_args(self):
        """Test task with no arguments."""
        def no_arg_task():
            return "success"

        task = create_task("no_args", no_arg_task)
        queue = TaskQueue(max_workers=1, name="TestQueue")
        result = queue.execute_task(task)

        assert result.success
        assert result.result == "success"


if __name__ == "__main__":
    pytest.main([__file__])