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

import asyncio
from unittest.mock import patch

import pytest

from src.utils.concurrency import (
    AsyncTaskQueue,
    create_task,
    run_concurrent_async,
)


class TestAsyncTaskQueue:
    """Test AsyncTaskQueue functionality."""

    @pytest.mark.asyncio
    async def test_async_task_queue_initialization(self):
        """Test async task queue initialization."""
        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        assert queue.max_concurrent == 2
        assert queue.name == "TestQueue"

    @pytest.mark.asyncio
    async def test_single_task_execution(self):
        """Test executing a single task."""

        async def add_task(a, b):
            return a + b

        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        task = create_task("add_task", add_task, 3, b=7)
        result = await queue.execute_task(task)

        assert result.success
        assert result.result == 10
        assert result.task_id == "add_task"

    @pytest.mark.asyncio
    async def test_multiple_tasks_execution(self):
        """Test executing multiple tasks."""

        async def power_task(base, exp=2):
            await asyncio.sleep(0.01)  # Simulate async work
            return base**exp

        tasks = [create_task(f"power_{i}", power_task, i, exp=2) for i in range(1, 6)]

        queue = AsyncTaskQueue(max_concurrent=3, name="TestQueue")
        results = await queue.execute_tasks(tasks)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == (i + 1) ** 2
            assert result.task_id == f"power_{i + 1}"

    @pytest.mark.asyncio
    async def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_calls = []

        def progress_callback(completed: int, total: int, result):
            progress_calls.append((completed, total, result.success))

        async def slow_task(delay):
            await asyncio.sleep(delay)
            return f"completed_{delay}"

        tasks = [create_task(f"slow_{i}", slow_task, 0.01 * i) for i in range(1, 4)]

        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        await queue.execute_tasks(tasks, progress_callback=progress_callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3, True)  # Final call

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in task execution."""

        async def error_task():
            raise RuntimeError("Test runtime error")

        async def good_task():
            return "success"

        async def another_good():
            return "success2"

        tasks = [
            create_task("good_task", good_task),
            create_task("error_task", error_task),
            create_task("another_good", another_good),
        ]

        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        results = await queue.execute_tasks(tasks)

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

        async def sample_func(a, b=10):
            return a + b

        task = create_task("test", sample_func, 5, b=15)
        assert task.id == "test"
        assert task.func == sample_func
        assert task.args == (5,)
        assert task.kwargs == {"b": 15}
        assert task.priority == 0

    @pytest.mark.asyncio
    async def test_run_concurrent_async(self):
        """Test run_concurrent_async utility function."""

        async def double_func(x):
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_concurrent_async(double_func, items, max_workers=3, task_prefix="double")

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == (i + 1) * 2
            assert result.task_id == f"double_{i}"

    @pytest.mark.asyncio
    async def test_run_concurrent_async_with_errors(self):
        """Test run_concurrent_async with some failing tasks."""

        async def risky_func(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_concurrent_async(risky_func, items, max_workers=2, task_prefix="risky")

        assert len(results) == 5

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 4
        assert len(failed) == 1
        assert isinstance(failed[0].error, ValueError)


class TestConcurrencyUtilities:
    """Test concurrency utility functions."""

    def test_create_task(self):
        """Test task creation utility."""

        async def sample_func(a, b=10):
            return a + b

        task = create_task("test", sample_func, 5, b=15)
        assert task.id == "test"
        assert task.func == sample_func
        assert task.args == (5,)
        assert task.kwargs == {"b": 15}
        assert task.priority == 0

    @pytest.mark.asyncio
    async def test_run_concurrent_async(self):
        """Test run_concurrent_async utility function."""

        async def double_func(x):
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_concurrent_async(double_func, items, max_workers=3, task_prefix="double")

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.success
            assert result.result == (i + 1) * 2
            assert result.task_id == f"double_{i}"

    @pytest.mark.asyncio
    async def test_run_concurrent_async_with_errors(self):
        """Test run_concurrent_async with some failing tasks."""

        async def risky_func(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2

        items = [1, 2, 3, 4, 5]
        results = await run_concurrent_async(risky_func, items, max_workers=2, task_prefix="risky")

        assert len(results) == 5

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        assert len(successful) == 4
        assert len(failed) == 1
        assert isinstance(failed[0].error, ValueError)


class TestConcurrencyEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        """Test handling of empty task list."""
        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        results = await queue.execute_tasks([])
        assert results == []

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task execution with timeout."""

        async def slow_task():
            await asyncio.sleep(2)
            return "completed"

        task = create_task("slow", slow_task)
        queue = AsyncTaskQueue(max_concurrent=1, name="TestQueue")

        # This should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.execute_tasks([task]), timeout=0.1)

    @pytest.mark.asyncio
    async def test_task_with_no_args(self):
        """Test task with no arguments."""

        async def no_arg_task():
            return "success"

        task = create_task("no_args", no_arg_task)
        queue = AsyncTaskQueue(max_concurrent=1, name="TestQueue")
        result = await queue.execute_task(task)

        assert result.success
        assert result.result == "success"


class TestIntegration:
    """Integration tests for concurrency with real workflows."""

    @patch("src.utils.fetcher.get_full_study")
    def test_fetcher_concurrency_integration(self, mock_get_full_study):
        """Test fetcher workflow with concurrent processing."""
        # Mock the API call
        mock_get_full_study.return_value = {"mock": "trial_data"}

        from src.utils.config import Config
        from src.workflows.trials_fetcher import TrialsFetcherWorkflow

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
        from src.utils.config import Config
        from src.workflows.trials_optimizer import TrialsOptimizerWorkflow

        # Create test trial data
        trial_data = [
            {"nct_id": "NCT001", "title": "Trial 1"},
            {"nct_id": "NCT002", "title": "Trial 2"},
            {"nct_id": "NCT003", "title": "Trial 3"},
        ]

        # Create workflow with test config
        Config()
        workflow = TrialsOptimizerWorkflow()

        # Test concurrent optimization
        result = workflow.execute(
            trials_data=trial_data,
            cv_folds=2,
            prompts=["direct_mcode_evidence_based_concise"],
            models=["deepseek-coder"],
            max_combinations=1,
        )

        assert result.success
        assert result.metadata["total_combinations_tested"] == 1


class TestConcurrencyEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_task_list(self):
        """Test handling of empty task list."""
        queue = AsyncTaskQueue(max_concurrent=2, name="TestQueue")
        results = await queue.execute_tasks([])
        assert results == []

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test task execution with timeout."""

        async def slow_task():
            await asyncio.sleep(2)
            return "completed"

        task = create_task("slow", slow_task)
        queue = AsyncTaskQueue(max_concurrent=1, name="TestQueue")

        # This should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(queue.execute_tasks([task]), timeout=0.1)

    @pytest.mark.asyncio
    async def test_task_with_no_args(self):
        """Test task with no arguments."""

        async def no_arg_task():
            return "success"

        task = create_task("no_args", no_arg_task)
        queue = AsyncTaskQueue(max_concurrent=1, name="TestQueue")
        result = await queue.execute_task(task)

        assert result.success
        assert result.result == "success"


if __name__ == "__main__":
    pytest.main([__file__])
