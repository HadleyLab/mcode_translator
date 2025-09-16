"""
Concurrent Clinical Trial Data Fetcher with Pipeline Integration
Provides high-performance concurrent processing using TaskQueue and Pipeline system
"""

import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import click

# Add src directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.fetcher import (ClinicalTrialsAPIError,
                                  calculate_total_studies, get_full_study,
                                  process_eligibility_criteria_with_mcode,
                                  search_trials)
from src.pipeline.task_queue import (BenchmarkTask, PipelineTaskQueue,
                                     get_global_task_queue,
                                     initialize_task_queue,
                                     shutdown_task_queue)
from src.shared.models import enhance_trial_with_mcode_results
from src.shared.types import TaskStatus
from src.utils import Config, get_logger

# Get logger instance
logger = get_logger(__name__)


@dataclass
class ConcurrentProcessingResult:
    """Result from concurrent trial processing"""

    total_trials: int
    successful_trials: int
    failed_trials: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    duration_seconds: float
    task_stats: Dict[str, Any]


@dataclass
class ProcessingConfig:
    """Configuration for concurrent processing"""

    max_workers: int = 5
    batch_size: int = 10
    process_criteria: bool = False
    process_trials: bool = False
    model_name: Optional[str] = None
    prompt_name: str = "direct_mcode"
    export_path: Optional[str] = None
    progress_updates: bool = True


class ConcurrentFetcher:
    """
    High-performance concurrent clinical trial fetcher with Pipeline integration
    """

    def __init__(self, config: ProcessingConfig = None):
        """
        Initialize the concurrent fetcher

        Args:
            config: Processing configuration settings
        """
        self.config = config or ProcessingConfig()
        self.task_queue: Optional[PipelineTaskQueue] = None
        self.task_results: Dict[str, Any] = {}
        self.processing_start_time: float = 0.0

    async def initialize(self) -> None:
        """Initialize the task queue and workers"""
        logger.info(
            f"🚀 Initializing concurrent fetcher with {self.config.max_workers} workers"
        )
        self.task_queue = await initialize_task_queue(
            max_workers=self.config.max_workers
        )

    async def shutdown(self) -> None:
        """Shutdown the task queue and workers"""
        if self.task_queue:
            logger.info("🛑 Shutting down concurrent fetcher")
            await shutdown_task_queue()
            self.task_queue = None

    async def search_and_process_trials(
        self, condition: str, limit: int = 100
    ) -> ConcurrentProcessingResult:
        """
        Search for trials and process them concurrently

        Args:
            condition: Medical condition to search for
            limit: Maximum number of trials to process

        Returns:
            ConcurrentProcessingResult with statistics and results
        """
        if not self.task_queue:
            await self.initialize()

        logger.info(f"🔍 Searching for '{condition}' trials with limit {limit}")
        self.processing_start_time = time.time()

        try:
            # Step 1: Search for trials
            search_result = search_trials(condition, fields=None, max_results=limit)
            trials = search_result.get("studies", [])

            if not trials:
                logger.warning("No trials found for the search criteria")
                return ConcurrentProcessingResult(
                    total_trials=0,
                    successful_trials=0,
                    failed_trials=0,
                    results=[],
                    errors=[],
                    duration_seconds=time.time() - self.processing_start_time,
                    task_stats={},
                )

            logger.info(f"📋 Found {len(trials)} trials to process")

            # Step 2: Process trials in batches
            return await self._process_trials_concurrently(trials)

        except Exception as e:
            logger.error(f"❌ Error in search and process: {str(e)}")
            raise ClinicalTrialsAPIError(f"Concurrent processing failed: {str(e)}")

    async def process_trial_list(
        self, nct_ids: List[str]
    ) -> ConcurrentProcessingResult:
        """
        Process a specific list of trial NCT IDs concurrently

        Args:
            nct_ids: List of NCT IDs to process

        Returns:
            ConcurrentProcessingResult with statistics and results
        """
        if not self.task_queue:
            await self.initialize()

        logger.info(f"📝 Processing {len(nct_ids)} specified trials")
        self.processing_start_time = time.time()

        try:
            # Fetch trial data for all NCT IDs
            trials = []
            failed_fetches = []

            for nct_id in nct_ids:
                try:
                    trial_data = get_full_study(nct_id)
                    trials.append(trial_data)
                    logger.info(f"✅ Fetched trial data for {nct_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to fetch {nct_id}: {str(e)}")
                    failed_fetches.append({"nct_id": nct_id, "error": str(e)})

            if not trials:
                logger.warning("No trials could be fetched")
                return ConcurrentProcessingResult(
                    total_trials=len(nct_ids),
                    successful_trials=0,
                    failed_trials=len(failed_fetches),
                    results=[],
                    errors=failed_fetches,
                    duration_seconds=time.time() - self.processing_start_time,
                    task_stats={},
                )

            logger.info(
                f"📋 Successfully fetched {len(trials)} trials, processing concurrently"
            )

            # Process the successfully fetched trials
            result = await self._process_trials_concurrently(trials)

            # Add fetch errors to the result
            result.errors.extend(failed_fetches)
            result.failed_trials += len(failed_fetches)

            return result

        except Exception as e:
            logger.error(f"❌ Error in process trial list: {str(e)}")
            raise ClinicalTrialsAPIError(
                f"Concurrent trial processing failed: {str(e)}"
            )

    async def _process_trials_concurrently(
        self, trials: List[Dict[str, Any]]
    ) -> ConcurrentProcessingResult:
        """
        Process a list of trials concurrently using the task queue

        Args:
            trials: List of trial data to process

        Returns:
            ConcurrentProcessingResult with processing statistics
        """
        if not self.config.process_criteria and not self.config.process_trials:
            # No mCODE processing requested, just return the trials
            return ConcurrentProcessingResult(
                total_trials=len(trials),
                successful_trials=len(trials),
                failed_trials=0,
                results=trials,
                errors=[],
                duration_seconds=time.time() - self.processing_start_time,
                task_stats={},
            )

        # Create benchmark tasks for concurrent processing
        tasks_created = []
        task_callbacks = {}

        for trial in trials:
            nct_id = self._extract_nct_id(trial)

            # Create a benchmark task for this trial
            task = BenchmarkTask(
                task_id=str(uuid.uuid4()),
                prompt_name=self.config.prompt_name,
                model_name=self.config.model_name,
                trial_id=nct_id,
                trial_data=trial,
                prompt_type="DIRECT_MCODE",
                pipeline_type="DIRECT_MCODE",
            )

            # Add callback to track results
            def create_callback(trial_data):
                def callback(completed_task: BenchmarkTask):
                    self.task_results[completed_task.task_id] = {
                        "trial_data": trial_data,
                        "task": completed_task,
                    }

                return callback

            task_callbacks[task.task_id] = create_callback(trial)

            # Submit task to queue
            task_id = await self.task_queue.add_task(task, task_callbacks[task.task_id])
            tasks_created.append((task_id, nct_id))

        logger.info(f"📤 Submitted {len(tasks_created)} tasks to queue")

        # Wait for all tasks to complete with progress updates
        await self._wait_for_completion_with_progress(len(tasks_created))

        # Collect results
        successful_results = []
        failed_results = []

        for task_id, nct_id in tasks_created:
            if task_id in self.task_results:
                task_result = self.task_results[task_id]
                task = task_result["task"]
                trial_data = task_result["trial_data"]

                if task.status == TaskStatus.SUCCESS:
                    # Add mCODE results to trial data using standardized utility
                    if task.result:
                        enhanced_trial = enhance_trial_with_mcode_results(
                            trial_data, task.result
                        )
                        successful_results.append(enhanced_trial)
                    else:
                        successful_results.append(trial_data)
                    logger.info(f"✅ Successfully processed {nct_id}")
                else:
                    failed_results.append(
                        {
                            "nct_id": nct_id,
                            "error": task.error_message,
                            "trial_data": trial_data,
                        }
                    )
                    logger.error(f"❌ Failed to process {nct_id}: {task.error_message}")
            else:
                failed_results.append(
                    {
                        "nct_id": nct_id,
                        "error": "Task result not found",
                        "trial_data": None,
                    }
                )
                logger.error(f"❌ Task result not found for {nct_id}")

        # Get final task statistics
        task_stats = self.task_queue.get_task_stats()

        duration = time.time() - self.processing_start_time
        logger.info(f"🏁 Concurrent processing completed in {duration:.2f} seconds")
        logger.info(f"   ✅ Successful: {len(successful_results)}")
        logger.info(f"   ❌ Failed: {len(failed_results)}")
        logger.info(
            f"   📊 Success rate: {len(successful_results)/len(trials)*100:.1f}%"
        )

        return ConcurrentProcessingResult(
            total_trials=len(trials),
            successful_trials=len(successful_results),
            failed_trials=len(failed_results),
            results=successful_results,
            errors=failed_results,
            duration_seconds=duration,
            task_stats=task_stats,
        )

    async def _wait_for_completion_with_progress(self, total_tasks: int) -> None:
        """
        Wait for all tasks to complete with periodic progress updates

        Args:
            total_tasks: Total number of tasks submitted
        """
        if not self.config.progress_updates:
            # Just wait for all tasks to complete without progress updates
            await self.task_queue.task_queue.join()
            return

        logger.info("📊 Starting progress monitoring...")

        completed_last_check = 0
        start_time = time.time()

        while True:
            await asyncio.sleep(2.0)  # Check every 2 seconds

            stats = self.task_queue.get_task_stats()
            completed = stats["completed_tasks"]

            if completed >= total_tasks:
                break

            # Calculate progress metrics
            elapsed = time.time() - start_time
            progress_pct = (completed / total_tasks) * 100

            # Calculate processing rate
            new_completions = completed - completed_last_check
            rate = new_completions / 2.0  # tasks per second (checked every 2 seconds)

            # Estimate time remaining
            if rate > 0:
                remaining_tasks = total_tasks - completed
                eta_seconds = remaining_tasks / rate
                eta_str = f"{eta_seconds:.0f}s"
            else:
                eta_str = "calculating..."

            logger.info(
                f"📈 Progress: {completed}/{total_tasks} ({progress_pct:.1f}%) "
                f"- Rate: {rate:.1f} tasks/sec - ETA: {eta_str} "
                f"- Workers: {stats['workers_running']}"
            )

            completed_last_check = completed

        # Final progress message
        final_elapsed = time.time() - start_time
        final_rate = total_tasks / final_elapsed if final_elapsed > 0 else 0
        logger.info(
            f"🏁 All tasks completed in {final_elapsed:.2f}s "
            f"(avg rate: {final_rate:.1f} tasks/sec)"
        )

    def _extract_nct_id(self, trial_data: Dict[str, Any]) -> str:
        """Extract NCT ID from trial data"""
        try:
            return trial_data["protocolSection"]["identificationModule"]["nctId"]
        except (KeyError, TypeError):
            return f"unknown_{str(uuid.uuid4())[:8]}"

    async def export_results(
        self, result: ConcurrentProcessingResult, export_path: str
    ) -> None:
        """
        Export processing results to a JSON file

        Args:
            result: Processing result to export
            export_path: Path to export file
        """
        export_data = {
            "summary": {
                "total_trials": result.total_trials,
                "successful_trials": result.successful_trials,
                "failed_trials": result.failed_trials,
                "success_rate": (
                    (result.successful_trials / result.total_trials * 100)
                    if result.total_trials > 0
                    else 0
                ),
                "duration_seconds": result.duration_seconds,
                "processing_rate": (
                    result.total_trials / result.duration_seconds
                    if result.duration_seconds > 0
                    else 0
                ),
            },
            "task_statistics": result.task_stats,
            "successful_trials": result.results,
            "failed_trials": result.errors,
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"📄 Results exported to {export_path}")


# Async context manager for automatic cleanup
class ConcurrentFetcherContext:
    """Context manager for automatic cleanup of ConcurrentFetcher"""

    def __init__(self, config: ProcessingConfig = None):
        self.config = config
        self.fetcher: Optional[ConcurrentFetcher] = None

    async def __aenter__(self) -> ConcurrentFetcher:
        self.fetcher = ConcurrentFetcher(self.config)
        await self.fetcher.initialize()
        return self.fetcher

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.fetcher:
            await self.fetcher.shutdown()


# Utility functions for easy integration
async def concurrent_search_and_process(
    condition: str,
    limit: int = 100,
    max_workers: int = 5,
    batch_size: int = 10,
    process_criteria: bool = False,
    process_trials: bool = False,
    model_name: Optional[str] = None,
    prompt_name: str = "direct_mcode",
    export_path: Optional[str] = None,
    progress_updates: bool = True,
) -> ConcurrentProcessingResult:
    """
    Convenience function for concurrent trial search and processing

    Args:
        condition: Medical condition to search for
        limit: Maximum number of trials to process
        max_workers: Number of concurrent workers
        batch_size: Batch size for processing
        process_criteria: Whether to process eligibility criteria
        process_trials: Whether to process complete trials
        model_name: Model to use for processing
        prompt_name: Prompt to use for processing
        export_path: Optional path to export results
        progress_updates: Whether to show progress updates

    Returns:
        ConcurrentProcessingResult
    """
    config = ProcessingConfig(
        max_workers=max_workers,
        batch_size=batch_size,
        process_criteria=process_criteria,
        process_trials=process_trials,
        model_name=model_name,
        prompt_name=prompt_name,
        export_path=export_path,
        progress_updates=progress_updates,
    )

    async with ConcurrentFetcherContext(config) as fetcher:
        result = await fetcher.search_and_process_trials(condition, limit)

        if export_path:
            await fetcher.export_results(result, export_path)

        return result


async def concurrent_process_trials(
    nct_ids: List[str],
    max_workers: int = 5,
    batch_size: int = 10,
    process_criteria: bool = False,
    process_trials: bool = False,
    model_name: Optional[str] = None,
    prompt_name: str = "direct_mcode",
    export_path: Optional[str] = None,
    progress_updates: bool = True,
) -> ConcurrentProcessingResult:
    """
    Convenience function for concurrent processing of specific trials

    Args:
        nct_ids: List of NCT IDs to process
        max_workers: Number of concurrent workers
        batch_size: Batch size for processing
        process_criteria: Whether to process eligibility criteria
        process_trials: Whether to process complete trials
        model_name: Model to use for processing
        prompt_name: Prompt to use for processing
        export_path: Optional path to export results
        progress_updates: Whether to show progress updates

    Returns:
        ConcurrentProcessingResult
    """
    config = ProcessingConfig(
        max_workers=max_workers,
        batch_size=batch_size,
        process_criteria=process_criteria,
        process_trials=process_trials,
        model_name=model_name,
        prompt_name=prompt_name,
        export_path=export_path,
        progress_updates=progress_updates,
    )

    async with ConcurrentFetcherContext(config) as fetcher:
        result = await fetcher.process_trial_list(nct_ids)

        if export_path:
            await fetcher.export_results(result, export_path)

        return result
