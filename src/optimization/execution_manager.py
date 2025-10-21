"""
Optimization Execution Manager - Handles concurrent optimization execution.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline import McodePipeline
from shared.extractors import DataExtractor
from utils.concurrency import AsyncTaskQueue, create_async_task_queue_from_args
from utils.metrics import PerformanceMetrics


class OptimizationExecutionManager:
    """Manages the execution of optimization trials with producer-consumer pattern."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.extractor = DataExtractor()

    async def execute_optimization(
        self,
        trials_data: List[Dict[str, Any]],
        combinations: List[Dict[str, str]],
        cv_folds: int,
        cli_args: Optional[Any] = None,
    ) -> Dict[int, Dict[str, List[Any]]]:
        """Execute optimization with concurrent cross validation."""
        # Create k-fold splits
        fold_indices = self._create_kfold_splits(len(trials_data), cv_folds)

        # Setup concurrency
        if cli_args:
            task_queue = create_async_task_queue_from_args(cli_args, "optimizer")
            workers = task_queue.max_concurrent
        else:
            task_queue = AsyncTaskQueue(max_concurrent=1, name="OptimizerAsyncQueue")
            workers = task_queue.max_concurrent

        # Initialize results storage
        combo_results: Dict[int, Dict[str, List[Any]]] = {
            i: {"scores": [], "errors": [], "metrics": [], "mcode_elements": []}
            for i in range(len(combinations))
        }

        # Progress tracking
        total_tasks = len(combinations) * len(trials_data)
        completed_tasks = {"count": 0}
        progress_lock = asyncio.Lock()

        # Producer-consumer execution
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        await self._run_producer(queue, combinations, trials_data, fold_indices, cv_folds, workers)
        await self._run_workers(
            queue, workers, combo_results, completed_tasks, progress_lock, total_tasks
        )

        return combo_results

    def _create_kfold_splits(self, n_samples: int, n_folds: int) -> List[List[int]]:
        """Create k-fold cross validation splits."""
        indices = list(range(n_samples))
        fold_sizes = [n_samples // n_folds] * n_folds
        remainder = n_samples % n_folds

        for i in range(remainder):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for size in fold_sizes:
            folds.append(indices[start : start + size])
            start += size

        return folds

    async def _run_producer(
        self,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
        combinations: List[Dict[str, str]],
        trials_data: List[Dict[str, Any]],
        fold_indices: List[List[int]],
        cv_folds: int,
        workers: int,
    ) -> None:
        """Producer: put all tasks in the queue."""
        self.logger.info("🔄 Producer: Creating concurrent tasks...")
        task_id = 0

        for combo_idx, combo in enumerate(combinations):
            for fold in range(cv_folds):
                val_indices = fold_indices[fold]
                fold_trials = [trials_data[i] for i in val_indices]
                for trial in fold_trials:
                    task_data = {
                        "task_id": f"trial_{task_id}",
                        "combination": combo,
                        "trial": trial,
                        "fold": fold,
                        "combo_idx": combo_idx,
                    }
                    await queue.put(task_data)
                    task_id += 1

        # Signal completion
        for _ in range(workers):
            await queue.put(None)

        self.logger.info(f"✅ Producer: Created {task_id} tasks")

    async def _run_workers(
        self,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
        workers: int,
        combo_results: Dict[int, Dict[str, List[Any]]],
        completed_tasks: Dict[str, int],
        progress_lock: asyncio.Lock,
        total_tasks: int,
    ) -> None:
        """Run worker tasks."""
        worker_tasks = [
            asyncio.create_task(
                self._worker(i, queue, combo_results, completed_tasks, progress_lock, total_tasks)
            )
            for i in range(workers)
        ]
        await asyncio.gather(*worker_tasks)
        self.logger.info("🎉 All workers completed")

    async def _worker(
        self,
        worker_id: int,
        queue: asyncio.Queue[Optional[Dict[str, Any]]],
        combo_results: Dict[int, Dict[str, List[Any]]],
        completed_tasks: Dict[str, int],
        progress_lock: asyncio.Lock,
        total_tasks: int,
    ) -> None:
        """Individual worker processing tasks."""
        self.logger.info(f"🤖 Worker {worker_id}: Starting...")
        worker_completed = 0
        quota_exceeded_models = set()

        while True:
            task_data = await queue.get()
            if task_data is None:
                break

            # Skip quota exceeded models
            model_name = task_data["combination"]["model"]
            if model_name in quota_exceeded_models:
                async with progress_lock:
                    completed_tasks["count"] += 1
                    total_completed = completed_tasks["count"]
                    remaining = total_tasks - total_completed
                self.logger.warning(
                    f"❌ Worker {worker_id}: Skipped {model_name} + {task_data['combination']['prompt']} (NCT{self.extractor.extract_trial_id(task_data['trial'])}) - Model quota exceeded"
                )
                continue

            try:
                result = await self._test_single_trial(
                    task_data["combination"],
                    task_data["trial"],
                    task_data["fold"],
                    task_data["combo_idx"],
                )

                # Store results
                combo_idx = result["combo_idx"]
                score = result.get("score", 0)
                metrics = result.get("quality_metrics", {})
                predicted_mcode = result.get("predicted_mcode", [])

                combo_results[combo_idx]["scores"].append(score)
                combo_results[combo_idx]["metrics"].append(metrics)
                combo_results[combo_idx]["mcode_elements"].extend(predicted_mcode)

                worker_completed += 1
                combo = task_data["combination"]

                # Update progress
                async with progress_lock:
                    completed_tasks["count"] += 1
                    total_completed = completed_tasks["count"]
                    remaining = total_tasks - total_completed

                # Log progress
                nctid = self.extractor.extract_trial_id(task_data["trial"])
                perf_data = result.get("performance_metrics", {})
                processing_time = perf_data.get("processing_time_seconds", 0)
                tokens_used = perf_data.get("tokens_used", 0)
                elements_found = metrics.get("element_count", 0)

                self.logger.info(
                    f"✅ Worker {worker_id}: Trial {worker_completed} - {combo['model']} + {combo['prompt']} (NCT{nctid}, score: {score:.3f}, {elements_found} elements, {processing_time:.1f}s, {tokens_used} tokens)"
                )
                self.logger.info(
                    f"📊 Progress: {total_completed}/{total_tasks} completed ({remaining} remaining)"
                )

            except Exception as e:
                combo_idx = task_data["combo_idx"]
                combo_name = (
                    f"{task_data['combination']['model']} + {task_data['combination']['prompt']}"
                )
                nctid = self.extractor.extract_trial_id(task_data["trial"])

                if "quota" in str(e).lower():
                    quota_exceeded_models.add(model_name)
                    async with progress_lock:
                        completed_tasks["count"] += 1
                    combo_results[combo_idx]["errors"].append(f"Quota exceeded for {model_name}")
                    continue
                else:
                    async with progress_lock:
                        completed_tasks["count"] += 1
                    combo_results[combo_idx]["errors"].append(str(e))
                    self.logger.exception(
                        f"❌ Worker {worker_id}: Failed {combo_name} (NCT{nctid}) - {e}"
                    )

        self.logger.info(f"🏁 Worker {worker_id}: Finished processing {worker_completed} tasks")

    async def _test_single_trial(
        self,
        combination: Dict[str, str],
        trial: Dict[str, Any],
        fold: int,
        combo_idx: int,
    ) -> Dict[str, Any]:
        """Test a single trial with a specific prompt×model combination."""
        prompt_name = combination["prompt"]
        model_name = combination["model"]

        try:
            perf_metrics = PerformanceMetrics()
            perf_metrics.start_tracking()

            pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)
            result = await pipeline.process(trial)

            predicted = [elem.model_dump() for elem in result.mcode_mappings]
            num_elements = len(predicted)

            from utils.token_tracker import global_token_tracker

            token_usage = global_token_tracker.get_total_usage()
            tokens_used = token_usage.total_tokens if token_usage else 0

            perf_metrics.stop_tracking(tokens_used=tokens_used, elements_processed=num_elements)
            score = min(num_elements / 10.0, 1.0)

            metrics = {
                "precision": score,
                "recall": score,
                "f1_score": score,
                "element_count": num_elements,
                **perf_metrics.get_metrics(),
            }

            return {
                "combination": combination,
                "combo_idx": combo_idx,
                "fold": fold,
                "trial_score": score,
                "score": score,
                "quality_metrics": metrics,
                "predicted_mcode": predicted,
                "performance_metrics": perf_metrics.to_dict(),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "combination": combination,
                "combo_idx": combo_idx,
                "fold": fold,
                "trial_score": 0,
                "score": 0,
                "quality_metrics": {
                    "precision": 0,
                    "recall": 0,
                    "f1_score": 0,
                    "element_count": 0,
                },
                "predicted_mcode": [],
                "performance_metrics": {},
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
