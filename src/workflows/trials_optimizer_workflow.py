"""
Trials Optimizer Workflow - Optimize mCODE translation parameters.

This workflow handles testing different combinations of prompts and models
to find optimal settings for mCODE translation. Results are saved to
configuration files, not to CORE Memory.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline import McodePipeline
from src.utils.logging_config import get_logger
from src.utils.concurrency import AsyncQueue, create_task, create_async_queue_from_args
from src.utils.metrics import BenchmarkMetrics, PerformanceMetrics

from .base_workflow import BaseWorkflow, WorkflowResult


class TrialsOptimizerWorkflow(BaseWorkflow):
    """
    Workflow for optimizing mCODE translation parameters.

    Tests different combinations of prompts and models to find optimal
    settings for mCODE processing. Results are saved to config files.
    """

    @property
    def memory_space(self) -> str:
        """Optimizer workflows use 'optimization' space."""
        return "optimization"

    async def execute(self, **kwargs) -> WorkflowResult:
        """
        Execute the optimization workflow with cross validation.

        Args:
            **kwargs: Workflow parameters including:
                - trials_data: List of trial data for testing
                - prompts: List of prompt templates to test
                - models: List of LLM models to test
                - max_combinations: Maximum combinations to test
                - cv_folds: Number of cross validation folds
                - output_config: Where to save optimal settings
                - cli_args: CLI arguments for concurrency configuration

        Returns:
            WorkflowResult: Optimization results
        """
        try:
            # Extract parameters
            trials_data = kwargs["trials_data"]
            prompts = kwargs.get("prompts", ["direct_mcode_evidence_based_concise"])
            models = kwargs.get("models", ["deepseek-coder"])
            max_combinations = kwargs.get("max_combinations", 5)
            cv_folds = kwargs["cv_folds"]
            output_config = kwargs.get("output_config")

            if not trials_data:
                return self._create_result(
                    success=False,
                    error_message="No trial data provided for optimization.",
                )

            if len(trials_data) < cv_folds:
                self.logger.warning(
                    f"Only {len(trials_data)} trials available, reducing CV folds to {len(trials_data)}"
                )
                cv_folds = len(trials_data)

            # Log optimization scope for clarity
            self.logger.info(f"🔬 OPTIMIZATION SCOPE:")
            self.logger.info(f"   📊 Prompts: {len(prompts)} ({', '.join(prompts)})")
            self.logger.info(f"   🤖 Models: {len(models)} ({', '.join(models)})")
            self.logger.info(f"   📈 Max combinations: {max_combinations}")
            self.logger.info(f"   📋 Trials: {len(trials_data)}")
            self.logger.info(f"   🔄 CV folds: {cv_folds}")

            # Generate combinations to test
            combinations = self._generate_combinations(
                prompts, models, max_combinations
            )

            # Log actual combinations generated
            total_possible = len(prompts) * len(models)
            self.logger.info(f"🎯 COMBINATIONS GENERATED:")
            self.logger.info(f"   📊 Total possible: {total_possible}")
            self.logger.info(f"   ✅ Actually testing: {len(combinations)}")
            if len(combinations) < total_possible:
                self.logger.info(f"   ✂️  Limited by max_combinations={max_combinations}")
            else:
                self.logger.info(f"   🎉 Testing ALL possible combinations!")

            self.logger.info(f"🧪 Generated {len(combinations)} combinations to test:")
            for i, combo in enumerate(combinations, 1):
                self.logger.info(f"  {i}. {combo['model']} + {combo['prompt']}")

            # Create proper k-fold splits
            fold_indices = self._create_kfold_splits(len(trials_data), cv_folds)
            total_trials_per_fold = [len(fold) for fold in fold_indices]
            total_trials_processed = sum(total_trials_per_fold)

            # Each combination is tested on each trial in each fold for maximum parallelism
            total_tasks = len(combinations) * total_trials_processed
            self.logger.info(
                f"🔬 Fully asynchronous optimization: {len(combinations)} combinations × {total_trials_processed} trials = {total_tasks} concurrent tasks"
            )
            self.logger.info(f"📊 Fold sizes: {total_trials_per_fold} (total: {total_trials_processed} trials)")

            # Run optimization with concurrent cross validation
            optimization_results = []
            best_result = None
            best_score = 0

            # Get concurrency configuration from CLI args
            cli_args = kwargs.get("cli_args")
            if cli_args:
                task_queue = create_async_queue_from_args(cli_args, "optimizer")
                workers = task_queue.max_concurrent
                self.logger.info(f"✅ Using CLI-configured concurrency: {workers} workers")
            else:
                # Fallback to default async queue
                task_queue = AsyncQueue(max_concurrent=1, name="OptimizerAsyncQueue")
                workers = task_queue.max_concurrent
                self.logger.warning("⚠️ No CLI args provided, falling back to 1 worker")

            self.logger.info(f"🤖 Using async queue with {workers} max concurrent tasks")

            # Simple producer-consumer pattern
            queue = asyncio.Queue()
            combo_results = {i: {"scores": [], "errors": [], "metrics": [], "mcode_elements": []} for i in range(len(combinations))}

            # Producer: put all tasks in the queue
            async def producer():
                self.logger.info("🔄 Producer: Creating concurrent tasks...")
                task_id = 0

                # Pre-compute fold trials
                fold_trials = []
                for fold in range(cv_folds):
                    val_indices = fold_indices[fold]
                    fold_trials.append([trials_data[i] for i in val_indices])

                # Put all tasks in queue
                for combo_idx, combo in enumerate(combinations):
                    for fold in range(cv_folds):
                        val_trials = fold_trials[fold]
                        for trial_idx, trial in enumerate(val_trials):
                            task_data = {
                                "task_id": f"trial_{task_id}",
                                "combination": combo,
                                "trial": trial,
                                "fold": fold,
                                "combo_idx": combo_idx
                            }
                            await queue.put(task_data)
                            task_id += 1

                # Signal completion
                await queue.put(None)
                self.logger.info(f"✅ Producer: Created {task_id} tasks")

            # Consumer: process tasks from queue
            async def worker(worker_id: int):
                self.logger.info(f"🤖 Worker {worker_id}: Starting...")
                completed = 0

                while True:
                    task_data = await queue.get()
                    if task_data is None:
                        await queue.put(None)  # Pass stop signal
                        break

                    try:
                        # Process the task
                        result = await self._test_single_trial(
                            task_data["combination"],
                            task_data["trial"],
                            task_data["fold"],
                            task_data["combo_idx"]
                        )

                        # Store results
                        combo_idx = result["combo_idx"]
                        score = result.get("score", 0)
                        metrics = result.get("quality_metrics", {})
                        predicted_mcode = result.get("predicted_mcode", [])

                        combo_results[combo_idx]["scores"].append(score)
                        combo_results[combo_idx]["metrics"].append(metrics)
                        combo_results[combo_idx]["mcode_elements"].extend(predicted_mcode)

                        completed += 1
                        combo = combinations[combo_idx]
                        self.logger.info(f"✅ Worker {worker_id}: Trial {completed} - {combo['model']} + {combo['prompt']} (score: {score:.3f})")

                    except Exception as e:
                        combo_idx = task_data["combo_idx"]
                        combo_results[combo_idx]["errors"].append(str(e))
                        combo_name = combinations[combo_idx]['model'] + " + " + combinations[combo_idx]['prompt']
                        self.logger.warning(f"❌ Worker {worker_id}: Failed {combo_name} - {e}")

                self.logger.info(f"🏁 Worker {worker_id}: Finished processing {completed} tasks")

            # Start producer and workers
            producer_task = asyncio.create_task(producer())
            worker_tasks = [asyncio.create_task(worker(i)) for i in range(workers)]

            # Wait for completion
            await asyncio.gather(producer_task, *worker_tasks)
            self.logger.info("🎉 All workers completed")

            # Create directory for saving individual runs
            runs_dir = Path("optimization_runs")
            runs_dir.mkdir(exist_ok=True)
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Aggregate results by combination
            for combo_idx, combo in enumerate(combinations):
                scores = combo_results[combo_idx]["scores"]
                errors = combo_results[combo_idx]["errors"]
                all_metrics = combo_results[combo_idx]["metrics"]

                if scores:
                    # Calculate CV statistics for this combination
                    cv_average_score = sum(scores) / len(scores)
                    cv_std = (sum((s - cv_average_score) ** 2 for s in scores) / len(scores)) ** 0.5

                    # Aggregate detailed metrics
                    avg_precision = sum(m.get("precision", 0) for m in all_metrics) / len(all_metrics)
                    avg_recall = sum(m.get("recall", 0) for m in all_metrics) / len(all_metrics)
                    avg_f1 = sum(m.get("f1_score", 0) for m in all_metrics) / len(all_metrics)

                    result = {
                        "combination": combo,
                        "success": True,
                        "cv_average_score": cv_average_score,
                        "cv_std_score": cv_std,
                        "fold_scores": scores,  # Individual trial scores
                        "cv_folds": cv_folds,
                        "total_trials": len(scores),
                        "total_elements": len(combo_results[combo_idx]["mcode_elements"]),
                        "errors": errors,
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {
                            "precision": avg_precision,
                            "recall": avg_recall,
                            "f1_score": avg_f1
                        },
                        "predicted_mcode": combo_results[combo_idx]["mcode_elements"]  # Include mCODE elements
                    }
                    optimization_results.append(result)

                    # Save individual run result
                    run_filename = f"run_{run_timestamp}_{combo['model']}_{combo['prompt'].replace('/', '_')}.json"
                    run_path = runs_dir / run_filename
                    with open(run_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"💾 Saved run result to: {run_path}")

                    # Track best result
                    if cv_average_score > best_score:
                        best_score = cv_average_score
                        best_result = result

                    self.logger.info(f"✅ Completed combination: {combo['model']} + {combo['prompt']} (CV score: {cv_average_score:.3f} ± {cv_std:.3f})")
                else:
                    # No successful trials for this combination
                    error_result = {
                        "combination": combo,
                        "success": False,
                        "error": f"All {len(errors)} trials failed: {errors[:3]}...",  # Show first 3 errors
                        "timestamp": datetime.now().isoformat(),
                    }
                    optimization_results.append(error_result)

                    # Save failed run result
                    run_filename = f"run_{run_timestamp}_{combo['model']}_{combo['prompt'].replace('/', '_')}_FAILED.json"
                    run_path = runs_dir / run_filename
                    with open(run_path, "w", encoding="utf-8") as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)
                    self.logger.info(f"💾 Saved failed run result to: {run_path}")

                    self.logger.error(f"❌ Failed combination {combo['model']} + {combo['prompt']}: All trials failed")

            # Set default LLM specification based on best result
            if best_result:
                self._set_default_llm_spec(best_result)

            # Save optimal settings if requested
            if output_config and best_result:
                self._save_optimal_config(best_result, output_config)

            # Save all processed mcode elements if requested
            save_mcode_elements = kwargs.get("save_mcode_elements")
            if save_mcode_elements:
                self._save_all_mcode_elements(combo_results, combinations, save_mcode_elements)

            # Final summary
            successful_combinations = len([r for r in optimization_results if r.get("success", False)])
            total_combinations = len(combinations)
            total_trials_processed = sum(len(r.get("fold_scores", [])) for r in optimization_results if r.get("success", False))

            self.logger.info(f"📊 Optimization complete!")
            self.logger.info(f"   ✅ Successful combinations: {successful_combinations}/{total_combinations}")
            self.logger.info(f"   📈 Total trials processed: {total_trials_processed}")
            if best_result:
                self.logger.info(f"   🏆 Best CV score: {best_score:.3f} ({best_result['combination']['model']} + {best_result['combination']['prompt']})")

            return self._create_result(
                success=len(optimization_results) > 0,
                data=optimization_results,
                metadata={
                    "total_combinations_tested": len(combinations),
                    "cv_folds": cv_folds,
                    "successful_tests": len(
                        [r for r in optimization_results if r.get("success", False)]
                    ),
                    "best_score": best_score,
                    "best_combination": (
                        best_result.get("combination") if best_result else None
                    ),
                    "config_saved": output_config is not None,
                },
            )

        except Exception as e:
            return self._handle_error(e, "trials optimization")

    def _create_kfold_splits(self, n_samples: int, n_folds: int) -> List[List[int]]:
        """
        Create k-fold cross validation splits.

        Args:
            n_samples: Total number of samples
            n_folds: Number of folds

        Returns:
            List of lists, where each inner list contains indices for that fold's validation set
        """
        indices = list(range(n_samples))
        fold_sizes = [n_samples // n_folds] * n_folds
        remainder = n_samples % n_folds

        # Distribute remainder across first few folds
        for i in range(remainder):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for size in fold_sizes:
            folds.append(indices[start:start + size])
            start += size

        return folds

    def _generate_combinations(
        self, prompts: List[str], models: List[str], max_combinations: int
    ) -> List[Dict[str, str]]:
        """Generate combinations of prompts and models to test."""
        combinations = []

        for prompt in prompts:
            for model in models:
                if max_combinations > 0 and len(combinations) >= max_combinations:
                    break
                combinations.append({"prompt": prompt, "model": model})

        return combinations

    async def _test_single_trial(
        self, combination: Dict[str, str], trial: Dict[str, Any], fold: int, combo_idx: int
    ) -> Dict[str, Any]:
        """Test a single trial with a specific prompt×model combination."""
        prompt_name = combination["prompt"]
        model_name = combination["model"]

        # Initialize performance tracking
        perf_metrics = PerformanceMetrics()
        perf_metrics.start_tracking()

        try:
            # Initialize pipeline with this combination
            pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)

            # Process single trial asynchronously
            result = await pipeline.process(trial)

            # Calculate quality metrics
            predicted = [elem.model_dump() for elem in result.mcode_mappings]
            num_elements = len(predicted)

            # Get token usage from global tracker
            from src.utils.token_tracker import global_token_tracker
            token_usage = global_token_tracker.get_total_usage()
            tokens_used = token_usage.total_tokens if token_usage else 0

            # Stop performance tracking
            perf_metrics.stop_tracking(tokens_used=tokens_used, elements_processed=num_elements)

            # Use number of mCODE elements as basic score (more elements = better performance)
            score = min(num_elements / 10.0, 1.0)  # Cap at 1.0

            # Get performance metrics
            perf_data = perf_metrics.get_metrics()

            # Enhanced quality metrics
            metrics = {
                "precision": score,
                "recall": score,
                "f1_score": score,
                "element_count": num_elements,
                **perf_data  # Include all performance metrics
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
            # Stop tracking even on failure
            perf_metrics.stop_tracking(tokens_used=0, elements_processed=0)

            return {
                "combination": combination,
                "combo_idx": combo_idx,
                "fold": fold,
                "success": False,
                "error": str(e),
                "score": 0.0,
                "performance_metrics": perf_metrics.to_dict(),
                "timestamp": datetime.now().isoformat(),
            }


    def _save_optimal_config(
        self, best_result: Dict[str, Any], output_path: str
    ) -> None:
        """Save optimal settings to configuration file."""
        try:
            config_data = {
                "optimal_settings": {
                    "model": best_result["combination"]["model"],
                    "prompt": best_result["combination"]["prompt"],
                    "cv_score": best_result.get("cv_average_score", best_result.get("average_score", 0)),
                    "cv_std": best_result.get("cv_std_score", 0),
                    "optimization_timestamp": datetime.now().isoformat(),
                    "optimizer_version": "2.0.0",
                },
                "metadata": {
                    "combinations_tested": best_result.get("combinations_tested", 0),
                    "cv_folds": best_result.get("cv_folds", 3),
                    "total_trials": best_result.get("total_trials", 0),
                    "fold_scores": best_result.get("fold_scores", []),
                    "metrics": best_result.get("metrics", {})
                },
            }

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved optimal config to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save optimal config: {e}")
            raise

    def _set_default_llm_spec(self, best_result: Dict[str, Any]) -> None:
        """Set the default LLM specification based on optimization results."""
        best_model = best_result["combination"]["model"]
        best_prompt = best_result["combination"]["prompt"]
        best_score = best_result.get("cv_average_score", best_result.get("average_score", 0))

        # Update the LLM config file
        llm_config_path = Path("src/config/llms_config.json")
        if not llm_config_path.exists():
            raise FileNotFoundError(f"LLM config file not found: {llm_config_path}")

        with open(llm_config_path, "r", encoding="utf-8") as f:
            llm_config = json.load(f)

        # Update the default model
        llm_config["models"]["default"] = best_model

        # Add optimization metadata
        if "optimization" not in llm_config["models"]:
            llm_config["models"]["optimization"] = {}

        llm_config["models"]["optimization"]["optimized_model"] = best_model
        llm_config["models"]["optimization"]["optimized_prompt"] = best_prompt
        llm_config["models"]["optimization"]["optimization_score"] = best_score
        llm_config["models"]["optimization"]["last_optimized"] = datetime.now().isoformat()

        with open(llm_config_path, "w", encoding="utf-8") as f:
            json.dump(llm_config, f, indent=2, ensure_ascii=False)

        # Update the prompts config file
        prompts_config_path = Path("src/config/prompts_config.json")
        if not prompts_config_path.exists():
            raise FileNotFoundError(f"Prompts config file not found: {prompts_config_path}")

        with open(prompts_config_path, "r", encoding="utf-8") as f:
            prompts_config = json.load(f)

        # Update the default prompt
        prompts_config["prompts"]["default"] = best_prompt

        # Add optimization metadata to prompts config
        if "optimization" not in prompts_config["prompts"]:
            prompts_config["prompts"]["optimization"] = {}

        prompts_config["prompts"]["optimization"]["optimized_prompt"] = best_prompt
        prompts_config["prompts"]["optimization"]["optimization_score"] = best_score
        prompts_config["prompts"]["optimization"]["last_optimized"] = datetime.now().isoformat()

        with open(prompts_config_path, "w", encoding="utf-8") as f:
            json.dump(prompts_config, f, indent=2, ensure_ascii=False)

        self.logger.info(f"🔧 Updated defaults - Model: {best_model} (in {llm_config_path}), Prompt: {best_prompt} (in {prompts_config_path}), CV score: {best_score:.3f}")

    def get_available_prompts(self) -> List[str]:
        """Get list of available prompt templates."""
        # This would typically scan the prompts directory
        return [
            "direct_mcode_evidence_based_concise",
            "direct_mcode_evidence_based",
            "direct_mcode_minimal",
            "direct_mcode_structured",
        ]

    def get_available_models(self) -> List[str]:
        """Get list of available LLM models."""
        # This would typically come from configuration
        return [
            # OpenAI models
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # DeepSeek models
            "deepseek-coder",
            "deepseek-chat",
            "deepseek-reasoner",
            # Other models (for compatibility)
            "claude-3",
            "llama-3"
        ]

    def validate_combination(self, prompt: str, model: str) -> bool:
        """
        Validate that a prompt×model combination is valid.

        Args:
            prompt: Prompt template name
            model: Model name

        Returns:
            bool: True if combination is valid
        """
        available_prompts = self.get_available_prompts()
        available_models = self.get_available_models()

        return prompt in available_prompts and model in available_models

    def summarize_benchmark_validations(self, runs_dir: str = "optimization_runs") -> Dict[str, Any]:
        """
        Review all saved benchmark validations and pick the best one.

        Args:
            runs_dir: Directory containing saved run results

        Returns:
            Dict containing summary of all runs and the best combination
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return {
                "success": False,
                "error": f"Runs directory not found: {runs_dir}",
                "total_runs": 0,
                "best_result": None
            }

        all_results = []
        best_result = None
        best_score = 0

        # Load all run files
        for run_file in runs_path.glob("run_*.json"):
            try:
                with open(run_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    all_results.append(result)

                    # Check if this is the best result
                    if result.get("success") and result.get("cv_average_score", 0) > best_score:
                        best_score = result["cv_average_score"]
                        best_result = result

            except Exception as e:
                self.logger.warning(f"Failed to load run file {run_file}: {e}")

        if not all_results:
            return {
                "success": False,
                "error": "No valid run files found",
                "total_runs": 0,
                "best_result": None
            }

        # Calculate summary statistics
        successful_runs = [r for r in all_results if r.get("success")]
        failed_runs = [r for r in all_results if not r.get("success")]

        scores = [r.get("cv_average_score", 0) for r in successful_runs]
        avg_score = sum(scores) / len(scores) if scores else 0
        std_score = (sum((s - avg_score) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 0

        # Calculate precision, recall, and F1 statistics
        precision_scores = [r["metrics"].get("precision", 0) for r in successful_runs if "metrics" in r]
        recall_scores = [r["metrics"].get("recall", 0) for r in successful_runs if "metrics" in r]
        f1_scores = [r["metrics"].get("f1_score", 0) for r in successful_runs if "metrics" in r]

        # Calculate averages for detailed metrics
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

        # Calculate standard deviations
        precision_std = (sum((p - avg_precision) ** 2 for p in precision_scores) / len(precision_scores)) ** 0.5 if precision_scores else 0
        recall_std = (sum((r - avg_recall) ** 2 for r in recall_scores) / len(recall_scores)) ** 0.5 if recall_scores else 0
        f1_std = (sum((f - avg_f1) ** 2 for f in f1_scores) / len(f1_scores)) ** 0.5 if f1_scores else 0

        # Aggregate performance metrics using PerformanceMetrics class
        all_perf_metrics = []
        for r in successful_runs:
            if "performance_metrics" in r:
                all_perf_metrics.append(r["performance_metrics"])

        # Calculate aggregated performance statistics
        if all_perf_metrics:
            avg_processing_time = sum(m.get("processing_time_seconds", 0) for m in all_perf_metrics) / len(all_perf_metrics)
            avg_tokens_used = sum(m.get("tokens_used", 0) for m in all_perf_metrics) / len(all_perf_metrics)
            avg_cost = sum(m.get("estimated_cost_usd", 0) for m in all_perf_metrics) / len(all_perf_metrics)
            avg_elements_per_second = sum(m.get("elements_per_second", 0) for m in all_perf_metrics) / len(all_perf_metrics)
            avg_tokens_per_second = sum(m.get("tokens_per_second", 0) for m in all_perf_metrics) / len(all_perf_metrics)

            # Calculate standard deviations
            processing_times = [m.get("processing_time_seconds", 0) for m in all_perf_metrics]
            tokens_list = [m.get("tokens_used", 0) for m in all_perf_metrics]
            costs_list = [m.get("estimated_cost_usd", 0) for m in all_perf_metrics]
            elements_per_second_list = [m.get("elements_per_second", 0) for m in all_perf_metrics]
            tokens_per_second_list = [m.get("tokens_per_second", 0) for m in all_perf_metrics]

            processing_time_std = (sum((t - avg_processing_time) ** 2 for t in processing_times) / len(processing_times)) ** 0.5 if processing_times else 0
            tokens_std = (sum((t - avg_tokens_used) ** 2 for t in tokens_list) / len(tokens_list)) ** 0.5 if tokens_list else 0
            cost_std = (sum((c - avg_cost) ** 2 for c in costs_list) / len(costs_list)) ** 0.5 if costs_list else 0
        else:
            avg_processing_time = avg_tokens_used = avg_cost = avg_elements_per_second = avg_tokens_per_second = 0
            processing_time_std = tokens_std = cost_std = 0
            processing_times = tokens_list = costs_list = elements_per_second_list = tokens_per_second_list = []

        summary = {
            "success": True,
            "total_runs": len(all_results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "average_score": avg_score,
            "std_score": std_score,
            # Detailed metrics
            "precision": {
                "mean": avg_precision,
                "std": precision_std,
                "count": len(precision_scores)
            },
            "recall": {
                "mean": avg_recall,
                "std": recall_std,
                "count": len(recall_scores)
            },
            "f1_score": {
                "mean": avg_f1,
                "std": f1_std,
                "count": len(f1_scores)
            },
            # Time and token analysis
            "processing_time": {
                "mean_seconds": avg_processing_time,
                "std_seconds": processing_time_std,
                "total_measurements": len(processing_times),
                "min_seconds": min(processing_times) if processing_times else 0,
                "max_seconds": max(processing_times) if processing_times else 0
            },
            "token_usage": {
                "mean_tokens": avg_tokens_used,
                "std_tokens": tokens_std,
                "total_measurements": len(tokens_list),
                "min_tokens": min(tokens_list) if tokens_list else 0,
                "max_tokens": max(tokens_list) if tokens_list else 0
            },
            "cost_analysis": {
                "mean_cost_usd": avg_cost,
                "std_cost_usd": cost_std,
                "total_measurements": len(costs_list),
                "min_cost_usd": min(costs_list) if costs_list else 0,
                "max_cost_usd": max(costs_list) if costs_list else 0,
                "total_estimated_cost_usd": sum(costs_list) if costs_list else 0
            },
            "performance_metrics": {
                "elements_per_second": {
                    "mean": avg_elements_per_second,
                    "total_measurements": len(elements_per_second_list)
                },
                "tokens_per_second": {
                    "mean": avg_tokens_per_second,
                    "total_measurements": len(tokens_per_second_list)
                }
            },
            "best_result": best_result,
            "all_results": all_results,
            "summary_timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"📊 Benchmark Summary:")
        self.logger.info(f"   📁 Total runs: {len(all_results)}")
        self.logger.info(f"   ✅ Successful: {len(successful_runs)}")
        self.logger.info(f"   ❌ Failed: {len(failed_runs)}")
        self.logger.info(f"   📈 Average score: {avg_score:.3f} ± {std_score:.3f}")

        # Log detailed metrics if available
        if precision_scores:
            self.logger.info(f"   🎯 Precision: {avg_precision:.3f} ± {precision_std:.3f} (n={len(precision_scores)})")
        if recall_scores:
            self.logger.info(f"   📊 Recall: {avg_recall:.3f} ± {recall_std:.3f} (n={len(recall_scores)})")
        if f1_scores:
            self.logger.info(f"   🏆 F1 Score: {avg_f1:.3f} ± {f1_std:.3f} (n={len(f1_scores)})")

        if best_result:
            combo = best_result.get("combination", {})
            self.logger.info(f"   🏆 Best: {combo.get('model')} + {combo.get('prompt')} (score: {best_score:.3f})")

        return summary

    def _save_all_mcode_elements(
        self, combo_results: Dict[int, Dict], combinations: List[Dict[str, str]], output_path: str
    ) -> None:
        """Save all processed mCODE elements from optimization."""
        try:
            all_mcode_data = {
                "optimization_summary": {
                    "total_combinations": len(combinations),
                    "timestamp": datetime.now().isoformat(),
                    "combinations": combinations
                },
                "mcode_elements": {}
            }

            for combo_idx, combo in enumerate(combinations):
                combo_key = f"{combo['model']}_{combo['prompt']}"
                all_mcode_data["mcode_elements"][combo_key] = {
                    "combination": combo,
                    "predicted_mcode": combo_results[combo_idx]["mcode_elements"],
                    "scores": combo_results[combo_idx]["scores"],
                    "metrics": combo_results[combo_idx]["metrics"]
                }

            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_mcode_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"💾 Saved all mCODE elements to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to save all mCODE elements: {e}")
            raise
