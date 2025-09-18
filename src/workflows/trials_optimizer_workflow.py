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
from src.utils.concurrency import TaskQueue, create_task, get_optimizer_pool, create_task_queue_from_args

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

    def execute(self, **kwargs) -> WorkflowResult:
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

            # Generate combinations to test
            combinations = self._generate_combinations(
                prompts, models, max_combinations
            )

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
                task_queue = create_task_queue_from_args(cli_args, "optimizer")
                workers = task_queue.worker_pool.max_workers
            else:
                # Fallback to default optimizer pool
                task_queue = TaskQueue(max_workers=get_optimizer_pool().max_workers, name="OptimizerQueue")
                workers = task_queue.worker_pool.max_workers

            self.logger.info(f"🚀 Using {workers} concurrent workers for optimization")

            # Create tasks for fully asynchronous execution
            # Break down to individual trial processing for maximum parallelism
            tasks = []
            task_id = 0

            for combo_idx, combo in enumerate(combinations):
                self.logger.debug(f"📋 Creating tasks for combination {combo_idx + 1}: {combo['model']} + {combo['prompt']}")
                for fold in range(cv_folds):
                    # Get validation trials for this fold
                    val_indices = fold_indices[fold]
                    val_trials = [trials_data[i] for i in val_indices]

                    # Create individual tasks for each trial in this fold
                    for trial_idx, trial in enumerate(val_trials):
                        task = create_task(
                            task_id=f"trial_{task_id}",
                            func=self._test_single_trial,
                            combination=combo,
                            trial=trial,
                            fold=fold,
                            combo_idx=combo_idx
                        )
                        tasks.append(task)
                        task_id += 1

            self.logger.info(f"✅ Created {len(tasks)} tasks for execution")
            if len(tasks) != total_tasks:
                self.logger.warning(f"⚠️  Task count mismatch: expected {total_tasks}, created {len(tasks)}")

            # Track results by combination for aggregation
            combo_results = {i: {"scores": [], "errors": []} for i in range(len(combinations))}

            def progress_callback(completed, total, result):
                if result.success:
                    combo_idx = result.result.get('combo_idx', 0)
                    score = result.result.get('score', 0)
                    combo = combinations[combo_idx]
                    combo_results[combo_idx]["scores"].append(score)
                    fold = result.result.get('fold', 0)
                    self.logger.info(f"✅ Trial {completed}/{total}: {combo['model']} + {combo['prompt']} (fold {fold}, score: {score:.3f})")
                else:
                    combo_idx = result.result.get('combo_idx', 0) if result.result else 0
                    combo_results[combo_idx]["errors"].append(str(result.error))
                    combo_name = combinations[combo_idx]['model'] + " + " + combinations[combo_idx]['prompt'] if combo_idx < len(combinations) else "unknown"
                    self.logger.warning(f"❌ Failed trial {completed}/{total}: {combo_name} - {result.error}")

            task_results = task_queue.execute_tasks(tasks, progress_callback=progress_callback)

            # Aggregate results by combination
            for combo_idx, combo in enumerate(combinations):
                scores = combo_results[combo_idx]["scores"]
                errors = combo_results[combo_idx]["errors"]

                if scores:
                    # Calculate CV statistics for this combination
                    cv_average_score = sum(scores) / len(scores)
                    cv_std = (sum((s - cv_average_score) ** 2 for s in scores) / len(scores)) ** 0.5

                    result = {
                        "combination": combo,
                        "success": True,
                        "cv_average_score": cv_average_score,
                        "cv_std_score": cv_std,
                        "fold_scores": scores,  # Individual trial scores
                        "cv_folds": cv_folds,
                        "total_trials": len(scores),
                        "errors": errors,
                        "timestamp": datetime.now().isoformat(),
                    }
                    optimization_results.append(result)

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
                    self.logger.error(f"❌ Failed combination {combo['model']} + {combo['prompt']}: All trials failed")

            # Set default LLM specification based on best result
            if best_result:
                self._set_default_llm_spec(best_result)

            # Save optimal settings if requested
            if output_config and best_result:
                self._save_optimal_config(best_result, output_config)

            # Final summary
            successful_combinations = len([r for r in optimization_results if r.get("success", False)])
            total_combinations = len(combinations)
            total_trials_processed = sum(len(r.get("fold_scores", [])) for r in optimization_results if r.get("success", False))

            self.logger.info(f"📊 Optimization complete!")
            self.logger.info(f"   ✅ Successful combinations: {successful_combinations}/{total_combinations}")
            self.logger.info(f"   📈 Total trials processed: {total_trials_processed}")
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
                if len(combinations) >= max_combinations:
                    break
                combinations.append({"prompt": prompt, "model": model})

        return combinations

    def _test_single_trial(
        self, combination: Dict[str, str], trial: Dict[str, Any], fold: int, combo_idx: int
    ) -> Dict[str, Any]:
        """Test a single trial with a specific prompt×model combination."""
        prompt_name = combination["prompt"]
        model_name = combination["model"]

        try:
            # Initialize pipeline with this combination
            pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)

            # Process single trial
            result = pipeline.process(trial)

            # Calculate quality score
            score = self._calculate_quality_score(result)

            return {
                "combination": combination,
                "combo_idx": combo_idx,
                "fold": fold,
                "trial_score": score,
                "score": score,  # For aggregation
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "combination": combination,
                "combo_idx": combo_idx,
                "fold": fold,
                "success": False,
                "error": str(e),
                "score": 0.0,  # Failed trials get 0 score
                "timestamp": datetime.now().isoformat(),
            }

    def _test_combination_cv(
        self, combination: Dict[str, str], trials_data: List[Dict[str, Any]], cv_folds: int
    ) -> Dict[str, Any]:
        """Test a specific prompt×model combination using cross validation (legacy method)."""
        prompt_name = combination["prompt"]
        model_name = combination["model"]

        try:
            # Initialize pipeline with this combination
            pipeline = McodePipeline(prompt_name=prompt_name, model_name=model_name)

            # Perform k-fold cross validation
            fold_scores = []
            fold_sizes = []

            for fold in range(cv_folds):
                # Split data into train/validation for this fold
                val_size = max(1, len(trials_data) // cv_folds)
                val_start = fold * val_size
                val_end = min((fold + 1) * val_size, len(trials_data))

                val_trials = trials_data[val_start:val_end]
                fold_sizes.append(len(val_trials))

                # Test on validation trials
                fold_trial_scores = []
                for trial in val_trials:
                    try:
                        # Process trial
                        result = pipeline.process(trial)

                        # Calculate quality score
                        score = self._calculate_quality_score(result)
                        fold_trial_scores.append(score)

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to process trial in fold {fold} with {model_name}: {e}"
                        )
                        fold_trial_scores.append(0.0)

                # Average score for this fold
                fold_avg_score = sum(fold_trial_scores) / len(fold_trial_scores) if fold_trial_scores else 0.0
                fold_scores.append(fold_avg_score)

            # Calculate cross validation statistics
            cv_average_score = sum(fold_scores) / len(fold_scores) if fold_scores else 0.0
            cv_std = (sum((s - cv_average_score) ** 2 for s in fold_scores) / len(fold_scores)) ** 0.5 if fold_scores else 0.0

            return {
                "combination": combination,
                "success": True,
                "cv_average_score": cv_average_score,
                "cv_std_score": cv_std,
                "fold_scores": fold_scores,
                "fold_sizes": fold_sizes,
                "cv_folds": cv_folds,
                "total_trials": len(trials_data),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "combination": combination,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _test_combination(
        self, combination: Dict[str, str], trials_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test a specific prompt×model combination (legacy method for backward compatibility)."""
        return self._test_combination_cv(combination, trials_data, cv_folds=3)

    def _calculate_quality_score(self, result) -> float:
        """
        Calculate a quality score for mCODE processing results.

        This is a simplified scoring function. In practice, this would
        use more sophisticated metrics.
        """
        try:
            # Base score from number of mappings
            mappings_count = (
                len(result.mcode_mappings) if hasattr(result, "mcode_mappings") else 0
            )

            # Bonus for validation results
            validation_bonus = 0
            if hasattr(result, "validation_results") and result.validation_results:
                # validation_results is a ValidationResult object, not a dict
                compliance = getattr(result.validation_results, "compliance_score", 0)
                validation_bonus = compliance * 0.2

            # Bonus for source references
            reference_bonus = 0
            if hasattr(result, "source_references") and result.source_references:
                reference_bonus = min(len(result.source_references) * 0.1, 0.3)

            # Calculate total score (0-1 scale)
            base_score = min(
                mappings_count / 20.0, 1.0
            )  # Expect ~20 mappings for good quality
            total_score = min(base_score + validation_bonus + reference_bonus, 1.0)

            return total_score

        except Exception as e:
            self.logger.warning(f"Error calculating quality score: {e}")
            return 0.0

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
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            # DeepSeek models
            "deepseek-coder",
            "deepseek-chat",
            "deepseek-instruction",
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
